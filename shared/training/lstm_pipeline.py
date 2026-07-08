from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from shared.cluster.model import fit_embedding_pca_features, transform_embedding_pca_features
from shared.common.utils import write_json
from shared.training.metrics import (
    build_supervised_frame,
    compute_signal_return_metrics,
    compute_strong_regime_metrics,
    compute_thresholded_binary_direction_metrics,
    compute_two_stage_signal_metrics,
    filter_feature_frame_by_min_date,
    serialize_timestamp,
    seed_everything,
    split_supervised_frame,
    split_supervised_frame_at_date,
    to_serializable_config,
)


class _LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.event_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.direction_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.return_head = nn.Linear(hidden_size, 1)
        self.event_head = nn.Linear(hidden_size, 1)
        self.direction_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        event_out, _ = self.event_lstm(x)
        direction_out, _ = self.direction_lstm(x)
        event_hidden = event_out[:, -1, :]
        direction_hidden = direction_out[:, -1, :]
        return (
            torch.nn.functional.softplus(self.return_head(event_hidden).squeeze(-1)),
            self.event_head(event_hidden).squeeze(-1),
            self.direction_head(direction_hidden).squeeze(-1),
        )


class _EventLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.event_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.event_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        event_out, _ = self.event_lstm(x)
        return self.event_head(event_out[:, -1, :]).squeeze(-1)


class _DirectionLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.direction_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.direction_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        direction_out, _ = self.direction_lstm(x)
        return self.direction_head(direction_out[:, -1, :]).squeeze(-1)


def _build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(X)
    if n < seq_len:
        raise ValueError(f"Not enough rows ({n}) for seq_len={seq_len}.")
    seqs = np.stack([X[i : i + seq_len] for i in range(n - seq_len + 1)])
    targets = y[seq_len - 1:]
    return seqs, targets


def _balanced_accuracy(
    predicted_positive: torch.Tensor,
    actual_positive: torch.Tensor,
) -> float:
    actual_positive = actual_positive.bool()
    predicted_positive = predicted_positive.bool()
    positive_count = int(actual_positive.sum().item())
    negative_count = int((~actual_positive).sum().item())
    if positive_count == 0 or negative_count == 0:
        return 0.0
    positive_recall = float(
        (predicted_positive & actual_positive).sum().item() / positive_count
    )
    negative_recall = float(
        ((~predicted_positive) & (~actual_positive)).sum().item() / negative_count
    )
    return (positive_recall + negative_recall) / 2.0


def _calibrate_probability_threshold(
    probabilities: torch.Tensor,
    actual_positive: torch.Tensor,
) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = float("-inf")
    for threshold in np.arange(0.20, 0.801, 0.01):
        score = _balanced_accuracy(probabilities >= threshold, actual_positive)
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12
            and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_threshold = float(threshold)
            best_score = score
    return best_threshold, best_score


def _calibrate_event_probability_threshold(
    probabilities: torch.Tensor,
    actual_positive: torch.Tensor,
    minimum_recall: float,
) -> tuple[float, dict]:
    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError("minimum_recall must be between 0 and 1.")

    actual_positive = actual_positive.bool()
    positive_count = int(actual_positive.sum().item())
    negative_count = int((~actual_positive).sum().item())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Event threshold calibration requires both event classes.")

    best: tuple[tuple[float, float, float], float, dict] | None = None
    for threshold in np.arange(0.10, 0.901, 0.01):
        predicted_positive = probabilities >= threshold
        predicted_count = int(predicted_positive.sum().item())
        if predicted_count == 0:
            continue
        true_positive = int((predicted_positive & actual_positive).sum().item())
        true_negative = int(((~predicted_positive) & (~actual_positive)).sum().item())
        precision = true_positive / predicted_count
        recall = true_positive / positive_count
        specificity = true_negative / negative_count
        balanced = (recall + specificity) / 2.0
        if recall + 1e-12 < minimum_recall:
            continue
        score = (precision, balanced, -abs(float(threshold) - 0.5))
        details = {
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "balanced_accuracy": float(balanced),
        }
        if best is None or score > best[0]:
            best = (score, float(threshold), details)

    if best is None:
        threshold, balanced = _calibrate_probability_threshold(
            probabilities,
            actual_positive,
        )
        predicted_positive = probabilities >= threshold
        predicted_count = int(predicted_positive.sum().item())
        true_positive = int((predicted_positive & actual_positive).sum().item())
        return threshold, {
            "precision": float(true_positive / predicted_count) if predicted_count else 0.0,
            "recall": float(true_positive / positive_count),
            "specificity": float(
                ((~predicted_positive) & (~actual_positive)).sum().item() / negative_count
            ),
            "balanced_accuracy": float(balanced),
        }

    return best[1], best[2]


def _binary_ranking_metrics(
    probabilities: torch.Tensor,
    actual_positive: torch.Tensor,
) -> tuple[float, float]:
    probability_values = probabilities.detach().cpu().numpy()
    target_values = actual_positive.detach().cpu().numpy().astype(int)
    if len(np.unique(target_values)) != 2:
        return 0.5, float(target_values.mean())
    return (
        float(roc_auc_score(target_values, probability_values)),
        float(average_precision_score(target_values, probability_values)),
    )


def _event_checkpoint_score(
    event_roc_auc: float,
    event_pr_auc: float,
    threshold_metrics: dict,
    objective: str,
) -> tuple[tuple[float, ...], float, str]:
    normalized = str(objective).lower()
    if normalized in {"ranking", "auc_pr", "auc_pr_average"}:
        score = 0.5 * event_roc_auc + 0.5 * event_pr_auc
        return (score,), score, "0.5*event_roc_auc+0.5*event_pr_auc"
    if normalized in {
        "precision_at_recall",
        "precision_at_min_recall",
        "precision_recall_floor",
    }:
        precision = float(threshold_metrics["precision"])
        balanced = float(threshold_metrics["balanced_accuracy"])
        recall = float(threshold_metrics["recall"])
        return (
            precision,
            balanced,
            event_pr_auc,
            event_roc_auc,
            recall,
        ), precision, "event_precision_at_min_recall"
    raise ValueError(
        "lstm_event_selection_objective must be one of: "
        "ranking, precision_at_recall"
    )


def _resolve_device(config) -> torch.device:
    requested = str(config.lstm_device).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("lstm_device='cuda' was requested, but CUDA is unavailable.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("lstm_device must be one of: cpu, cuda, auto.")
    return torch.device(requested)


def _train_lstm(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    sequence_origin_dates: np.ndarray,
    sequence_target_dates: np.ndarray,
    config,
) -> tuple[_LSTMRegressor, dict]:
    seed_everything(config.random_seed)

    device = _resolve_device(config)
    model = _LSTMRegressor(
        input_size=X_train_seq.shape[2],
        hidden_size=config.lstm_hidden_size,
        num_layers=config.lstm_num_layers,
        dropout=config.lstm_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lstm_learning_rate,
        weight_decay=config.lstm_weight_decay,
    )
    huber = nn.HuberLoss(delta=config.lstm_huber_delta)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    val_size = max(1, int(len(X_train_seq) * 0.1))
    val_start = len(X_train_seq) - val_size
    validation_start_date = sequence_origin_dates[val_start]
    fit_mask = sequence_target_dates[:val_start] < validation_start_date
    if not bool(fit_mask.any()):
        raise ValueError("No training sequences remain after validation purge.")

    X_tr = torch.as_tensor(X_train_seq[:val_start][fit_mask], dtype=torch.float32, device=device)
    y_tr = torch.as_tensor(y_train_seq[:val_start][fit_mask], dtype=torch.float32, device=device)
    X_va = torch.as_tensor(X_train_seq[val_start:], dtype=torch.float32, device=device)
    y_va = torch.as_tensor(y_train_seq[val_start:], dtype=torch.float32, device=device)

    threshold = float(config.lstm_direction_return_threshold)
    train_event = y_tr.abs() > threshold
    train_direction = y_tr > 0
    event_counts = torch.bincount(train_event.long(), minlength=2)
    direction_counts = torch.bincount(train_direction[train_event].long(), minlength=2)
    if bool((event_counts == 0).any()) or bool((direction_counts == 0).any()):
        raise ValueError("Two-stage training requires both event classes and both directions.")
    event_weights = len(y_tr) / (2.0 * event_counts.float())
    direction_weights = int(train_event.sum()) / (2.0 * direction_counts.float())

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, y_tr),
        batch_size=config.lstm_batch_size,
        shuffle=False,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[LSTM] device={device.type} parameters={parameter_count:,} "
        f"train={len(X_tr):,} validation={len(X_va):,} "
        f"threshold=+/-{threshold:g}% strong_train={int(train_event.sum()):,} "
        f"loss={config.lstm_return_loss_weight:g}*magnitude_huber+"
        f"{config.lstm_event_loss_weight:g}*event_bce+"
        f"{config.lstm_direction_loss_weight:g}*direction_bce",
        flush=True,
    )

    best_score_key: tuple[float, ...] | None = None
    best_score = float("-inf")
    best_val_loss = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    patience = 0
    history: list[dict] = []
    selection_label = "0.5*event_roc_auc+0.5*event_pr_auc"

    def loss_parts(
        return_pred: torch.Tensor,
        event_logit: torch.Tensor,
        direction_logit: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        event_target = (target.abs() > threshold).float()
        direction_target = (target > 0).float()
        return_loss = huber(return_pred, target.abs())
        raw_event = bce(event_logit, event_target)
        event_loss = (raw_event * event_weights[event_target.long()]).mean()
        event_mask = event_target.bool()
        if bool(event_mask.any()):
            raw_direction = bce(
                direction_logit[event_mask],
                direction_target[event_mask],
            )
            direction_loss = (
                raw_direction
                * direction_weights[direction_target[event_mask].long()]
            ).mean()
        else:
            direction_loss = direction_logit.sum() * 0.0
        total = (
            config.lstm_return_loss_weight * return_loss
            + config.lstm_event_loss_weight * event_loss
            + config.lstm_direction_loss_weight * direction_loss
        )
        return total, return_loss, event_loss, direction_loss

    for epoch in range(1, config.lstm_epochs + 1):
        model.train()
        train_total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            outputs = model(xb)
            total, _, _, _ = loss_parts(*outputs, yb)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_total += float(total.item()) * len(xb)

        model.eval()
        with torch.no_grad():
            val_return, val_event_logit, val_direction_logit = model(X_va)
            val_total, val_return_loss, val_event_loss, val_direction_loss = loss_parts(
                val_return,
                val_event_logit,
                val_direction_logit,
                y_va,
            )
            val_event_target = y_va.abs() > threshold
            val_direction_target = y_va > 0
            val_event_probability = torch.sigmoid(val_event_logit)
            validation_event_threshold, event_threshold_metrics = (
                _calibrate_event_probability_threshold(
                    val_event_probability,
                    val_event_target,
                    config.lstm_event_min_recall,
                )
            )
            event_balanced = event_threshold_metrics["balanced_accuracy"]
            event_roc_auc, event_pr_auc = _binary_ranking_metrics(
                val_event_probability,
                val_event_target,
            )
            validation_direction_threshold, direction_balanced = _calibrate_probability_threshold(
                torch.sigmoid(val_direction_logit[val_event_target]),
                val_direction_target[val_event_target],
            )
            event_score_key, event_selection_score, selection_label = (
                _event_checkpoint_score(
                    event_roc_auc,
                    event_pr_auc,
                    event_threshold_metrics,
                    config.lstm_event_selection_objective,
                )
            )

        val_loss_value = float(val_total.item())
        improved = best_score_key is None or event_score_key > best_score_key or (
            event_score_key == best_score_key
            and val_loss_value < best_val_loss
        )
        if improved:
            best_score_key = event_score_key
            best_score = event_selection_score
            best_val_loss = val_loss_value
            best_epoch = epoch
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_total / len(X_tr),
                "validation_loss": val_loss_value,
                "validation_return_loss": float(val_return_loss.item()),
                "validation_event_loss": float(val_event_loss.item()),
                "validation_direction_loss": float(val_direction_loss.item()),
                "validation_event_roc_auc": event_roc_auc,
                "validation_event_pr_auc": event_pr_auc,
                "validation_event_selection_score": event_selection_score,
                "validation_event_selection_key": list(event_score_key),
                "validation_event_selection_objective": selection_label,
                "validation_event_precision": event_threshold_metrics["precision"],
                "validation_event_recall": event_threshold_metrics["recall"],
                "validation_event_balanced_accuracy": event_balanced,
                "validation_direction_balanced_accuracy": direction_balanced,
                "validation_event_probability_threshold": validation_event_threshold,
                "validation_direction_probability_threshold": validation_direction_threshold,
                "patience_counter": patience,
            }
        )
        marker = " *" if improved else ""
        if getattr(config, "verbose_output", False):
            print(
                f"[LSTM] epoch {epoch:03d}/{config.lstm_epochs:03d} "
                f"train={train_total / len(X_tr):.6f} val={val_loss_value:.6f} "
                f"event_auc={event_roc_auc * 100:.1f}% "
                f"event_pr={event_pr_auc * 100:.1f}% "
                f"event_precision={event_threshold_metrics['precision'] * 100:.1f}% "
                f"event_recall={event_threshold_metrics['recall'] * 100:.1f}% "
                f"patience={patience}/{config.lstm_early_stopping_patience}{marker}",
                flush=True,
            )
        if patience >= config.lstm_early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError("Two-stage LSTM did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, val_event_logit, val_direction_logit = model(X_va)
        val_event_target = y_va.abs() > threshold
        val_direction_target = y_va > 0
        val_event_probability = torch.sigmoid(val_event_logit)
        calibrated_event_threshold, calibrated_event_metrics = (
            _calibrate_event_probability_threshold(
                val_event_probability,
                val_event_target,
                config.lstm_event_min_recall,
            )
        )
        calibrated_event_roc_auc, calibrated_event_pr_auc = _binary_ranking_metrics(
            val_event_probability,
            val_event_target,
        )
        calibrated_direction_threshold, calibrated_direction_score = (
            _calibrate_probability_threshold(
                torch.sigmoid(val_direction_logit[val_event_target]),
                val_direction_target[val_event_target],
            )
        )
    event_probability_threshold = (
        float(config.lstm_event_probability_threshold)
        if config.lstm_event_probability_threshold is not None
        else calibrated_event_threshold
    )
    direction_probability_threshold = (
        float(config.lstm_direction_probability_threshold)
        if config.lstm_direction_probability_threshold is not None
        else calibrated_direction_threshold
    )
    print(
        f"[LSTM] completed best_epoch={best_epoch} best_event_score={best_score * 100:.1f}% "
        f"event_threshold={event_probability_threshold:.2f} "
        f"direction_threshold={direction_probability_threshold:.2f}",
        flush=True,
    )
    return model, {
        "device": str(device),
        "parameter_count": int(parameter_count),
        "epochs_requested": int(config.lstm_epochs),
        "epochs_completed": int(len(history)),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_val_loss),
        "best_validation_event_score": float(best_score),
        "model_selection_objective": selection_label,
        "model_selection_score_key": list(best_score_key or ()),
        "calibrated_event_roc_auc": float(calibrated_event_roc_auc),
        "calibrated_event_pr_auc": float(calibrated_event_pr_auc),
        "calibrated_event_precision": calibrated_event_metrics["precision"],
        "calibrated_event_recall": calibrated_event_metrics["recall"],
        "calibrated_event_balanced_accuracy": calibrated_event_metrics[
            "balanced_accuracy"
        ],
        "calibrated_direction_balanced_accuracy": float(calibrated_direction_score),
        "event_probability_threshold": event_probability_threshold,
        "direction_probability_threshold": direction_probability_threshold,
        "direction_return_threshold_pct": threshold,
        "event_minimum_recall": float(config.lstm_event_min_recall),
        "loss_function": (
            "magnitude_huber_plus_event_bce_plus_thresholded_direction_bce"
        ),
        "huber_delta": float(config.lstm_huber_delta),
        "return_loss_weight": float(config.lstm_return_loss_weight),
        "event_loss_weight": float(config.lstm_event_loss_weight),
        "direction_loss_weight": float(config.lstm_direction_loss_weight),
        "event_class_counts": event_counts.tolist(),
        "direction_class_counts": direction_counts.tolist(),
        "fit_sequences": int(len(X_tr)),
        "validation_sequences": int(len(X_va)),
        "validation_boundary_purged_sequences": int(val_start - int(fit_mask.sum())),
        "stopped_early": len(history) < config.lstm_epochs,
        "history": history,
    }


def _train_event_lstm(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    sequence_origin_dates: np.ndarray,
    sequence_target_dates: np.ndarray,
    config,
) -> tuple[_EventLSTMClassifier, dict]:
    seed_everything(config.random_seed)
    device = _resolve_device(config)
    model = _EventLSTMClassifier(
        input_size=X_train_seq.shape[2],
        hidden_size=config.lstm_hidden_size,
        num_layers=config.lstm_num_layers,
        dropout=config.lstm_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lstm_learning_rate,
        weight_decay=config.lstm_weight_decay,
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")

    val_size = max(1, int(len(X_train_seq) * 0.1))
    val_start = len(X_train_seq) - val_size
    validation_start_date = sequence_origin_dates[val_start]
    fit_mask = sequence_target_dates[:val_start] < validation_start_date
    if not bool(fit_mask.any()):
        raise ValueError("No Event LSTM sequences remain after validation purge.")

    X_tr = torch.as_tensor(
        X_train_seq[:val_start][fit_mask], dtype=torch.float32, device=device
    )
    y_tr = torch.as_tensor(
        y_train_seq[:val_start][fit_mask], dtype=torch.float32, device=device
    )
    X_va = torch.as_tensor(X_train_seq[val_start:], dtype=torch.float32, device=device)
    y_va = torch.as_tensor(y_train_seq[val_start:], dtype=torch.float32, device=device)

    threshold = float(config.lstm_direction_return_threshold)
    train_event = y_tr.abs() > threshold
    event_counts = torch.bincount(train_event.long(), minlength=2)
    if bool((event_counts == 0).any()):
        raise ValueError("Event LSTM training requires both event classes.")
    event_weights = len(y_tr) / (2.0 * event_counts.float())
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, train_event.float()),
        batch_size=config.lstm_batch_size,
        shuffle=False,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[EVENT-LSTM] device={device.type} parameters={parameter_count:,} "
        f"train={len(X_tr):,} validation={len(X_va):,} "
        f"threshold=+/-{threshold:g}% events={int(train_event.sum()):,} "
        "loss=class_weighted_event_bce",
        flush=True,
    )

    best_score_key: tuple[float, ...] | None = None
    best_score = float("-inf")
    best_val_loss = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    patience = 0
    history: list[dict] = []
    val_event_target = y_va.abs() > threshold
    selection_label = "0.5*event_roc_auc+0.5*event_pr_auc"

    for epoch in range(1, config.lstm_epochs + 1):
        model.train()
        train_total = 0.0
        for xb, event_target in loader:
            optimizer.zero_grad()
            event_logit = model(xb)
            raw_loss = bce(event_logit, event_target)
            event_loss = (
                raw_loss * event_weights[event_target.long()]
            ).mean()
            event_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_total += float(event_loss.item()) * len(xb)

        model.eval()
        with torch.no_grad():
            val_event_logit = model(X_va)
            raw_val_loss = bce(val_event_logit, val_event_target.float())
            val_loss = (
                raw_val_loss * event_weights[val_event_target.long()]
            ).mean()
            val_probability = torch.sigmoid(val_event_logit)
            validation_threshold, threshold_metrics = (
                _calibrate_event_probability_threshold(
                    val_probability,
                    val_event_target,
                    config.lstm_event_min_recall,
                )
            )
            event_roc_auc, event_pr_auc = _binary_ranking_metrics(
                val_probability,
                val_event_target,
            )
            event_score_key, event_selection_score, selection_label = (
                _event_checkpoint_score(
                    event_roc_auc,
                    event_pr_auc,
                    threshold_metrics,
                    config.lstm_event_selection_objective,
                )
            )

        val_loss_value = float(val_loss.item())
        improved = best_score_key is None or event_score_key > best_score_key or (
            event_score_key == best_score_key
            and val_loss_value < best_val_loss
        )
        if improved:
            best_score_key = event_score_key
            best_score = event_selection_score
            best_val_loss = val_loss_value
            best_epoch = epoch
            best_state = {
                key: value.cpu().clone() for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_total / len(X_tr),
                "validation_loss": val_loss_value,
                "validation_event_roc_auc": event_roc_auc,
                "validation_event_pr_auc": event_pr_auc,
                "validation_event_selection_score": event_selection_score,
                "validation_event_selection_key": list(event_score_key),
                "validation_event_selection_objective": selection_label,
                "validation_event_precision": threshold_metrics["precision"],
                "validation_event_recall": threshold_metrics["recall"],
                "validation_event_balanced_accuracy": threshold_metrics[
                    "balanced_accuracy"
                ],
                "validation_event_probability_threshold": validation_threshold,
                "patience_counter": patience,
            }
        )
        if getattr(config, "verbose_output", False):
            marker = " *" if improved else ""
            print(
                f"[EVENT-LSTM] epoch {epoch:03d}/{config.lstm_epochs:03d} "
                f"train={train_total / len(X_tr):.6f} val={val_loss_value:.6f} "
                f"auc={event_roc_auc * 100:.1f}% pr={event_pr_auc * 100:.1f}% "
                f"precision={threshold_metrics['precision'] * 100:.1f}% "
                f"recall={threshold_metrics['recall'] * 100:.1f}% "
                f"patience={patience}/{config.lstm_early_stopping_patience}{marker}",
                flush=True,
            )
        if patience >= config.lstm_early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError("Event LSTM did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probability = torch.sigmoid(model(X_va))
        calibrated_threshold, calibrated_metrics = (
            _calibrate_event_probability_threshold(
                val_probability,
                val_event_target,
                config.lstm_event_min_recall,
            )
        )
        calibrated_roc_auc, calibrated_pr_auc = _binary_ranking_metrics(
            val_probability,
            val_event_target,
        )
    event_probability_threshold = (
        float(config.lstm_event_probability_threshold)
        if config.lstm_event_probability_threshold is not None
        else calibrated_threshold
    )
    print(
        f"[EVENT-LSTM] completed best_epoch={best_epoch} "
        f"best_event_score={best_score * 100:.1f}% "
        f"event_threshold={event_probability_threshold:.2f}",
        flush=True,
    )
    return model, {
        "device": str(device),
        "parameter_count": int(parameter_count),
        "epochs_requested": int(config.lstm_epochs),
        "epochs_completed": int(len(history)),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_val_loss),
        "best_validation_event_score": float(best_score),
        "model_selection_objective": selection_label,
        "model_selection_score_key": list(best_score_key or ()),
        "calibrated_event_roc_auc": float(calibrated_roc_auc),
        "calibrated_event_pr_auc": float(calibrated_pr_auc),
        "calibrated_event_precision": calibrated_metrics["precision"],
        "calibrated_event_recall": calibrated_metrics["recall"],
        "calibrated_event_balanced_accuracy": calibrated_metrics[
            "balanced_accuracy"
        ],
        "calibrated_direction_balanced_accuracy": None,
        "event_probability_threshold": event_probability_threshold,
        "direction_probability_threshold": None,
        "direction_return_threshold_pct": threshold,
        "event_minimum_recall": float(config.lstm_event_min_recall),
        "loss_function": "class_weighted_event_bce",
        "huber_delta": None,
        "return_loss_weight": 0.0,
        "event_loss_weight": 1.0,
        "direction_loss_weight": 0.0,
        "event_class_counts": event_counts.tolist(),
        "direction_class_counts": None,
        "fit_sequences": int(len(X_tr)),
        "validation_sequences": int(len(X_va)),
        "validation_boundary_purged_sequences": int(
            val_start - int(fit_mask.sum())
        ),
        "stopped_early": len(history) < config.lstm_epochs,
        "history": history,
    }


def _train_direction_lstm(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    sequence_origin_dates: np.ndarray,
    sequence_target_dates: np.ndarray,
    config,
) -> tuple[_DirectionLSTMClassifier, dict]:
    seed_everything(config.random_seed)
    device = _resolve_device(config)
    model = _DirectionLSTMClassifier(
        input_size=X_train_seq.shape[2],
        hidden_size=config.lstm_hidden_size,
        num_layers=config.lstm_num_layers,
        dropout=config.lstm_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lstm_learning_rate,
        weight_decay=config.lstm_weight_decay,
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")

    val_size = max(1, int(len(X_train_seq) * 0.1))
    val_start = len(X_train_seq) - val_size
    validation_start_date = sequence_origin_dates[val_start]
    fit_mask = sequence_target_dates[:val_start] < validation_start_date
    if not bool(fit_mask.any()):
        raise ValueError("No Direction LSTM sequences remain after validation purge.")

    X_tr_all = torch.as_tensor(
        X_train_seq[:val_start][fit_mask], dtype=torch.float32, device=device
    )
    y_tr_all = torch.as_tensor(
        y_train_seq[:val_start][fit_mask], dtype=torch.float32, device=device
    )
    X_va = torch.as_tensor(X_train_seq[val_start:], dtype=torch.float32, device=device)
    y_va = torch.as_tensor(y_train_seq[val_start:], dtype=torch.float32, device=device)

    threshold = float(config.lstm_direction_return_threshold)
    train_event = y_tr_all.abs() > threshold
    if int(train_event.sum().item()) == 0:
        raise ValueError("Direction LSTM training requires at least one event sample.")
    direction_target_all = y_tr_all > 0
    direction_target = direction_target_all[train_event].float()
    direction_counts = torch.bincount(direction_target.long(), minlength=2)
    if bool((direction_counts == 0).any()):
        raise ValueError("Direction LSTM training requires both up and down events.")
    direction_weights = int(train_event.sum()) / (2.0 * direction_counts.float())

    X_tr = X_tr_all[train_event]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, direction_target),
        batch_size=config.lstm_batch_size,
        shuffle=False,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[DIRECTION-LSTM] device={device.type} parameters={parameter_count:,} "
        f"event_train={len(X_tr):,} validation={len(X_va):,} "
        f"threshold=+/-{threshold:g}% loss=class_weighted_direction_bce",
        flush=True,
    )

    best_score_key: tuple[float, ...] | None = None
    best_score = float("-inf")
    best_val_loss = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    patience = 0
    history: list[dict] = []
    val_event = y_va.abs() > threshold
    val_direction_target = y_va[val_event] > 0

    def weighted_direction_loss(
        direction_logit: torch.Tensor,
        direction_target_values: torch.Tensor,
    ) -> torch.Tensor:
        raw_loss = bce(direction_logit, direction_target_values.float())
        return (
            raw_loss * direction_weights[direction_target_values.long()]
        ).mean()

    for epoch in range(1, config.lstm_epochs + 1):
        model.train()
        train_total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            direction_logit = model(xb)
            direction_loss = weighted_direction_loss(direction_logit, yb)
            direction_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_total += float(direction_loss.item()) * len(xb)

        model.eval()
        with torch.no_grad():
            if bool(val_event.any()):
                val_direction_logit = model(X_va[val_event])
                val_loss = weighted_direction_loss(
                    val_direction_logit,
                    val_direction_target.float(),
                )
                val_probability = torch.sigmoid(val_direction_logit)
                validation_direction_threshold, direction_balanced = (
                    _calibrate_probability_threshold(
                        val_probability,
                        val_direction_target,
                    )
                )
                val_predicted_up = val_probability >= validation_direction_threshold
                direction_accuracy = float(
                    (val_predicted_up == val_direction_target).float().mean().item()
                )
                direction_roc_auc, direction_pr_auc = _binary_ranking_metrics(
                    val_probability,
                    val_direction_target,
                )
            else:
                val_loss = torch.tensor(float("inf"), device=device)
                validation_direction_threshold = 0.5
                direction_balanced = 0.0
                direction_accuracy = 0.0
                direction_roc_auc = 0.5
                direction_pr_auc = 0.0

        val_loss_value = float(val_loss.item())
        direction_score_key = (
            direction_balanced,
            direction_accuracy,
            direction_roc_auc,
            direction_pr_auc,
            -val_loss_value,
        )
        improved = best_score_key is None or direction_score_key > best_score_key
        if improved:
            best_score_key = direction_score_key
            best_score = direction_balanced
            best_val_loss = val_loss_value
            best_epoch = epoch
            best_state = {
                key: value.cpu().clone() for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_total / len(X_tr),
                "validation_loss": val_loss_value,
                "validation_direction_balanced_accuracy": direction_balanced,
                "validation_direction_accuracy": direction_accuracy,
                "validation_direction_roc_auc": direction_roc_auc,
                "validation_direction_pr_auc": direction_pr_auc,
                "validation_direction_probability_threshold": (
                    validation_direction_threshold
                ),
                "validation_direction_event_count": int(val_event.sum().item()),
                "validation_direction_selection_key": list(direction_score_key),
                "patience_counter": patience,
            }
        )
        if getattr(config, "verbose_output", False):
            marker = " *" if improved else ""
            print(
                f"[DIRECTION-LSTM] epoch {epoch:03d}/{config.lstm_epochs:03d} "
                f"train={train_total / len(X_tr):.6f} val={val_loss_value:.6f} "
                f"balanced={direction_balanced * 100:.1f}% "
                f"accuracy={direction_accuracy * 100:.1f}% "
                f"patience={patience}/{config.lstm_early_stopping_patience}{marker}",
                flush=True,
            )
        if patience >= config.lstm_early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError("Direction LSTM did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        if bool(val_event.any()):
            val_direction_logit = model(X_va[val_event])
            val_probability = torch.sigmoid(val_direction_logit)
            calibrated_direction_threshold, calibrated_direction_score = (
                _calibrate_probability_threshold(
                    val_probability,
                    val_direction_target,
                )
            )
            calibrated_direction_roc_auc, calibrated_direction_pr_auc = (
                _binary_ranking_metrics(val_probability, val_direction_target)
            )
        else:
            calibrated_direction_threshold = 0.5
            calibrated_direction_score = 0.0
            calibrated_direction_roc_auc = 0.5
            calibrated_direction_pr_auc = 0.0
    direction_probability_threshold = (
        float(config.lstm_direction_probability_threshold)
        if config.lstm_direction_probability_threshold is not None
        else calibrated_direction_threshold
    )
    print(
        f"[DIRECTION-LSTM] completed best_epoch={best_epoch} "
        f"best_direction_score={best_score * 100:.1f}% "
        f"direction_threshold={direction_probability_threshold:.2f}",
        flush=True,
    )
    return model, {
        "device": str(device),
        "parameter_count": int(parameter_count),
        "epochs_requested": int(config.lstm_epochs),
        "epochs_completed": int(len(history)),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_val_loss),
        "best_validation_event_score": None,
        "best_validation_direction_score": float(best_score),
        "model_selection_objective": (
            "direction_balanced_accuracy_on_validation_events"
        ),
        "model_selection_score_key": list(best_score_key or ()),
        "calibrated_event_roc_auc": None,
        "calibrated_event_pr_auc": None,
        "calibrated_event_precision": None,
        "calibrated_event_recall": None,
        "calibrated_event_balanced_accuracy": None,
        "calibrated_direction_balanced_accuracy": float(
            calibrated_direction_score
        ),
        "calibrated_direction_roc_auc": float(calibrated_direction_roc_auc),
        "calibrated_direction_pr_auc": float(calibrated_direction_pr_auc),
        "event_probability_threshold": None,
        "direction_probability_threshold": direction_probability_threshold,
        "direction_return_threshold_pct": threshold,
        "event_minimum_recall": None,
        "loss_function": "class_weighted_direction_bce_on_actual_events",
        "huber_delta": None,
        "return_loss_weight": 0.0,
        "event_loss_weight": 0.0,
        "direction_loss_weight": 1.0,
        "event_class_counts": [
            int((~train_event).sum().item()),
            int(train_event.sum().item()),
        ],
        "direction_class_counts": direction_counts.tolist(),
        "fit_sequences": int(len(X_tr)),
        "validation_sequences": int(len(X_va)),
        "validation_event_sequences": int(val_event.sum().item()),
        "validation_boundary_purged_sequences": int(
            val_start - int(fit_mask.sum())
        ),
        "stopped_early": len(history) < config.lstm_epochs,
        "history": history,
    }


def _predict_event(
    model: _EventLSTMClassifier,
    X_seq: np.ndarray,
) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X_seq, dtype=torch.float32, device=device)).cpu().numpy()


def _predict_direction(
    model: _DirectionLSTMClassifier,
    X_seq: np.ndarray,
) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X_seq, dtype=torch.float32, device=device)).cpu().numpy()


def _evaluate_direction_only(
    model: _DirectionLSTMClassifier,
    X_test_seq: np.ndarray,
    test_frame: pd.DataFrame,
    direction_threshold: float,
    direction_probability_threshold: float,
) -> tuple[dict, pd.DataFrame]:
    direction_logit = _predict_direction(model, X_test_seq)
    direction_probability = 1.0 / (
        1.0 + np.exp(-np.clip(direction_logit, -30.0, 30.0))
    )
    predicted_up = direction_probability >= direction_probability_threshold
    y_test = test_frame["target_logret"].to_numpy(dtype=float)
    actual_event = np.abs(y_test) > direction_threshold
    actual_up = y_test > direction_threshold
    actual_down = y_test < -direction_threshold
    metrics = compute_thresholded_binary_direction_metrics(
        predicted_up,
        y_test,
        direction_threshold,
    )
    if int(actual_event.sum()) > 0 and len(np.unique(actual_up[actual_event])) == 2:
        metrics["direction_roc_auc"] = float(
            roc_auc_score(
                actual_up[actual_event].astype(int),
                direction_probability[actual_event],
            )
        )
        metrics["direction_pr_auc"] = float(
            average_precision_score(
                actual_up[actual_event].astype(int),
                direction_probability[actual_event],
            )
        )
        metrics["direction_brier_score"] = float(
            brier_score_loss(
                actual_up[actual_event].astype(int),
                direction_probability[actual_event],
            )
        )
    else:
        metrics["direction_roc_auc"] = None
        metrics["direction_pr_auc"] = None
        metrics["direction_brier_score"] = None
    metrics["direction_probability_threshold"] = float(
        direction_probability_threshold
    )
    metrics["actual_event_count"] = int(actual_event.sum())
    metrics["actual_event_rate"] = float(actual_event.mean())

    current_price = test_frame["target_price"].to_numpy(dtype=float)
    future_price = test_frame["target_future_price"].to_numpy(dtype=float)
    actual_simple_return = future_price / current_price - 1.0
    actual_direction_class = np.where(
        actual_up,
        "up",
        np.where(actual_down, "down", "ignored"),
    )
    predictions = pd.DataFrame({
        "Current_Date": pd.to_datetime(test_frame["Date"]),
        "Target_Date": pd.to_datetime(test_frame["target_date"]),
        "Current_Price": current_price,
        "Actual_Future_Price": future_price,
        "Actual_LogRet": y_test,
        "Actual_Return": actual_simple_return,
        "Actual_Event": actual_event,
        "Direction_Logit": direction_logit,
        "Direction_Probability": direction_probability,
        "Direction_Class": np.where(predicted_up, "up", "down"),
        "Actual_Direction_Class": actual_direction_class,
        "Model_Signal": np.where(predicted_up, 1.0, -1.0),
        "Model_Signal_Return": np.where(predicted_up, 1.0, -1.0)
        * actual_simple_return,
        "Confidence": np.abs(direction_probability - 0.5) * 2.0,
    })
    return metrics, predictions


def _evaluate_event_only(
    model: _EventLSTMClassifier,
    X_test_seq: np.ndarray,
    test_frame: pd.DataFrame,
    direction_threshold: float,
    event_probability_threshold: float,
    event_training_base_rate: float,
) -> tuple[dict, pd.DataFrame]:
    event_logit = _predict_event(model, X_test_seq)
    event_probability = 1.0 / (1.0 + np.exp(-np.clip(event_logit, -30.0, 30.0)))
    predicted_event = event_probability >= event_probability_threshold
    y_test = test_frame["target_logret"].to_numpy(dtype=float)
    actual_event = np.abs(y_test) > direction_threshold
    target = actual_event.astype(int)
    prediction = predicted_event.astype(int)
    tp = int(((prediction == 1) & (target == 1)).sum())
    fp = int(((prediction == 1) & (target == 0)).sum())
    fn = int(((prediction == 0) & (target == 1)).sum())
    tn = int(((prediction == 0) & (target == 0)).sum())
    has_both_classes = len(np.unique(target)) == 2
    brier = float(brier_score_loss(target, event_probability))
    baseline_probability = np.full(len(target), float(event_training_base_rate))
    brier_baseline = float(brier_score_loss(target, baseline_probability))
    metrics = {
        "event_accuracy": float((prediction == target).mean()),
        "event_balanced_accuracy": (
            float(balanced_accuracy_score(target, prediction))
            if has_both_classes
            else None
        ),
        "event_precision": float(precision_score(target, prediction, zero_division=0)),
        "event_recall": float(recall_score(target, prediction, zero_division=0)),
        "event_specificity": float(tn / (tn + fp)) if tn + fp else None,
        "event_f1": float(f1_score(target, prediction, zero_division=0)),
        "actual_event_count": int(target.sum()),
        "actual_event_rate": float(target.mean()),
        "predicted_event_count": int(prediction.sum()),
        "predicted_event_rate": float(prediction.mean()),
        "true_positive_event_count": tp,
        "false_positive_event_count": fp,
        "false_negative_event_count": fn,
        "true_negative_event_count": tn,
        "event_roc_auc": (
            float(roc_auc_score(target, event_probability))
            if has_both_classes
            else None
        ),
        "event_pr_auc": float(average_precision_score(target, event_probability)),
        "event_pr_auc_baseline": float(target.mean()),
        "event_pr_auc_lift": float(
            average_precision_score(target, event_probability) - target.mean()
        ),
        "event_brier_score": brier,
        "event_brier_baseline": brier_baseline,
        "event_brier_skill_score": (
            1.0 - brier / brier_baseline if brier_baseline > 0.0 else None
        ),
        "event_probability_threshold": float(event_probability_threshold),
        "direction_return_threshold_pct": float(direction_threshold),
    }
    predictions = pd.DataFrame(
        {
            "Current_Date": pd.to_datetime(test_frame["Date"]),
            "Target_Date": pd.to_datetime(test_frame["target_date"]),
            "Current_Price": test_frame["target_price"].to_numpy(dtype=float),
            "Actual_Future_Price": test_frame["target_future_price"].to_numpy(
                dtype=float
            ),
            "Actual_LogRet": y_test,
            "Event_Logit": event_logit,
            "Event_Probability": event_probability,
            "Predicted_Event": predicted_event,
            "Actual_Event": actual_event,
        }
    )
    return metrics, predictions


def _predict(
    model: _LSTMRegressor,
    X_seq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    model.to(device)
    with torch.no_grad():
        return tuple(
            output.cpu().numpy()
            for output in model(torch.FloatTensor(X_seq).to(device))
        )


def _evaluate(
    model: _LSTMRegressor,
    X_test_seq: np.ndarray,
    test_frame: pd.DataFrame,
    horizon: int,
    direction_threshold: float,
    event_probability_threshold: float,
    direction_probability_threshold: float,
    event_training_base_rate: float,
) -> tuple[dict, pd.DataFrame]:
    raw_predicted_logret, event_logit, direction_logit = _predict(model, X_test_seq)
    event_probability = 1.0 / (1.0 + np.exp(-np.clip(event_logit, -30.0, 30.0)))
    direction_probability = 1.0 / (
        1.0 + np.exp(-np.clip(direction_logit, -30.0, 30.0))
    )
    predicted_event = event_probability >= event_probability_threshold
    predicted_up = direction_probability >= direction_probability_threshold
    model_signal = np.where(
        predicted_event,
        np.where(predicted_up, 1.0, -1.0),
        0.0,
    )
    predicted_logret = np.abs(raw_predicted_logret) * model_signal
    y_test = test_frame["target_logret"].to_numpy()
    current_price = test_frame["target_price"].to_numpy()
    future_price = test_frame["target_future_price"].to_numpy()

    predicted_future_price = current_price * np.exp(predicted_logret / 100.0)
    predicted_simple_return = predicted_future_price / current_price - 1.0
    actual_simple_return = future_price / current_price - 1.0

    actual_actionable = np.abs(y_test) > direction_threshold
    actual_direction_class = np.where(
        ~actual_actionable,
        "ignored",
        np.where(y_test > 0, "up", "down"),
    )
    model_signal_return = model_signal * actual_simple_return

    mae = float(mean_absolute_error(future_price, predicted_future_price))
    rmse = float(np.sqrt(mean_squared_error(future_price, predicted_future_price)))
    logret_rmse = float(np.sqrt(mean_squared_error(y_test, predicted_logret)))
    r2 = float(r2_score(future_price, predicted_future_price))
    mape = float(np.mean(np.abs((future_price - predicted_future_price) / future_price)) * 100)

    baseline_future_price = current_price.copy()
    baseline_rmse = float(np.sqrt(mean_squared_error(future_price, baseline_future_price)))
    baseline_mae = float(mean_absolute_error(future_price, baseline_future_price))
    baseline_mape = float(
        np.mean(np.abs((future_price - baseline_future_price) / future_price)) * 100
    )

    direction_confidence = np.abs(direction_probability - direction_probability_threshold)
    confidence = event_probability * direction_confidence
    conf_cutoff = float(np.quantile(confidence, 0.7))
    high_conf_mask = (confidence >= conf_cutoff) & actual_actionable & predicted_event
    long_mask = high_conf_mask & predicted_up
    short_mask = high_conf_mask & ~predicted_up
    long_count = int(long_mask.sum())
    short_count = int(short_mask.sum())

    direction_metrics = compute_thresholded_binary_direction_metrics(
        predicted_up,
        y_test,
        direction_threshold,
    )
    two_stage_metrics = compute_two_stage_signal_metrics(
        predicted_event,
        predicted_up,
        y_test,
        direction_threshold,
    )

    metrics: dict = {
        "mae": mae,
        "rmse": rmse,
        "logret_rmse": logret_rmse,
        "r2_score": r2,
        "direction_accuracy": direction_metrics["direction_accuracy"],
        "mape": mape,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "baseline_mape": baseline_mape,
        "direction_objective_score": direction_metrics["macro_balanced_accuracy"],
        "event_probability_threshold": float(event_probability_threshold),
        "direction_probability_threshold": float(direction_probability_threshold),
        "high_conf_threshold": conf_cutoff,
        "high_conf_long_accuracy": (
            float((y_test[long_mask] > direction_threshold).mean()) if long_count > 0 else None
        ),
        "high_conf_long_count": long_count,
        "high_conf_short_accuracy": (
            float((y_test[short_mask] < -direction_threshold).mean()) if short_count > 0 else None
        ),
        "high_conf_short_count": short_count,
    }
    metrics.update(direction_metrics)
    metrics.update(two_stage_metrics)
    actionable_direction = (y_test[actual_actionable] > direction_threshold).astype(int)
    metrics["direction_roc_auc"] = (
        float(roc_auc_score(actionable_direction, direction_probability[actual_actionable]))
        if len(np.unique(actionable_direction)) == 2
        else None
    )
    event_target = actual_actionable.astype(int)
    metrics["event_roc_auc"] = (
        float(roc_auc_score(event_target, event_probability))
        if len(np.unique(event_target)) == 2
        else None
    )
    metrics["event_pr_auc"] = float(
        average_precision_score(event_target, event_probability)
    )
    metrics["event_pr_auc_baseline"] = float(event_target.mean())
    metrics["event_pr_auc_lift"] = (
        metrics["event_pr_auc"] - metrics["event_pr_auc_baseline"]
    )
    metrics["event_brier_score"] = float(
        brier_score_loss(event_target, event_probability)
    )
    baseline_event_probability = np.full(
        len(event_target),
        float(event_training_base_rate),
    )
    event_brier_baseline = float(
        brier_score_loss(event_target, baseline_event_probability)
    )
    metrics["event_brier_baseline"] = event_brier_baseline
    metrics["event_brier_skill_score"] = (
        1.0 - metrics["event_brier_score"] / event_brier_baseline
        if event_brier_baseline > 0.0
        else None
    )
    metrics.update(compute_strong_regime_metrics(predicted_logret, y_test))
    metrics.update(
        compute_signal_return_metrics(
            predicted_logret,
            y_test,
            horizon=horizon,
            confidence_values=confidence,
        )
    )

    predictions = pd.DataFrame({
        "Current_Date": pd.to_datetime(test_frame["Date"]),
        "Target_Date": pd.to_datetime(test_frame["target_date"]),
        "Current_Price": current_price,
        "Actual_Future_Price": future_price,
        "Pred_Future_Price": predicted_future_price,
        "Pred_LogRet": predicted_logret,
        "Raw_Magnitude_Head_LogRet": raw_predicted_logret,
        "Actual_LogRet": y_test,
        "Pred_Return": predicted_simple_return,
        "Actual_Return": actual_simple_return,
        "Event_Logit": event_logit,
        "Event_Probability": event_probability,
        "Predicted_Event": predicted_event,
        "Actual_Event": actual_actionable,
        "Direction_Logit": direction_logit,
        "Direction_Probability": direction_probability,
        "Direction_Class": np.where(
            predicted_event,
            np.where(predicted_up, "up", "down"),
            "hold",
        ),
        "Actual_Direction_Class": actual_direction_class,
        "Model_Signal": model_signal,
        "Model_Signal_Return": model_signal_return,
        "Confidence": confidence,
    })
    return metrics, predictions


def run_training_experiment(
    experiment_name: str,
    feature_df: pd.DataFrame,
    candidate_feature_columns: list[str],
    training_frame_output_path: Path | None,
    predictions_output_path: Path | None,
    model_output_path: Path | None,
    metadata_output_path: Path | None,
    config,
    forced_horizon: int,
    forced_selected_features: list[str] | None = None,
    embedding_columns_for_pca: list[str] | None = None,
    n_embedding_pca_components: int = 5,
    min_date: str | pd.Timestamp | None = None,
    test_start_date: str | pd.Timestamp | None = None,
    test_end_date: str | pd.Timestamp | None = None,
    persist_artifacts: bool = True,
    training_mode: str = "multitask",
) -> dict:
    if training_mode not in {"multitask", "event_only", "direction_only"}:
        raise ValueError(
            "training_mode must be one of: multitask, event_only, direction_only"
        )
    horizon = int(forced_horizon)
    active_features = (
        list(forced_selected_features)
        if forced_selected_features is not None
        else list(candidate_feature_columns)
    )

    filtered_df = filter_feature_frame_by_min_date(feature_df, min_date)
    supervised_frame = build_supervised_frame(filtered_df, active_features, horizon)

    if test_start_date is None:
        train_frame, test_frame, pre_test_frame, purged_rows = split_supervised_frame(
            supervised_frame,
            config.train_ratio,
        )
    else:
        train_frame, test_frame, pre_test_frame, purged_rows = (
            split_supervised_frame_at_date(supervised_frame, test_start_date)
        )
    if test_end_date is not None:
        resolved_test_end = pd.to_datetime(test_end_date, errors="coerce")
        if pd.isna(resolved_test_end):
            raise ValueError(f"Invalid test_end_date: {test_end_date}")
        test_frame = test_frame.loc[
            pd.to_datetime(test_frame["Date"]) <= resolved_test_end
        ].reset_index(drop=True)
        if test_frame.empty:
            raise ValueError("test_end_date leaves no test rows.")

    seq_len = config.lstm_seq_len
    n_pre_test_seq = len(pre_test_frame) - seq_len + 1
    if n_pre_test_seq <= 0:
        raise ValueError(
            f"Training data too short ({len(pre_test_frame)} rows) for seq_len={seq_len}."
        )
    pre_test_sequence_rows = pre_test_frame.iloc[seq_len - 1:].reset_index(drop=True)
    test_start_date = pd.to_datetime(test_frame["Date"].iloc[0])
    eligible_train_mask = (
        pd.to_datetime(pre_test_sequence_rows["target_date"]) < test_start_date
    ).to_numpy()
    eligible_sequence_count = int(eligible_train_mask.sum())
    validation_size = max(1, int(eligible_sequence_count * 0.1))
    validation_start = eligible_sequence_count - validation_size
    if validation_start <= 0:
        raise ValueError("Not enough training sequences for an internal validation split.")
    eligible_sequence_rows = pre_test_sequence_rows.loc[eligible_train_mask].reset_index(
        drop=True
    )
    validation_origin_date = pd.to_datetime(
        eligible_sequence_rows["Date"].iloc[validation_start]
    )
    representation_fit_mask = (
        pd.to_datetime(pre_test_frame["Date"]) < validation_origin_date
    ).to_numpy()
    if not bool(representation_fit_mask.any()):
        raise ValueError("No rows remain for leakage-safe feature preprocessing.")

    embedding_pca_payload: dict | None = None
    if embedding_columns_for_pca:
        fit_emb = pre_test_frame.loc[
            representation_fit_mask,
            embedding_columns_for_pca,
        ].to_numpy(dtype=float)
        _, embedding_pca_payload = fit_embedding_pca_features(
            fit_emb,
            source_columns=embedding_columns_for_pca,
            n_components=n_embedding_pca_components,
        )
        pc_columns: list[str] = embedding_pca_payload["feature_columns"]
        pre_test_emb = pre_test_frame[embedding_columns_for_pca].to_numpy(dtype=float)
        emb_pc_vectors = transform_embedding_pca_features(
            pre_test_emb,
            embedding_pca_payload,
        )
        for i, col in enumerate(pc_columns):
            pre_test_frame[col] = emb_pc_vectors[:, i]
        test_emb = test_frame[embedding_columns_for_pca].to_numpy(dtype=float)
        test_pc_vectors = transform_embedding_pca_features(test_emb, embedding_pca_payload)
        for i, col in enumerate(pc_columns):
            test_frame[col] = test_pc_vectors[:, i]
        active_features = [
            f for f in active_features if f not in embedding_columns_for_pca
        ] + pc_columns

    scaler = StandardScaler()
    scaler.fit(
        pre_test_frame.loc[representation_fit_mask, active_features].to_numpy(
            dtype=float
        )
    )
    X_pre_test = scaler.transform(pre_test_frame[active_features].to_numpy(dtype=float))
    X_test = scaler.transform(test_frame[active_features].to_numpy(dtype=float))
    y_pre_test = pre_test_frame["target_logret"].to_numpy(dtype=float)
    y_test_arr = test_frame["target_logret"].to_numpy(dtype=float)

    X_combined = np.concatenate([X_pre_test, X_test], axis=0)
    y_combined = np.concatenate([y_pre_test, y_test_arr], axis=0)
    X_all_seq, y_all_seq = _build_sequences(X_combined, y_combined, seq_len)

    X_train_seq = X_all_seq[:n_pre_test_seq][eligible_train_mask]
    y_train_seq = y_all_seq[:n_pre_test_seq][eligible_train_mask]
    sequence_origin_dates = pd.to_datetime(
        pre_test_sequence_rows.loc[eligible_train_mask, "Date"]
    ).to_numpy()
    sequence_target_dates = pd.to_datetime(
        pre_test_sequence_rows.loc[eligible_train_mask, "target_date"]
    ).to_numpy()
    X_test_seq = X_all_seq[n_pre_test_seq : n_pre_test_seq + len(X_test)]

    if training_mode == "event_only":
        model, training_summary = _train_event_lstm(
            X_train_seq,
            y_train_seq,
            sequence_origin_dates,
            sequence_target_dates,
            config,
        )
        metrics, predictions = _evaluate_event_only(
            model,
            X_test_seq,
            test_frame,
            config.lstm_direction_return_threshold,
            training_summary["event_probability_threshold"],
            training_summary["event_class_counts"][1]
            / sum(training_summary["event_class_counts"]),
        )
    elif training_mode == "direction_only":
        model, training_summary = _train_direction_lstm(
            X_train_seq,
            y_train_seq,
            sequence_origin_dates,
            sequence_target_dates,
            config,
        )
        metrics, predictions = _evaluate_direction_only(
            model,
            X_test_seq,
            test_frame,
            config.lstm_direction_return_threshold,
            training_summary["direction_probability_threshold"],
        )
    else:
        model, training_summary = _train_lstm(
            X_train_seq,
            y_train_seq,
            sequence_origin_dates,
            sequence_target_dates,
            config,
        )
        metrics, predictions = _evaluate(
            model,
            X_test_seq,
            test_frame,
            horizon,
            config.lstm_direction_return_threshold,
            training_summary["event_probability_threshold"],
            training_summary["direction_probability_threshold"],
            training_summary["event_class_counts"][1]
            / sum(training_summary["event_class_counts"]),
        )

    metadata: dict = {
        "experiment_name": experiment_name,
        "model_type": (
            "event_lstm"
            if training_mode == "event_only"
            else "direction_lstm"
            if training_mode == "direction_only"
            else "lstm"
        ),
        "training_mode": training_mode,
        "best_horizon": horizon,
        "best_horizon_direction_score": None,
        "selected_feature_count": len(active_features),
        "selected_features": active_features,
        "lstm_device": training_summary["device"],
        "lstm_seq_len": seq_len,
        "lstm_hidden_size": config.lstm_hidden_size,
        "lstm_num_layers": config.lstm_num_layers,
        "lstm_dropout": config.lstm_dropout,
        "loss_function": training_summary["loss_function"],
        "huber_delta": training_summary["huber_delta"],
        "epochs_completed": training_summary["epochs_completed"],
        "best_epoch": training_summary["best_epoch"],
        "best_validation_loss": training_summary["best_validation_loss"],
        "best_validation_event_score": training_summary[
            "best_validation_event_score"
        ],
        "model_selection_objective": training_summary["model_selection_objective"],
        "model_selection_score_key": training_summary["model_selection_score_key"],
        "event_minimum_recall": training_summary["event_minimum_recall"],
        "direction_return_threshold_pct": training_summary[
            "direction_return_threshold_pct"
        ],
        "event_probability_threshold": training_summary[
            "event_probability_threshold"
        ],
        "direction_probability_threshold": training_summary[
            "direction_probability_threshold"
        ],
        "return_loss_weight": training_summary["return_loss_weight"],
        "event_loss_weight": training_summary["event_loss_weight"],
        "direction_loss_weight": training_summary["direction_loss_weight"],
        "stopped_early": training_summary["stopped_early"],
        "feature_frame_start_date": serialize_timestamp(supervised_frame["Date"].iloc[0]),
        "feature_frame_end_date": serialize_timestamp(supervised_frame["Date"].iloc[-1]),
        "train_rows": int(len(train_frame)),
        "train_sequences": int(len(X_train_seq)),
        "purged_train_rows": purged_rows,
        "representation_fit_rows": int(representation_fit_mask.sum()),
        "representation_fit_end_date": serialize_timestamp(
            pre_test_frame.loc[representation_fit_mask, "Date"].iloc[-1]
        ),
        "internal_validation_start_date": serialize_timestamp(validation_origin_date),
        "test_rows": int(len(test_frame)),
        "train_start_date": serialize_timestamp(train_frame["Date"].iloc[0]),
        "train_end_date": serialize_timestamp(train_frame["Date"].iloc[-1]),
        "test_start_date": serialize_timestamp(test_frame["Date"].iloc[0]),
        "test_end_date": serialize_timestamp(test_frame["Date"].iloc[-1]),
        "metrics": metrics,
        "config": to_serializable_config(config),
        "horizon_selection_mode": "fixed",
        "feature_selection_mode": "fixed_regression_style",
        "training": training_summary,
    }
    if embedding_pca_payload is not None:
        metadata["embedding_pca"] = embedding_pca_payload

    if persist_artifacts:
        if training_frame_output_path is not None:
            training_frame_output_path.parent.mkdir(parents=True, exist_ok=True)
            supervised_frame.to_csv(
                training_frame_output_path, index=False, encoding="utf-8-sig"
            )
        if predictions_output_path is not None:
            predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(predictions_output_path, index=False, encoding="utf-8-sig")
        if model_output_path is not None:
            lstm_path = model_output_path.with_suffix(".pt")
            lstm_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": len(active_features),
                    "hidden_size": config.lstm_hidden_size,
                    "num_layers": config.lstm_num_layers,
                    "dropout": config.lstm_dropout,
                    "architecture": (
                        "event_only_lstm_classifier"
                        if training_mode == "event_only"
                        else "direction_only_lstm_classifier"
                        if training_mode == "direction_only"
                        else "separate_event_and_direction_lstm_encoders"
                    ),
                    "feature_columns": active_features,
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "seq_len": seq_len,
                    "training": training_summary,
                },
                lstm_path,
            )
        if metadata_output_path is not None:
            write_json(metadata, metadata_output_path)

    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()
        model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    return metadata
