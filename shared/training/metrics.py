from __future__ import annotations

import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def seed_everything(seed: int = 42) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def serialize_timestamp(value: object) -> str | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.strftime("%Y-%m-%d")


def filter_feature_frame_by_min_date(
    feature_df: pd.DataFrame,
    min_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    filtered = feature_df.copy()
    if "Date" not in filtered.columns:
        raise ValueError("Feature dataframe must include a Date column.")
    filtered["Date"] = pd.to_datetime(filtered["Date"], errors="coerce")
    filtered = filtered.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if min_date is None:
        return filtered
    minimum_timestamp = pd.to_datetime(min_date, errors="coerce")
    if pd.isna(minimum_timestamp):
        raise ValueError(f"Invalid min_date value: {min_date}")
    filtered = filtered[filtered["Date"] >= minimum_timestamp].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No rows remain after applying the comparison start date filter.")
    return filtered


def to_serializable_config(config) -> dict:
    raw_config = asdict(config)
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in raw_config.items()
    }


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _zero_if_none(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def _simple_return_from_logret(logret_pct: np.ndarray) -> np.ndarray:
    return np.exp(logret_pct / 100.0) - 1.0


def _compute_direction_accuracy(
    predicted_values: np.ndarray,
    actual_values: np.ndarray,
) -> float:
    predicted_direction = (predicted_values > 0).astype(int)
    actual_direction = (actual_values > 0).astype(int)
    return float((predicted_direction == actual_direction).mean())


def compute_direction_metrics(
    predicted_values: np.ndarray,
    actual_values: np.ndarray,
) -> dict:
    predicted_up = predicted_values > 0
    predicted_down = predicted_values < 0
    actual_up = actual_values > 0
    actual_down = actual_values < 0

    predicted_up_count = int(predicted_up.sum())
    predicted_down_count = int(predicted_down.sum())
    actual_up_count = int(actual_up.sum())
    actual_down_count = int(actual_down.sum())
    total_count = int(len(actual_values))

    true_up_count = int((predicted_up & actual_up).sum())
    true_down_count = int((predicted_down & actual_down).sum())

    return {
        "direction_accuracy": _compute_direction_accuracy(predicted_values, actual_values),
        "actual_up_count": actual_up_count,
        "actual_down_count": actual_down_count,
        "actual_up_rate": _safe_rate(actual_up_count, total_count),
        "actual_down_rate": _safe_rate(actual_down_count, total_count),
        "predicted_up_count": predicted_up_count,
        "predicted_down_count": predicted_down_count,
        "predicted_up_rate": _safe_rate(predicted_up_count, total_count),
        "predicted_down_rate": _safe_rate(predicted_down_count, total_count),
        "up_precision": _safe_rate(true_up_count, predicted_up_count),
        "down_precision": _safe_rate(true_down_count, predicted_down_count),
        "up_recall": _safe_rate(true_up_count, actual_up_count),
        "down_recall": _safe_rate(true_down_count, actual_down_count),
    }


def compute_thresholded_binary_direction_metrics(
    predicted_up_values: np.ndarray,
    actual_logret: np.ndarray,
    direction_threshold: float,
) -> dict:
    predicted_up_all = np.asarray(predicted_up_values, dtype=bool)
    actual_values = np.asarray(actual_logret, dtype=float)
    if len(predicted_up_all) != len(actual_values):
        raise ValueError("predicted_up_values and actual_logret must have equal length.")

    actionable = np.abs(actual_values) > direction_threshold
    predicted_up = predicted_up_all[actionable]
    actual_up = actual_values[actionable] > direction_threshold
    actual_down = actual_values[actionable] < -direction_threshold
    predicted_down = ~predicted_up

    true_up = int((predicted_up & actual_up).sum())
    true_down = int((predicted_down & actual_down).sum())
    actual_up_count = int(actual_up.sum())
    actual_down_count = int(actual_down.sum())
    predicted_up_count = int(predicted_up.sum())
    predicted_down_count = int(predicted_down.sum())
    up_recall = _safe_rate(true_up, actual_up_count)
    down_recall = _safe_rate(true_down, actual_down_count)
    valid_recalls = [value for value in (down_recall, up_recall) if value is not None]
    actionable_count = int(actionable.sum())
    if actionable_count == 0:
        return {
            "direction_accuracy": None,
            "macro_balanced_accuracy": None,
            "actionable_direction_accuracy": None,
            "actionable_label_coverage": 0.0,
            "actual_down_count": 0,
            "actual_up_count": 0,
            "actual_down_rate": None,
            "actual_up_rate": None,
            "predicted_down_count": 0,
            "predicted_up_count": 0,
            "predicted_down_rate": None,
            "predicted_up_rate": None,
            "all_predicted_down_count": int((~predicted_up_all).sum()),
            "all_predicted_up_count": int(predicted_up_all.sum()),
            "down_precision": None,
            "up_precision": None,
            "down_recall": None,
            "up_recall": None,
            "direction_confusion_matrix": [[0, 0], [0, 0]],
            "direction_class_order": ["down", "up"],
            "direction_return_threshold_pct": float(direction_threshold),
            "direction_sample_count": 0,
            "ignored_small_move_count": int(len(actual_values)),
            "direction_label_mode": "thresholded_binary",
        }

    return {
        "direction_accuracy": float((predicted_up == actual_up).mean()),
        "macro_balanced_accuracy": float(np.mean(valid_recalls)),
        "actionable_direction_accuracy": float((predicted_up == actual_up).mean()),
        "actionable_label_coverage": float(actionable.mean()),
        "actual_down_count": actual_down_count,
        "actual_up_count": actual_up_count,
        "actual_down_rate": _safe_rate(actual_down_count, actionable_count),
        "actual_up_rate": _safe_rate(actual_up_count, actionable_count),
        "predicted_down_count": predicted_down_count,
        "predicted_up_count": predicted_up_count,
        "predicted_down_rate": _safe_rate(predicted_down_count, actionable_count),
        "predicted_up_rate": _safe_rate(predicted_up_count, actionable_count),
        "all_predicted_down_count": int((~predicted_up_all).sum()),
        "all_predicted_up_count": int(predicted_up_all.sum()),
        "down_precision": _safe_rate(true_down, predicted_down_count),
        "up_precision": _safe_rate(true_up, predicted_up_count),
        "down_recall": down_recall,
        "up_recall": up_recall,
        "direction_confusion_matrix": [
            [true_down, actual_down_count - true_down],
            [actual_up_count - true_up, true_up],
        ],
        "direction_class_order": ["down", "up"],
        "direction_return_threshold_pct": float(direction_threshold),
        "direction_sample_count": actionable_count,
        "ignored_small_move_count": int((~actionable).sum()),
        "direction_label_mode": "thresholded_binary",
    }


def compute_two_stage_signal_metrics(
    predicted_event_values: np.ndarray,
    predicted_up_values: np.ndarray,
    actual_logret: np.ndarray,
    direction_threshold: float,
) -> dict:
    predicted_event = np.asarray(predicted_event_values, dtype=bool)
    predicted_up = np.asarray(predicted_up_values, dtype=bool)
    actual_values = np.asarray(actual_logret, dtype=float)
    if not (len(predicted_event) == len(predicted_up) == len(actual_values)):
        raise ValueError("Two-stage prediction arrays must have equal length.")

    actual_event = np.abs(actual_values) > direction_threshold
    actual_up = actual_values > direction_threshold
    actual_down = actual_values < -direction_threshold

    event_tp = int((predicted_event & actual_event).sum())
    event_fp = int((predicted_event & ~actual_event).sum())
    event_fn = int((~predicted_event & actual_event).sum())
    event_tn = int((~predicted_event & ~actual_event).sum())
    event_precision = _safe_rate(event_tp, event_tp + event_fp)
    event_recall = _safe_rate(event_tp, event_tp + event_fn)
    event_specificity = _safe_rate(event_tn, event_tn + event_fp)
    event_f1 = (
        None
        if event_precision is None
        or event_recall is None
        or event_precision + event_recall == 0
        else float(2 * event_precision * event_recall / (event_precision + event_recall))
    )
    event_balanced_accuracy = float(
        np.mean(
            [
                value
                for value in (event_recall, event_specificity)
                if value is not None
            ]
        )
    )

    true_positive_event_mask = predicted_event & actual_event
    direction_correct = (predicted_up & actual_up) | ((~predicted_up) & actual_down)
    correct_strong_signals = int((true_positive_event_mask & direction_correct).sum())
    predicted_event_count = int(predicted_event.sum())
    predicted_event_up_count = int((predicted_event & predicted_up).sum())
    predicted_event_down_count = int((predicted_event & ~predicted_up).sum())
    conditional_direction_accuracy = _safe_rate(correct_strong_signals, event_tp)
    strong_signal_accuracy = _safe_rate(correct_strong_signals, predicted_event_count)

    tp_event_actual_up = true_positive_event_mask & actual_up
    tp_event_actual_down = true_positive_event_mask & actual_down
    conditional_up_recall = _safe_rate(
        int((tp_event_actual_up & predicted_up).sum()),
        int(tp_event_actual_up.sum()),
    )
    conditional_down_recall = _safe_rate(
        int((tp_event_actual_down & ~predicted_up).sum()),
        int(tp_event_actual_down.sum()),
    )
    valid_direction_recalls = [
        value
        for value in (conditional_down_recall, conditional_up_recall)
        if value is not None
    ]

    return {
        "event_accuracy": float((predicted_event == actual_event).mean()),
        "event_balanced_accuracy": event_balanced_accuracy,
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_specificity": event_specificity,
        "event_f1": event_f1,
        "actual_event_count": int(actual_event.sum()),
        "actual_event_rate": float(actual_event.mean()),
        "predicted_event_count": predicted_event_count,
        "predicted_event_up_count": predicted_event_up_count,
        "predicted_event_down_count": predicted_event_down_count,
        "predicted_event_rate": float(predicted_event.mean()),
        "true_positive_event_count": event_tp,
        "false_positive_event_count": event_fp,
        "false_negative_event_count": event_fn,
        "true_negative_event_count": event_tn,
        "conditional_direction_accuracy": conditional_direction_accuracy,
        "conditional_direction_balanced_accuracy": (
            None
            if not valid_direction_recalls
            else float(np.mean(valid_direction_recalls))
        ),
        "conditional_down_recall": conditional_down_recall,
        "conditional_up_recall": conditional_up_recall,
        "strong_signal_accuracy": strong_signal_accuracy,
        "correct_strong_signal_count": correct_strong_signals,
        "hold_count": int((~predicted_event).sum()),
        "two_stage_sample_count": int(len(actual_values)),
    }


def compute_strong_regime_metrics(
    predicted_logret: np.ndarray,
    actual_logret: np.ndarray,
) -> dict:
    predicted_return = _simple_return_from_logret(predicted_logret)
    actual_return = _simple_return_from_logret(actual_logret)

    fall_strong = predicted_return < -0.003
    rise_strong = predicted_return >= 0.006
    strong = fall_strong | rise_strong

    fall_strong_count = int(fall_strong.sum())
    rise_strong_count = int(rise_strong.sum())
    strong_count = int(strong.sum())
    total_count = int(len(predicted_return))

    fall_strong_hits = int((fall_strong & (actual_return < 0.0)).sum())
    rise_strong_hits = int((rise_strong & (actual_return > 0.0)).sum())
    strong_hits = fall_strong_hits + rise_strong_hits

    return {
        "fall_strong_count": fall_strong_count,
        "rise_strong_count": rise_strong_count,
        "strong_regime_count": strong_count,
        "strong_regime_rate": _safe_rate(strong_count, total_count),
        "fall_strong_precision": _safe_rate(fall_strong_hits, fall_strong_count),
        "rise_strong_precision": _safe_rate(rise_strong_hits, rise_strong_count),
        "strong_regime_precision": _safe_rate(strong_hits, strong_count),
    }


def compute_signal_return_metrics(
    predicted_logret: np.ndarray,
    actual_logret: np.ndarray,
    horizon: int = 1,
    confidence_values: np.ndarray | None = None,
) -> dict:
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    actual_return = _simple_return_from_logret(actual_logret)
    signal = np.sign(predicted_logret)
    signal_return = signal * actual_return
    active_mask = signal != 0.0
    long_mask = signal > 0.0
    short_mask = signal < 0.0

    def average_pct(mask: np.ndarray) -> float | None:
        if int(mask.sum()) == 0:
            return None
        return float(signal_return[mask].mean() * 100.0)

    def staggered_cumulative_pct(
        returns: np.ndarray,
        mask: np.ndarray,
    ) -> float | None:
        if int(mask.sum()) == 0:
            return None

        # T+N forward returns from adjacent rows overlap. Treat each offset as
        # one independently rebalanced sleeve and average their terminal values.
        sleeve_terminal_values: list[float] = []
        for offset in range(min(horizon, len(returns))):
            sleeve_returns = returns[offset::horizon]
            sleeve_mask = mask[offset::horizon]
            invested_returns = np.where(sleeve_mask, sleeve_returns, 0.0)
            sleeve_terminal_values.append(float(np.prod(1.0 + invested_returns)))

        return float((np.mean(sleeve_terminal_values) - 1.0) * 100.0)

    resolved_confidence = (
        np.abs(predicted_logret)
        if confidence_values is None
        else np.asarray(confidence_values, dtype=float)
    )
    if len(resolved_confidence) != len(predicted_logret):
        raise ValueError("confidence_values must match predicted_logret length.")
    high_conf_cutoff = float(np.quantile(resolved_confidence, 0.7))
    high_conf_mask = active_mask & (resolved_confidence >= high_conf_cutoff)

    return {
        "buy_and_hold_avg_return_pct": float(actual_return.mean() * 100.0),
        "buy_and_hold_cumulative_return_pct": staggered_cumulative_pct(
            actual_return,
            np.ones(len(actual_return), dtype=bool),
        ),
        "model_signal_avg_return_pct": average_pct(active_mask),
        "model_signal_cumulative_return_pct": staggered_cumulative_pct(
            signal_return,
            active_mask,
        ),
        "model_signal_long_avg_return_pct": average_pct(long_mask),
        "model_signal_short_avg_return_pct": average_pct(short_mask),
        "model_signal_high_conf_avg_return_pct": average_pct(high_conf_mask),
        "model_signal_high_conf_cumulative_return_pct": staggered_cumulative_pct(
            signal_return,
            high_conf_mask,
        ),
        "model_signal_active_count": int(active_mask.sum()),
        "model_signal_high_conf_count": int(high_conf_mask.sum()),
        "cumulative_return_method": "staggered_non_overlapping_sleeves",
        "cumulative_return_horizon": int(horizon),
    }


def score_asymmetric_direction_objective(
    predicted_values: np.ndarray,
    actual_values: np.ndarray,
    rmse: float,
) -> float:
    metrics = compute_direction_metrics(predicted_values, actual_values)

    up_precision = _zero_if_none(metrics["up_precision"])
    down_precision = _zero_if_none(metrics["down_precision"])
    up_recall = _zero_if_none(metrics["up_recall"])
    down_recall = _zero_if_none(metrics["down_recall"])
    predicted_down_rate = _zero_if_none(metrics["predicted_down_rate"])
    actual_down_rate = _zero_if_none(metrics["actual_down_rate"])

    strong_metrics = compute_strong_regime_metrics(predicted_values, actual_values)
    side_precisions = [
        float(value)
        for value in (
            strong_metrics["fall_strong_precision"],
            strong_metrics["rise_strong_precision"],
        )
        if value is not None
    ]
    side_strong_precision = 0.0 if not side_precisions else float(np.mean(side_precisions))
    strong_regime_precision = _zero_if_none(strong_metrics["strong_regime_precision"])

    balanced_precision = (up_precision + down_precision) / 2.0
    balanced_recall = (up_recall + down_recall) / 2.0
    direction_mix_balance = 1.0 - abs(predicted_down_rate - actual_down_rate)
    min_strong_count = max(5, int(len(actual_values) * 0.05))
    strong_support = min(
        1.0,
        strong_metrics["strong_regime_count"] / max(1, min_strong_count),
    )
    strong_precision = (
        (strong_regime_precision * 0.6) + (side_strong_precision * 0.4)
    ) * strong_support

    return float(
        (metrics["direction_accuracy"] * 0.25)
        + (balanced_precision * 0.25)
        + (strong_precision * 0.20)
        + (balanced_recall * 0.13)
        + (down_precision * 0.09)
        + (down_recall * 0.04)
        + (direction_mix_balance * 0.04)
        - (rmse * 0.05)
    )


def build_supervised_frame(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
) -> pd.DataFrame:
    supervised = feature_df.copy()
    supervised["target_logret"] = (
        np.log(supervised["target_price"].shift(-horizon) / supervised["target_price"]) * 100
    )
    supervised["target_future_price"] = supervised["target_price"].shift(-horizon)
    supervised["target_date"] = supervised["Date"].shift(-horizon)

    required_columns = feature_columns + [
        "target_logret",
        "target_future_price",
        "target_date",
        "target_price",
    ]
    return supervised.dropna(subset=required_columns).copy()


def split_supervised_frame(
    supervised: pd.DataFrame,
    train_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Split by origin date and purge train labels that reach the test period."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    split_index = int(len(supervised) * train_ratio)
    if split_index <= 0 or split_index >= len(supervised):
        raise ValueError("The train/test split must leave rows on both sides.")

    pre_test_frame = supervised.iloc[:split_index].copy().reset_index(drop=True)
    test_frame = supervised.iloc[split_index:].copy().reset_index(drop=True)
    test_start_date = pd.to_datetime(test_frame["Date"].iloc[0], errors="coerce")
    train_target_dates = pd.to_datetime(pre_test_frame["target_date"], errors="coerce")
    train_frame = pre_test_frame.loc[train_target_dates < test_start_date].copy()
    train_frame = train_frame.reset_index(drop=True)
    purged_rows = int(len(pre_test_frame) - len(train_frame))

    if train_frame.empty:
        raise ValueError("No train rows remain after purging cross-boundary targets.")
    return train_frame, test_frame, pre_test_frame, purged_rows


def split_supervised_frame_at_date(
    supervised: pd.DataFrame,
    test_start_date: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Split at an explicit origin date and purge labels crossing the boundary."""
    boundary = pd.to_datetime(test_start_date, errors="coerce")
    if pd.isna(boundary):
        raise ValueError(f"Invalid test_start_date: {test_start_date}")

    origin_dates = pd.to_datetime(supervised["Date"], errors="coerce")
    split_candidates = np.flatnonzero((origin_dates >= boundary).to_numpy())
    if len(split_candidates) == 0:
        raise ValueError("test_start_date is after the supervised frame.")
    split_index = int(split_candidates[0])
    if split_index <= 0:
        raise ValueError("test_start_date must leave training rows before the test period.")

    pre_test_frame = supervised.iloc[:split_index].copy().reset_index(drop=True)
    test_frame = supervised.iloc[split_index:].copy().reset_index(drop=True)
    resolved_test_start = pd.to_datetime(test_frame["Date"].iloc[0], errors="coerce")
    train_target_dates = pd.to_datetime(pre_test_frame["target_date"], errors="coerce")
    train_frame = pre_test_frame.loc[train_target_dates < resolved_test_start].copy()
    train_frame = train_frame.reset_index(drop=True)
    purged_rows = int(len(pre_test_frame) - len(train_frame))
    if train_frame.empty or test_frame.empty:
        raise ValueError("Explicit date split must leave both train and test rows.")
    return train_frame, test_frame, pre_test_frame, purged_rows


def run_mean_return_baseline(feature_df: pd.DataFrame, config) -> dict:
    horizon = config.regression_style_fixed_horizon
    supervised = build_supervised_frame(feature_df, [], horizon)

    train_frame, test_frame, _pre_test_frame, purged_rows = split_supervised_frame(
        supervised,
        config.train_ratio,
    )

    mean_train_logret = float(train_frame["target_logret"].mean())
    predicted_logret = np.full(len(test_frame), mean_train_logret)
    current_price = test_frame["target_price"].to_numpy()
    future_price = test_frame["target_future_price"].to_numpy()
    predicted_future_price = current_price * np.exp(predicted_logret / 100.0)

    mae = float(mean_absolute_error(future_price, predicted_future_price))
    rmse = float(np.sqrt(mean_squared_error(future_price, predicted_future_price)))
    r2 = float(r2_score(future_price, predicted_future_price))
    direction_accuracy = _compute_direction_accuracy(
        predicted_future_price - current_price,
        future_price - current_price,
    )
    mape = float(np.mean(np.abs((future_price - predicted_future_price) / future_price)) * 100)

    baseline_future_price = current_price.copy()
    baseline_rmse = float(np.sqrt(mean_squared_error(future_price, baseline_future_price)))
    baseline_mae = float(mean_absolute_error(future_price, baseline_future_price))
    baseline_mape = float(
        np.mean(np.abs((future_price - baseline_future_price) / future_price)) * 100
    )

    actual_logret = test_frame["target_logret"].to_numpy()
    predicted_up_count = int((predicted_logret >= 0).sum())
    predicted_down_count = int((predicted_logret < 0).sum())
    predicted_up_precision: float | None = (
        direction_accuracy if predicted_up_count > 0 else None
    )

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2_score": r2,
        "direction_accuracy": direction_accuracy,
        "mape": mape,
        "actual_downside_count": int((actual_logret < 0).sum()),
        "actual_upside_count": int((actual_logret >= 0).sum()),
        "predicted_down_count": predicted_down_count,
        "predicted_down_rate": float(predicted_down_count / len(test_frame)),
        "predicted_down_precision": None,
        "predicted_up_count": predicted_up_count,
        "predicted_up_rate": float(predicted_up_count / len(test_frame)),
        "predicted_up_precision": predicted_up_precision,
        "mean_train_logret": mean_train_logret,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "baseline_mape": baseline_mape,
        "high_conf_threshold": None,
        "high_conf_long_accuracy": None,
        "high_conf_short_accuracy": None,
        "high_conf_long_count": None,
        "high_conf_short_count": None,
    }

    return {
        "experiment_name": "mean_return_baseline",
        "best_horizon": int(horizon),
        "best_horizon_direction_score": None,
        "selected_feature_count": 0,
        "selected_features": [],
        "feature_frame_start_date": serialize_timestamp(supervised["Date"].iloc[0]),
        "feature_frame_end_date": serialize_timestamp(supervised["Date"].iloc[-1]),
        "train_rows": int(len(train_frame)),
        "purged_train_rows": purged_rows,
        "test_rows": int(len(test_frame)),
        "train_start_date": serialize_timestamp(train_frame["Date"].iloc[0]),
        "train_end_date": serialize_timestamp(train_frame["Date"].iloc[-1]),
        "test_start_date": serialize_timestamp(test_frame["Date"].iloc[0]),
        "test_end_date": serialize_timestamp(test_frame["Date"].iloc[-1]),
        "metrics": metrics,
    }


def _build_delta_metrics(market_only_metrics: dict, market_news_metrics: dict) -> dict:
    delta = {
        "direction_accuracy": (
            market_news_metrics["direction_accuracy"] - market_only_metrics["direction_accuracy"]
        ),
        "rmse": market_news_metrics["rmse"] - market_only_metrics["rmse"],
        "mae": market_news_metrics["mae"] - market_only_metrics["mae"],
        "r2_score": market_news_metrics["r2_score"] - market_only_metrics["r2_score"],
        "mape": market_news_metrics["mape"] - market_only_metrics["mape"],
    }
    for key in (
        "model_signal_avg_return_pct",
        "model_signal_cumulative_return_pct",
        "model_signal_high_conf_avg_return_pct",
        "buy_and_hold_avg_return_pct",
    ):
        if key in market_news_metrics and key in market_only_metrics:
            mn = market_news_metrics[key]
            mo = market_only_metrics[key]
            if mn is not None and mo is not None:
                delta[key] = float(mn) - float(mo)
    return delta


def build_comparison_artifacts(
    market_only_result: dict,
    market_news_result: dict,
) -> tuple[pd.DataFrame, dict]:
    delta_metrics = _build_delta_metrics(
        market_only_result["metrics"],
        market_news_result["metrics"],
    )
    mo_m = market_only_result["metrics"]
    mn_m = market_news_result["metrics"]

    rows = [
        {
            "experiment_name": market_only_result["experiment_name"],
            "best_horizon": market_only_result["best_horizon"],
            "selected_feature_count": market_only_result["selected_feature_count"],
            "direction_accuracy": mo_m["direction_accuracy"],
            "rmse": mo_m["rmse"],
            "mae": mo_m["mae"],
            "r2_score": mo_m["r2_score"],
            "mape": mo_m["mape"],
            "model_signal_avg_return_pct": mo_m.get("model_signal_avg_return_pct"),
            "model_signal_cumulative_return_pct": mo_m.get("model_signal_cumulative_return_pct"),
            "model_signal_high_conf_avg_return_pct": mo_m.get(
                "model_signal_high_conf_avg_return_pct"
            ),
        },
        {
            "experiment_name": market_news_result["experiment_name"],
            "best_horizon": market_news_result["best_horizon"],
            "selected_feature_count": market_news_result["selected_feature_count"],
            "direction_accuracy": mn_m["direction_accuracy"],
            "rmse": mn_m["rmse"],
            "mae": mn_m["mae"],
            "r2_score": mn_m["r2_score"],
            "mape": mn_m["mape"],
            "model_signal_avg_return_pct": mn_m.get("model_signal_avg_return_pct"),
            "model_signal_cumulative_return_pct": mn_m.get("model_signal_cumulative_return_pct"),
            "model_signal_high_conf_avg_return_pct": mn_m.get(
                "model_signal_high_conf_avg_return_pct"
            ),
        },
        {
            "experiment_name": "market_news_minus_market_only",
            "best_horizon": (
                market_news_result["best_horizon"] - market_only_result["best_horizon"]
            ),
            "selected_feature_count": (
                market_news_result["selected_feature_count"]
                - market_only_result["selected_feature_count"]
            ),
            "direction_accuracy": delta_metrics["direction_accuracy"],
            "rmse": delta_metrics["rmse"],
            "mae": delta_metrics["mae"],
            "r2_score": delta_metrics["r2_score"],
            "mape": delta_metrics["mape"],
            "model_signal_avg_return_pct": delta_metrics.get("model_signal_avg_return_pct"),
            "model_signal_cumulative_return_pct": delta_metrics.get(
                "model_signal_cumulative_return_pct"
            ),
            "model_signal_high_conf_avg_return_pct": delta_metrics.get(
                "model_signal_high_conf_avg_return_pct"
            ),
        },
    ]

    comparison_payload = {
        "market_only": market_only_result,
        "market_news": market_news_result,
        "delta_market_news_minus_market_only": delta_metrics,
    }
    return pd.DataFrame(rows), comparison_payload
