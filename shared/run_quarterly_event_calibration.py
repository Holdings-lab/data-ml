from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.common.utils import write_json
from shared.training.metrics import seed_everything


@dataclass(frozen=True)
class _PlattScaler:
    coefficient: float
    intercept: float
    method: str


def _event_target(frame: pd.DataFrame, threshold: float) -> np.ndarray:
    return (frame["Actual_LogRet"].abs().to_numpy(dtype=float) > threshold).astype(int)


def _as_binary(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.lower().eq("true").to_numpy(dtype=int)


def _logit_feature(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def _fit_platt_scaler(
    probability: np.ndarray,
    target: np.ndarray,
    random_seed: int,
) -> _PlattScaler:
    if len(np.unique(target)) != 2:
        raise ValueError("Probability calibration requires both event classes.")
    model = LogisticRegression(solver="lbfgs", random_state=random_seed)
    feature = _logit_feature(probability)
    target = np.asarray(target, dtype=int)
    model.fit(feature, target)
    coefficient = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])
    if coefficient > 0.0:
        return _PlattScaler(coefficient, intercept, "platt")

    # A probability calibration layer must not reverse the model's risk order.
    # If the unconstrained fit has a negative slope, preserve the original
    # ranking and only shift it to the observed past-quarter event rate.
    target_rate = float(target.mean())
    low, high = -30.0, 30.0
    raw_logit = feature[:, 0]
    for _ in range(100):
        midpoint = (low + high) / 2.0
        mean_probability = float(
            (1.0 / (1.0 + np.exp(-(raw_logit + midpoint)))).mean()
        )
        if mean_probability < target_rate:
            low = midpoint
        else:
            high = midpoint
    return _PlattScaler(1.0, (low + high) / 2.0, "intercept_shift_fallback")


def _apply_platt_scaler(
    model: _PlattScaler,
    probability: np.ndarray,
) -> np.ndarray:
    linear = model.coefficient * _logit_feature(probability)[:, 0] + model.intercept
    return 1.0 / (1.0 + np.exp(-linear))


def _select_threshold(
    target: np.ndarray,
    probability: np.ndarray,
    minimum_recall: float,
) -> tuple[float, dict]:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if int(target.sum()) == 0:
        raise ValueError("Threshold selection requires at least one event.")

    best: tuple[tuple[float, float, float], float, dict] | None = None
    for threshold in np.arange(0.0, 1.001, 0.01):
        prediction = (probability >= threshold).astype(int)
        if int(prediction.sum()) == 0:
            continue
        recall = float(recall_score(target, prediction, zero_division=0))
        if recall + 1e-12 < minimum_recall:
            continue
        precision = float(precision_score(target, prediction, zero_division=0))
        balanced = float(balanced_accuracy_score(target, prediction))
        score = (precision, balanced, float(threshold))
        details = {
            "precision": precision,
            "recall": recall,
            "balanced_accuracy": balanced,
        }
        if best is None or score > best[0]:
            best = (score, float(threshold), details)
    if best is None:
        raise ValueError("No threshold satisfies the minimum recall.")
    return best[1], best[2]


def _binary_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
) -> dict:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = np.asarray(prediction, dtype=int)
    has_both_classes = len(np.unique(target)) == 2
    tp = int(((prediction == 1) & (target == 1)).sum())
    fp = int(((prediction == 1) & (target == 0)).sum())
    fn = int(((prediction == 0) & (target == 1)).sum())
    tn = int(((prediction == 0) & (target == 0)).sum())
    event_rate = float(target.mean())
    brier = float(brier_score_loss(target, probability))
    brier_baseline = event_rate * (1.0 - event_rate)
    return {
        "sample_count": int(len(target)),
        "actual_event_count": int(target.sum()),
        "actual_event_rate": event_rate,
        "predicted_event_count": int(prediction.sum()),
        "true_positive_event_count": tp,
        "false_positive_event_count": fp,
        "false_negative_event_count": fn,
        "true_negative_event_count": tn,
        "event_roc_auc": (
            float(roc_auc_score(target, probability)) if has_both_classes else None
        ),
        "event_pr_auc": (
            float(average_precision_score(target, probability))
            if has_both_classes
            else None
        ),
        "event_pr_auc_baseline": event_rate,
        "event_balanced_accuracy": (
            float(balanced_accuracy_score(target, prediction))
            if has_both_classes
            else None
        ),
        "event_precision": float(precision_score(target, prediction, zero_division=0)),
        "event_recall": float(recall_score(target, prediction, zero_division=0)),
        "event_f1": float(f1_score(target, prediction, zero_division=0)),
        "event_brier_score": brier,
        "event_brier_baseline": brier_baseline,
        "event_brier_skill_score": (
            1.0 - brier / brier_baseline if brier_baseline > 0.0 else None
        ),
        "mean_predicted_probability": float(probability.mean()),
    }


def _reliability_rows(
    variant: str,
    mode: str,
    target: np.ndarray,
    probability: np.ndarray,
) -> list[dict]:
    edges = np.array([0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.000001])
    labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-80%", "80-100%"]
    band_index = np.digitize(probability, edges[1:-1], right=False)
    rows: list[dict] = []
    for index, label in enumerate(labels):
        mask = band_index == index
        if not bool(mask.any()):
            continue
        predicted_mean = float(probability[mask].mean())
        actual_rate = float(target[mask].mean())
        rows.append(
            {
                "variant": variant,
                "mode": mode,
                "probability_band": label,
                "sample_count": int(mask.sum()),
                "mean_predicted_probability": predicted_mean,
                "actual_event_rate": actual_rate,
                "calibration_gap": predicted_mean - actual_rate,
            }
        )
    return rows


def _load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        parse_dates=["Current_Date", "Target_Date"],
    )
    required = {
        "Quarter",
        "Current_Date",
        "Target_Date",
        "Actual_LogRet",
        "Event_Probability",
        "Predicted_Event",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction file is missing columns: {missing}")
    return frame.sort_values("Current_Date").reset_index(drop=True)


def run_quarterly_calibration(
    quarterly_root: Path,
    output_root: Path | None = None,
    warmup_quarters: int = 4,
    calibration_quarters: int = 4,
    event_return_threshold: float = 2.0,
    minimum_recall: float = 0.5,
    random_seed: int = 42,
) -> dict:
    seed_everything(random_seed)
    output = output_root or quarterly_root.with_name(f"{quarterly_root.name}_calibrated")
    output.mkdir(parents=True, exist_ok=True)

    variant_paths = {
        "market_only": quarterly_root / "market_only_oos_predictions.csv",
        "activity_sentiment": quarterly_root / "activity_sentiment_oos_predictions.csv",
    }
    fold_rows: list[dict] = []
    pooled_rows: list[dict] = []
    stability_rows: list[dict] = []
    selection_history: list[dict] = []
    reliability_rows: list[dict] = []

    for variant, path in variant_paths.items():
        frame = _load_predictions(path)
        frame["Actual_Event"] = _event_target(frame, event_return_threshold)
        quarters = list(dict.fromkeys(frame["Quarter"].tolist()))
        if len(quarters) <= warmup_quarters:
            raise ValueError(f"Not enough quarterly folds for {variant} calibration.")

        evaluated_parts: list[pd.DataFrame] = []
        for index, quarter in enumerate(quarters):
            if index < warmup_quarters:
                continue
            test = frame.loc[frame["Quarter"] == quarter].copy()
            test_start = pd.to_datetime(test["Current_Date"].min())
            allowed_quarters = quarters[max(0, index - calibration_quarters) : index]
            calibration = frame.loc[
                frame["Quarter"].isin(allowed_quarters)
                & (frame["Target_Date"] < test_start)
            ].copy()
            calibration_target = calibration["Actual_Event"].to_numpy(dtype=int)
            calibration_raw = calibration["Event_Probability"].to_numpy(dtype=float)
            scaler = _fit_platt_scaler(
                calibration_raw,
                calibration_target,
                random_seed,
            )
            calibration_probability = _apply_platt_scaler(scaler, calibration_raw)
            selected_threshold, threshold_metrics = _select_threshold(
                calibration_target,
                calibration_probability,
                minimum_recall,
            )

            test["Raw_Event_Probability"] = test["Event_Probability"].astype(float)
            test["Calibrated_Event_Probability"] = _apply_platt_scaler(
                scaler,
                test["Raw_Event_Probability"].to_numpy(dtype=float),
            )
            test["Raw_Predicted_Event"] = _as_binary(test["Predicted_Event"])
            test["Calibrated_Predicted_Event"] = (
                test["Calibrated_Event_Probability"] >= selected_threshold
            ).astype(int)
            test["Selected_Threshold"] = selected_threshold
            evaluated_parts.append(test)
            selection_history.append(
                {
                    "variant": variant,
                    "quarter": quarter,
                    "calibration_quarters": allowed_quarters,
                    "calibration_rows": int(len(calibration)),
                    "latest_calibration_target_date": calibration["Target_Date"].max().strftime(
                        "%Y-%m-%d"
                    ),
                    "selected_threshold": selected_threshold,
                    "calibration_precision": threshold_metrics["precision"],
                    "calibration_recall": threshold_metrics["recall"],
                    "calibration_balanced_accuracy": threshold_metrics[
                        "balanced_accuracy"
                    ],
                    "calibration_method": scaler.method,
                    "platt_coefficient": scaler.coefficient,
                    "platt_intercept": scaler.intercept,
                }
            )

        evaluated = pd.concat(evaluated_parts, ignore_index=True)
        evaluated.to_csv(
            output / f"{variant}_calibrated_oos_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        modes = {
            "raw": ("Raw_Event_Probability", "Raw_Predicted_Event"),
            "calibrated": (
                "Calibrated_Event_Probability",
                "Calibrated_Predicted_Event",
            ),
        }
        for quarter, group in evaluated.groupby("Quarter", sort=False):
            target = group["Actual_Event"].to_numpy(dtype=int)
            for mode, (probability_column, prediction_column) in modes.items():
                fold_rows.append(
                    {
                        "quarter": quarter,
                        "variant": variant,
                        "mode": mode,
                        **_binary_metrics(
                            target,
                            group[probability_column].to_numpy(dtype=float),
                            group[prediction_column].to_numpy(dtype=int),
                        ),
                    }
                )

        target = evaluated["Actual_Event"].to_numpy(dtype=int)
        for mode, (probability_column, prediction_column) in modes.items():
            probability = evaluated[probability_column].to_numpy(dtype=float)
            prediction = evaluated[prediction_column].to_numpy(dtype=int)
            pooled_rows.append(
                {
                    "variant": variant,
                    "mode": mode,
                    **_binary_metrics(target, probability, prediction),
                }
            )
            reliability_rows.extend(
                _reliability_rows(variant, mode, target, probability)
            )

    fold_frame = pd.DataFrame(fold_rows)
    for (variant, mode), group in fold_frame.groupby(["variant", "mode"], sort=False):
        valid_auc = group["event_roc_auc"].dropna()
        stability_rows.append(
            {
                "variant": variant,
                "mode": mode,
                "fold_count": int(len(group)),
                "valid_auc_fold_count": int(len(valid_auc)),
                "mean_event_roc_auc": float(valid_auc.mean()),
                "std_event_roc_auc": float(valid_auc.std(ddof=0)),
                "minimum_event_roc_auc": float(valid_auc.min()),
                "auc_above_random_fold_rate": float((valid_auc > 0.5).mean()),
                "mean_event_pr_auc": float(group["event_pr_auc"].mean()),
                "mean_event_precision": float(group["event_precision"].mean()),
                "mean_event_recall": float(group["event_recall"].mean()),
            }
        )

    fold_frame.to_csv(output / "fold_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pooled_rows).to_csv(
        output / "pooled_oos_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(stability_rows).to_csv(
        output / "stability_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(selection_history).to_csv(
        output / "selection_history.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(reliability_rows).to_csv(
        output / "probability_reliability.csv", index=False, encoding="utf-8-sig"
    )

    payload = {
        "quarterly_root": str(quarterly_root),
        "output_root": str(output),
        "warmup_quarters": warmup_quarters,
        "calibration_quarters": calibration_quarters,
        "event_return_threshold": event_return_threshold,
        "minimum_recall": minimum_recall,
        "random_seed": random_seed,
        "method": "platt_logistic_calibration_on_prior_quarters",
        "selection_history": selection_history,
        "fold_comparison": fold_rows,
        "pooled_oos_summary": pooled_rows,
        "stability_summary": stability_rows,
        "probability_reliability": reliability_rows,
    }
    write_json(payload, output / "report.json")
    return payload


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _print_report(payload: dict) -> None:
    print("\n과거 4개 분기 기반 확률 보정 - 합산 OOS")
    for row in payload["pooled_oos_summary"]:
        print(
            f"  {row['variant']:20s} {row['mode']:10s} "
            f"Precision={_pct(row['event_precision'])} "
            f"Recall={_pct(row['event_recall'])} "
            f"Brier={row['event_brier_score']:.3f}"
        )
    print("\n분기 평균 AUC (보정 전후 순서가 같으므로 동일해야 함)")
    for row in payload["stability_summary"]:
        print(
            f"  {row['variant']:20s} {row['mode']:10s} "
            f"AUC={_pct(row['mean_event_roc_auc'])}"
        )
    print(f"\n  결과 위치: {payload['output_root']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe quarterly Platt probability calibration."
    )
    parser.add_argument(
        "--quarterly-root",
        default="data/training/qqq/event_walkforward_quarterly_h1",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--warmup-quarters", type=int, default=4)
    parser.add_argument("--calibration-quarters", type=int, default=4)
    parser.add_argument("--event-return-threshold", type=float, default=2.0)
    parser.add_argument("--minimum-recall", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_quarterly_calibration(
        quarterly_root=Path(args.quarterly_root),
        output_root=Path(args.output_root) if args.output_root else None,
        warmup_quarters=args.warmup_quarters,
        calibration_quarters=args.calibration_quarters,
        event_return_threshold=args.event_return_threshold,
        minimum_recall=args.minimum_recall,
        random_seed=args.random_seed,
    )
    _print_report(payload)


if __name__ == "__main__":
    main()
