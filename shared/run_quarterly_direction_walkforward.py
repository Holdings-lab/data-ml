from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.common.utils import training_data_path, write_json
from shared.config.schema import make_training_config
from shared.run_event_walkforward import _load_feature_groups
from shared.run_quarterly_event_walkforward import _completed_quarters
from shared.training.lstm_pipeline import run_training_experiment
from shared.training.metrics import compute_thresholded_binary_direction_metrics


DIRECTION_METRICS = (
    "direction_accuracy",
    "macro_balanced_accuracy",
    "down_precision",
    "up_precision",
    "down_recall",
    "up_recall",
    "direction_roc_auc",
    "direction_pr_auc",
    "direction_brier_score",
)


def _pooled_direction_metrics(
    predictions: pd.DataFrame,
    direction_threshold: float,
) -> dict:
    actual_logret = predictions["Actual_LogRet"].to_numpy(dtype=float)
    predicted_up = (
        predictions["Direction_Class"].astype(str).str.lower().eq("up").to_numpy()
    )
    metrics = compute_thresholded_binary_direction_metrics(
        predicted_up,
        actual_logret,
        direction_threshold,
    )
    actual_event = np.abs(actual_logret) > direction_threshold
    actual_up = actual_logret > direction_threshold
    probability = predictions["Direction_Probability"].to_numpy(dtype=float)
    if int(actual_event.sum()) > 0 and len(np.unique(actual_up[actual_event])) == 2:
        metrics["direction_roc_auc"] = float(
            roc_auc_score(actual_up[actual_event].astype(int), probability[actual_event])
        )
        metrics["direction_pr_auc"] = float(
            average_precision_score(
                actual_up[actual_event].astype(int),
                probability[actual_event],
            )
        )
        metrics["direction_brier_score"] = float(
            brier_score_loss(
                actual_up[actual_event].astype(int),
                probability[actual_event],
            )
        )
    else:
        metrics["direction_roc_auc"] = None
        metrics["direction_pr_auc"] = None
        metrics["direction_brier_score"] = None
    metrics["sample_count"] = int(len(predictions))
    metrics["actual_event_count"] = int(actual_event.sum())
    return metrics


def _safe_nanmean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    if not bool(np.isfinite(numeric).any()):
        return None
    return float(np.nanmean(numeric))


def _safe_nanmin(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    if not bool(np.isfinite(numeric).any()):
        return None
    return float(np.nanmin(numeric))


def run_quarterly_direction_walk_forward(
    config,
    training_frame_path: Path,
    output_root: Path | None = None,
    start_quarter: str = "2022Q1",
    common_start_date: str = "2017-03-10",
    news_feature_profile: str = "none",
    include_market_only: bool = True,
) -> dict:
    frame, market_columns, scalar_news_columns = _load_feature_groups(
        config,
        training_frame_path,
        news_feature_profile,
    )
    frame = frame.loc[frame["Date"] >= pd.Timestamp(common_start_date)].copy()
    frame = frame.sort_values("Date").reset_index(drop=True)
    quarters = _completed_quarters(start_quarter, pd.to_datetime(frame["Date"].max()))
    ticker_slug = config.target_ticker.lower().replace("^", "")
    output_name = (
        "direction_walkforward_quarterly"
        if config.regression_style_fixed_horizon == 5
        else f"direction_walkforward_quarterly_h{config.regression_style_fixed_horizon}"
    )
    if news_feature_profile not in {"none", "market_only"}:
        output_name = f"{output_name}_{news_feature_profile}_news"
    root = output_root or training_data_path(
        ticker_slug,
        output_name,
        "report.json",
    ).parent
    root.mkdir(parents=True, exist_ok=True)

    variants: dict[str, list[str]] = {}
    if include_market_only or not scalar_news_columns:
        variants["market_only"] = market_columns
    if scalar_news_columns:
        news_variant = (
            "activity_sentiment"
            if news_feature_profile == "legacy"
            else f"{news_feature_profile}_news"
        )
        variants[news_variant] = market_columns + scalar_news_columns

    fold_rows: list[dict] = []
    all_predictions: dict[str, list[pd.DataFrame]] = {name: [] for name in variants}
    for quarter in quarters:
        fold_start = quarter.start_time.normalize()
        fold_end = quarter.end_time.normalize()
        for variant, features in variants.items():
            print(
                f"[DIRECTION-QUARTERLY] {quarter} {variant} "
                f"train<{fold_start.date()} test<={fold_end.date()}",
                flush=True,
            )
            fold_dir = root / variant / str(quarter)
            predictions_path = fold_dir / "predictions.csv"
            result = run_training_experiment(
                experiment_name=f"quarterly_direction_{variant}_{quarter}",
                feature_df=frame,
                candidate_feature_columns=features,
                training_frame_output_path=None,
                predictions_output_path=predictions_path,
                model_output_path=None,
                metadata_output_path=fold_dir / "metadata.json",
                config=config,
                forced_horizon=config.regression_style_fixed_horizon,
                forced_selected_features=features,
                min_date=common_start_date,
                test_start_date=fold_start,
                test_end_date=fold_end,
                training_mode="direction_only",
            )
            metrics = result["metrics"]
            row = {
                "quarter": str(quarter),
                "variant": variant,
                "train_rows": result["train_rows"],
                "test_rows": result["test_rows"],
                "actual_event_count": metrics["actual_event_count"],
                "direction_sample_count": metrics["direction_sample_count"],
            }
            row.update({key: metrics.get(key) for key in DIRECTION_METRICS})
            fold_rows.append(row)

            predictions = pd.read_csv(
                predictions_path,
                encoding="utf-8-sig",
                parse_dates=["Current_Date", "Target_Date"],
            )
            predictions.insert(0, "Quarter", str(quarter))
            all_predictions[variant].append(predictions)

    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(root / "fold_comparison.csv", index=False, encoding="utf-8-sig")

    stability_rows: list[dict] = []
    pooled_rows: list[dict] = []
    for variant, group in fold_frame.groupby("variant", sort=False):
        stability = {
            "variant": variant,
            "fold_count": int(len(group)),
            "total_test_rows": int(group["test_rows"].sum()),
            "total_direction_sample_count": int(group["direction_sample_count"].sum()),
        }
        for key in DIRECTION_METRICS:
            stability[f"mean_{key}"] = _safe_nanmean(group[key])
            stability[f"minimum_{key}"] = _safe_nanmin(group[key])
        latest_four = group.tail(4)
        stability["latest_4_mean_direction_accuracy"] = _safe_nanmean(
            latest_four["direction_accuracy"]
        )
        stability["latest_4_mean_macro_balanced_accuracy"] = _safe_nanmean(
            latest_four["macro_balanced_accuracy"]
        )
        stability_rows.append(stability)

        pooled_predictions = pd.concat(all_predictions[variant], ignore_index=True)
        if pooled_predictions["Current_Date"].duplicated().any():
            raise ValueError(f"Quarterly predictions overlap for {variant}.")
        pooled_predictions = pooled_predictions.sort_values("Current_Date").reset_index(
            drop=True
        )
        pooled_predictions.to_csv(
            root / f"{variant}_oos_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pooled = {
            "variant": variant,
            **_pooled_direction_metrics(
                pooled_predictions,
                config.lstm_direction_return_threshold,
            ),
        }
        pooled_rows.append(pooled)

    pd.DataFrame(stability_rows).to_csv(
        root / "stability_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(pooled_rows).to_csv(
        root / "pooled_oos_summary.csv", index=False, encoding="utf-8-sig"
    )
    payload = {
        "target_ticker": config.target_ticker,
        "horizon": config.regression_style_fixed_horizon,
        "common_start_date": common_start_date,
        "news_feature_profile": news_feature_profile,
        "training_mode": "direction_only",
        "random_seed": config.random_seed,
        "quarters": [str(quarter) for quarter in quarters],
        "fold_comparison": fold_rows,
        "stability_summary": stability_rows,
        "pooled_oos_summary": pooled_rows,
    }
    write_json(payload, root / "report.json")
    return payload


def _pct(value: float | None) -> str:
    if value is None or not np.isfinite(float(value)):
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def _print_report(payload: dict) -> None:
    print("\nQuarterly direction walk-forward - pooled OOS")
    for pooled in payload["pooled_oos_summary"]:
        print(
            f"  {pooled['variant']:20s} "
            f"Accuracy={_pct(pooled.get('direction_accuracy'))} "
            f"Balanced={_pct(pooled.get('macro_balanced_accuracy'))} "
            f"DownRecall={_pct(pooled.get('down_recall'))} "
            f"UpRecall={_pct(pooled.get('up_recall'))} "
            f"AUC={_pct(pooled.get('direction_roc_auc'))}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quarterly Direction walk-forward.")
    parser.add_argument(
        "--training-frame",
        default="data/training/qqq/market_only/training_frame.csv",
    )
    parser.add_argument("--target-ticker", default="QQQ")
    parser.add_argument("--ticker-preset", default="auto")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--start-quarter", default="2022Q1")
    parser.add_argument("--common-start-date", default="2017-03-10")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--news-feature-profile",
        choices=("none", "market_only", "legacy", "t1", "t1_hybrid"),
        default="none",
    )
    parser.add_argument("--news-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--horizon", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = make_training_config(
        args.target_ticker,
        preset=args.ticker_preset,
        verbose_output=args.verbose,
        random_seed=args.random_seed,
        regression_style_fixed_horizon=args.horizon,
    )
    payload = run_quarterly_direction_walk_forward(
        config,
        Path(args.training_frame),
        Path(args.output_root) if args.output_root else None,
        args.start_quarter,
        args.common_start_date,
        args.news_feature_profile,
        not args.news_only,
    )
    _print_report(payload)


if __name__ == "__main__":
    main()
