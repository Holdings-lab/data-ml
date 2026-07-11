from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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

from shared.common.utils import training_data_path, write_json
from shared.config.schema import make_training_config
from shared.run_event_walkforward import EVENT_METRICS, _load_feature_groups
from shared.training.lstm_pipeline import run_training_experiment


def _completed_quarters(
    start_quarter: str,
    maximum_date: pd.Timestamp,
) -> list[pd.Period]:
    first = pd.Period(start_quarter, freq="Q")
    last = maximum_date.to_period("Q")
    if maximum_date.month not in (3, 6, 9, 12) or maximum_date.day < 25:
        last -= 1
    if last < first:
        raise ValueError("No completed quarter remains after start_quarter.")
    return list(pd.period_range(first, last, freq="Q"))


def _pooled_metrics(
    predictions: pd.DataFrame,
    event_return_threshold: float,
) -> dict:
    actual_event = (
        predictions["Actual_LogRet"].abs() > event_return_threshold
    ).astype(int)
    predicted_event = (
        predictions["Predicted_Event"].astype(str).str.lower().eq("true").astype(int)
    )
    probability = predictions["Event_Probability"].astype(float)
    tp = int(((predicted_event == 1) & (actual_event == 1)).sum())
    fp = int(((predicted_event == 1) & (actual_event == 0)).sum())
    fn = int(((predicted_event == 0) & (actual_event == 1)).sum())
    tn = int(((predicted_event == 0) & (actual_event == 0)).sum())
    return {
        "sample_count": int(len(predictions)),
        "actual_event_count": int(actual_event.sum()),
        "predicted_event_count": int(predicted_event.sum()),
        "true_positive_event_count": tp,
        "false_positive_event_count": fp,
        "false_negative_event_count": fn,
        "true_negative_event_count": tn,
        "event_roc_auc": float(roc_auc_score(actual_event, probability)),
        "event_pr_auc": float(average_precision_score(actual_event, probability)),
        "event_pr_auc_baseline": float(actual_event.mean()),
        "event_balanced_accuracy": float(
            balanced_accuracy_score(actual_event, predicted_event)
        ),
        "event_precision": float(
            precision_score(actual_event, predicted_event, zero_division=0)
        ),
        "event_recall": float(recall_score(actual_event, predicted_event)),
        "event_f1": float(f1_score(actual_event, predicted_event)),
        "event_brier_score": float(brier_score_loss(actual_event, probability)),
    }


def run_quarterly_walk_forward(
    config,
    training_frame_path: Path,
    output_root: Path | None = None,
    start_quarter: str = "2022Q1",
    common_start_date: str = "2017-03-10",
    news_feature_profile: str = "legacy",
    include_market_only: bool = True,
    training_mode: str = "multitask",
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
        "event_walkforward_quarterly"
        if config.regression_style_fixed_horizon == 5
        else f"event_walkforward_quarterly_h{config.regression_style_fixed_horizon}"
    )
    if news_feature_profile not in {"legacy", "none", "market_only"}:
        output_name = f"{output_name}_{news_feature_profile}_news"
    if training_mode == "event_only":
        output_name = f"{output_name}_event_only"
    root = output_root or training_data_path(
        ticker_slug,
        output_name,
        "report.json",
    ).parent
    root.mkdir(parents=True, exist_ok=True)

    news_variant = (
        "activity_sentiment"
        if news_feature_profile == "legacy"
        else f"{news_feature_profile}_news"
    )
    if scalar_news_columns:
        variants = {news_variant: market_columns + scalar_news_columns}
    else:
        variants = {}
    if include_market_only or not variants:
        variants = {"market_only": market_columns, **variants}
    fold_rows: list[dict] = []
    all_predictions: dict[str, list[pd.DataFrame]] = {name: [] for name in variants}
    for quarter in quarters:
        fold_start = quarter.start_time.normalize()
        fold_end = quarter.end_time.normalize()
        for variant, features in variants.items():
            print(
                f"[QUARTERLY] {quarter} {variant} "
                f"train<{fold_start.date()} test<={fold_end.date()}",
                flush=True,
            )
            fold_dir = root / variant / str(quarter)
            predictions_path = fold_dir / "predictions.csv"
            result = run_training_experiment(
                experiment_name=f"quarterly_{variant}_{quarter}",
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
                training_mode=training_mode,
            )
            metrics = result["metrics"]
            row = {
                "quarter": str(quarter),
                "variant": variant,
                "train_rows": result["train_rows"],
                "test_rows": result["test_rows"],
                "actual_event_count": metrics["actual_event_count"],
                "predicted_event_count": metrics["predicted_event_count"],
            }
            row.update({key: metrics.get(key) for key in EVENT_METRICS})
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
        weights = group["test_rows"].to_numpy(dtype=float)
        valid_auc = group["event_roc_auc"].dropna()
        stability = {
            "variant": variant,
            "fold_count": int(len(group)),
            "total_test_rows": int(group["test_rows"].sum()),
            "valid_auc_fold_count": int(len(valid_auc)),
            "auc_above_random_fold_rate": float((valid_auc > 0.5).mean()),
            "auc_above_60_fold_rate": float((valid_auc >= 0.6).mean()),
        }
        for key in EVENT_METRICS:
            values = group[key].to_numpy(dtype=float)
            valid = np.isfinite(values)
            stability[f"mean_{key}"] = float(np.nanmean(values))
            stability[f"weighted_mean_{key}"] = float(
                np.average(values[valid], weights=weights[valid])
            )
            stability[f"std_{key}"] = float(np.nanstd(values, ddof=0))
            stability[f"minimum_{key}"] = float(np.nanmin(values))
        latest_four = group.tail(4)
        stability["latest_4_mean_event_roc_auc"] = float(
            latest_four["event_roc_auc"].mean()
        )
        stability["latest_4_mean_event_pr_auc"] = float(
            latest_four["event_pr_auc"].mean()
        )
        stability_rows.append(stability)

        pooled_predictions = pd.concat(all_predictions[variant], ignore_index=True)
        if pooled_predictions["Current_Date"].duplicated().any():
            raise ValueError(f"Quarterly predictions overlap for {variant}.")
        pooled_predictions = pooled_predictions.sort_values("Current_Date").reset_index(drop=True)
        pooled_predictions.to_csv(
            root / f"{variant}_oos_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pooled = {
            "variant": variant,
            **_pooled_metrics(
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
        "training_mode": training_mode,
        "event_selection_objective": config.lstm_event_selection_objective,
        "random_seed": config.random_seed,
        "quarters": [str(quarter) for quarter in quarters],
        "fold_comparison": fold_rows,
        "stability_summary": stability_rows,
        "pooled_oos_summary": pooled_rows,
    }
    write_json(payload, root / "report.json")
    return payload


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _print_report(payload: dict) -> None:
    print("\nQuarterly walk-forward - pooled OOS")
    for pooled in payload["pooled_oos_summary"]:
        print(
            f"  {pooled['variant']:20s} "
            f"Precision={_pct(pooled['event_precision'])} "
            f"Recall={_pct(pooled['event_recall'])} "
            f"Balanced={_pct(pooled['event_balanced_accuracy'])} "
            f"Brier={pooled['event_brier_score']:.3f}"
        )
    print("\nQuarter stability")
    for row in payload["stability_summary"]:
        print(
            f"  {row['variant']:20s} "
            f"AUC mean={_pct(row['mean_event_roc_auc'])} "
            f"std={_pct(row['std_event_roc_auc'])} "
            f"min={_pct(row['minimum_event_roc_auc'])} "
            f">50% folds={_pct(row['auc_above_random_fold_rate'])} "
            f"latest4={_pct(row['latest_4_mean_event_roc_auc'])}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quarterly expanding Event walk-forward.")
    parser.add_argument(
        "--training-frame",
        default="data/training/qqq/market_news/training_frame.csv",
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
        default="legacy",
    )
    parser.add_argument("--news-only", action="store_true")
    parser.add_argument("--event-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument(
        "--event-selection-objective",
        choices=("ranking", "precision_at_recall"),
        default="ranking",
    )
    parser.add_argument("--event-min-recall", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = make_training_config(
        args.target_ticker,
        preset=args.ticker_preset,
        verbose_output=args.verbose,
        random_seed=args.random_seed,
        regression_style_fixed_horizon=args.horizon,
        lstm_event_selection_objective=args.event_selection_objective,
        lstm_event_min_recall=args.event_min_recall,
    )
    payload = run_quarterly_walk_forward(
        config,
        Path(args.training_frame),
        Path(args.output_root) if args.output_root else None,
        args.start_quarter,
        args.common_start_date,
        args.news_feature_profile,
        not args.news_only,
        "event_only" if args.event_only else "multitask",
    )
    _print_report(payload)


if __name__ == "__main__":
    main()

