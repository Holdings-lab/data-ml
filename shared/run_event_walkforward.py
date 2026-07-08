from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.common.utils import training_data_path, write_json
from shared.config.schema import make_training_config
from shared.market.data import supplementary_ticker_feature_columns
from shared.news.merge import (
    NEWS_ACTIVITY_FEATURE_COLUMNS,
    NEWS_SENTIMENT_FEATURE_COLUMNS,
    T1_HYBRID_NEWS_FEATURE_COLUMNS,
    build_t1_event_news_features,
)
from shared.training.lstm_pipeline import run_training_experiment


EVENT_METRICS = (
    "event_roc_auc",
    "event_pr_auc",
    "event_balanced_accuracy",
    "event_precision",
    "event_recall",
    "event_f1",
    "event_brier_score",
    "event_brier_skill_score",
)


def _dedupe(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def _load_feature_groups(
    config,
    training_frame_path: Path,
    news_feature_profile: str = "legacy",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = pd.read_csv(training_frame_path, encoding="utf-8-sig")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    market_columns = _dedupe(
        list(config.market_feature_columns)
        + supplementary_ticker_feature_columns(
            config.macro_tickers,
            config.supplementary_ticker_feature_suffixes,
        )
    )
    if news_feature_profile in {"none", "market_only"}:
        scalar_news_columns = []
    elif news_feature_profile == "legacy":
        scalar_news_columns = list(NEWS_ACTIVITY_FEATURE_COLUMNS) + list(
            NEWS_SENTIMENT_FEATURE_COLUMNS
        )
    elif news_feature_profile == "t1":
        frame, scalar_news_columns = build_t1_event_news_features(frame)
    elif news_feature_profile == "t1_hybrid":
        frame, _ = build_t1_event_news_features(frame)
        scalar_news_columns = list(T1_HYBRID_NEWS_FEATURE_COLUMNS)
    else:
        raise ValueError(
            "news_feature_profile must be one of: none, market_only, legacy, t1, t1_hybrid"
        )
    missing = [
        column
        for column in market_columns + scalar_news_columns
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Walk-forward frame is missing columns: {missing}")
    return frame, market_columns, scalar_news_columns


def run_walk_forward(
    config,
    training_frame_path: Path,
    output_root: Path | None = None,
    years: tuple[int, ...] = (2022, 2023, 2024, 2025, 2026),
    common_start_date: str = "2017-03-10",
) -> dict:
    frame, market_columns, scalar_news_columns = _load_feature_groups(
        config,
        training_frame_path,
    )
    frame = frame.loc[frame["Date"] >= pd.Timestamp(common_start_date)].copy()
    frame = frame.sort_values("Date").reset_index(drop=True)
    maximum_date = pd.to_datetime(frame["Date"].max())
    ticker_slug = config.target_ticker.lower().replace("^", "")
    root = output_root or training_data_path(
        ticker_slug,
        "event_walkforward",
        "report.json",
    ).parent
    root.mkdir(parents=True, exist_ok=True)

    variants = {
        "market_only": market_columns,
        "activity_sentiment": _dedupe(market_columns + scalar_news_columns),
    }
    rows: list[dict] = []
    results: dict[str, dict[str, dict]] = {name: {} for name in variants}
    for year in years:
        fold_start = pd.Timestamp(year=year, month=1, day=1)
        fold_end = min(pd.Timestamp(year=year, month=12, day=31), maximum_date)
        if fold_start > maximum_date:
            continue
        for variant, features in variants.items():
            print(
                f"[WALK-FORWARD] {year} {variant} "
                f"train<{fold_start.date()} test<={fold_end.date()}",
                flush=True,
            )
            fold_dir = root / variant / str(year)
            result = run_training_experiment(
                experiment_name=f"walkforward_{variant}_{year}",
                feature_df=frame,
                candidate_feature_columns=features,
                training_frame_output_path=None,
                predictions_output_path=fold_dir / "predictions.csv",
                model_output_path=None,
                metadata_output_path=fold_dir / "metadata.json",
                config=config,
                forced_horizon=config.regression_style_fixed_horizon,
                forced_selected_features=features,
                min_date=common_start_date,
                test_start_date=fold_start,
                test_end_date=fold_end,
            )
            results[variant][str(year)] = result
            metrics = result["metrics"]
            row = {
                "year": year,
                "variant": variant,
                "train_rows": result["train_rows"],
                "test_rows": result["test_rows"],
                "actual_event_count": metrics["actual_event_count"],
                "predicted_event_count": metrics["predicted_event_count"],
            }
            row.update({key: metrics.get(key) for key in EVENT_METRICS})
            rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(root / "fold_comparison.csv", index=False, encoding="utf-8-sig")
    summary_rows: list[dict] = []
    for variant, group in comparison.groupby("variant", sort=False):
        summary = {
            "variant": variant,
            "fold_count": int(len(group)),
            "total_test_rows": int(group["test_rows"].sum()),
            "total_actual_event_count": int(group["actual_event_count"].sum()),
        }
        for key in EVENT_METRICS:
            summary[f"mean_{key}"] = float(group[key].mean())
            summary[f"std_{key}"] = float(group[key].std(ddof=0))
            summary[f"minimum_{key}"] = float(group[key].min())
        summary_rows.append(summary)

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(root / "summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "target_ticker": config.target_ticker,
        "horizon": config.regression_style_fixed_horizon,
        "common_start_date": common_start_date,
        "years": sorted(comparison["year"].unique().tolist()),
        "feature_groups": {
            "market": market_columns,
            "activity_sentiment": scalar_news_columns,
        },
        "fold_comparison": rows,
        "summary": summary_rows,
        "results": results,
    }
    write_json(payload, root / "report.json")
    return payload


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _print_report(payload: dict) -> None:
    print("\nEvent walk-forward")
    print("  Year  Variant                  AUC  PR-AUC Balanced Precision Recall")
    print("  " + "-" * 72)
    for row in payload["fold_comparison"]:
        print(
            f"  {row['year']}  {row['variant']:20s} "
            f"{_pct(row['event_roc_auc']):>6s} "
            f"{_pct(row['event_pr_auc']):>7s} "
            f"{_pct(row['event_balanced_accuracy']):>8s} "
            f"{_pct(row['event_precision']):>9s} "
            f"{_pct(row['event_recall']):>6s}"
        )
    print("\n  Fold mean")
    for row in payload["summary"]:
        print(
            f"  {row['variant']:20s} "
            f"AUC={_pct(row['mean_event_roc_auc'])} "
            f"PR={_pct(row['mean_event_pr_auc'])} "
            f"Balanced={_pct(row['mean_event_balanced_accuracy'])} "
            f"Precision={_pct(row['mean_event_precision'])} "
            f"Recall={_pct(row['mean_event_recall'])}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expanding-window Event walk-forward.")
    parser.add_argument(
        "--training-frame",
        default="data/training/qqq/market_news/training_frame.csv",
    )
    parser.add_argument("--target-ticker", default="QQQ")
    parser.add_argument("--ticker-preset", default="auto")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--years", default="2022,2023,2024,2025,2026")
    parser.add_argument("--common-start-date", default="2017-03-10")
    parser.add_argument("--lstm-epochs", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    years = tuple(int(value.strip()) for value in args.years.split(",") if value.strip())
    overrides = {"verbose_output": args.verbose}
    if args.lstm_epochs is not None:
        overrides["lstm_epochs"] = args.lstm_epochs
    config = make_training_config(
        args.target_ticker,
        preset=args.ticker_preset,
        **overrides,
    )
    payload = run_walk_forward(
        config,
        Path(args.training_frame),
        Path(args.output_root) if args.output_root else None,
        years,
        args.common_start_date,
    )
    _print_report(payload)


if __name__ == "__main__":
    main()
