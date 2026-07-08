from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.common.utils import training_data_path, write_json
from shared.config.schema import make_training_config
from shared.run_event_walkforward import EVENT_METRICS, _load_feature_groups
from shared.training.lstm_pipeline import run_training_experiment


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def run_scalar_leave_one_out(
    config,
    training_frame_path: Path,
    output_root: Path | None = None,
    years: tuple[int, ...] = (2024, 2025, 2026),
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
        "scalar_news_loo",
        "report.json",
    ).parent
    root.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, str | None, list[str]]] = [
        ("all_scalar", None, market_columns + scalar_news_columns)
    ]
    variants.extend(
        (
            f"drop_{_slug(feature)}",
            feature,
            market_columns + [column for column in scalar_news_columns if column != feature],
        )
        for feature in scalar_news_columns
    )

    rows: list[dict] = []
    results: dict[str, dict[str, dict]] = {name: {} for name, _, _ in variants}
    for variant, dropped_feature, features in variants:
        for year in years:
            fold_start = pd.Timestamp(year=year, month=1, day=1)
            fold_end = min(pd.Timestamp(year=year, month=12, day=31), maximum_date)
            if fold_start > maximum_date:
                continue
            print(
                f"[SCALAR-LOO] {variant} year={year} drop={dropped_feature or 'none'}",
                flush=True,
            )
            fold_dir = root / variant / str(year)
            result = run_training_experiment(
                experiment_name=f"scalar_loo_{variant}_{year}",
                feature_df=frame,
                candidate_feature_columns=features,
                training_frame_output_path=None,
                predictions_output_path=None,
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
                "variant": variant,
                "dropped_feature": dropped_feature,
                "year": year,
                "test_rows": result["test_rows"],
            }
            row.update({key: metrics.get(key) for key in EVENT_METRICS})
            rows.append(row)

    fold_frame = pd.DataFrame(rows)
    fold_frame.to_csv(root / "fold_results.csv", index=False, encoding="utf-8-sig")
    summary_rows: list[dict] = []
    for (variant, dropped_feature), group in fold_frame.groupby(
        ["variant", "dropped_feature"],
        dropna=False,
        sort=False,
    ):
        row = {
            "variant": variant,
            "dropped_feature": None if pd.isna(dropped_feature) else dropped_feature,
            "fold_count": int(len(group)),
        }
        for key in EVENT_METRICS:
            row[f"mean_{key}"] = float(group[key].mean())
            row[f"minimum_{key}"] = float(group[key].min())
        summary_rows.append(row)

    baseline = next(row for row in summary_rows if row["variant"] == "all_scalar")
    for row in summary_rows:
        for key in EVENT_METRICS:
            row[f"delta_mean_{key}"] = (
                row[f"mean_{key}"] - baseline[f"mean_{key}"]
            )
        row["delta_ranking_score"] = 0.5 * (
            row["delta_mean_event_roc_auc"] + row["delta_mean_event_pr_auc"]
        )
        row["delta_operating_score"] = (
            0.5 * row["delta_mean_event_balanced_accuracy"]
            + 0.25 * row["delta_mean_event_precision"]
            + 0.25 * row["delta_mean_event_recall"]
        )

    summary_rows.sort(key=lambda row: row["delta_ranking_score"], reverse=True)
    pd.DataFrame(summary_rows).to_csv(
        root / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "target_ticker": config.target_ticker,
        "horizon": config.regression_style_fixed_horizon,
        "years": list(years),
        "common_start_date": common_start_date,
        "scalar_news_features": scalar_news_columns,
        "fold_results": rows,
        "summary": summary_rows,
        "results": results,
    }
    write_json(payload, root / "report.json")
    return payload


def _pct(value: float) -> str:
    return f"{value * 100:+.1f}%p"


def _print_report(payload: dict) -> None:
    print("\nScalar news leave-one-out (delta vs all scalar)")
    print("  Dropped feature                  Rank     Balanced Precision Recall   Brier")
    print("  " + "-" * 78)
    for row in payload["summary"]:
        feature = row["dropped_feature"] or "none (baseline)"
        print(
            f"  {feature:30s} "
            f"{_pct(row['delta_ranking_score']):>8s} "
            f"{_pct(row['delta_mean_event_balanced_accuracy']):>9s} "
            f"{_pct(row['delta_mean_event_precision']):>9s} "
            f"{_pct(row['delta_mean_event_recall']):>6s} "
            f"{row['delta_mean_event_brier_score']:+.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scalar news leave-one-out walk-forward.")
    parser.add_argument(
        "--training-frame",
        default="data/training/qqq/market_news/training_frame.csv",
    )
    parser.add_argument("--target-ticker", default="QQQ")
    parser.add_argument("--ticker-preset", default="auto")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--years", default="2024,2025,2026")
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
    payload = run_scalar_leave_one_out(
        config,
        Path(args.training_frame),
        Path(args.output_root) if args.output_root else None,
        years,
        args.common_start_date,
    )
    _print_report(payload)


if __name__ == "__main__":
    main()
