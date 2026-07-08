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
from shared.market.data import (
    build_market_feature_frame,
    download_market_data,
    supplementary_ticker_feature_columns,
)
from shared.news.features import build_daily_news_feature_table, load_news_source_table
from shared.news.merge import (
    NEWS_ACTIVITY_FEATURE_COLUMNS,
    NEWS_SENTIMENT_FEATURE_COLUMNS,
    merge_news_features_into_market_frame,
)
from shared.training.lstm_pipeline import run_training_experiment
from shared.training.metrics import seed_everything


def _dedupe(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Ablation frame is missing columns: {missing}")
    return _dedupe(columns)


def _metric_delta(metrics: dict, baseline: dict, key: str) -> float | None:
    value = metrics.get(key)
    base_value = baseline.get(key)
    if value is None or base_value is None:
        return None
    return float(value) - float(base_value)


def run_news_ablation(
    config,
    output_root: Path | None = None,
    training_frame_path: Path | None = None,
    common_start_date_override: str | None = None,
) -> dict:
    seed_everything(config.random_seed)
    ticker_slug = config.target_ticker.lower().replace("^", "")
    root = output_root or training_data_path(ticker_slug, "news_ablation", "report.json").parent
    root.mkdir(parents=True, exist_ok=True)

    if training_frame_path is None:
        news_source = load_news_source_table(config.news_input_path)
        daily_news = build_daily_news_feature_table(news_source)
        raw_market = download_market_data(config)
        market_frame, _ = build_market_feature_frame(
            raw_market,
            config.target_ticker,
            supplementary_tickers=config.macro_tickers,
        )
        merged_frame, all_news_columns = merge_news_features_into_market_frame(
            market_frame,
            daily_news,
        )
    else:
        merged_frame = pd.read_csv(training_frame_path, encoding="utf-8-sig")
        merged_frame["Date"] = pd.to_datetime(merged_frame["Date"], errors="coerce")
        all_news_columns = list(NEWS_ACTIVITY_FEATURE_COLUMNS) + list(
            NEWS_SENTIMENT_FEATURE_COLUMNS
        ) + [
            column for column in merged_frame.columns if column.startswith("body_emb_")
        ]

    market_columns = _require_columns(
        merged_frame,
        list(config.market_feature_columns)
        + supplementary_ticker_feature_columns(
            config.macro_tickers,
            config.supplementary_ticker_feature_suffixes,
        ),
    )
    activity_columns = _require_columns(
        merged_frame,
        list(NEWS_ACTIVITY_FEATURE_COLUMNS),
    )
    sentiment_columns = _require_columns(
        merged_frame,
        list(NEWS_SENTIMENT_FEATURE_COLUMNS),
    )
    embedding_columns = _require_columns(
        merged_frame,
        [column for column in all_news_columns if column.startswith("body_emb_")],
    )

    common_frame = merged_frame.dropna(subset=market_columns).copy()
    if common_start_date_override is not None:
        common_frame = common_frame[
            pd.to_datetime(common_frame["Date"]) >= pd.Timestamp(common_start_date_override)
        ].copy()
    common_frame = common_frame.sort_values("Date").reset_index(drop=True)
    if common_frame.empty:
        raise ValueError("No common rows remain for news ablation.")
    common_start_date = pd.to_datetime(common_frame["Date"].iloc[0]).strftime("%Y-%m-%d")

    variants = [
        ("market_only", [], False),
        ("activity", activity_columns, False),
        ("sentiment", sentiment_columns, False),
        ("embedding", [], True),
        ("activity_sentiment", activity_columns + sentiment_columns, False),
        ("all_news", activity_columns + sentiment_columns, True),
    ]

    results: dict[str, dict] = {}
    for name, scalar_news_columns, use_embeddings in variants:
        variant_dir = root / name
        selected_scalar_columns = _dedupe(market_columns + scalar_news_columns)
        active_embedding_columns = embedding_columns if use_embeddings else None
        print(
            f"[ABLATION] {name}: scalar={len(selected_scalar_columns)} "
            f"embedding_pca={config.training_embedding_pca_components if use_embeddings else 0}",
            flush=True,
        )
        results[name] = run_training_experiment(
            experiment_name=f"news_ablation_{name}",
            feature_df=common_frame,
            candidate_feature_columns=_dedupe(
                selected_scalar_columns + (embedding_columns if use_embeddings else [])
            ),
            training_frame_output_path=None,
            predictions_output_path=variant_dir / "predictions.csv",
            model_output_path=variant_dir / "model.json",
            metadata_output_path=variant_dir / "metadata.json",
            config=config,
            forced_horizon=config.regression_style_fixed_horizon,
            forced_selected_features=selected_scalar_columns,
            embedding_columns_for_pca=active_embedding_columns,
            n_embedding_pca_components=config.training_embedding_pca_components,
            min_date=common_start_date,
        )

    baseline_metrics = results["market_only"]["metrics"]
    metric_keys = [
        "event_roc_auc",
        "event_pr_auc",
        "event_balanced_accuracy",
        "event_precision",
        "event_recall",
        "event_f1",
        "event_brier_score",
        "event_brier_skill_score",
    ]
    rows: list[dict] = []
    for name, _, _ in variants:
        result = results[name]
        metrics = result["metrics"]
        row = {
            "variant": name,
            "feature_count": result["selected_feature_count"],
            "test_start_date": result["test_start_date"],
            "test_end_date": result["test_end_date"],
            "test_rows": result["test_rows"],
            "actual_event_count": metrics.get("actual_event_count"),
            "predicted_event_count": metrics.get("predicted_event_count"),
        }
        for key in metric_keys:
            row[key] = metrics.get(key)
            row[f"delta_{key}"] = _metric_delta(metrics, baseline_metrics, key)
        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(root / "comparison.csv", index=False, encoding="utf-8-sig")
    payload = {
        "target_ticker": config.target_ticker,
        "horizon": config.regression_style_fixed_horizon,
        "common_start_date": common_start_date,
        "feature_blocks": {
            "market": market_columns,
            "activity": activity_columns,
            "sentiment": sentiment_columns,
            "embedding": embedding_columns,
        },
        "results": results,
        "comparison": rows,
    }
    write_json(payload, root / "report.json")
    return payload


def _format_rate(value: object) -> str:
    return "N/A" if value is None else f"{float(value) * 100:.1f}%"


def _print_report(payload: dict) -> None:
    print("\nNews ablation - common-period comparison")
    print("  Variant              AUC    PR-AUC  Balanced Precision Recall  Brier Skill")
    print("  " + "-" * 76)
    for row in payload["comparison"]:
        print(
            f"  {row['variant']:18s} "
            f"{_format_rate(row['event_roc_auc']):>7s} "
            f"{_format_rate(row['event_pr_auc']):>8s} "
            f"{_format_rate(row['event_balanced_accuracy']):>9s} "
            f"{_format_rate(row['event_precision']):>9s} "
            f"{_format_rate(row['event_recall']):>6s} "
            f"{row['event_brier_score']:.3f} "
            f"{_format_rate(row['event_brier_skill_score']):>6s}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run common-period news feature block ablations for Event prediction."
    )
    parser.add_argument("--target-ticker", default="QQQ")
    parser.add_argument("--ticker-preset", default="auto")
    parser.add_argument("--news-input", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--training-frame", default=None)
    parser.add_argument("--common-start-date", default=None)
    parser.add_argument("--lstm-epochs", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {"verbose_output": args.verbose}
    if args.lstm_epochs is not None:
        overrides["lstm_epochs"] = args.lstm_epochs
    config = make_training_config(
        ticker=args.target_ticker,
        news_input_path=args.news_input,
        preset=args.ticker_preset,
        **overrides,
    )
    payload = run_news_ablation(
        config,
        Path(args.output_root) if args.output_root else None,
        Path(args.training_frame) if args.training_frame else None,
        args.common_start_date,
    )
    _print_report(payload)


if __name__ == "__main__":
    main()
