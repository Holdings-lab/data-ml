from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from shared.config.schema import make_training_config
from shared.config.ticker_presets import available_ticker_presets
from shared.pipelines.market_news import run_market_news_training_pipeline  # noqa: E402


def _parse_horizon_candidates(raw_value: str) -> tuple[int, ...]:
    """
    CLI 입력 문자열을 horizon 튜플로 바꾼다.

    예:
    - "5,10,15" -> (5, 10, 15)
    """
    horizon_values = []

    for value in raw_value.split(","):
        stripped = value.strip()
        if not stripped:
            continue
        horizon_values.append(int(stripped))

    if not horizon_values:
        raise ValueError("At least one horizon value is required.")

    return tuple(horizon_values)


def _parse_csv_values(raw_value: str) -> tuple[str, ...]:
    values = tuple(
        value.strip().upper()
        for value in raw_value.split(",")
        if value.strip()
    )
    if not values:
        raise ValueError("At least one comma-separated value is required.")
    return values


def _parse_macro_tickers(raw_value: str) -> tuple[str, ...]:
    return _parse_csv_values(raw_value)


def _parse_market_feature_columns(raw_value: str) -> tuple[str, ...]:
    columns = tuple(
        value.strip()
        for value in raw_value.split(",")
        if value.strip()
    )
    if not columns:
        raise ValueError("At least one market feature column is required.")
    return columns


def _parse_supplementary_ticker_feature_suffixes(raw_value: str) -> tuple[str, ...]:
    if raw_value.strip().lower() in {"", "none"}:
        return ()
    return _parse_market_feature_columns(raw_value)


def _optional_path_overrides(args: argparse.Namespace) -> dict:
    output_arg_to_config_field = {
        "market_only_training_output": "market_only_training_frame_output_path",
        "market_only_predictions_output": "market_only_predictions_output_path",
        "market_only_model_output": "market_only_model_output_path",
        "market_only_metadata_output": "market_only_metadata_output_path",
        "daily_news_output": "daily_news_features_output_path",
        "merged_training_output": "merged_training_frame_output_path",
        "predictions_output": "predictions_output_path",
        "model_output": "model_output_path",
        "metadata_output": "metadata_output_path",
        "comparison_output": "comparison_output_path",
        "comparison_metadata_output": "comparison_metadata_output_path",
        "aligned_comparison_output": "aligned_comparison_output_path",
        "aligned_comparison_metadata_output": "aligned_comparison_metadata_output_path",
        "cluster_model_output": "cluster_model_output_path",
        "cluster_report_output": "cluster_report_output_path",
        "cluster_visualization_output": "cluster_visualization_output_path",
    }
    return {
        config_field: Path(raw_path)
        for output_arg, config_field in output_arg_to_config_field.items()
        if (raw_path := getattr(args, output_arg)) is not None
    }


def _config_overrides_from_args(args: argparse.Namespace) -> dict:
    optional_scalar_overrides = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "top_feature_count": args.top_feature_count,
        "training_embedding_pca_components": args.training_embedding_pca_components,
        "optuna_trials": args.optuna_trials,
        "train_ratio": args.train_ratio,
        "random_seed": args.random_seed,
        "aligned_comparison_start_date": args.aligned_start_date,
        "regression_style_fixed_horizon": args.regression_style_fixed_horizon,
    }
    overrides = {
        key: value
        for key, value in optional_scalar_overrides.items()
        if value is not None
    }

    if args.macro_tickers is not None:
        overrides["macro_tickers"] = _parse_macro_tickers(args.macro_tickers)
    if args.horizons is not None:
        overrides["horizon_candidates"] = _parse_horizon_candidates(args.horizons)
    if args.market_feature_columns is not None:
        overrides["market_feature_columns"] = _parse_market_feature_columns(
            args.market_feature_columns
        )
    if args.supplementary_ticker_feature_suffixes is not None:
        overrides["supplementary_ticker_feature_suffixes"] = (
            _parse_supplementary_ticker_feature_suffixes(
                args.supplementary_ticker_feature_suffixes
            )
        )

    overrides["market_news_only"] = args.market_news_only
    overrides.update(_optional_path_overrides(args))
    return overrides


def parse_args() -> argparse.Namespace:
    """
    shared 실행 엔트리포인트에서 사용할 CLI 인자를 정의한다.

    기본값만으로도 돌아가게 해두되, 팀원이 필요할 때는
    경로와 학습 범위를 쉽게 바꿀 수 있게 만드는 것이 목표다.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare two XGBoost target-ticker regression experiments: "
            "market-only vs market-plus-crawler-news."
        )
    )
    parser.add_argument("--target-ticker", default="QQQ")
    parser.add_argument(
        "--ticker-preset",
        default="auto",
        help=(
            "Ticker preset to apply before CLI overrides. Use auto, none, or one of: "
            f"{', '.join(available_ticker_presets())}."
        ),
    )
    parser.add_argument(
        "--macro-tickers",
        default=None,
        help=(
            "Comma-separated macro tickers. If omitted, the ticker preset supplies them."
        ),
    )
    parser.add_argument(
        "--market-feature-columns",
        default=None,
        help=(
            "Comma-separated explicit market feature columns. If omitted, "
            "the ticker preset supplies the feature set."
        ),
    )
    parser.add_argument(
        "--supplementary-ticker-feature-suffixes",
        default=None,
        help=(
            "Comma-separated suffixes auto-added for non-fixed macro tickers, "
            "for example ret_5,ret_20,shock_5. Use none to rely only on explicit "
            "market feature columns."
        ),
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--news-input",
        default=None,
        help=(
            "News embedding CSV path. If omitted, the ticker-specific "
            "data/crawler/features/{ticker}/merged_finbert_with_embeddings.csv "
            "is used when available."
        ),
    )
    parser.add_argument(
        "--market-only-training-output",
        default=None,
    )
    parser.add_argument(
        "--market-only-predictions-output",
        default=None,
    )
    parser.add_argument("--market-only-model-output", default=None)
    parser.add_argument("--market-only-metadata-output", default=None)
    parser.add_argument(
        "--daily-news-output",
        default=None,
    )
    parser.add_argument(
        "--merged-training-output",
        default=None,
    )
    parser.add_argument(
        "--predictions-output",
        default=None,
    )
    parser.add_argument("--model-output", default=None)
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--comparison-output", default=None)
    parser.add_argument("--comparison-metadata-output", default=None)
    parser.add_argument(
        "--aligned-comparison-output",
        default=None,
    )
    parser.add_argument(
        "--aligned-comparison-metadata-output",
        default=None,
    )
    parser.add_argument("--cluster-model-output", default=None)
    parser.add_argument("--cluster-report-output", default=None)
    parser.add_argument("--cluster-visualization-output", default=None)
    parser.add_argument(
        "--horizons",
        default=None,
        help=(
            "Comma-separated horizon candidates, for example: 5,10,15,20. "
            "If omitted, the ticker preset supplies them."
        ),
    )
    parser.add_argument(
        "--aligned-start-date",
        default=None,
        help=(
            "Optional override for the fair-comparison start date. "
            "If omitted, the first trading day with lagged news coverage is used."
        ),
    )
    parser.add_argument("--top-feature-count", type=int, default=None)
    parser.add_argument(
        "--training-embedding-pca-components",
        type=int,
        default=None,
        help="Number of body_emb_* PCA components added to the main market_news training.",
    )
    parser.add_argument("--optuna-trials", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument(
        "--regression-style-fixed-horizon",
        type=int,
        default=None,
        help=(
            "Fixed horizon used for the main shared experiments so they stay close to "
            "training/train_regression.py."
        ),
    )
    parser.add_argument(
        "--market-news-only",
        action="store_true",
        default=False,
        help="Skip market-only baseline and aligned comparison - run only market+news training.",
    )
    return parser.parse_args()


def _print_high_conf_report(metrics: dict) -> None:
    threshold = metrics.get("high_conf_threshold")
    long_acc = metrics.get("high_conf_long_accuracy")
    long_n = metrics.get("high_conf_long_count", 0)
    short_acc = metrics.get("high_conf_short_accuracy")
    short_n = metrics.get("high_conf_short_count", 0)
    if threshold is None:
        return
    print(f"  [고확신 상위 30%] 기준 문턱값(LogRet 절대값): {threshold:.4f}")
    if long_acc is not None:
        print(f"  상승(Long) 확신 시 정확도: {long_acc * 100:.2f}%  (n={long_n})")
    if short_acc is not None:
        print(f"  하락(Short) 확신 시 정확도: {short_acc * 100:.2f}%  (n={short_n})")


def main() -> None:
    args = parse_args()

    config = make_training_config(
        ticker=args.target_ticker,
        news_input_path=(Path(args.news_input) if args.news_input is not None else None),
        preset=args.ticker_preset,
        **_config_overrides_from_args(args),
    )

    result = run_market_news_training_pipeline(config)
    market_news = result["market_news"]
    mean_baseline = result.get("mean_return_baseline", {})

    print("=== Shared XGBoost Training Completed ===")
    print(f"Target ticker: {config.target_ticker}")
    print(f"Ticker preset: {config.preset_name}")

    if mean_baseline:
        b_metrics = mean_baseline.get("metrics", {})
        print("--- Mean return baseline (항상 평균 수익률 예측) ---")
        print(f"  Direction accuracy : {b_metrics.get('direction_accuracy', 0) * 100:.2f}%")
        print(f"  RMSE               : {b_metrics.get('rmse', 0):.4f}")
        print(f"  Mean train logret  : {b_metrics.get('mean_train_logret', 0):+.4f}")
        print(f"  Test rows          : {mean_baseline.get('test_rows', 0)}")

    if not config.market_news_only:
        market_only = result["market_only"]
        delta = result["delta_market_news_minus_market_only"]
        aligned = result["aligned_shared_period_comparison"]
        aligned_summary = aligned["summary"]

        print("--- Market only ---")
        print(f"Best horizon: {market_only['best_horizon']} trading days")
        print(f"Selected features: {market_only['selected_feature_count']}")
        print(f"RMSE: {market_only['metrics']['rmse']:.4f}")
        print(f"Direction accuracy: {market_only['metrics']['direction_accuracy'] * 100:.2f}%")
        _print_high_conf_report(market_only["metrics"])

    print("--- Market + crawler news ---")
    print(f"Best horizon: {market_news['best_horizon']} trading days")
    print(f"Selected features: {market_news['selected_feature_count']}")
    print(f"RMSE: {market_news['metrics']['rmse']:.4f}")
    print(f"Direction accuracy: {market_news['metrics']['direction_accuracy'] * 100:.2f}%")
    _print_high_conf_report(market_news["metrics"])

    if not config.market_news_only:
        print("--- Delta (market+news - market_only) ---")
        print(f"RMSE delta: {delta['rmse']:.4f}")
        print(f"Direction accuracy delta: {delta['direction_accuracy'] * 100:.2f}%")
        print("--- Fair aligned comparison (shared horizon + shared period) ---")
        print(f"Aligned start date: {aligned['aligned_start_date']}")
        print(
            "Best shared horizon by direction accuracy delta: "
            f"{aligned_summary['best_shared_horizon_by_direction_accuracy_delta']} trading days"
        )
        print(
            "Aligned direction accuracy delta: "
            f"{aligned_summary['direction_accuracy_delta'] * 100:.2f}%"
        )
        print(f"Aligned RMSE delta: {aligned_summary['rmse_delta']:.4f}")
        print(f"Comparison CSV saved to: {config.comparison_output_path}")
        print(f"Comparison JSON saved to: {config.comparison_metadata_output_path}")
        print(f"Aligned comparison CSV saved to: {config.aligned_comparison_output_path}")
        print(
            "Aligned comparison JSON saved to: "
            f"{config.aligned_comparison_metadata_output_path}"
        )

    cluster_report = (
        result.get("predicted_return_cluster_report")
        or result.get("volatility_prediction_report")
        or result.get("volatility_cluster_report", {})
    )
    predicted_groups = cluster_report.get("predicted_groups", [])
    if predicted_groups:
        metrics = cluster_report.get("source_model_metrics", {})
        print(
            "--- Predicted return regimes "
            f"({cluster_report.get('horizon', config.cluster_horizon)}-trading-day "
            "model forecast vs current price, "
            f"{config.cluster_window_days}-calendar-day news window) ---"
        )
        if metrics:
            print(
                "Source model direction accuracy: "
                f"{metrics.get('direction_accuracy', 0.0) * 100:.2f}%"
            )
            print(f"Source model RMSE: {metrics.get('rmse', 0.0):.4f}")
        for group in predicted_groups:
            label = group["label"]
            count = group["count"]
            avg_pred_return = group.get("average_predicted_return_pct")
            avg_return_str = (
                "N/A"
                if avg_pred_return is None
                else f"{avg_pred_return:+.2f}% avg pred"
            )
            direction_accuracy = group.get("direction_accuracy")
            direction_str = (
                "N/A"
                if direction_accuracy is None
                else f"{direction_accuracy * 100:.1f}% dir acc"
            )
            dr = group.get("date_range", {})
            date_str = f"{dr.get('first', '?')} ~ {dr.get('last', '?')}" if dr else "N/A"
            print(
                f"  [{label:13s}]  n={count:4d}  {avg_return_str}  "
                f"{direction_str}  ({date_str})"
            )
        print(f"Cluster model saved to:         {config.cluster_model_output_path}")
        print(f"Cluster report saved to:        {config.cluster_report_output_path}")
        print(f"Cluster visualization saved to: {config.cluster_visualization_output_path}")


if __name__ == "__main__":
    main()
