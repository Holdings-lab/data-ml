from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from shared.config.schema import MarketNewsTrainingConfig
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


def parse_args() -> argparse.Namespace:
    """
    shared 실행 엔트리포인트에서 사용할 CLI 인자를 정의한다.

    기본값만으로도 돌아가게 해두되, 팀원이 필요할 때는
    경로와 학습 범위를 쉽게 바꿀 수 있게 만드는 것이 목표다.
    """
    default_config = MarketNewsTrainingConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Compare two XGBoost QQQ regression experiments: "
            "market-only vs market-plus-crawler-news."
        )
    )
    parser.add_argument("--target-ticker", default=default_config.target_ticker)
    parser.add_argument("--start-date", default=default_config.start_date)
    parser.add_argument("--end-date", default=default_config.end_date)
    parser.add_argument("--news-input", default=str(default_config.news_input_path))
    parser.add_argument(
        "--daily-news-output",
        default=str(default_config.daily_news_features_output_path),
    )
    parser.add_argument(
        "--merged-training-output",
        default=str(default_config.merged_training_frame_output_path),
    )
    parser.add_argument(
        "--predictions-output",
        default=str(default_config.predictions_output_path),
    )
    parser.add_argument("--model-output", default=str(default_config.model_output_path))
    parser.add_argument("--metadata-output", default=str(default_config.metadata_output_path))
    parser.add_argument(
        "--aligned-comparison-output",
        default=str(default_config.aligned_comparison_output_path),
    )
    parser.add_argument(
        "--aligned-comparison-metadata-output",
        default=str(default_config.aligned_comparison_metadata_output_path),
    )
    parser.add_argument(
        "--horizons",
        default=",".join(str(horizon) for horizon in default_config.horizon_candidates),
        help="Comma-separated horizon candidates, for example: 5,10,15,20",
    )
    parser.add_argument(
        "--aligned-start-date",
        default=default_config.aligned_comparison_start_date,
        help=(
            "Optional override for the fair-comparison start date. "
            "If omitted, the first trading day with lagged news coverage is used."
        ),
    )
    parser.add_argument("--top-feature-count", type=int, default=default_config.top_feature_count)
    parser.add_argument("--optuna-trials", type=int, default=default_config.optuna_trials)
    parser.add_argument("--train-ratio", type=float, default=default_config.train_ratio)
    parser.add_argument("--random-seed", type=int, default=default_config.random_seed)
    parser.add_argument(
        "--regression-style-fixed-horizon",
        type=int,
        default=default_config.regression_style_fixed_horizon,
        help=(
            "Fixed horizon used for the main shared experiments so they stay close to "
            "training/train_regression.py."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = MarketNewsTrainingConfig(
        target_ticker=args.target_ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        news_input_path=Path(args.news_input),
        daily_news_features_output_path=Path(args.daily_news_output),
        merged_training_frame_output_path=Path(args.merged_training_output),
        predictions_output_path=Path(args.predictions_output),
        model_output_path=Path(args.model_output),
        metadata_output_path=Path(args.metadata_output),
        aligned_comparison_output_path=Path(args.aligned_comparison_output),
        aligned_comparison_metadata_output_path=Path(args.aligned_comparison_metadata_output),
        horizon_candidates=_parse_horizon_candidates(args.horizons),
        top_feature_count=args.top_feature_count,
        optuna_trials=args.optuna_trials,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        aligned_comparison_start_date=args.aligned_start_date,
        regression_style_fixed_horizon=args.regression_style_fixed_horizon,
    )

    result = run_market_news_training_pipeline(config)
    market_only = result["market_only"]
    market_news = result["market_news"]
    delta = result["delta_market_news_minus_market_only"]
    aligned = result["aligned_shared_period_comparison"]
    aligned_summary = aligned["summary"]

    print("=== Shared XGBoost Comparison Completed ===")
    print(f"Target ticker: {config.target_ticker}")
    print("--- Market only ---")
    print(f"Best horizon: {market_only['best_horizon']} trading days")
    print(f"Selected features: {market_only['selected_feature_count']}")
    print(f"RMSE: {market_only['metrics']['rmse']:.4f}")
    print(f"Direction accuracy: {market_only['metrics']['direction_accuracy'] * 100:.2f}%")
    print("--- Market + crawler news ---")
    print(f"Best horizon: {market_news['best_horizon']} trading days")
    print(f"Selected features: {market_news['selected_feature_count']}")
    print(f"RMSE: {market_news['metrics']['rmse']:.4f}")
    print(f"Direction accuracy: {market_news['metrics']['direction_accuracy'] * 100:.2f}%")
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


if __name__ == "__main__":
    main()
