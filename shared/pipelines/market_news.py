from __future__ import annotations

"""
shared 통합 학습 오케스트레이션 레이어.

이 파일의 역할은 세부 구현을 모두 담는 것이 아니라,
shared 파이프라인의 큰 흐름을 한눈에 보여주는 것이다.

세부 책임은 아래 하위 패키지로 분리했다.
- `shared.market.data`: 시장 데이터 다운로드와 가격 피처 생성
- `shared.news.features`: 문서 단위 뉴스 로드와 일자별 뉴스 집계
- `shared.news.merge`: 뉴스 피처를 시장 거래일 프레임에 병합
- `shared.training.xgboost_pipeline`: horizon 선택, XGBoost 학습, 평가, 비교 결과 생성

팀원이 이 파일을 먼저 읽으면 "실행 순서"를 파악할 수 있고,
필요할 때만 세부 모듈로 내려가도록 의도한 구조다.
"""

import pandas as pd

from shared.common.utils import write_json
from shared.config.schema import MarketNewsTrainingConfig
from shared.market.data import build_market_feature_frame, download_market_data
from shared.news.features import build_daily_news_feature_table, load_news_source_table
from shared.news.merge import merge_news_features_into_market_frame
from shared.training.xgboost_pipeline import (
    build_comparison_artifacts,
    run_aligned_horizon_comparison_suite,
    run_training_experiment,
    seed_everything,
)

__all__ = [
    "build_market_feature_frame",
    "build_daily_news_feature_table",
    "build_comparison_artifacts",
    "download_market_data",
    "load_news_source_table",
    "merge_news_features_into_market_frame",
    "run_market_news_training_pipeline",
    "run_aligned_horizon_comparison_suite",
    "run_training_experiment",
    "seed_everything",
]


REGRESSION_STYLE_MARKET_FEATURE_COLUMNS = [
    "ret_5",
    "ret_accel",
    "dist_to_ma5",
    "bb_pos",
    "rsi_14",
    "vol_shock",
    "vix_z_score_5",
    "drawdown",
    "vol_ratio",
    "rel_strength_5",
    "uup_shock_5",
    "tlt_shock_5",
    "hyg_ret",
    "target_spy_rel_ret",
]

REGRESSION_STYLE_NEWS_FEATURE_COLUMNS = [
    "sentiment_gap",
    "body_sentiment_gap",
    "sentiment_shock",
    "body_sentiment_score",
    "days_since_news",
    "body_sentiment_decay_3d",
]


def _require_feature_columns(
    feature_df: pd.DataFrame,
    candidate_columns: list[str],
    source_name: str,
) -> list[str]:
    """
    팀원 스크립트에서 고정으로 쓰는 피처들이 현재 프레임에 모두 존재하는지 확인한다.
    """
    missing_columns = [column for column in candidate_columns if column not in feature_df.columns]
    if missing_columns:
        raise ValueError(
            f"{source_name} is missing required regression-style feature columns: {missing_columns}"
        )
    return candidate_columns.copy()


def _resolve_aligned_comparison_start_date(
    merged_feature_df: pd.DataFrame,
    config: MarketNewsTrainingConfig,
) -> pd.Timestamp:
    """
    공정 비교용 시작일을 결정한다.

    기본값은 merged 프레임에서 실제 lagged 뉴스가 처음 관측되는 거래일이다.
    사용자가 강제로 시작일을 주면 그 값을 우선한다.
    """
    if config.aligned_comparison_start_date is not None:
        aligned_start_date = pd.to_datetime(
            config.aligned_comparison_start_date,
            errors="coerce",
        )
        if pd.isna(aligned_start_date):
            raise ValueError(
                f"Invalid aligned comparison start date: {config.aligned_comparison_start_date}"
            )
        return aligned_start_date

    if "news_count_lag1" not in merged_feature_df.columns:
        raise ValueError("Merged feature dataframe does not include news_count_lag1.")

    candidate_dates = merged_feature_df.loc[
        merged_feature_df["news_count_lag1"] > 0,
        "Date",
    ]
    if candidate_dates.empty:
        raise ValueError("Could not determine an aligned comparison start date from news coverage.")

    return pd.to_datetime(candidate_dates.iloc[0], errors="coerce")


def run_market_news_training_pipeline(
    config: MarketNewsTrainingConfig,
) -> dict:
    """
    뉴스 + 시장 데이터 기반 XGBoost 비교 실험을 end-to-end로 실행한다.

    실행 순서:
    1. 뉴스 원본 테이블 로드
    2. 일자별 뉴스 피처 생성
    3. 시장 가격 피처 생성
    4. market-only 실험 실행
    5. market+news 실험 실행
    6. 두 실험 결과를 비교 저장

    이 함수는 shared 파이프라인의 최상위 진입점이다.
    따라서 세부 계산식보다 "무엇이 어떤 순서로 연결되는가"가 드러나도록 유지한다.
    """
    seed_everything(config.random_seed)

    # 1) 크롤러 산출물 로드 후, 문서 단위 뉴스를 거래일 기준 숫자 피처로 압축한다.
    news_source_df = load_news_source_table(config.news_input_path)
    daily_news_features = build_daily_news_feature_table(news_source_df)
    daily_news_features.to_csv(
        config.daily_news_features_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # 2) 가격/거시 자산 시계열로부터 공통 시장 피처 프레임을 만든다.
    raw_market_df = download_market_data(config)
    market_feature_df, _market_feature_columns = build_market_feature_frame(
        raw_market_df,
        config.target_ticker,
    )
    regression_market_feature_columns = _require_feature_columns(
        market_feature_df,
        REGRESSION_STYLE_MARKET_FEATURE_COLUMNS,
        "market feature frame",
    )

    # 3) team regression script와 같은 고정 horizon/고정 피처로 baseline 실험을 수행한다.
    market_only_result = run_training_experiment(
        experiment_name="market_only",
        feature_df=market_feature_df,
        candidate_feature_columns=regression_market_feature_columns,
        training_frame_output_path=config.market_only_training_frame_output_path,
        predictions_output_path=config.market_only_predictions_output_path,
        model_output_path=config.market_only_model_output_path,
        metadata_output_path=config.market_only_metadata_output_path,
        config=config,
        forced_horizon=config.regression_style_fixed_horizon,
        forced_selected_features=regression_market_feature_columns,
    )

    # 4) team regression script 방식으로 뉴스 피처를 병합하고, 같은 horizon/피처 구조로 재학습한다.
    merged_feature_df, _news_feature_columns = merge_news_features_into_market_frame(
        market_feature_df,
        daily_news_features,
    )
    regression_news_feature_columns = _require_feature_columns(
        merged_feature_df,
        REGRESSION_STYLE_NEWS_FEATURE_COLUMNS,
        "market+news feature frame",
    )
    market_news_result = run_training_experiment(
        experiment_name="market_news",
        feature_df=merged_feature_df,
        candidate_feature_columns=(
            regression_market_feature_columns + regression_news_feature_columns
        ),
        training_frame_output_path=config.merged_training_frame_output_path,
        predictions_output_path=config.predictions_output_path,
        model_output_path=config.model_output_path,
        metadata_output_path=config.metadata_output_path,
        config=config,
        forced_horizon=config.regression_style_fixed_horizon,
        forced_selected_features=(
            regression_market_feature_columns + regression_news_feature_columns
        ),
    )

    # 5) 뉴스 커버리지가 실제로 존재하는 기간 + 동일 horizon 기준의 공정 비교 결과를 만든다.
    aligned_comparison_start_date = _resolve_aligned_comparison_start_date(
        merged_feature_df,
        config,
    )
    aligned_comparison_df, aligned_comparison_payload = run_aligned_horizon_comparison_suite(
        market_only_feature_df=market_feature_df,
        market_news_feature_df=merged_feature_df,
        market_only_feature_columns=regression_market_feature_columns,
        market_news_feature_columns=(
            regression_market_feature_columns + regression_news_feature_columns
        ),
        aligned_start_date=aligned_comparison_start_date,
        config=config,
        forced_market_only_features=regression_market_feature_columns,
        forced_market_news_features=(
            regression_market_feature_columns + regression_news_feature_columns
        ),
    )
    aligned_comparison_df.to_csv(
        config.aligned_comparison_output_path,
        index=False,
        encoding="utf-8-sig",
    )
    write_json(aligned_comparison_payload, config.aligned_comparison_metadata_output_path)

    # 6) 기존 best-horizon 결과도 그대로 비교표 형태로 저장한다.
    comparison_df, comparison_payload = build_comparison_artifacts(
        market_only_result,
        market_news_result,
    )
    comparison_df.to_csv(config.comparison_output_path, index=False, encoding="utf-8-sig")
    comparison_payload["aligned_shared_period_comparison"] = aligned_comparison_payload
    write_json(comparison_payload, config.comparison_metadata_output_path)
    return comparison_payload
