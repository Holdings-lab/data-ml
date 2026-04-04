from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from shared.common.utils import crawler_data_path, training_data_path


class FedDocument(TypedDict, total=False):
    """
    레거시 FOMC 크롤러가 반환하던 원시 문서 스키마.

    기존 코드와의 호환성을 유지하기 위해 남겨둔다.
    """
    release_date: str
    release_time: str
    is_sep: bool
    doc_type: str
    label: str
    url: str
    title: str
    body_text: str


class StandardNewsDocument(TypedDict, total=False):
    """
    크롤러 후처리가 끝난 뒤, 학습 파이프라인으로 넘길 때 사용하는 표준 스키마.

    원본 수집기별 컬럼명이 달라도 이 스키마로 맞춘 뒤 사용하면
    이후 단계에서는 "문서 출처"보다 "문서가 어떤 속성을 갖는지"에 집중할 수 있다.
    """
    date: str
    category: str
    doc_type: str
    title: str
    body: str
    link: str
    body_original_length: int
    title_sentiment_score: float
    body_sentiment_score: float
    body_n_chunks: int


class DailyNewsFeatureRow(TypedDict, total=False):
    """
    문서 단위 데이터를 일자 단위 숫자 피처로 집계한 뒤의 스키마.

    학습 모델은 텍스트 문서를 직접 받기보다, 날짜별 이벤트 밀도와 감성 강도처럼
    숫자로 압축된 입력을 받는 편이 훨씬 안정적이다.
    """
    date: str
    news_count: int
    news_body_sentiment_mean: float
    news_body_sentiment_min: float
    news_body_sentiment_max: float
    fomc_news_count: int
    bis_news_count: int
    white_house_news_count: int


@dataclass(frozen=True)
class MarketNewsTrainingConfig:
    """
    뉴스 피처와 시장 가격 피처를 함께 학습할 때 사용하는 실행 설정.

    한곳에서 기본값을 관리해 두면, 팀원이 스크립트를 실행할 때
    "어떤 입력을 읽고 어떤 결과를 어디에 쓰는지"를 훨씬 빠르게 이해할 수 있다.
    """
    target_ticker: str = "QQQ"
    macro_tickers: tuple[str, ...] = ("SPY", "^VIX", "TLT", "HYG", "UUP")
    start_date: str = "2015-01-01"
    end_date: str = "2026-01-01"
    news_input_path: Path = field(
        default_factory=lambda: crawler_data_path("features", "merged_finbert.csv")
    )
    market_only_training_frame_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "market_only",
            "qqq_market_only_training_frame.csv",
        )
    )
    market_only_predictions_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "market_only",
            "qqq_market_only_predictions.csv",
        )
    )
    market_only_model_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "market_only",
            "qqq_market_only_xgboost_model.json",
        )
    )
    market_only_metadata_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "market_only",
            "qqq_market_only_metadata.json",
        )
    )
    daily_news_features_output_path: Path = field(
        default_factory=lambda: crawler_data_path("features", "daily_news_features.csv")
    )
    merged_training_frame_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "market_news",
            "qqq_market_news_training_frame.csv",
        )
    )
    predictions_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "market_news",
            "qqq_market_news_predictions.csv",
        )
    )
    model_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "market_news",
            "qqq_market_news_xgboost_model.json",
        )
    )
    metadata_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "market_news",
            "qqq_market_news_metadata.json",
        )
    )
    comparison_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "comparison",
            "qqq_market_model_comparison.csv",
        )
    )
    comparison_metadata_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "comparison",
            "qqq_market_model_comparison.json",
        )
    )
    aligned_comparison_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "comparison",
            "qqq_market_model_comparison_aligned.csv",
        )
    )
    aligned_comparison_metadata_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "comparison",
            "qqq_market_model_comparison_aligned.json",
        )
    )
    horizon_candidates: tuple[int, ...] = (5, 10, 15, 20)
    top_feature_count: int = 25
    optuna_trials: int = 50
    train_ratio: float = 0.8
    random_seed: int = 42
    aligned_comparison_start_date: str | None = None
    regression_style_fixed_horizon: int = 15
