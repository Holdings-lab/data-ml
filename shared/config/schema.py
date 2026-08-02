from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from shared.common.utils import crawler_data_path, project_root, training_data_path
from shared.config.ticker_presets import (
    BASE_MARKET_FEATURE_COLUMNS,
    DEFAULT_SUPPLEMENTARY_TICKER_FEATURE_SUFFIXES,
    get_ticker_training_preset,
    ticker_slug,
)


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
    ucsb_news_count: int


def _crawler_data_path_no_create(*parts: str) -> Path:
    return project_root().joinpath("data", "crawler", *parts)


def _training_data_path_no_create(*parts: str) -> Path:
    return project_root().joinpath("data", "training", *parts)


@dataclass(frozen=True)
class MarketNewsTrainingConfig:
    """
    뉴스 피처와 시장 가격 피처를 함께 학습할 때 사용하는 실행 설정.

    한곳에서 기본값을 관리해 두면, 팀원이 스크립트를 실행할 때
    "어떤 입력을 읽고 어떤 결과를 어디에 쓰는지"를 훨씬 빠르게 이해할 수 있다.
    """
    target_ticker: str = "QQQ"
    preset_name: str = "default"
    macro_tickers: tuple[str, ...] = ("SPY", "^VIX", "TLT", "HYG", "UUP")
    supplementary_ticker_feature_suffixes: tuple[str, ...] = (
        DEFAULT_SUPPLEMENTARY_TICKER_FEATURE_SUFFIXES
    )
    market_feature_columns: tuple[str, ...] = BASE_MARKET_FEATURE_COLUMNS
    start_date: str = "2017-01-12"
    end_date: str = "2026-05-02"
    news_input_path: Path = field(
        default_factory=lambda: crawler_data_path(
            "features", "qqq", "merged_finbert_with_embeddings.csv"
        )
    )
    market_only_training_frame_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "qqq",
            "market_only",
            "training_frame.csv",
        )
    )
    market_only_predictions_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "qqq",
            "market_only",
            "predictions.csv",
        )
    )
    market_only_model_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "qqq",
            "market_only",
            "xgboost_model.json",
        )
    )
    market_only_metadata_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "qqq",
            "market_only",
            "metadata.json",
        )
    )
    daily_news_features_output_path: Path = field(
        default_factory=lambda: crawler_data_path("features", "qqq", "daily_news_features.csv")
    )
    merged_training_frame_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "qqq",
            "market_news",
            "training_frame.csv",
        )
    )
    predictions_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "qqq",
            "market_news",
            "predictions.csv",
        )
    )
    model_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "qqq",
            "market_news",
            "xgboost_model.json",
        )
    )
    metadata_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "qqq",
            "market_news",
            "metadata.json",
        )
    )
    comparison_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "qqq",
            "comparison",
            "market_model_comparison.csv",
        )
    )
    comparison_metadata_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "qqq",
            "comparison",
            "market_model_comparison.json",
        )
    )
    aligned_comparison_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "qqq",
            "comparison",
            "market_model_comparison_aligned.csv",
        )
    )
    aligned_comparison_metadata_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "qqq",
            "comparison",
            "market_model_comparison_aligned.json",
        )
    )
    cluster_model_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "qqq",
            "comparison",
            "volatility_cluster_model.json",
        )
    )
    cluster_report_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "qqq",
            "comparison",
            "volatility_cluster_report.json",
        )
    )
    cluster_visualization_output_path: Path = field(
        default_factory=lambda: training_data_path(
            "qqq",
            "comparison",
            "cluster_visualization.png",
        )
    )
    cluster_horizon: int = 5
    cluster_window_days: int = 5
    horizon_candidates: tuple[int, ...] = (5, 7, 10, 15)
    top_feature_count: int = 30
    training_embedding_pca_components: int = 5
    optuna_trials: int = 200
    train_ratio: float = 0.8
    random_seed: int = 42
    aligned_comparison_start_date: str | None = None
    regression_style_fixed_horizon: int = 5
    market_news_only: bool = False


def make_training_config(
    ticker: str,
    news_input_path: Path | str | None = None,
    preset: str | None = "auto",
    **overrides,
) -> MarketNewsTrainingConfig:
    """
    티커 이름 기반으로 출력 경로를 자동 생성한 MarketNewsTrainingConfig를 반환한다.

    QQQ 외 다른 ETF(XLE, XLK 등)를 추가할 때 출력 파일이 겹치지 않도록
    모든 경로를 ticker 이름으로 prefix한다.

    Parameters
    ----------
    ticker        : 예측 대상 티커 (예: "XLE", "QQQ")
    news_input_path : 해당 티커의 뉴스 임베딩 CSV 경로.
                    None이면 data/crawler/features/{ticker}/ 아래의
                    merged_finbert_with_embeddings.csv를 우선 사용하고,
                    없으면 예전 flat 파일명으로 fallback한다.
    **overrides   : MarketNewsTrainingConfig 필드 직접 덮어쓰기
                    (예: macro_tickers=(...), optuna_trials=300)
    """
    normalized_ticker = ticker.upper()
    t = ticker_slug(normalized_ticker)
    preset_overrides = get_ticker_training_preset(
        normalized_ticker,
        preset=preset,
    ).as_config_overrides()

    if news_input_path is None:
        nested_news_input_path = _crawler_data_path_no_create(
            "features",
            t,
            "merged_finbert_with_embeddings.csv",
        )
        legacy_news_filename = (
            "merged_finbert_with_embeddings.csv"
            if t == "qqq"
            else f"{t}_merged_finbert_with_embeddings.csv"
        )
        legacy_news_input_path = _crawler_data_path_no_create(
            "features",
            legacy_news_filename,
        )
        resolved_news_input_path = (
            nested_news_input_path
            if nested_news_input_path.exists()
            else legacy_news_input_path
        )
    else:
        resolved_news_input_path = Path(news_input_path)

    config_values = {
        "target_ticker": normalized_ticker,
        "news_input_path": resolved_news_input_path,
        "daily_news_features_output_path": _crawler_data_path_no_create(
            "features", t, "daily_news_features.csv"
        ),
        "market_only_training_frame_output_path": _training_data_path_no_create(
            t, "market_only", "training_frame.csv"
        ),
        "market_only_predictions_output_path": _training_data_path_no_create(
            t, "market_only", "predictions.csv"
        ),
        "market_only_model_output_path": _training_data_path_no_create(
            t, "market_only", "xgboost_model.json"
        ),
        "market_only_metadata_output_path": _training_data_path_no_create(
            t, "market_only", "metadata.json"
        ),
        "merged_training_frame_output_path": _training_data_path_no_create(
            t, "market_news", "training_frame.csv"
        ),
        "predictions_output_path": _training_data_path_no_create(
            t, "market_news", "predictions.csv"
        ),
        "model_output_path": _training_data_path_no_create(
            t, "market_news", "xgboost_model.json"
        ),
        "metadata_output_path": _training_data_path_no_create(
            t, "market_news", "metadata.json"
        ),
        "comparison_output_path": _training_data_path_no_create(
            t, "comparison", "market_model_comparison.csv"
        ),
        "comparison_metadata_output_path": _training_data_path_no_create(
            t, "comparison", "market_model_comparison.json"
        ),
        "aligned_comparison_output_path": _training_data_path_no_create(
            t, "comparison", "market_model_comparison_aligned.csv"
        ),
        "aligned_comparison_metadata_output_path": _training_data_path_no_create(
            t, "comparison", "market_model_comparison_aligned.json"
        ),
        "cluster_model_output_path": _training_data_path_no_create(
            t, "comparison", "volatility_cluster_model.json"
        ),
        "cluster_report_output_path": _training_data_path_no_create(
            t, "comparison", "volatility_cluster_report.json"
        ),
        "cluster_visualization_output_path": _training_data_path_no_create(
            t, "comparison", "cluster_visualization.png"
        ),
    }
    config_values.update(preset_overrides)
    config_values.update(overrides)

    return MarketNewsTrainingConfig(**config_values)

