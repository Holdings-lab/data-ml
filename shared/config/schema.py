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
    ?덇굅??FOMC ?щ·?ш? 諛섑솚?섎뜕 ?먯떆 臾몄꽌 ?ㅽ궎留?

    湲곗〈 肄붾뱶????명솚?깆쓣 ?좎??섍린 ?꾪빐 ?④꺼?붾떎.
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
    ?щ·???꾩쿂由ш? ?앸궃 ?? ?숈뒿 ?뚯씠?꾨씪?몄쑝濡??섍만 ???ъ슜?섎뒗 ?쒖? ?ㅽ궎留?

    ?먮낯 ?섏쭛湲곕퀎 而щ읆紐낆씠 ?щ씪?????ㅽ궎留덈줈 留욎텣 ???ъ슜?섎㈃
    ?댄썑 ?④퀎?먯꽌??"臾몄꽌 異쒖쿂"蹂대떎 "臾몄꽌媛 ?대뼡 ?띿꽦??媛뽯뒗吏"??吏묒쨷?????덈떎.
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
    臾몄꽌 ?⑥쐞 ?곗씠?곕? ?쇱옄 ?⑥쐞 ?レ옄 ?쇱쿂濡?吏묎퀎???ㅼ쓽 ?ㅽ궎留?

    ?숈뒿 紐⑤뜽? ?띿뒪??臾몄꽌瑜?吏곸젒 諛쏄린蹂대떎, ?좎쭨蹂??대깽??諛?꾩? 媛먯꽦 媛뺣룄泥섎읆
    ?レ옄濡??뺤텞???낅젰??諛쏅뒗 ?몄씠 ?⑥뵮 ?덉젙?곸씠??
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
    ?댁뒪 ?쇱쿂? ?쒖옣 媛寃??쇱쿂瑜??④퍡 ?숈뒿?????ъ슜?섎뒗 ?ㅽ뻾 ?ㅼ젙.

    ?쒓납?먯꽌 湲곕낯媛믪쓣 愿由ы빐 ?먮㈃, ??먯씠 ?ㅽ겕由쏀듃瑜??ㅽ뻾????
    "?대뼡 ?낅젰???쎄퀬 ?대뼡 寃곌낵瑜??대뵒???곕뒗吏"瑜??⑥뵮 鍮좊Ⅴ寃??댄빐?????덈떎.
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
    cluster_horizon: int = 2
    cluster_window_days: int = 5
    horizon_candidates: tuple[int, ...] = (5, 7, 10, 15)
    top_feature_count: int = 30
    training_embedding_pca_components: int = 5
    use_news_embeddings: bool = False
    optuna_trials: int = 200
    train_ratio: float = 0.8
    random_seed: int = 42
    aligned_comparison_start_date: str | None = None
    regression_style_fixed_horizon: int = 2
    market_news_only: bool = False
    verbose_output: bool = False
    lstm_device: str = "auto"
    lstm_seq_len: int = 10
    lstm_hidden_size: int = 32
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_batch_size: int = 32
    lstm_epochs: int = 100
    lstm_learning_rate: float = 1e-3
    lstm_early_stopping_patience: int = 15
    lstm_huber_delta: float = 1.0
    lstm_return_loss_weight: float = 0.2
    lstm_event_loss_weight: float = 1.0
    lstm_direction_loss_weight: float = 1.0
    lstm_direction_return_threshold: float = 2.0
    lstm_event_min_recall: float = 0.4
    lstm_event_selection_objective: str = "ranking"
    lstm_event_probability_threshold: float | None = None
    lstm_direction_probability_threshold: float | None = 0.5
    lstm_weight_decay: float = 1e-4

def make_training_config(
    ticker: str,
    news_input_path: Path | str | None = None,
    preset: str | None = "auto",
    **overrides,
) -> MarketNewsTrainingConfig:
    """
    ?곗빱 ?대쫫 湲곕컲?쇰줈 異쒕젰 寃쎈줈瑜??먮룞 ?앹꽦??MarketNewsTrainingConfig瑜?諛섑솚?쒕떎.

    QQQ ???ㅻⅨ ETF(XLE, XLK ??瑜?異붽?????異쒕젰 ?뚯씪??寃뱀튂吏 ?딅룄濡?
    紐⑤뱺 寃쎈줈瑜?ticker ?대쫫?쇰줈 prefix?쒕떎.

    Parameters
    ----------
    ticker        : ?덉륫 ????곗빱 (?? "XLE", "QQQ")
    news_input_path : ?대떦 ?곗빱???댁뒪 ?꾨쿋??CSV 寃쎈줈.
                    None?대㈃ data/crawler/features/{ticker}/ ?꾨옒??
                    merged_finbert_with_embeddings.csv瑜??곗꽑 ?ъ슜?섍퀬,
                    ?놁쑝硫??덉쟾 flat ?뚯씪紐낆쑝濡?fallback?쒕떎.
    **overrides   : MarketNewsTrainingConfig ?꾨뱶 吏곸젒 ??뼱?곌린
                    (?? macro_tickers=(...), optuna_trials=300)
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





