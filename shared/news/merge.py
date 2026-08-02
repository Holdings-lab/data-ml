from __future__ import annotations

import pandas as pd


NEUTRAL_FILL_DEFAULTS = {
    "title_neutral_prob": 1.0,
    "body_neutral_prob": 1.0,
}

CATEGORY_FEATURE_COLUMN_PREFIX = "category_"
BASE_CATEGORY_FILL_ZERO_COLUMNS = [
    "category_BIS",
    "category_EIA",
    "category_FOMC",
    "category_FRASER",
    "category_UCSB",
]

EMBEDDING_DECAY_MAX_DAYS = 5
EMBEDDING_DECAY_HALF_LIFE_DAYS = 3.0

SENTIMENT_FILL_ZERO_COLUMNS = [
    "news_count",
    "negative_news_count",
    "positive_news_count",
    "negative_news_ratio",
    "positive_news_ratio",
    "title_positive_prob",
    "title_negative_prob",
    "title_sentiment_score",
    "body_positive_prob",
    "body_negative_prob",
    "body_sentiment_score",
]

REGRESSION_STYLE_NEWS_FEATURE_COLUMNS = [
    "news_count_5d",
    "days_since_news",
    "sentiment_gap",
    "body_sentiment_gap",
    "sentiment_shock",
    "body_sentiment_5d_mean",
    "title_sentiment_5d_mean",
    "negative_news_spike_5d",
    "body_sentiment_decay_3d",
    "fomc_sentiment",
    "fomc_recent_5d",
    "sentiment_divergence",
]


def _rolling_zscore(series: pd.Series, window: int, min_periods: int = 5) -> pd.Series:
    rolling = series.rolling(window, min_periods=min_periods)
    mean = rolling.mean()
    std = rolling.std(ddof=0)
    zscore = (series - mean) / (std + 1e-9)
    return zscore.where(std > 0, 0.0).fillna(0.0)


def _zero_fill_columns(df: pd.DataFrame) -> list[str]:
    category_columns = [
        column
        for column in df.columns
        if column.startswith(CATEGORY_FEATURE_COLUMN_PREFIX)
    ]
    return list(
        dict.fromkeys(
            SENTIMENT_FILL_ZERO_COLUMNS
            + BASE_CATEGORY_FILL_ZERO_COLUMNS
            + category_columns
        )
    )


def _merge_daily_news_table(
    market_df: pd.DataFrame,
    daily_news_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    팀원 `train_regression.py`와 같은 방식으로 시장 데이터와 뉴스 일자 테이블을 병합한다.

    핵심은 날짜 기준 left join 뒤에 뉴스 결측을 보수적으로 채우는 것이다.
    """
    merged = market_df.copy()
    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce").dt.tz_localize(None)

    news = daily_news_df.copy()
    news["date"] = pd.to_datetime(news["date"], errors="coerce").dt.tz_localize(None)

    merged = merged.merge(news, left_on="Date", right_on="date", how="left")

    for column, default_value in NEUTRAL_FILL_DEFAULTS.items():
        if column not in merged.columns:
            merged[column] = default_value
        merged[column] = merged[column].fillna(default_value)

    zero_fill_columns = _zero_fill_columns(merged)
    for column in zero_fill_columns:
        if column not in merged.columns:
            merged[column] = 0.0

    # 스칼라 감성/카테고리: ffill 대신 0으로 채워 뉴스 없는 날은 무신호(0)로 처리한다.
    # ffill을 쓰면 며칠 전 뉴스 감성이 아무 뉴스 없는 날까지 그대로 전파돼
    # "뉴스 없음"과 "예전 뉴스 잔존 영향"이 섞여 왜곡될 수 있다.
    merged[zero_fill_columns] = merged[zero_fill_columns].fillna(0.0)

    # 임베딩은 마지막 뉴스 벡터를 최대 5일까지만 지수 감쇠해서 이어간다.
    # 5일 이후에는 오래된 문맥을 무신호(0)로 끊어 노이즈 누적을 줄인다.
    # 마지막 뉴스 이후 경과 일수 계산 (train_regression.py와 동일)
    # 모델이 "오늘 뉴스"와 "며칠 전 뉴스"를 구분해서 학습할 수 있도록 돕는다.
    last_news_date = merged["Date"].where(merged["news_count"] > 0).ffill()
    merged["days_since_news"] = (
        (merged["Date"] - last_news_date).dt.days.clip(upper=30).fillna(30).astype(float)
    )

    emb_cols = [c for c in merged.columns if c.startswith("body_emb_")]
    if emb_cols:
        last_embedding_values = merged[emb_cols].ffill().fillna(0.0)
        embedding_decay = 0.5 ** (
            merged["days_since_news"] / EMBEDDING_DECAY_HALF_LIFE_DAYS
        )
        embedding_decay = embedding_decay.where(
            merged["days_since_news"] <= EMBEDDING_DECAY_MAX_DAYS,
            0.0,
        )
        merged[emb_cols] = last_embedding_values.mul(embedding_decay, axis=0)

    merged = merged.fillna(0.0)
    merged["news_count_lag1"] = merged["news_count"].shift(1).fillna(0.0)
    return merged.drop(columns=["date"], errors="ignore")


def _build_regression_style_news_features(merged: pd.DataFrame) -> list[str]:
    """
    팀원 스크립트의 감성 파생 피처를 그대로 만든다.
    """
    last_body_sentiment = (
        merged["body_sentiment_score"].where(merged["news_count"] > 0).ffill().fillna(0.0)
    )

    merged["body_sentiment_3d_mean"] = merged["body_sentiment_score"].rolling(3).mean()
    merged["body_sentiment_5d_mean"] = merged["body_sentiment_score"].rolling(5).mean()
    merged["title_sentiment_3d_mean"] = merged["title_sentiment_score"].rolling(3).mean()
    merged["title_sentiment_5d_mean"] = merged["title_sentiment_score"].rolling(5).mean()
    merged["body_sentiment_trend"] = (
        merged["body_sentiment_score"] - merged["body_sentiment_score"].shift(3)
    )
    merged["title_sentiment_trend"] = (
        merged["title_sentiment_score"] - merged["title_sentiment_score"].shift(3)
    )
    merged["negative_news_spike"] = (
        merged["body_negative_prob"]
        / (merged["body_negative_prob"].rolling(10).mean() + 1e-9)
    )
    merged["negative_news_spike_5d"] = (
        merged["body_negative_prob"]
        / (merged["body_negative_prob"].rolling(5).mean() + 1e-9)
    )
    merged["news_count_5d"] = merged["news_count"].rolling(5).sum()
    merged["news_count_zscore_20d"] = _rolling_zscore(merged["news_count"], 20)
    merged["negative_count_ratio_5d"] = (
        merged["negative_news_count"].rolling(5).sum()
        / (merged["news_count_5d"] + 1e-9)
    )
    merged["sentiment_gap"] = merged["title_positive_prob"] - merged["title_negative_prob"]
    merged["body_sentiment_gap"] = (
        merged["body_positive_prob"] - merged["body_negative_prob"]
    )
    merged["sentiment_shock"] = (
        merged["sentiment_gap"] - merged["sentiment_gap"].rolling(5).mean()
    )
    merged["sentiment_shock_zscore_20d"] = _rolling_zscore(
        merged["sentiment_shock"],
        20,
    )
    merged["fomc_sentiment"] = merged["body_sentiment_score"] * merged["category_FOMC"]
    merged["fomc_sentiment_shock"] = (
        merged["fomc_sentiment"] - merged["fomc_sentiment"].rolling(20).mean()
    )
    merged["fomc_recent_5d"] = merged["category_FOMC"].rolling(5).max()
    merged["sentiment_divergence"] = (
        merged["title_sentiment_score"] - merged["body_sentiment_score"]
    ).abs()
    # 반감기 3/7/15일 감쇠 — days_since_news 는 항상 현재 날짜 이전 뉴스 기준이므로 lookahead 없음
    for half_life in (3, 5, 7, 15):
        merged[f"body_sentiment_decay_{half_life}d"] = last_body_sentiment * (
            0.5 ** (merged["days_since_news"] / half_life)
        )
    merged.fillna(0.0, inplace=True)
    emb_cols = [c for c in merged.columns if c.startswith("body_emb_")]
    return REGRESSION_STYLE_NEWS_FEATURE_COLUMNS + emb_cols


def merge_news_features_into_market_frame(
    market_df: pd.DataFrame,
    daily_news_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    시장 프레임에 회귀 스크립트 스타일 뉴스 피처를 붙인다.

    기존 shared 구조는 유지하되, 실제 피처 생성 로직은 팀원 스크립트 기준을 따른다.
    """
    merged = _merge_daily_news_table(market_df, daily_news_df)
    model_news_feature_columns = _build_regression_style_news_features(merged)
    return merged, model_news_feature_columns
