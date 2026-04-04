from __future__ import annotations

import pandas as pd


NEUTRAL_FILL_DEFAULTS = {
    "title_neutral_prob": 1.0,
    "body_neutral_prob": 1.0,
}

SENTIMENT_FILL_ZERO_COLUMNS = [
    "news_count",
    "title_positive_prob",
    "title_negative_prob",
    "title_sentiment_score",
    "body_positive_prob",
    "body_negative_prob",
    "body_sentiment_score",
    "category_BIS",
    "category_FOMC",
    "category_White House",
    "body_n_chunks",
]

REGRESSION_STYLE_NEWS_FEATURE_COLUMNS = [
    "sentiment_gap",
    "body_sentiment_gap",
    "sentiment_shock",
    "body_sentiment_score",
]


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

    for column in SENTIMENT_FILL_ZERO_COLUMNS:
        if column not in merged.columns:
            merged[column] = 0.0

    merged[SENTIMENT_FILL_ZERO_COLUMNS] = (
        merged[SENTIMENT_FILL_ZERO_COLUMNS]
        .ffill()
        .fillna(0.0)
    )

    merged = merged.ffill().fillna(0.0)
    merged["news_count_lag1"] = merged["news_count"].shift(1).fillna(0.0)
    return merged.drop(columns=["date"], errors="ignore")


def _build_regression_style_news_features(merged: pd.DataFrame) -> list[str]:
    """
    팀원 스크립트의 감성 파생 피처를 그대로 만든다.
    """
    merged["body_sentiment_3d_mean"] = merged["body_sentiment_score"].rolling(3).mean()
    merged["body_sentiment_5d_mean"] = merged["body_sentiment_score"].rolling(5).mean()
    merged["title_sentiment_3d_mean"] = merged["title_sentiment_score"].rolling(3).mean()
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
    merged["sentiment_gap"] = merged["title_positive_prob"] - merged["title_negative_prob"]
    merged["body_sentiment_gap"] = (
        merged["body_positive_prob"] - merged["body_negative_prob"]
    )
    merged["sentiment_shock"] = (
        merged["sentiment_gap"] - merged["sentiment_gap"].rolling(5).mean()
    )
    merged = merged.fillna(0.0)
    return REGRESSION_STYLE_NEWS_FEATURE_COLUMNS.copy()


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
