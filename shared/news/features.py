from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


CATEGORY_TO_PREFIX = {
    "FOMC": "fomc",
    "BIS": "bis",
    "UCSB": "ucsb",
}

DOC_TYPE_TO_FEATURE = {
    "statement": "fomc_statement_count",
    "minutes": "fomc_minutes_count",
    "implementation_note": "fomc_implementation_note_count",
    "press_release": "bis_press_release_count",
    "presidential_actions": "ucsb_presidential_actions_count",
    "briefings_statements": "ucsb_briefings_statements_count",
    "executive_orders": "ucsb_executive_orders_count",
}


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    source_name: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            f"{source_name} is missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )


def _normalize_doc_type(raw_value: object) -> str:
    """
    문서 타입 문자열을 피처 이름으로 쓰기 쉬운 형태로 정규화한다.

    예:
    - "Presidential Actions" -> "presidential_actions"
    - "Briefings & Statements" -> "briefings_statements"
    """
    text = "" if pd.isna(raw_value) else str(raw_value).strip().lower()
    text = text.replace("&", "and")

    normalized_chars = []
    previous_was_underscore = False

    for char in text:
        if char.isalnum():
            normalized_chars.append(char)
            previous_was_underscore = False
            continue

        if previous_was_underscore:
            continue

        normalized_chars.append("_")
        previous_was_underscore = True

    normalized = "".join(normalized_chars).strip("_")
    normalized = normalized.replace("and", "")
    normalized = normalized.replace("__", "_").strip("_")
    return normalized


def _count_matched_keywords(raw_value: object) -> int:
    """
    White House 크롤러가 기록한 matched_keywords를 개수형 피처로 바꾼다.

    문자열 자체를 모델에 바로 넣기보다 "이 날 몇 개의 관련 키워드가 잡혔는가"를
    수치로 주는 편이 현재 XGBoost 구조와 더 잘 맞는다.
    """
    if pd.isna(raw_value):
        return 0

    keywords = [keyword.strip() for keyword in str(raw_value).split(",") if keyword.strip()]
    return len(keywords)


def load_news_source_table(input_path) -> pd.DataFrame:
    """
    crawler 후처리 결과인 merged_finbert.csv를 읽어 표준 형태로 정리한다.

    여기서 한 번 컬럼을 정리해 두면, 아래 단계는 크롤러 출처에 상관없이
    동일한 규칙으로 일자별 피처를 만들 수 있다.
    """
    news_df = pd.read_csv(input_path, encoding="utf-8-sig")

    required_columns = [
        "date",
        "category",
        "doc_type",
        "title",
        "body",
    ]
    _validate_required_columns(news_df, required_columns, str(input_path))

    prepared = news_df.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared = prepared.dropna(subset=["date"]).copy()

    prepared["category"] = prepared["category"].fillna("Unknown").astype(str)
    prepared["doc_type"] = prepared["doc_type"].fillna("unknown").astype(str)
    prepared["title"] = prepared["title"].fillna("").astype(str)
    prepared["body"] = prepared["body"].fillna("").astype(str)

    if "body_original_length" not in prepared.columns:
        prepared["body_original_length"] = prepared["body"].str.len()

    if "title_sentiment_score" not in prepared.columns:
        prepared["title_sentiment_score"] = 0.0

    if "body_sentiment_score" not in prepared.columns:
        prepared["body_sentiment_score"] = 0.0

    if "body_n_chunks" not in prepared.columns:
        prepared["body_n_chunks"] = 0

    prepared["title_sentiment_score"] = prepared["title_sentiment_score"].fillna(0.0)
    prepared["body_sentiment_score"] = prepared["body_sentiment_score"].fillna(0.0)
    prepared["body_n_chunks"] = prepared["body_n_chunks"].fillna(0).astype(int)
    prepared["body_original_length"] = prepared["body_original_length"].fillna(0).astype(int)
    prepared["doc_type_key"] = prepared["doc_type"].map(_normalize_doc_type)

    if "matched_keywords" in prepared.columns:
        prepared["matched_keywords_count"] = prepared["matched_keywords"].map(_count_matched_keywords)
    else:
        prepared["matched_keywords_count"] = 0

    prepared["is_negative_news"] = (prepared["body_sentiment_score"] <= -0.15).astype(int)
    prepared["is_positive_news"] = (prepared["body_sentiment_score"] >= 0.15).astype(int)
    return prepared


def build_daily_news_feature_table(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    팀원 `train_regression.py`가 기대하는 형태로 일자별 뉴스 피처를 만든다.

    핵심 규칙은 아래 두 가지다.
    1. 주말 뉴스는 다음 영업일(월요일)로 미룬다.
    2. 같은 날짜에 나온 문서는 평균을 내어 하루 1행으로 압축한다.
    """
    prepared = news_df.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.tz_localize(None)
    prepared = prepared.dropna(subset=["date"]).copy()

    if "category_BIS" not in prepared.columns:
        prepared["category_BIS"] = prepared["category"].eq("BIS").astype(int)
    if "category_FOMC" not in prepared.columns:
        prepared["category_FOMC"] = prepared["category"].eq("FOMC").astype(int)
    if "category_UCSB" not in prepared.columns:
        prepared["category_UCSB"] = prepared["category"].eq("UCSB").astype(int)

    probability_defaults = {
        "title_positive_prob": 0.0,
        "title_negative_prob": 0.0,
        "title_neutral_prob": 1.0,
        "body_positive_prob": 0.0,
        "body_negative_prob": 0.0,
        "body_neutral_prob": 1.0,
        "title_sentiment_score": 0.0,
        "body_sentiment_score": 0.0,
        "body_n_chunks": 0.0,
    }
    for column, default_value in probability_defaults.items():
        if column not in prepared.columns:
            prepared[column] = default_value
        prepared[column] = prepared[column].fillna(default_value)

    day_of_week = prepared["date"].dt.dayofweek
    if "is_weekend" not in prepared.columns:
        prepared["is_weekend"] = day_of_week.isin([5, 6]).astype(int)
    if "day_of_week_sin" not in prepared.columns:
        prepared["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    if "day_of_week_cos" not in prepared.columns:
        prepared["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    month = prepared["date"].dt.month
    if "month_sin" not in prepared.columns:
        prepared["month_sin"] = np.sin(2 * np.pi * month / 12)
    if "month_cos" not in prepared.columns:
        prepared["month_cos"] = np.cos(2 * np.pi * month / 12)

    # 팀원 스크립트와 같은 방식으로 주말 뉴스는 다음 월요일에 반영한다.
    prepared["date"] = prepared["date"] + pd.to_timedelta(
        np.where(
            day_of_week == 5,
            2,
            np.where(day_of_week == 6, 1, 0),
        ),
        unit="D",
    )

    regression_daily_columns = [
        "category_BIS",
        "category_FOMC",
        "category_UCSB",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "title_positive_prob",
        "title_negative_prob",
        "title_neutral_prob",
        "title_sentiment_score",
        "body_positive_prob",
        "body_negative_prob",
        "body_neutral_prob",
        "body_sentiment_score",
        "body_n_chunks",
    ]

    grouped = prepared.groupby("date", sort=True)
    daily = grouped[regression_daily_columns].mean(numeric_only=True).reset_index()
    daily["news_count"] = grouped.size().to_numpy()

    ordered_columns = ["date", "news_count", *regression_daily_columns]
    daily = daily[ordered_columns]
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily
