from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


CATEGORY_FEATURE_COLUMN_PREFIX = "category_"
BASE_CATEGORY_FEATURE_COLUMNS: tuple[str, ...] = (
    "category_BIS",
    "category_EIA",
    "category_FOMC",
    "category_FRASER",
    "category_UCSB",
)


def _parse_embedding_string(x: object) -> "list[float] | None":
    if x is None:
        return None
    try:
        if bool(pd.isna(x)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(x, str) and not x.strip():
        return None
    import ast
    try:
        parsed = ast.literal_eval(str(x))
        if not isinstance(parsed, (list, tuple)):
            return None
        return [float(v) for v in parsed]
    except (TypeError, ValueError, SyntaxError):
        return None


def _validate_embedding_lists(
    emb_lists: pd.Series,
    source_name: str,
    column_name: str,
) -> int:
    invalid_rows = emb_lists[emb_lists.map(lambda value: value is None)].index.tolist()
    if invalid_rows:
        raise ValueError(
            f"{source_name} has invalid or missing {column_name} values. "
            f"Example row indices: {invalid_rows[:5]}"
        )

    lengths = emb_lists.map(len)
    empty_rows = lengths[lengths <= 0].index.tolist()
    if empty_rows:
        raise ValueError(
            f"{source_name} has empty {column_name} vectors. "
            f"Example row indices: {empty_rows[:5]}"
        )

    dimension_counts = lengths.value_counts().sort_index()
    if len(dimension_counts) != 1:
        expected_dim = int(dimension_counts.idxmax())
        mismatched_rows = lengths[lengths != expected_dim].index.tolist()
        raise ValueError(
            f"{source_name} has inconsistent {column_name} dimensions: "
            f"{dimension_counts.to_dict()}. "
            f"Expected the most common dimension {expected_dim}; "
            f"example mismatched row indices: {mismatched_rows[:5]}"
        )

    return int(lengths.iloc[0])


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


def _normalize_category(raw_value: object) -> str:
    text = "" if pd.isna(raw_value) else str(raw_value).strip()
    if not text:
        return "Unknown"

    upper_text = text.upper()
    if upper_text == "FOMC":
        return "FOMC"
    if upper_text == "BIS":
        return "BIS"
    if upper_text == "EIA" or "ENERGY INFORMATION ADMINISTRATION" in upper_text:
        return "EIA"
    if upper_text == "FRASER" or "FEDERAL RESERVE ARCHIVAL SYSTEM" in upper_text:
        return "FRASER"
    if "UCSB" in upper_text or "PRESIDENCY PROJECT" in upper_text:
        return "UCSB"
    return text


def _category_feature_column(category: object) -> str:
    category_text = _normalize_category(category)
    normalized_chars: list[str] = []
    previous_was_underscore = False

    for char in category_text.upper():
        if char.isalnum():
            normalized_chars.append(char)
            previous_was_underscore = False
            continue

        if previous_was_underscore:
            continue

        normalized_chars.append("_")
        previous_was_underscore = True

    suffix = "".join(normalized_chars).strip("_") or "UNKNOWN"
    return f"{CATEGORY_FEATURE_COLUMN_PREFIX}{suffix}"


def _ordered_category_feature_columns(categories: pd.Series) -> list[str]:
    observed_columns = {
        _category_feature_column(category)
        for category in categories.dropna().unique()
    }
    ordered_base_columns = [
        column for column in BASE_CATEGORY_FEATURE_COLUMNS if column in observed_columns
    ]
    extra_columns = sorted(observed_columns - set(BASE_CATEGORY_FEATURE_COLUMNS))
    return ordered_base_columns + extra_columns


def _ensure_category_feature_columns(prepared: pd.DataFrame) -> list[str]:
    category_columns = _ordered_category_feature_columns(prepared["category"])
    row_category_columns = prepared["category"].map(_category_feature_column)

    for column in category_columns:
        prepared[column] = row_category_columns.eq(column).astype(int)

    return category_columns


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
    crawler 후처리 결과인 merged_finbert_with_embeddings.csv를 읽어 표준 형태로 정리한다.

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

    prepared["category"] = prepared["category"].map(_normalize_category)
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

    if "body_summary_embedding" in prepared.columns:
        embedding_column = "body_summary_embedding"
        emb_lists = prepared["body_summary_embedding"].map(_parse_embedding_string)
        emb_dim = _validate_embedding_lists(emb_lists, str(input_path), embedding_column)
        emb_df = pd.DataFrame(
            emb_lists.tolist(),
            index=prepared.index,
            columns=[f"body_emb_{i}" for i in range(emb_dim)],
        )
        prepared = pd.concat([prepared, emb_df], axis=1)

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

    category_feature_columns = _ensure_category_feature_columns(prepared)

    probability_defaults = {
        "title_positive_prob": 0.0,
        "title_negative_prob": 0.0,
        "title_neutral_prob": 1.0,
        "body_positive_prob": 0.0,
        "body_negative_prob": 0.0,
        "body_neutral_prob": 1.0,
        "title_sentiment_score": 0.0,
        "body_sentiment_score": 0.0,
    }
    for column, default_value in probability_defaults.items():
        if column not in prepared.columns:
            prepared[column] = default_value
        prepared[column] = prepared[column].fillna(default_value)

    if "is_negative_news" not in prepared.columns:
        prepared["is_negative_news"] = (prepared["body_sentiment_score"] <= -0.15).astype(int)
    if "is_positive_news" not in prepared.columns:
        prepared["is_positive_news"] = (prepared["body_sentiment_score"] >= 0.15).astype(int)

    day_of_week = prepared["date"].dt.dayofweek
    prepared["date"] = prepared["date"] + pd.to_timedelta(
        np.where(day_of_week == 5, 2, np.where(day_of_week == 6, 1, 0)), unit="D"
    )

    emb_cols = [c for c in prepared.columns if c.startswith("body_emb_")]
    for col in emb_cols:
        prepared[col] = prepared[col].fillna(0.0)

    regression_daily_columns = [
        *category_feature_columns,
        "title_positive_prob",
        "title_negative_prob",
        "title_neutral_prob",
        "title_sentiment_score",
        "body_positive_prob",
        "body_negative_prob",
        "body_neutral_prob",
        "body_sentiment_score",
        *emb_cols,
    ]

    grouped = prepared.groupby("date", sort=True)
    daily = grouped[regression_daily_columns].mean(numeric_only=True).reset_index()
    daily["news_count"] = grouped.size().to_numpy()
    daily["negative_news_count"] = grouped["is_negative_news"].sum().to_numpy(dtype=float)
    daily["positive_news_count"] = grouped["is_positive_news"].sum().to_numpy(dtype=float)
    daily["negative_news_ratio"] = daily["negative_news_count"] / (
        daily["news_count"] + 1e-9
    )
    daily["positive_news_ratio"] = daily["positive_news_count"] / (
        daily["news_count"] + 1e-9
    )

    ordered_columns = [
        "date",
        "news_count",
        "negative_news_count",
        "positive_news_count",
        "negative_news_ratio",
        "positive_news_ratio",
        *regression_daily_columns,
    ]
    daily = daily[ordered_columns]
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily
