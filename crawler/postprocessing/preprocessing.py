from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.support_legacy.data_paths import feature_csv_path, summarized_csv_path

DEFAULT_MERGED_OUTPUT_CSV = feature_csv_path("xlf_merged_table_sorted.csv")
DEFAULT_ENCODED_OUTPUT_CSV = feature_csv_path("xlf_merged_table_sorted_encoded.csv")

# 존재하는 파일 경로만 반환
def _existing_csv_paths(csv_paths: Iterable[str]) -> list[str]:
    return [path for path in csv_paths if Path(path).exists()]


def _pick_first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_date_series(s: pd.Series) -> pd.Series:
    """
    Normalize a date-like series to 'YYYY-MM-DD' strings when possible.
    If parsing fails, keep the original non-empty string value.
    """
    # Convert to string and handle missing values
    raw = s.fillna("").astype(str).str.strip()
    raw_clean = raw.where(raw != "", other=pd.NA)

    # Parse dates
    dt = pd.to_datetime(raw_clean, errors="coerce")
    # Format valid dates as YYYY-MM-DD, keep original for parsing failures
    out = dt.dt.strftime("%Y-%m-%d")
    out = out.where(~dt.isna(), other=raw_clean)
    return out


def merge_csvs_to_table(
    csv_paths: List[str],
    encoding: str = "utf-8-sig",
    drop_duplicates: bool = True,
    sort_by_date: bool = True,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    csv_paths의 csv 파일을 읽고 하나의 테이블로 병합
    기본 세팅은 중복 제거 및 날짜 내림차순 정렬

    Output columns:
    - category: category
    - doc_type: doc_type
    - release_date: release_date / published_date
    - url: link / url
    - title: title
    - body: body
    """
    tables: List[pd.DataFrame] = []

    for path in csv_paths:
        try:
            df = pd.read_csv(path, encoding=encoding)
        except Exception as e:
            raise ValueError(f"[MERGED] CSV 파일 읽기 실패 {path}: {e}")

        category = _pick_first_existing(df, ["category"])
        doc_type = _pick_first_existing(df, ["doc_type"])
        release_date = _pick_first_existing(df, ["release_date", "published_date"])
        url = _pick_first_existing(df, ["link", "url"])
        title = "title" if "title" in df.columns else None
        body = _pick_first_existing(df, ["body"])

        missing = [
            name
            for name, col in [
                ("category", category),
                ("doc_type", doc_type),
                ("release_date", release_date),
                ("url", url),
                ("title", title),
                ("body", body),
            ]
            if col is None
        ]

        if missing:
            raise ValueError(
                f"[MERGED] {path} 존재하지 않는 컬럼 : {missing}. "
                f"사용 가능한 컬럼: {list(df.columns)}"
            )

        body_series = df[body].fillna("").astype(str)
        body_length_series = body_series.str.len()

        output = pd.DataFrame(
            {
                "category": df[category],
                "doc_type": df[doc_type],
                "release_date": _normalize_date_series(df[release_date]),
                "url": df[url],
                "body_original_length": body_length_series,
                "title": df[title],
                "body": body_series,
            }
        )
        tables.append(output)

    merged = pd.concat(tables, ignore_index=True)
    if drop_duplicates:
        merged = merged.drop_duplicates()

    merged = merged[["category", "doc_type", "release_date", "url", "body_original_length", "title", "body"]]

    if sort_by_date:
        merged["release_date"] = pd.to_datetime(merged["release_date"], errors="coerce")
        merged = merged.sort_values(by="release_date", ascending=ascending).reset_index(drop=True)

    return merged


def one_hot_encode_category(
    df: pd.DataFrame,
    prefix: str = "category",
    dtype: str = "int64",
    expected_categories: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    One-hot encode the `category` column from an existing DataFrame.
    """
    if "category" not in df.columns:
        raise ValueError(
            "[ONEHOT] 'category' 칼럼을 찾을 수 없습니다. "
            f"사용 가능한 컬럼 : {list(df.columns)}"
        )

    encoded = pd.get_dummies(df["category"], prefix=prefix, dtype=dtype)

    if expected_categories is not None:
        for category in expected_categories:
            column_name = f"{prefix}_{category}" if prefix else str(category)
            if column_name not in encoded.columns:
                encoded[column_name] = 0

        ordered_columns = [f"{prefix}_{category}" if prefix else str(category) for category in expected_categories]
        ordered_columns.extend([column for column in encoded.columns if column not in ordered_columns])
        encoded = encoded[ordered_columns]

    return pd.concat([df, encoded], axis=1)


def main() -> None:
    csv_candidates = [
        summarized_csv_path("fed_fomc_links_summarized.csv"),
        summarized_csv_path("xlf_presidential_documents_summarized.csv"),
        summarized_csv_path("fraser_sample_summarized.csv"),
    ]
    csv_paths = _existing_csv_paths(csv_candidates)

    if not csv_paths:
        raise FileNotFoundError(
            "csv 파일을 찾을 수 없습니다. "
            f"확인된 경로: {csv_candidates}"
        )

    merged = merge_csvs_to_table(csv_paths)
    print("[INFO] merged_rows=", len(merged))
    merged.to_csv(DEFAULT_MERGED_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] saved merged file: {DEFAULT_MERGED_OUTPUT_CSV}")

    encoded = one_hot_encode_category(merged)
    encoded.to_csv(DEFAULT_ENCODED_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] saved encoded file: {DEFAULT_ENCODED_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
