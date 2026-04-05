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


DEFAULT_MERGED_OUTPUT_CSV = feature_csv_path("merged_table_sorted.csv")
DEFAULT_ENCODED_OUTPUT_CSV = feature_csv_path("merged_table_sorted_encoded.csv")
DEFAULT_TIME_FEATURES_OUTPUT_CSV = feature_csv_path("merged_table_sorted_time_features.csv")
DATE_COL = "date"
BODY_COL = "body"
BODY_LENGTH_COL = "body_original_length"


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
    raw = s.astype(str)
    raw_clean = raw.where(~raw.str.lower().isin(["nan", "none", "nat"]), other=pd.NA)

    dt = pd.to_datetime(raw_clean, errors="coerce")
    out = dt.dt.strftime("%Y-%m-%d")
    out = out.where(~dt.isna(), other=raw_clean)
    return out


def merge_csvs_to_table(
    csv_paths: List[str],
    encoding: str = "utf-8-sig",
    drop_duplicates: bool = True,
    sort_by_date: bool = True,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Read multiple CSV files and merge them into a standardized table.

    Output columns:
    - date: date / release_date / published_date
    - category: category
    - doc_type: doc_type
    - title: title
    - body: body
    - link: link / url
    """
    tables: List[pd.DataFrame] = []

    for path in csv_paths:
        df = pd.read_csv(path, encoding=encoding)

        date_col = _pick_first_existing(df, ["date", "release_date", "published_date"])
        category_col = _pick_first_existing(df, ["category"])
        doc_type_col = _pick_first_existing(df, ["doc_type"])
        title_col = "title" if "title" in df.columns else None
        body_col = _pick_first_existing(df, ["body"])
        link_col = _pick_first_existing(df, ["link", "url"])

        missing = [
            name
            for name, col in [
                ("date", date_col),
                ("category", category_col),
                ("doc_type", doc_type_col),
                ("title", title_col),
                ("body", body_col),
                ("link", link_col),
            ]
            if col is None
        ]
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        body_series = df[body_col].fillna("").astype(str)
        out = pd.DataFrame(
            {
                "date": _normalize_date_series(df[date_col]),
                "category": df[category_col],
                "doc_type": df[doc_type_col],
                "title": df[title_col],
                "body": body_series,
                "body_original_length": body_series.str.len(),
                "link": df[link_col],
            }
        )
        tables.append(out)

    merged = pd.concat(tables, ignore_index=True)
    if drop_duplicates:
        merged = merged.drop_duplicates()

    merged = merged[["date", "category", "doc_type", "title", "body", "body_original_length", "link"]]

    if sort_by_date:
        sort_key = pd.to_datetime(merged["date"], errors="coerce")
        merged = merged.assign(_sort_date=sort_key).sort_values(
            by=["_sort_date", "date"],
            ascending=ascending,
            na_position="last",
            kind="mergesort",
        )
        merged = merged.drop(columns=["_sort_date"]).reset_index(drop=True)

    return merged


def one_hot_encode_category(
    df: pd.DataFrame,
    keep_category: bool = True,
    prefix: str = "category",
    dtype: str = "int64",
) -> pd.DataFrame:
    """
    One-hot encode the `category` column from an existing DataFrame.
    """
    if "category" not in df.columns:
        raise ValueError(
            "`category` column was not found. "
            f"Available columns: {list(df.columns)}"
        )

    encoded = pd.get_dummies(df["category"], prefix=prefix, dtype=dtype)

    if keep_category:
        return pd.concat([df, encoded], axis=1)

    return pd.concat([df.drop(columns=["category"]), encoded], axis=1)


def read_csv_and_one_hot_encode_category(
    csv_path: str,
    encoding: str = "utf-8-sig",
    keep_category: bool = True,
    prefix: str = "category",
    dtype: str = "int64",
) -> pd.DataFrame:
    """
    Read a CSV file and one-hot encode the `category` column.
    """
    df = pd.read_csv(csv_path, encoding=encoding)
    return one_hot_encode_category(
        df,
        keep_category=keep_category,
        prefix=prefix,
        dtype=dtype,
    )


def add_cyclical_time_features(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    """
    Add calendar-based and cyclical time features derived from a date column.
    """
    if date_col not in df.columns:
        raise ValueError(
            f"`{date_col}` column was not found. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    out["day_of_week"] = out[date_col].dt.dayofweek
    out["month"] = out[date_col].dt.month

    out["day_of_week_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["day_of_week_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)

    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    out["is_weekend"] = (out["day_of_week"] >= 5).astype("Int64")
    return out


def read_csv_and_add_cyclical_time_features(
    csv_path: str,
    encoding: str = "utf-8-sig",
    date_col: str = DATE_COL,
) -> pd.DataFrame:
    """
    Read a CSV file and add cyclical time features from a date column.
    """
    df = pd.read_csv(csv_path, encoding=encoding)
    return add_cyclical_time_features(df, date_col=date_col)


def main() -> None:
    csv_candidates = [
        summarized_csv_path("fed_fomc_links_summarized.csv"),
        summarized_csv_path("ucsb_presidential_documents_summarized.csv"),
        summarized_csv_path("bis_press_releases_summarized.csv"),
    ]
    csv_paths = _existing_csv_paths(csv_candidates)

    if not csv_paths:
        raise FileNotFoundError(
            "No summarized crawler outputs were found. "
            f"Checked: {csv_candidates}"
        )

    merged = merge_csvs_to_table(csv_paths)
    print("[INFO] merged_rows=", len(merged))
    merged.to_csv(DEFAULT_MERGED_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] saved merged file: {DEFAULT_MERGED_OUTPUT_CSV}")

    encoded = one_hot_encode_category(merged, keep_category=True)
    encoded.to_csv(DEFAULT_ENCODED_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] saved encoded file: {DEFAULT_ENCODED_OUTPUT_CSV}")

    time_features = add_cyclical_time_features(encoded, date_col=DATE_COL)
    time_features.to_csv(DEFAULT_TIME_FEATURES_OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] saved time features file: {DEFAULT_TIME_FEATURES_OUTPUT_CSV}")

    preview_cols = [
        "date",
        "day_of_week", "day_of_week_sin", "day_of_week_cos",
        "month", "month_sin", "month_cos",
        "is_weekend",
    ]
    print(time_features[preview_cols].head(10))


if __name__ == "__main__":
    main()
