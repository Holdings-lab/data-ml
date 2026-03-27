from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd


def _pick_first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_date_series(s: pd.Series) -> pd.Series:
    """
    날짜 컬럼을 최대한 'YYYY-MM-DD' 문자열로 맞춘다.
    변환 실패한 값은 원본 문자열을 그대로 둔다.
    """
    raw = s.astype(str)
    raw_clean = raw.where(~raw.str.lower().isin(["nan", "none", "nat"]), other=pd.NA)

    dt = pd.to_datetime(raw_clean, errors="coerce")
    out = dt.dt.strftime("%Y-%m-%d")

    # 변환 실패한 행은 원본 문자열 유지
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
    여러 CSV를 읽어서 아래 4컬럼만 남긴 테이블로 병합한다.

    - date: release_date / published_date / date (가능한 것 우선)
    - category: category
    - doc_type: doc_type
    - title: title
    - body: body_text / body
    - link: url / link
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
            raise ValueError(f"{path}에 필요한 컬럼이 없습니다: {missing}. 현재 컬럼: {list(df.columns)}")

        out = pd.DataFrame(
            {
                "date": _normalize_date_series(df[date_col]),
                "category": df[category_col],
                "doc_type": df[doc_type_col],
                "title": df[title_col],
                "body": df[body_col],
                "link": df[link_col],
            }
        )
        tables.append(out)

    merged = pd.concat(tables, ignore_index=True)
    if drop_duplicates:
        merged = merged.drop_duplicates()

    merged = merged[["date", "category", "doc_type", "title", "body", "link"]]

    if sort_by_date:
        # date는 문자열일 수 있으므로 안전하게 datetime으로 변환해 정렬한다.
        # 파싱 실패(NaT)는 맨 뒤로 보낸다.
        sort_key = pd.to_datetime(merged["date"], errors="coerce")
        merged = merged.assign(_sort_date=sort_key).sort_values(
            by=["_sort_date", "date"],
            ascending=ascending,
            na_position="last",
            kind="mergesort",
        )
        merged = merged.drop(columns=["_sort_date"]).reset_index(drop=True)

    return merged


if __name__ == "__main__":
    # 예시 실행: 로컬에서 CSV 파일 경로를 직접 지정하세요.
    csv_paths = [
        "fed_fomc_links_summarized.csv",
        "whitehouse_qqq_policy_summarized.csv",
        "bis_press_releases.csv",
    ]
    merged = merge_csvs_to_table(csv_paths)
    print(merged.head(20))
    print("[INFO] merged_rows=", len(merged))
    merged.to_csv("merged_table_sorted.csv", index=False, encoding="utf-8-sig")
    print("[INFO] saved=merged_table_sorted.csv")

