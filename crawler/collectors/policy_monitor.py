from __future__ import annotations

import argparse
import json
import re
import os
import tempfile
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.collectors.collection_status import attempt, collect_item, propagate_errors, report_failure, summarize
from crawler.monitor_paths import OUTPUT_ROOT
from crawler.collectors.common import fetch_soup
from crawler.collectors.common import clean_text as _clean_text
from crawler.collectors.bis import create_requests_session, crawl_bis_news_index_selenium, extract_article
from crawler.collectors.eia import get_steo_body, get_steo_items, get_today_body, get_today_items
from crawler.collectors.fraser import DEFAULT_KEYWORDS as FRASER_DEFAULT_KEYWORDS, collect_fraser_documents
from crawler.collectors.fed import crawl_statement, crawl_minutes
from crawler.collectors.yfinance import scrape_news_sync as scrape_yahoo_news
from crawler.collectors.yfinance_support import scrape_news_sync as scrape_yahoo_news_support
from crawler.collectors.fda import crawl_fda_press_releases_24
from crawler.collectors.ucsb import (
    DOC_TYPE_URLS,
    crawl_listing,
    load_keyword_dictionary,
    parse_article,
)

BASE_URL = "https://www.federalreserve.gov"
FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

DEFAULT_OUTPUT_CSV = str(OUTPUT_ROOT / "policy_updates_features.csv")
DEFAULT_OLD_DATA_CSV = str(OUTPUT_ROOT / "old_data.csv")

DEFAULT_INTERVAL_SEC = 24 * 60 * 60 # 하루 1회 실행 주기 (초 단위)
US_EASTERN_TZ = ZoneInfo("America/New_York")    # 미국 동부시간 기준
KEYWORD_CONFIG_DIR = Path(__file__).with_name("keywords")   # American Presidential Project 관련 키워드 JSON 파일이 있는 디렉토리
MONITOR_SECTORS = ("qqq", "xlf", "xle", "xlv")

SECTOR_MONITOR_CONFIGS: dict[str, dict[str, Any]] = {
    "qqq": {
        "sources": ("FOMC", "BIS", "UCSB", "YAHOO"),
        "keyword_config_path": KEYWORD_CONFIG_DIR / "qqq_keywords.json",
    },
    "xlf": {
        "sources": ("FOMC", "FRASER", "UCSB", "YAHOO"),
        "keyword_config_path": KEYWORD_CONFIG_DIR / "xlf_keywords.json",
    },
    "xle": {
        "sources": ("FOMC", "EIA", "UCSB", "YAHOO"),
        "keyword_config_path": KEYWORD_CONFIG_DIR / "xle_keywords.json",
    },
    "xlv": {
        "sources": ("FOMC", "FDA", "UCSB", "YAHOO"),
        "keyword_config_path": KEYWORD_CONFIG_DIR / "xlv_keywords.json",
    },
}

CANONICAL_COLUMNS = [
    # 새 기록과 기존 기록을 같은 스키마로 합치기 위한 표준 컬럼 순서.
    "sector",
    "source",
    "category",
    "doc_type",
    "release_date",
    "url",
    "title",
    "image",
    "body"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0",
}


def _parse_us_date(raw_value: str | None) -> date | None:
    """
    args: raw_value (str) - Month Day, Year 형식의 날짜 문자열
    returns: date | None - date 객체 또는 None
    """
    try:
        return datetime.strptime(raw_value, "%B %d, %Y").date()
    except (TypeError, ValueError):
        return None


def _parse_iso_date(raw_value: str | None) -> date | None:
    if not isinstance(raw_value, str):
        return None
    try:
        return date.fromisoformat(raw_value)
    except ValueError:
        try:
            published = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if published.tzinfo is not None:
            published = published.astimezone(US_EASTERN_TZ)
        return published.date()


def _format_iso_date(raw_value: str | None) -> str:
    """
    args: raw_value (str | None) - Month Day, Year 형식의 날짜 문자열
    returns: str - ISO 형식의 날짜 문자열 (YYYY-MM-DD)
    """
    parsed = _parse_us_date(raw_value)
    return parsed.isoformat() if parsed else ""


def _us_eastern_now() -> datetime:
    # 모든 날짜 기준을 미국 동부시간으로 통일한다.
    return datetime.now(US_EASTERN_TZ)


def _get_target_date(reference_dt: date | None = None) -> date:
    current_dt = reference_dt or _us_eastern_now().date()
    # 모니터링 대상은 "현재 시점의 전날"로 고정한다.
    return current_dt - timedelta(days=1)



def _seconds_until_next_us_eastern_midnight() -> float:
    current_dt = _us_eastern_now()
    # 다음 자정까지 남은 초를 계산해 하루 1회 실행 주기를 만든다.
    next_midnight = current_dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return max(0.0, next_midnight.timestamp() - current_dt.timestamp())


def _normalise_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    df = pd.DataFrame(records)

    if "url" in df.columns:
        dedupe_columns = ["sector", "url"] if "sector" in df.columns else ["url"]
        df = df.drop_duplicates(subset=dedupe_columns, keep="last")

    # Ensure canonical columns exist
    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    # Normalize release_date to ISO format (YYYY-MM-DD) when possible.
    if "release_date" in df.columns:
        parsed = pd.to_datetime(df["release_date"], format="mixed", errors="coerce", utc=True)
        df["release_date"] = parsed.dt.strftime("%Y-%m-%d").fillna("")

    ordered_columns = [column for column in CANONICAL_COLUMNS if column in df.columns]
    remaining_columns = [column for column in df.columns if column not in ordered_columns]
    df = df[ordered_columns + remaining_columns]

    return df


def _sector_keyword_config_path(sector: str) -> Path:
    """
    주어진 sector에 대한 키워드 JSON 파일 경로를 반환한다.
    args: sector (str) - 섹터 이름
    returns: Path - 키워드 JSON 파일의 경로
    """
    config = SECTOR_MONITOR_CONFIGS.get(sector)
    if config is None:
        raise ValueError(f"Unsupported sector: {sector}")
    return config["keyword_config_path"]


# sector 태그를 각 레코드에 붙이는 헬퍼. 이후 중복 제거 시 sector+url 조합으로 고유성을 판단할 때 유용하다.
def _tag_records(records: list[dict[str, Any]], sector: str) -> list[dict[str, Any]]:
    tagged_records: list[dict[str, Any]] = []
    for record in records:
        tagged = record.copy()
        tagged["sector"] = sector
        tagged_records.append(tagged)
    return tagged_records


def _collect_fomc_records(target_date: date) -> list[dict[str, Any]]:
    # FOMC는 캘린더 페이지를 읽고 statement, minutes, implementation note를 구분한다.
    soup = fetch_soup(FOMC_CALENDAR_URL, headers=HEADERS, timeout=30, client=requests)
    records: list[dict[str, Any]] = []

    meetings = soup.find_all("div", class_="fomc-meeting")
    if not meetings:
        raise ValueError("FOMC calendar contains no meeting entries")

    for meeting in meetings:
        for link in meeting.find_all("a", href=True):
            label = _clean_text(link.get_text(" ", strip=True)).lower()
            url = urljoin(BASE_URL, link["href"])

            doc_type = None
            article: dict[str, Any] | None = None

            if "implementation note" in label:
                doc_type = "implementation_note"
                article = collect_item(lambda: crawl_statement(url))
            elif label == "html":
                parent_strong = link.parent.find("strong") if link.parent else None
                if parent_strong is None:
                    continue

                parent_title = _clean_text(parent_strong.get_text(" ", strip=True)).lower()

                if "statement:" in parent_title:
                    doc_type = "statement"
                    article = collect_item(lambda: crawl_statement(url))
                elif "minutes:" in parent_title:
                    doc_type = "minutes"
                    article = collect_item(lambda: crawl_minutes(url))

                    release_match = re.search(
                        r"Released ([A-Za-z]+ \d{1,2}, \d{4})",
                        link.parent.get_text(" ", strip=True),
                    )
                    if release_match and article:
                        article["release_date"] = release_match.group(1)

            if not doc_type or not article:
                continue

            published_date_value = _parse_us_date(article.get("release_date"))
            if published_date_value != target_date:
                continue

            records.append(
                {
                    "source": "FOMC",
                    "category": "FOMC",
                    "doc_type": doc_type,
                    "release_date": _format_iso_date(article.get("release_date")),
                    "url": url,
                    "title": article.get("title", ""),
                    "image": "",
                    "body": article.get("body", ""),    
                }
            )

    return records


def _collect_fraser_records(target_date: date) -> list[dict[str, Any]]:
    frame = collect_fraser_documents(keywords=FRASER_DEFAULT_KEYWORDS, start_date=target_date.isoformat())

    if frame.empty:
        return []

    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        release_date = str(row.get("release_date", ""))
        if _parse_iso_date(release_date) != target_date:
            continue

        records.append(
            {
                "source": "FRASER",
                "category": "FRASER",
                "doc_type": str(row.get("doc_type", "fraser")),
                "release_date": release_date,
                "url": str(row.get("url", "")),
                "title": str(row.get("title", "")),
                "image": "",
                "body": str(row.get("body", ""))
            }
        )

    return records


def _collect_eia_records(target_date: date) -> list[dict[str, Any]]:
    start_date = target_date.isoformat()
    records: list[dict[str, Any]] = []

    for item in collect_item(lambda: get_steo_items(start_date=start_date)) or []:
        release_date = str(item.get("release_date", ""))
        if _parse_iso_date(release_date) != target_date:
            continue

        url = str(item.get("url", ""))
        records.append(
            {
                "source": "EIA",
                "category": "EIA",
                "doc_type": str(item.get("doc_type", "STEO")),
                "release_date": release_date,
                "url": url,
                "title": str(item.get("title", "")),
                "image": "",
                "body": collect_item(lambda: get_steo_body(url)) or ""
            }
        )

    for item in collect_item(lambda: get_today_items(start_date=start_date)) or []:
        release_date = str(item.get("release_date", ""))
        if _parse_iso_date(release_date) != target_date:
            continue

        url = str(item.get("url", ""))
        records.append(
            {
                "source": "EIA",
                "category": "EIA",
                "doc_type": str(item.get("doc_type", "TODAY_IN_ENERGY")),
                "release_date": release_date,
                "url": url,
                "title": str(item.get("title", "")),
                "image": "",
                "body": collect_item(lambda: get_today_body(url)) or "",
            }
        )

    return records


def _collect_bis_records(target_date: date, max_pages: int, sleep_sec: float) -> list[dict[str, Any]]:
    # BIS는 목록 페이지에서 링크를 모은 뒤 상세 페이지를 다시 요청한다.
    listing_items = crawl_bis_news_index_selenium(max_pages=max_pages, sleep_sec=sleep_sec)
    if not listing_items:
        return []

    session = create_requests_session()
    records: list[dict[str, Any]] = []
    # listing_items are expected newest->oldest; no need for consecutive counters

    for item in listing_items:
        article = collect_item(lambda: extract_article(item["url"], session=session, sleep_sec=sleep_sec))
        if not article:
            continue

        # 전날 기사만 CSV에 남기기 위해 날짜 기준으로 걸러낸다.
        published_date_value = None
        published_date_raw = str(article.get("published_date", "")).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", published_date_raw):
            try:
                published_date_value = date.fromisoformat(published_date_raw)
            except ValueError:
                published_date_value = None

        if published_date_value is None:
            continue

        # 목록은 최신순으로 내려오기 때문에, 타겟 날짜보다 이전인 기사를 만나면
        # 그 이후의 항목들도 모두 이전 날짜일 가능성이 높다. 즉시 중단해도 안전하다.
        if published_date_value < target_date:
            break

        # 타겟 날짜보다 미래(더 최근)인 것은 건너뛴다.
        if published_date_value > target_date:
            continue

        records.append(
            {
                "source": "BIS",
                "category": "BIS",
                "doc_type": article.get("doc_type", "press_release"),
                "release_date": article.get("published_date", ""),
                "url": item["url"],
                "title": article.get("title", ""),
                "image": "",
                "body": article.get("body", ""),
            }
        )

    return records


def _collect_fda_records(target_date: date) -> list[dict[str, Any]]:
    # FDA 수집기는 목록의 datetime 및 target_date.date()를 사용한다.
    fda_target = datetime.combine(target_date, datetime.min.time())
    fda_records = crawl_fda_press_releases_24(target_date=fda_target)

    if not fda_records:
        return []

    records: list[dict[str, Any]] = []
    for row in fda_records:
        release_date = str(row.get("release_date", ""))

        published_date = _parse_iso_date(release_date)
        if published_date != target_date:
            continue

        records.append(
            {
                "source": "FDA",
                "category": "FDA",
                "doc_type": "press_release",
                "release_date": published_date.isoformat(),
                "url": str(row.get("url", "")),
                "title": str(row.get("title", "")),
                "image": "",
                "body": str(row.get("body", ""))
            }
        )

    return records


def _collect_ucsb_records(
    target_date: date,
    sleep_sec: float,
    keyword_config_path: str | Path | None,
) -> list[dict[str, Any]]:
    # UCSB는 키워드 매칭이 끝난 문서만 모니터링 CSV에 넣는다.
    # Allow passing either a path or a preloaded keyword dictionary.
    if isinstance(keyword_config_path, (str, Path)):
        keyword_dictionary = load_keyword_dictionary(Path(keyword_config_path))
    elif isinstance(keyword_config_path, dict):
        keyword_dictionary = keyword_config_path
    else:
        raise ValueError("keyword_config_path must be a path or a keyword dictionary mapping")
    selected_doc_types = list(DOC_TYPE_URLS.keys())

    records: list[dict[str, Any]] = []

    for doc_type in selected_doc_types:
        listing_items = crawl_listing(
            base_url=DOC_TYPE_URLS[doc_type],
            doc_type=doc_type,
            start_date=target_date,
            sleep_sec=sleep_sec,
        )

        for item in listing_items:
            article = parse_article(item, keyword_dictionary=keyword_dictionary)
            if article is None:
                continue

            published_date_value = _parse_iso_date(article.get("published_date"))
            if published_date_value != target_date:
                continue

            records.append(
                {
                    "source": "UCSB",
                    "category": "UCSB",
                    "doc_type": article.get("doc_type", doc_type),
                    "release_date": article.get("published_date", ""),
                    "url": item["url"],
                    "title": article.get("title", ""),
                    "image": "",
                    "body": article.get("body", "")
                }
            )

        time.sleep(sleep_sec)

    return records


def _collect_yahoo_records(target_date: date, ticker: str) -> list[dict[str, Any]]:
    ticker = ticker.lower()
    yahoo_records, primary_errors = attempt(
        lambda: scrape_yahoo_news(ticker=ticker.upper(), target_date=target_date.isoformat())
    )
    if not yahoo_records or primary_errors:
        fallback_records, fallback_errors = attempt(
            lambda: scrape_yahoo_news_support(tickers=[ticker.upper()], target_date=target_date.isoformat())
        )
        by_url = {row.get("url"): row for row in yahoo_records + fallback_records}
        yahoo_records = list(by_url.values())
        # A completed fallback (including zero matches) resolves primary failures.
        if fallback_errors:
            propagate_errors(primary_errors + fallback_errors)

    records: list[dict[str, Any]] = []
    for row in yahoo_records:
        row_ticker = str(row.get("sector", "")).strip().lower()
        if row_ticker != ticker:
            continue

        release_date = str(row.get("release_date", ""))

        if _parse_iso_date(release_date) != target_date:
            continue

        records.append(
            {
                "source": "YAHOO",
                "category": "YAHOO",
                "doc_type": "news",
                "release_date": release_date,
                "url": str(row.get("url", "")),
                "title": str(row.get("title", "")),
                "image": str(row.get("image", "")),
                "body": str(row.get("body", row.get("full_text", ""))),
            }
        )

    return records


def _collect_sector_records(
    sector: str,
    target_date: date,
    bis_max_pages: int,
    sleep_sec: float,
    shared_records: dict | None = None,
    source_results: list | None = None,
) -> list[dict[str, Any]]:
    if sector not in SECTOR_MONITOR_CONFIGS:
        raise ValueError(f"Unsupported sector: {sector}")
    config = SECTOR_MONITOR_CONFIGS[sector]
    callbacks = {
        "FOMC": lambda: _collect_fomc_records(target_date),
        "BIS": lambda: _collect_bis_records(target_date, max_pages=bis_max_pages, sleep_sec=sleep_sec),
        "FRASER": lambda: _collect_fraser_records(target_date),
        "EIA": lambda: _collect_eia_records(target_date),
        "FDA": lambda: _collect_fda_records(target_date),
        "YAHOO": lambda: _collect_yahoo_records(target_date, sector),
        "UCSB": lambda: _collect_ucsb_records(
            target_date, sleep_sec=sleep_sec, keyword_config_path=_sector_keyword_config_path(sector)
        ),
    }
    sector_records = []
    for source in config["sources"]:
        if source == "FOMC" and shared_records is not None:
            if source not in shared_records:
                shared_records[source] = attempt(callbacks[source])
            records, errors = shared_records[source]
        else:
            records, errors = attempt(callbacks[source])
        sector_records.extend(_tag_records(records, sector))
        if source_results is not None:
            source_results.append({
                "sector": sector, "source": source, "count": len(records),
                "status": ("partial_failure" if records else "failed") if errors
                          else ("success" if records else "empty"),
                "errors": errors,
            })
    return sector_records


def collect_policy_updates(
    bis_max_pages: int = 1,
    sleep_sec: float = 1.0,
    target_date: date | None = None,
) -> pd.DataFrame:
    # 이전 파일을 읽지 않고, 이번 사이클에서 새로 수집된 레코드만 반환한다.
    target_date = target_date or _get_target_date()

    new_records: list[dict[str, Any]] = []
    seen_urls = set()
    shared_records: dict = {}
    source_results: list = []

    for sector in MONITOR_SECTORS:
        print(f"[MONITOR] Starting {sector.upper()} crawl for {target_date.isoformat()}")
        sector_records = _collect_sector_records(
            sector=sector,
            target_date=target_date,
            bis_max_pages=bis_max_pages,
            sleep_sec=sleep_sec,
            shared_records=shared_records,
            source_results=source_results,
        )

        for record in sector_records:
            record_key = (record.get("sector", ""), record.get("url", ""))
            if record_key in seen_urls:
                continue
            seen_urls.add(record_key)
            new_records.append(record)

    new_df = _normalise_records(new_records)

    new_df.attrs["collection"] = {
        "status": summarize((item["status"] for item in source_results), len(new_df)),
        "sources": source_results,
        "sectors": [
            {"sector": sector,
             "errors": [error for item in source_results if item["sector"] == sector
                        for error in item["errors"]],
             "status": summarize(
                 (item["status"] for item in source_results if item["sector"] == sector),
                 int((new_df["sector"] == sector).sum()))}
            for sector in MONITOR_SECTORS
        ],
    }
    return new_df


def run_postprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    주어진 DataFrame에 통합 후처리를 적용하고 처리된 DataFrame을 반환한다.
    """
    from crawler.postprocessing.unified_pipeline import apply_unified_pipeline
    # Yahoo 요약 후 신뢰도 검증은 일시 비활성화.
    # from crawler.validating.validation_pipeline import validate_yahoo_summaries

    print("[MONITOR] 통합 후처리 파이프라인을 실행 중...")
    try:
        result_df = apply_unified_pipeline(
            df=df,
            include_summarization=True,
            include_encoding=True,
            include_sentiment=True,
            include_embeddings=True,
            # post_summary_hook=lambda summarized: validate_yahoo_summaries(
            #     summarized, source_records=df,
            # ),
        )
        print(f"[MONITOR] 통합 후처리 파이프라인 완료: {len(result_df)} rows 처리됨")
        return result_df
    except Exception as e:
        print(f"[MONITOR] ERROR: 통합 후처리 파이프라인 실패: {e}")
        raise


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
        frame.to_csv(temporary_path, index=False, encoding="utf-8-sig")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _save_monitor_results(processed_news: pd.DataFrame, processed_path: Path, old_news_path: Path) -> None:
    # 읽기 실패는 전파한다. 손상되거나 잠긴 기존 파일을 빈 데이터로 덮어쓰지 않는다.
    existing = pd.read_csv(processed_path, encoding="utf-8-sig") if processed_path.exists() else pd.DataFrame()
    combined = pd.concat([existing, processed_news], ignore_index=True, sort=False) if not existing.empty else processed_news.copy()
    if {"sector", "url"}.issubset(combined.columns):
        combined = combined.drop_duplicates(subset=["sector", "url"], keep="last")
    if "sector" in combined.columns:
        combined = combined[["sector"] + [column for column in combined.columns if column != "sector"]]

    recent = combined
    if not combined.empty and "release_date" in combined.columns:
        parsed = pd.to_datetime(combined["release_date"], format="mixed", errors="coerce", utc=True)
        cutoff = pd.Timestamp(_us_eastern_now().date(), tz="UTC") - pd.Timedelta(days=20)
        is_old = parsed < cutoff
        aged = combined[is_old]
        recent = combined[~is_old]
        if not aged.empty:
            if old_news_path.exists():
                archive = pd.read_csv(old_news_path, encoding="utf-8-sig")
                aged = pd.concat([archive, aged], ignore_index=True, sort=False)
            if {"sector", "url"}.issubset(aged.columns):
                aged = aged.drop_duplicates(subset=["sector", "url"], keep="last")
            # 보관 파일을 먼저 확정한다. 이후 실패해도 재실행 시 중복 제거로 복구한다.
            _write_csv_atomic(aged, old_news_path)
            print(f"[MONITOR] 보관 파일: {old_news_path} / {len(aged)} rows")
    _write_csv_atomic(recent, processed_path)
    print(f"[MONITOR] 모니터링 결과: {processed_path} / {len(recent)} rows")


def _write_json_atomic(state: dict | list, path: Path) -> None:
    if isinstance(state, list):
        # Deduplicate only the JSON snapshot; keep full diagnostics in logs.
        snapshot = []
        for item in state:
            errors_by_code = {}
            for error in item["errors"]:
                errors_by_code.setdefault(error["code"], error)
            snapshot.append({**item, "errors": list(errors_by_code.values())})
        state = snapshot
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            json.dump(state, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _append_monitor_log(path: Path, target_date: date, **details) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"logged_at": _us_eastern_now().isoformat(),
                                 "target_date": target_date.isoformat(), **details},
                                ensure_ascii=False) + "\n")


def run_monitor(
    output_path: str = DEFAULT_OUTPUT_CSV,
    interval_sec: float = DEFAULT_INTERVAL_SEC,
    bis_max_pages: int = 1,                                # 거의 업데이트 없음, 기본값 1페이지로 충분
    sleep_sec: float = 1.0,                                # Rate limit을 피하기 위해 요청 사이에 1초 정도 쉬는 것이 안전
    max_cycles: int | None = None,
    status_path: str | None = None,
) -> None:
    if max_cycles is not None and max_cycles < 1:
        raise ValueError("max_cycles must be at least 1")
    if sleep_sec < 0 or bis_max_pages < 1:
        raise ValueError("sleep_sec must be nonnegative and bis_max_pages positive")
    cycle = 0
    
    processed_path = Path(output_path)
    # old_news.csv는 메인 출력 파일과 같은 디렉토리에 저장하도록 경로 설정
    old_news_path = processed_path.parent / Path(DEFAULT_OLD_DATA_CSV).name
    if processed_path.resolve() == old_news_path.resolve():
        raise ValueError("Output and archive paths must differ")

    state_path = Path(status_path) if status_path else processed_path.with_suffix(".status.json")
    if state_path.resolve() in (processed_path.resolve(), old_news_path.resolve()):
        raise ValueError("Status path must differ from CSV paths")
    log_path = processed_path.parent / "log" / state_path.with_suffix(".log").name
    if log_path.resolve() in (processed_path.resolve(), old_news_path.resolve(), state_path.resolve()):
        raise ValueError("Log path must differ from CSV and status paths")

    while True:
        target_date = _get_target_date()
        cycle += 1

        print(f"[MONITOR] cycle={cycle} target_date={target_date.isoformat()} ")

        state = [{"sector": sector, "status": "running", "errors": []}
                 for sector in MONITOR_SECTORS]
        stage = "collection"
        _write_json_atomic(state, state_path)
        _append_monitor_log(log_path, target_date, stage=stage, status="running")
        try:
            news_list = collect_policy_updates(
                bis_max_pages=bis_max_pages, sleep_sec=sleep_sec, target_date=target_date,
            )
            collection = news_list.attrs.get("collection")
            if collection is None:
                # Preserve compatibility with callers supplying a plain DataFrame.
                collection = {"sources": [], "sectors": [
                    {"sector": sector, "status": "success" if (
                        "sector" in news_list and (news_list["sector"] == sector).any()
                    ) else "empty", "errors": []} for sector in MONITOR_SECTORS
                ]}
            state = [{"sector": item["sector"], "status": item["status"], "errors": item["errors"]}
                     for item in collection["sectors"]]
            for item in collection["sources"]:
                _append_monitor_log(log_path, target_date, **item)
            stage = "postprocessing"
            processed_news = (run_postprocessing_pipeline(df=news_list) if len(news_list)
                              else pd.DataFrame(columns=CANONICAL_COLUMNS))
            stage = "saving"
            if not all(item["status"] == "failed" for item in state) and (
                not processed_news.empty or processed_path.exists()
            ):
                _save_monitor_results(processed_news, processed_path, old_news_path)
        except Exception as exc:
            error = report_failure(exc, code=stage.upper() + "_FAILED")
            for item in state:
                item["status"] = "failed"
                item["errors"].append(error)
            _write_json_atomic(state, state_path)
            _append_monitor_log(log_path, target_date, stage=stage, status="failed", errors=[error])
            raise
        _write_json_atomic(state, state_path)
        _append_monitor_log(log_path, target_date, stage="completed", sectors=state)
        print(f"[MONITOR] status saved / {state_path}")

        if max_cycles is not None and cycle >= max_cycles:
            break

        if interval_sec <= 0:
            break

        if interval_sec == DEFAULT_INTERVAL_SEC:
            time.sleep(_seconds_until_next_us_eastern_midnight())
        else:
            time.sleep(interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="[Monitor] FOMC, BIS, FRASER, EIA, UCSB, FDA, YAHOO updates for the previous America/New_York day")
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_CSV, help="수집 결과 CSV 파일 경로")
    parser.add_argument("--status-path", default=None, help="수집 상태 JSON 경로; 기본값: 출력 CSV의 .status.json 파일")
    parser.add_argument("--interval-sec", type=float, default=DEFAULT_INTERVAL_SEC, help="모니터링 간격 (초 단위); 기본값은 다음 America/New_York 자정에 스케줄링됩니다")
    parser.add_argument("--bis-max-pages", type=int, default=1)
    parser.add_argument("--sleep-sec", type=float, default=1.0, help="Rate limit을 피하기 위해 요청 사이에 쉬는 시간 (초 단위)")
    parser.add_argument("--max-cycles", type=int, default=None, help="모니터링 주기 반복 횟수 제한; None이면 무한 반복")
    args = parser.parse_args()

    run_monitor(
        output_path=args.output_path,
        status_path=args.status_path,
        interval_sec=args.interval_sec,
        bis_max_pages=args.bis_max_pages,
        sleep_sec=args.sleep_sec,
        max_cycles=args.max_cycles,
    )

if __name__ == "__main__":
    main()
