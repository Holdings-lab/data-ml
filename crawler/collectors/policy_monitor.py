from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.collectors.bis import create_requests_session, crawl_bis_news_index_selenium, extract_article
from crawler.collectors.eia import get_steo_body, get_steo_items, get_today_body, get_today_items
from crawler.collectors.fraser import DEFAULT_KEYWORDS as FRASER_DEFAULT_KEYWORDS, collect_fraser_documents
from crawler.collectors.fed import crawl_implementation_note, crawl_fomc_statement, crawl_minutes
from crawler.collectors.yfinance import scrape_news_sync as scrape_yahoo_news
from crawler.collectors.yfinance_support import scrape_news_sync as scrape_yahoo_news_support
from crawler.collectors.fda import crawl_fda_press_releases_24
from crawler.collectors.ucsb import (
    DOC_TYPE_URLS,
    crawl_listing,
    load_keyword_dictionary,
    parse_article,
)
from crawler.postprocessing.unified_pipeline import apply_unified_pipeline

BASE_URL = "https://www.federalreserve.gov"
FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

DEFAULT_OUTPUT_CSV = "/opt/riseai/data/crawler/policy_updates_features.csv"
DEFAULT_OLD_DATA_CSV = "/opt/riseai/data/crawler/old_data.csv"
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


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _parse_us_date(raw_value: str) -> date | None:
    """
    args: raw_value (str) - Month Day, Year 형식의 날짜 문자열
    returns: date | None - date 객체 또는 None
    """
    try:
        return datetime.strptime(raw_value, "%B %d, %Y").date()
    except ValueError:
        return None


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
    return max(0.0, (next_midnight - current_dt).total_seconds())


def _normalise_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

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
        parsed = pd.to_datetime(df["release_date"], errors="coerce")
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


def _collect_fomc_records(target_date: datetime) -> list[dict[str, Any]]:
    # FOMC는 캘린더 페이지를 읽고 statement, minutes, implementation note를 구분한다.
    response = requests.get(FOMC_CALENDAR_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    records: list[dict[str, Any]] = []

    section = soup.find("div", class_="panel-default")
    meetings = section.find_all("div", class_="fomc-meeting")

    for meeting in meetings:
        for link in meeting.find_all("a", href=True):
            label = _clean_text(link.get_text(" ", strip=True)).lower()
            url = urljoin(BASE_URL, link["href"])

            doc_type = None
            article: dict[str, Any] | None = None

            if "implementation note" in label:
                doc_type = "implementation_note"
                article = crawl_implementation_note(url)
            elif label == "html":
                parent_strong = link.parent.find("strong") if link.parent else None
                if parent_strong is None:
                    continue

                parent_title = _clean_text(parent_strong.get_text(" ", strip=True)).lower()

                if "statement:" in parent_title:
                    doc_type = "statement"
                    article = crawl_fomc_statement(url)
                elif "minutes:" in parent_title:
                    doc_type = "minutes"
                    article = crawl_minutes(url)

                    release_match = re.search(
                        r"Released ([A-Za-z]+ \d{1,2}, \d{4})",
                        link.parent.get_text(" ", strip=True),
                    )
                    if release_match:
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
        if date.fromisoformat(release_date) != target_date:
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

    for item in get_steo_items(start_date=start_date):
        release_date = str(item.get("release_date", ""))
        if date.fromisoformat(release_date) != target_date:
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
                "body": get_steo_body(url)
            }
        )

    for item in get_today_items(start_date=start_date):
        release_date = str(item.get("release_date", ""))
        if date.fromisoformat(release_date) != target_date:
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
                "body": get_today_body(url),
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
        article = extract_article(item["url"], session=session, sleep_sec=sleep_sec)
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
    fda_records = crawl_fda_press_releases_24(target_date=target_date)

    if not fda_records:
        return []

    records: list[dict[str, Any]] = []
    for row in fda_records:
        release_date = str(row.get("release_date", ""))

        if date.fromisoformat(release_date) != target_date:
            continue

        records.append(
            {
                "source": "FDA",
                "category": "FDA",
                "doc_type": "press_release",
                "release_date": release_date,
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

            published_date_value = _parse_us_date(article.get("published_date"))
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
    yahoo_records = scrape_yahoo_news(ticker=ticker.upper(), target_date=target_date.isoformat())

    if not yahoo_records:
        yahoo_records = scrape_yahoo_news_support(tickers=[ticker.upper()], target_date=target_date.isoformat())

    if not yahoo_records:
        return []

    records: list[dict[str, Any]] = []
    for row in yahoo_records:
        row_ticker = str(row.get("sector", "")).strip().lower()
        if row_ticker != ticker:
            continue

        release_date = str(row.get("release_date", ""))

        if date.fromisoformat(release_date) != target_date:
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
) -> list[dict[str, Any]]:
    sector_config = SECTOR_MONITOR_CONFIGS.get(sector)
    if sector_config is None:
        raise ValueError(f"Unsupported sector: {sector}")

    sector_sources = set(sector_config["sources"])
    sector_records: list[dict[str, Any]] = []

    if "FOMC" in sector_sources:
        sector_records.extend(_tag_records(_collect_fomc_records(target_date), sector))
    if "BIS" in sector_sources:
        sector_records.extend(
            _tag_records(
                _collect_bis_records(
                    target_date,
                    max_pages=bis_max_pages,
                    sleep_sec=sleep_sec
                ),
                sector,
            )
        )
    if "FRASER" in sector_sources:
        sector_records.extend(_tag_records(_collect_fraser_records(target_date), sector))
    if "EIA" in sector_sources:
        sector_records.extend(_tag_records(_collect_eia_records(target_date), sector))
    if "FDA" in sector_sources:
        sector_records.extend(_tag_records(_collect_fda_records(target_date), sector))
    if "YAHOO" in sector_sources:
        sector_records.extend(_tag_records(_collect_yahoo_records(target_date, sector), sector))

    sector_records.extend(
        _tag_records(
            _collect_ucsb_records(
                target_date,
                sleep_sec=sleep_sec,
                keyword_config_path=_sector_keyword_config_path(sector),
            ),
            sector,
        )
    )
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

    for sector in MONITOR_SECTORS:
        print(f"[MONITOR] Starting {sector.upper()} crawl for {target_date.isoformat()}")
        sector_records = _collect_sector_records(
            sector=sector,
            target_date=target_date,
            bis_max_pages=bis_max_pages,
            sleep_sec=sleep_sec,
        )

        for record in sector_records:
            record_key = (record.get("sector", ""), record.get("url", ""))
            if record_key in seen_urls:
                continue
            seen_urls.add(record_key)
            new_records.append(record)

    new_df = _normalise_records(new_records)

    # 반환: 이번 사이클에서 새로 수집된 행들(중복 제거된 상태)
    return new_df


def run_postprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    주어진 DataFrame에 통합 후처리를 적용하고 처리된 DataFrame을 반환한다.
    """
    print("[MONITOR] 통합 후처리 파이프라인을 실행 중...")
    try:
        result_df = apply_unified_pipeline(
            df=df,
            include_summarization=True,
            include_encoding=True,
            include_sentiment=True,
            include_embeddings=True,
        )
        print(f"[MONITOR] 통합 후처리 파이프라인 완료: {len(result_df)} rows 처리됨")
        return result_df
    except Exception as e:
        print(f"[MONITOR] ERROR: 통합 후처리 파이프라인 실패: {e}")
        raise


def run_monitor(
    output_path: str = DEFAULT_OUTPUT_CSV,
    interval_sec: float = DEFAULT_INTERVAL_SEC,
    bis_max_pages: int = 1,                                # 거의 업데이트 없음, 기본값 1페이지로 충분
    sleep_sec: float = 1.0,                                # Rate limit을 피하기 위해 요청 사이에 1초 정도 쉬는 것이 안전
    max_cycles: int | None = None,
) -> None:
    cycle = 0
    
    processed_path = Path(output_path)
    # old_news.csv는 메인 출력 파일과 같은 디렉토리에 저장하도록 경로 설정
    old_news_path = Path(DEFAULT_OLD_DATA_CSV)

    while True:
        target_date = _get_target_date()
        cycle += 1

        print(f"[MONITOR] cycle={cycle} target_date={target_date.isoformat()} ")

        # 각 주기마다 전날 데이터를 한 번만 모은 뒤, interval_sec 값이 기본값이면 다음 자정까지 쉰다.
        news_list = collect_policy_updates(
            bis_max_pages=bis_max_pages,
            sleep_sec=sleep_sec,
            target_date=target_date,
        )

        news_count = len(news_list)
        if news_count > 0:
            print(f"[MONITOR] 수집한 뉴스 개수 : {news_count}개")
            processed_news = run_postprocessing_pipeline(df=news_list)

            # 1. 이전에 저장된 csv 파일 읽기
            if processed_path.exists():
                try:
                    existing_news = pd.read_csv(processed_path, encoding="utf-8-sig")
                except Exception:
                    existing_news = pd.DataFrame()
            else:
                existing_news = pd.DataFrame()

            # 2. 데이터 병합
            if existing_news.empty:
                combined_news = processed_news.copy()
            else:
                combined_news = pd.concat([existing_news, processed_news], ignore_index=True, sort=False)

            # 3. 중복 제거 및 컬럼 정렬
            if not combined_news.empty and {"sector", "url"}.issubset(combined_news.columns):
                combined_news = combined_news.drop_duplicates(subset=["sector", "url"], keep="last")

            if "sector" in combined_news.columns:
                ordered_columns = ["sector"] + [column for column in combined_news.columns if column != "sector"]
                combined_news = combined_news[ordered_columns]

            # 4. 20일 경과 데이터 필터링 및 old_news 분리

            if not combined_news.empty and "release_date" in combined_news.columns:
                # 안전한 비교를 위해 datetime으로 변환 후 시간대(timezone) 정보 제거
                parsed_dates = pd.to_datetime(combined_news["release_date"], errors='coerce').dt.tz_localize(None)
                cutoff_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=20)
                
                # 20일이 지난 데이터 마스킹 (NaT 등 파싱 실패한 데이터는 최신으로 간주하여 보존)
                is_old_mask = parsed_dates < cutoff_date
                
                aged_out_news = combined_news[is_old_mask]
                recent_news = combined_news[~is_old_mask]

                # [I/O 최소화 포인트] 20일 지난 데이터가 "실제로 존재할 때만" old_news.csv를 읽고 씁니다.
                if not aged_out_news.empty:
                    if old_news_path.exists():
                        try:
                            old_news_df = pd.read_csv(old_news_path, encoding="utf-8-sig")
                            aged_out_news = pd.concat([old_news_df, aged_out_news], ignore_index=True, sort=False)
                            # old_news에서도 중복 방지
                            if {"sector", "url"}.issubset(aged_out_news.columns):
                                aged_out_news = aged_out_news.drop_duplicates(subset=["sector", "url"], keep="last")
                        except Exception:
                            pass
                    
                    # old_news.csv 저장
                    aged_out_news.to_csv(old_news_path, index=False, encoding="utf-8-sig")
                    print(f"[MONITOR] 20일 경과 뉴스 {len(aged_out_news)}건을 {old_news_path.name}로 이동했습니다.")
            else:
                recent_news = combined_news

            # 최신 데이터만 원본 파일에 덮어쓰기 저장
            recent_news.to_csv(processed_path, index=False, encoding="utf-8-sig")
            print(f"[MONITOR] 모니터링 결과 저장 위치 : {processed_path} / 개수 : {len(recent_news)}")
            
        else:
            print("[MONITOR] 수집된 뉴스가 없습니다.")
            # 수집된 뉴스가 없을 때는 기존 파일을 열어서 날짜를 확인하는 작업을 생략하여 I/O를 최소화했습니다.

        if max_cycles is not None and cycle >= max_cycles:
            break

        if interval_sec <= 0:
            break

        if interval_sec >= DEFAULT_INTERVAL_SEC:
            time.sleep(_seconds_until_next_us_eastern_midnight())
        else:
            time.sleep(interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="[Monitor] FOMC, BIS, FRASER, EIA, UCSB, YAHOO updates for the previous America/New_York day")
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_CSV, help="수집 결과 CSV 파일 경로")
    parser.add_argument("--interval-sec", type=float, default=DEFAULT_INTERVAL_SEC, help="모니터링 간격 (초 단위); 기본값은 다음 America/New_York 자정에 스케줄링됩니다")
    parser.add_argument("--bis-max-pages", type=int, default=1)
    parser.add_argument("--sleep-sec", type=float, default=1.0, help="Rate limit을 피하기 위해 요청 사이에 쉬는 시간 (초 단위)")
    parser.add_argument("--max-cycles", type=int, default=None, help="모니터링 주기 반복 횟수 제한; None이면 무한 반복")
    args = parser.parse_args()

    run_monitor(
        output_path=args.output_path,
        interval_sec=args.interval_sec,
        bis_max_pages=args.bis_max_pages,
        sleep_sec=args.sleep_sec,
        max_cycles=args.max_cycles,
    )


if __name__ == "__main__":
    main()