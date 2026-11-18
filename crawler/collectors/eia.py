from __future__ import annotations

import argparse
import logging
import sys
import re
from pathlib import Path
from urllib.parse import urljoin
from datetime import date, datetime

import requests
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crawler.collectors.common import fetch_soup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


EIA_ROOT_URL = "https://www.eia.gov"
STEO_URL = "https://www.eia.gov/analysis/reports.php#/T186,T1139"
BASE_URL = "https://www.eia.gov/todayinenergy/"
TODAY_IN_ENERGY_URL = "https://www.eia.gov/todayinenergy/archive.php?my=all"
DEFAULT_START_DATE = "2017-01-01"
DEFAULT_OUTPUT_PATH = "data/crawler/collected/eia_collected.csv"


headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _fetch_dynamic_soup(url: str) -> BeautifulSoup | None:
    """
    브라우저를 가상(Headless)으로 띄워 JavaScript 렌더링이 
    완료된 완성형 HTML을 BeautifulSoup 객체로 반환합니다.
    """
    try:
        with sync_playwright() as p:
            # 브라우저 초기화 성능 최적화 옵션 추가
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # 봇 탐지 우회를 위한 가벼운 User-Agent 설정
            page.set_extra_http_headers(headers)
            
            # 페이지 이동 및 네트워크 안정화 대기
            page.goto(url, wait_until="networkidle") 

            # 실제 DOM id 기준으로 대기
            page.wait_for_selector("#reports-holder", timeout=10000)

            html = page.content()
            browser.close()
            return BeautifulSoup(html, "html.parser")
            
    except Exception:
        # 실패 시 페이지 구조 후보를 더 확인할 수 있도록 상세 예외를 남깁니다.
        logger.exception("STEO: Playwright 렌더링 중 오류 발생")
        return None


def get_steo_items(start_date: str = DEFAULT_START_DATE) -> list[dict]:
    """
    Playwright를 통해 동적으로 로드된 STEO 리포트 목록을 
    시작 날짜 이후 데이터 위주로 안전하게 수집합니다.
    """
    contents: list[dict] = []

    try:
        start_date_value = datetime.strptime(start_date, "%Y-%m-%d").date()
    except ValueError as e:
        logger.error(f"STEO: 시작 날짜 포맷 에러 (YYYY-MM-DD 필요): {e}")
        return contents

    soup = _fetch_dynamic_soup(STEO_URL)
    if soup is None:
        logger.warning("STEO: 페이지 소스를 가져오지 못했습니다. Playwright 브라우저가 설치되어 있는지 확인하세요. 예: 'python -m playwright install' 또는 'playwright install chromium'")
        return contents

    results_table = soup.find("div", id="reports-holder")
    if not results_table:
        candidate_ids = [tag.get("id") for tag in soup.find_all(attrs={"id": True}) if tag.get("id") and "report" in tag.get("id")]
        logger.info("STEO: no reports-holder container found; candidate ids=%s", candidate_ids)
        return contents

    results = results_table.find_all("div", class_="b_content")
    for result in results:
        title_tag = result.find("h3")
        title = title_tag.get_text(" ", strip=True) if title_tag else ""

        a_tag = title_tag.find("a") if title_tag else None
        href = a_tag.get("href") if a_tag else None
        # urljoin 시 도메인 루트(EIA_ROOT_URL) 기준으로 정확히 매핑
        url = urljoin(EIA_ROOT_URL, href) if href else None

        release_date_tag = result.find("h4")
        release_date = release_date_tag.get_text(" ", strip=True) if release_date_tag else ""
        release_date_value = _parse_eia_date(release_date)

        if release_date_value is None:
            logger.info("STEO: skipped item with unparsable date: %s", release_date)
            continue

        # 시작 날짜 이전 자료 진입 시 루프 종료 탈출
        if release_date_value < start_date_value:
            break

        contents.append({
            "doc_type": "STEO",
            "title": title,
            "url": url,
            "release_date": str(release_date_value),
        })

    return contents


def get_steo_body(url: str) -> str:
    """STEO 리포트 세부 페이지에서 본문 텍스트를 추출합니다."""
    # 메인 모듈에 선언되어 있는 공통 본문 크롤러 함수 호출
    return _get_article_body(url)


def _parse_eia_date(raw_value: str) -> date | None:
    """EIA 목록 날짜(예: 'May 12, 2026', 'May12, 2026')를 date 객체로 변환합니다."""
    if not raw_value:
        return None

    normalized = raw_value.strip()
    # Handle compact month-day forms like 'May12, 2026' -> 'May 12, 2026'.
    normalized = re.sub(r"^([A-Za-z]+)(\d{1,2},\s*\d{4})$", r"\1 \2", normalized)

    for fmt in ("%B %d, %Y",):
        try:
            return datetime.strptime(normalized, fmt).date()
        except ValueError:
            continue
    return None


def get_today_items(start_date: str = DEFAULT_START_DATE):
    """
    TODAY IN ENERGY 아카이브를 시작 날짜 이후로 수집합니다.
    """
    contents: list[dict] = []
    start_date_value = datetime.strptime(start_date, "%Y-%m-%d").date()

    soup = _fetch_soup(TODAY_IN_ENERGY_URL)
    if soup is None:
        logger.warning("TODAY IN ENERGY: unable to fetch page")
        return contents

    results_table = soup.find("div", class_="accordion")
    if not results_table:
        logger.info("TODAY IN ENERGY: no accordion container found")
        return contents

    spans = results_table.find_all("span")
    title_tags = results_table.find_all("h2")

    # lengths may differ depending on page structure; iterate over the minimum
    for span, title_tag in zip(spans, title_tags):
        title = title_tag.get_text(" ", strip=True) if title_tag else ""
        a_tag = title_tag.find("a") if title_tag else None
        href = a_tag.get("href") if a_tag else None
        url = urljoin(BASE_URL, href) if href else None
        release_date = span.get_text(" ", strip=True) if span else ""
        release_date_value = _parse_eia_date(release_date)

        if release_date_value is None:
            logger.info("TODAY IN ENERGY: skipped item with unparsable date: %s", release_date)
            continue

        if release_date_value < start_date_value:
            break

        contents.append({
            "doc_type": "TODAY_IN_ENERGY",
            "title": title,
            "url": url,
            "release_date": str(release_date_value),
        })

    return contents
    
def get_today_body(url):
    return _get_article_body(url)


def _fetch_soup(url: str) -> BeautifulSoup | None:
    """공통: URL을 요청하고 BeautifulSoup 객체를 반환합니다. 실패 시 None을 반환합니다."""
    try:
        return fetch_soup(url, headers=headers, timeout=(5, 30), client=requests)
    except requests.RequestException as exc:
        logger.exception("Request failed for %s: %s", url, exc)
        return None


def _get_article_body(url: str) -> str:
    """공통: 기사/리포트 페이지에서 본문을 추출합니다. 실패 시 빈 문자열을 반환합니다."""
    if not url:
        return ""

    soup = _fetch_soup(url)
    if soup is None:
        return ""

    body_tag = soup.find("div", class_="tie-article")
    body = body_tag.get_text(" ", strip=True) if body_tag else ""
    return body


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EIA(STEO/TODAY) crawler runner")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="결과 CSV 저장 경로 (기본값: data/crawler/collected/eia_collected.csv)",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="수집 시작일 (YYYY-MM-DD)",
    )
    return parser.parse_args()

# ==========================================
# 실행 테스트 및 DataFrame 변환 예시
# ==========================================
if __name__ == "__main__":
    args = _parse_args()
    results: list[dict] = []

    steo_data = get_steo_items(start_date=args.start_date)
    if steo_data:
        results.extend(steo_data)

    today_data = get_today_items(start_date=args.start_date)
    if today_data:
        results.extend(today_data)

    # 간소화된 진행 로그: 총 건수 및 구간별 요약만 출력
    total = len(results)
    logger.info("Collected total=%d (STEO=%d, TODAY=%d)", total, len(steo_data) if steo_data else 0, len(today_data) if today_data else 0)

    # 기사 1건 추출 시마다 진행 로그 출력
    for idx, rec in enumerate(results, start=1):
        rec_url = rec.get("url")
        doc_type = rec.get("doc_type", "UNKNOWN")
        title = rec.get("title", "")
        try:
            rec["content"] = _get_article_body(rec_url) if rec_url else ""
            logger.info("Article extracted: %d/%d [%s] %s", idx, total, doc_type, title)
        except Exception:
            logger.exception("Failed to fetch body for %s (%s)", rec.get("doc_type", "UNKNOWN"), rec_url)

    # Pandas DataFrame으로 정제 (Spring Boot로 던지거나 DB에 바로 박기 편한 형태)
    df = pd.DataFrame(results)
    print("\n[수집 완료 데이터 레이아웃]")
    if not df.empty and "doc_type" in df.columns:
        print(df[["doc_type", "release_date"]])
    else:
        print(df.head())

    # 첫 번째 수집 결과물 샘플 출력
    if not df.empty and "content" in df.columns:
        print("\n[TODAY IN ENERGY 수집 데이터 샘플 텍스트 (FinBERT 입력용)]")
        print(df.iloc[0]["content"][:300] + "...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Saved result CSV: %s", output_path)