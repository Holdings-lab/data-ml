import argparse
import re
import time
import random
import certifi
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, ElementClickInterceptedException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.support_legacy.data_paths import collected_csv_path


BASE_URL = "https://www.bis.gov"
START_URL = f"{BASE_URL}/news-updates"

HEADLESS = True
WAIT_SEC = 15

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

MONTH_DATE_PATTERN = (
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+\d{4}"
)


def clean_text(text: Optional[str]) -> str:
    """연속 공백/줄바꿈을 1칸 공백으로 정리한다."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def normalize_date(date_text: str) -> Optional[str]:
    """
    다양한 날짜 문자열에서 'Month DD, YYYY' 형식을 찾아
    YYYY-MM-DD 문자열로 정규화한다.
    """
    if not date_text:
        return None

    match = re.search(MONTH_DATE_PATTERN, date_text)
    if not match:
        return None

    raw_date = match.group(0)

    try:
        return datetime.strptime(raw_date, "%B %d, %Y").strftime("%Y-%m-%d")
    except ValueError:
        return raw_date


def create_driver(headless: bool = True) -> webdriver.Chrome:
    """Selenium Chrome 드라이버를 생성한다."""
    options = Options()

    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--window-size=1600,2200")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--user-agent={HEADERS['User-Agent']}")

    return webdriver.Chrome(options=options)


def wait_for_page_ready(driver: webdriver.Chrome, wait_sec: int = WAIT_SEC) -> None:
    """문서 로딩이 complete 상태가 될 때까지 대기한다."""
    WebDriverWait(driver, wait_sec).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )


def get_first_press_release_url(driver: webdriver.Chrome) -> Optional[str]:
    """
    현재 목록 페이지에서 첫 번째 보도자료 링크를 반환한다.
    페이지 이동 여부 확인용 기준값으로 사용한다.
    """
    soup = BeautifulSoup(driver.page_source, "html.parser")
    first_link = soup.select_one('a[href*="/press-release/"]')

    if not first_link:
        return None

    href = first_link.get("href", "")
    if not href:
        return None

    return urljoin(BASE_URL, href)


def extract_card_links_from_page(driver: webdriver.Chrome) -> List[Dict]:
    """
    현재 목록 페이지에서 보도자료 상세 링크를 추출한다.

    - /press-release/ 를 포함한 a 태그를 직접 찾는다.
    - 제목은 a 내부의 h3/h2 또는 a 텍스트에서 추출한다.
    """
    soup = BeautifulSoup(driver.page_source, "html.parser")

    items: List[Dict] = []
    seen = set()

    link_tags = soup.select('a[href*="/press-release/"]')

    for a_tag in link_tags:
        href = a_tag.get("href", "")
        if not href:
            continue

        title_tag = a_tag.find("h3")
        if title_tag:
            title = clean_text(title_tag.get_text(" ", strip=True))

        if not title:
            continue

        url = urljoin(BASE_URL, href)

        if url in seen:
            continue

        seen.add(url)
        items.append(
            {
                "title": title,
                "url": url,
            }
        )

    return items


def extract_body_text(container) -> str:
    """
    본문 컨테이너 내부의 p 태그들을 합쳐 본문 문자열을 만든다.
    빈 문단은 제외한다.
    """
    if not container:
        return ""

    body_parts: List[str] = []

    for p_tag in container.find_all("p"):
        text = clean_text(p_tag.get_text(" ", strip=True))
        if text:
            body_parts.append(text)

    return "\n".join(body_parts)


def parse_bis_article_html(html: str, url: str) -> Optional[Dict]:
    """BIS 상세 페이지 HTML에서 제목/날짜/본문을 추출한다."""
    
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.select_one("h2.leading-none.mb-4")
    title = clean_text(title_tag.get_text(" ", strip=True)) if title_tag else ""

    if not title:
        return None

    # 날짜 후보를 여러 군데서 찾음
    date_tag = soup.find("span", class_="date")
    date_text = clean_text(date_tag.get_text(" ", strip=True))
    
    published_date = normalize_date(date_text)

    # 본문 컨테이너 후보를 여러 개 시도
    body_container =  soup.find("div", class_="press-release-container")
    body = extract_body_text(body_container)

    return {
        "published_date": published_date,
        "category": "BIS",
        "doc_type": "press_release",
        "title": title,
        "url": url,
        "body": body,
    }


def create_requests_session() -> requests.Session:
    """
    BIS는 간헐적으로 느리거나(그에 따른 read timeout) rate limit(429)을 걸 수 있어,
    재시도/백오프/커넥션 재사용이 있는 세션을 사용한다.
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    retry = Retry(
        total=5,
        connect=3,
        read=5,
        backoff_factor=1.0,
        # 429/5xx는 재시도 대상으로 둔다.
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def extract_article(
    url: str,
    session: requests.Session,
    sleep_sec: float = 0.5,
) -> Optional[Dict]:
    """
    상세 페이지를 requests로 가져오되,
    SSL 인증서 검증용 CA 번들을 명시한다.
    """
    try:
        response = session.get(
            url,
            # connect와 read를 분리해서 read timeout 빈도를 줄인다.
            timeout=(10, 60),
            verify=certifi.where(),
        )
        response.raise_for_status()
    except requests.exceptions.SSLError as e:
        print(f"  -> SSL 인증서 검증 실패: {e}")
        return None
    except requests.RequestException as e:
        print(f"  -> 상세 페이지 요청 실패: {e}")
        return None

    # 레이트리밋을 덜 맞기 위해 약간의 지터를 추가한다.
    time.sleep(sleep_sec + random.uniform(0, 0.5))
    return parse_bis_article_html(response.text, url)


def click_next(driver, wait_sec: int = 15) -> bool:
    """
    Next 버튼을 클릭하고 실제 목록이 바뀌는지 확인한다.
    """
    before_first_url = get_first_press_release_url(driver)

    try:
        next_btn = WebDriverWait(driver, wait_sec).until(
            EC.presence_of_element_located(
                (By.XPATH, "//button[normalize-space()='Next']")
            )
        )
    except TimeoutException:
        print("  -> Next 버튼을 찾지 못함")
        return False

    # 실제 disabled 속성만 확인
    disabled_attr = next_btn.get_attribute("disabled")
    aria_disabled = next_btn.get_attribute("aria-disabled")

    if disabled_attr is not None or aria_disabled == "true":
        print("  -> Next 버튼이 실제 비활성 상태")
        return False

    # 화면 중앙으로 스크롤
    driver.execute_script(
        "arguments[0].scrollIntoView({block: 'center', inline: 'center'});",
        next_btn
    )
    time.sleep(1)

    # 일반 클릭 -> JS 클릭 순서
    try:
        next_btn.click()
    except (ElementClickInterceptedException, StaleElementReferenceException, Exception):
        try:
            driver.execute_script("arguments[0].click();", next_btn)
        except Exception as e:
            print(f"  -> 클릭 실패: {e}")
            return False

    # 클릭 후 목록 첫 링크가 바뀌는지 확인
    try:
        WebDriverWait(driver, wait_sec).until(
            lambda d: get_first_press_release_url(d) != before_first_url
        )
        time.sleep(1)
        return True
    except TimeoutException:
        print("  -> 클릭은 됐지만 페이지 변화 감지 실패")
        return False


def crawl_bis_news_index_selenium(
    max_pages: int = 5,
    sleep_sec: float = 1.0,
) -> List[Dict]:
    """
    목록 페이지를 순회하며 보도자료 상세 링크를 수집한다.
    """
    driver = create_driver(headless=HEADLESS)
    all_items: List[Dict] = []
    seen_urls = set()

    try:
        driver.get(START_URL)
        wait_for_page_ready(driver)
        time.sleep(2)

        for page_no in range(1, max_pages + 1):
            print(f"[INDEX] page {page_no}")

            page_items = extract_card_links_from_page(driver)
            new_count = 0

            for item in page_items:
                if item["url"] in seen_urls:
                    continue

                seen_urls.add(item["url"])
                all_items.append(item)
                new_count += 1

            print(f"  -> 새 링크 수집: {new_count}건")

            if page_no == max_pages:
                break

            moved = click_next(driver)
            if not moved:
                print("  -> Next 버튼 이동 실패 또는 마지막 페이지")
                break

            time.sleep(sleep_sec)

    finally:
        driver.quit()

    return all_items


def crawl_bis_press_releases(
    max_pages: int = 5,
    sleep_sec: float = 1.0,
    output_csv: str = collected_csv_path("bis_press_releases.csv"),
) -> pd.DataFrame:
    """
    1) Selenium으로 목록 링크 수집
    2) requests로 상세 기사 파싱
    3) CSV 저장
    """
    listing_items = crawl_bis_news_index_selenium(
        max_pages=max_pages,
        sleep_sec=sleep_sec,
    )

    print(f"\n총 목록 수집 개수: {len(listing_items)}")

    results: List[Dict] = []
    session = create_requests_session()

    for i, item in enumerate(listing_items, start=1):
        print(f"[ARTICLE {i}/{len(listing_items)}] {item['url']}")
        article = extract_article(item["url"], session=session, sleep_sec=sleep_sec)

        if not article:
            print("  -> 본문 추출 실패")
            continue

        results.append(article)
        print(f"  -> 저장: {article['title']}")

    df = pd.DataFrame(results)

    if not df.empty:
        keep_cols = [
            "published_date",
            "category",
            "doc_type",
            "title",
            "url",
            "body",
        ]
        df = df[keep_cols]

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {output_csv} / 건수: {len(df)}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIS press release crawler")
    parser.add_argument("--max-pages", type=int, default=9, help="목록 페이지 최대 탐색 수")
    parser.add_argument("--sleep-sec", type=float, default=1.0, help="항목 간 대기(지터 포함)")
    parser.add_argument(
        "--output-csv",
        type=str,
        default=collected_csv_path("bis_press_releases.csv"),
        help="저장 CSV 파일명",
    )
    args = parser.parse_args()

    df = crawl_bis_press_releases(
        max_pages=args.max_pages,
        sleep_sec=args.sleep_sec,
        output_csv=args.output_csv,
    )
    print(df.head())
