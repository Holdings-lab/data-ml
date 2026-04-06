from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.support_legacy.data_paths import collected_csv_path

BASE_URL = "https://www.presidency.ucsb.edu"
DEFAULT_OUTPUT_CSV = collected_csv_path("ucsb_presidential_documents.csv")
DEFAULT_KEYWORD_CONFIG_PATH = Path(__file__).with_name("ucsb_keywords.json")

# 날짜 기반 수집의 기본 시작점.
# 별도 인자를 주지 않으면 이 날짜 이후 문서만 모은다.
DEFAULT_START_DATE = "2025-01-01"

# UCSB 목록 페이지에서 한 번에 노출할 문서 수.
# 페이지 수를 기준으로 자르지는 않지만, 페이지 요청 URL을 만들 때는 필요하다.
ITEMS_PER_PAGE = 20

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}

# UCSB에서 현재 수집 대상으로 삼는 문서 카테고리의 목록 페이지 URL.
# 각 카테고리는 최신 문서가 먼저 보인다는 전제를 두고,
# 시작 날짜보다 오래된 문서가 나오면 이후 페이지 탐색을 멈춘다.
DOC_TYPE_URLS = {
    "Executive Orders": (
        f"{BASE_URL}/documents/app-categories/"
        "written-presidential-orders/presidential/executive-orders"
    ),
    "Press Conferences": f"{BASE_URL}/documents/app-categories/presidential/news-conferences",
    "Fact Sheets": f"{BASE_URL}/documents/app-attributes/fact-sheets",
}

# 본문 뒤쪽의 메타데이터/탐색 영역이 시작되는 제목들.
# 이런 구간까지 body에 포함하면 키워드 매칭 품질이 떨어지므로 수집을 중단한다.
STOP_SECTION_TITLES = {
    "filed under",
    "categories",
    "simple search of our archives",
}

# UCSB 목록 페이지의 날짜 헤더 형식 예: "April 5, 2026"
DATE_PATTERN = re.compile(r"^[A-Z][a-z]+ \d{1,2}, \d{4}$")


def clean_text(text: str) -> str:
    """
    연속 공백과 줄바꿈을 하나의 공백으로 정리한 뒤 양끝 공백을 제거한다.

    UCSB 페이지는 줄바꿈, 탭, 중복 공백이 섞여 있는 경우가 많아서
    대부분의 텍스트 비교와 저장 전에 이 정규화를 거친다.
    """
    return re.sub(r"\s+", " ", text).strip()


def parse_published_date(raw_value: str) -> str:
    """
    UCSB 목록 페이지의 날짜 문자열을 YYYY-MM-DD 형식으로 변환한다.

    입력 예:
    - "April 5, 2026"
    출력 예:
    - "2026-04-05"
    """
    parsed = datetime.strptime(raw_value, "%B %d, %Y")
    return parsed.strftime("%Y-%m-%d")


def parse_start_date(raw_value: str) -> datetime.date:
    """
    CLI로 받은 시작 날짜 문자열을 date 객체로 변환한다.

    이 date 객체는 목록 페이지 순회 중
    "이 문서가 수집 대상 기간 안에 있는가?"를 판단하는 기준점으로 사용된다.
    """
    return datetime.strptime(raw_value, "%Y-%m-%d").date()


def normalize_keyword_dictionary(
    keyword_dictionary: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """
    키워드 JSON을 내부에서 쓰기 쉬운 형태로 정리한다.

    처리 규칙:
    - 그룹명과 키워드 문자열의 공백을 정리한다.
    - 빈 문자열은 제거한다.
    - 중복 키워드는 제거한다.
    - 대소문자 구분 없는 정렬로 결과를 고정한다.
    - 최종적으로 유효한 그룹이 하나도 없으면 예외를 발생시킨다.
    """
    normalized: dict[str, list[str]] = {}

    for group_name, raw_keywords in keyword_dictionary.items():
        group = clean_text(str(group_name))
        keywords = [
            clean_text(str(keyword))
            for keyword in raw_keywords
            if clean_text(str(keyword))
        ]
        if group and keywords:
            normalized[group] = sorted(set(keywords), key=str.lower)

    if not normalized:
        raise ValueError("Keyword dictionary must include at least one non-empty group.")

    return normalized


def load_keyword_dictionary(keyword_config_path: str | Path | None) -> dict[str, list[str]]:
    """
    키워드 설정 JSON 파일을 읽고 정규화된 사전 형태로 반환한다.

    경로가 주어지지 않으면 수집기 옆의 기본 키워드 파일을 사용한다.
    """
    config_path = (
        Path(keyword_config_path)
        if keyword_config_path is not None
        else DEFAULT_KEYWORD_CONFIG_PATH
    )

    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("Keyword config must be a JSON object of group -> keyword list.")

    return normalize_keyword_dictionary(payload)


def fetch_page(url: str) -> BeautifulSoup:
    """
    UCSB 페이지를 요청하고 BeautifulSoup 객체로 반환한다.

    네트워크 오류나 HTTP 오류는 호출한 쪽에서 처리할 수 있도록 그대로 올린다.
    """
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def build_listing_url(base_url: str, page_number: int) -> str:
    """
    UCSB 카테고리 목록 URL에 페이지 번호와 페이지 크기 파라미터를 붙인다.

    이 사이트는 첫 페이지를 `page=1` 없이도 제공하므로
    1페이지는 `items_per_page`만 붙이고,
    이후 페이지부터 `page=N`을 붙인다.
    """
    if page_number <= 1:
        return f"{base_url}?items_per_page={ITEMS_PER_PAGE}"
    return f"{base_url}?page={page_number}&items_per_page={ITEMS_PER_PAGE}"


def parse_listing_page(soup: BeautifulSoup, doc_type: str) -> list[dict[str, str]]:
    """
    UCSB 목록 페이지에서 문서 메타데이터를 추출한다.

    이 함수가 만드는 필드:
    - title
    - url
    - published_date
    - doc_type
    - category

    UCSB 목록 페이지는 다음처럼 날짜 헤더와 문서 링크가 섞여 있다.
    - h4: "April 5, 2026"
    - a: "Executive Order ..."
    - a: "Fact Sheet ..."

    따라서 가장 최근에 본 날짜 헤더를 현재 링크의 published_date로 사용한다.
    """
    items: list[dict[str, str]] = []
    current_date = ""

    # 실제 문서 리스트는 view-content 컨테이너 아래에 붙는다.
    listing_container = soup.find("div", class_="view-content")
    if listing_container is None:
        return items

    for tag in listing_container.find_all(["h4", "a"]):
        if tag.name == "h4":
            heading_text = clean_text(tag.get_text(" ", strip=True))
            if DATE_PATTERN.match(heading_text):
                current_date = heading_text
            continue

        href = tag.get("href")
        title = clean_text(tag.get_text(" ", strip=True))

        if not href or not title:
            continue

        # 실제 문서 링크만 수집한다.
        # 목록 안에는 필터/페이지네이션 링크도 섞일 수 있어서 URL 패턴을 먼저 본다.
        if not href.startswith("/documents/"):
            continue

        # 페이지네이션 UI 텍스트는 문자 인코딩에 따라 깨질 수 있으므로
        # 완전일치 대신 대표 접두어(next/last) 기준으로 제외한다.
        lowered_title = title.lower()
        if lowered_title.startswith("next") or lowered_title.startswith("last"):
            continue

        # 날짜 헤더 이전에 나온 링크는 게시일을 알 수 없으므로 버린다.
        # 이 크롤러는 published_date를 기준으로 시작 날짜 필터를 적용하므로,
        # 날짜 없는 레코드는 뒤 단계에서 다루기 어렵다.
        if not current_date:
            continue

        items.append(
            {
                "title": title,
                "url": urljoin(BASE_URL, href),
                "published_date": parse_published_date(current_date),
                "doc_type": doc_type,
                "category": "UCSB Presidency Project",
            }
        )

    # 같은 페이지 안에서 같은 문서가 반복 노출될 수 있어 URL 기준으로 중복 제거한다.
    deduped: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for item in items:
        if item["url"] in seen_urls:
            continue
        seen_urls.add(item["url"])
        deduped.append(item)

    return deduped


def crawl_listing(
    base_url: str,
    doc_type: str,
    start_date: datetime.date,
    sleep_sec: float,
) -> list[dict[str, str]]:
    """
    시작 날짜를 기준으로 목록 페이지를 계속 내려가며 문서 메타데이터를 수집한다.

    기존의 max-pages 방식 대신 날짜 기반으로 멈추는 이유:
    - 어떤 기간에는 문서가 많고 어떤 기간에는 적다.
    - 페이지 수로 자르면 기간별 편차가 심하다.
    - "YYYY-MM-DD 이후 문서"라는 기준이 훨씬 재현 가능하고 해석이 쉽다.

    동작 방식:
    1. 1페이지부터 시작한다.
    2. 각 페이지에서 문서를 파싱한다.
    3. start_date 이상인 문서만 보관한다.
    4. start_date보다 오래된 문서가 보이기 시작하면 이후 페이지는 더 오래됐을 가능성이 높으므로 중단한다.
    """
    items: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    page_number = 1

    while True:
        page_url = build_listing_url(base_url, page_number)
        print(f"[INDEX] {doc_type}: {page_url}")

        try:
            soup = fetch_page(page_url)
        except Exception as exc:
            print(f"  -> failed to fetch listing page: {exc}")
            break

        page_items = parse_listing_page(soup, doc_type)
        if not page_items:
            print("  -> no listing items found, stopping")
            break

        # 현재 페이지 안에서도 시작 날짜보다 이전 문서는 버리고,
        # 기준 날짜 이상인 문서만 다음 단계(상세 페이지 파싱)로 넘긴다.
        eligible_items: list[dict[str, str]] = []
        reached_older_documents = False

        for item in page_items:
            published_date = datetime.strptime(item["published_date"], "%Y-%m-%d").date()
            if published_date < start_date:
                # 목록이 최신순이라는 전제 하에,
                # 한 번 오래된 문서가 나오기 시작하면 이후 페이지는 더 오래됐을 가능성이 크다.
                reached_older_documents = True
                continue
            eligible_items.append(item)

        added_count = 0
        for item in eligible_items:
            if item["url"] in seen_urls:
                continue
            seen_urls.add(item["url"])
            items.append(item)
            added_count += 1

        print(f"  -> found {added_count} new items on or after {start_date.isoformat()}")

        if reached_older_documents:
            print("  -> reached documents older than start date, stopping")
            break

        # 아직 오래된 문서를 만나지 않았으면 다음 페이지로 진행한다.
        page_number += 1
        time.sleep(sleep_sec)

    return items


def extract_article_body(soup: BeautifulSoup) -> str:
    """
    문서 상세 페이지에서 실제 본문 텍스트만 추출한다.

    예전처럼 제목(h1)의 형제 노드를 따라가면 body가 비는 페이지가 있어서,
    현재는 실제 본문 래퍼인 `div.field-docs-content`를 직접 기준점으로 사용한다.

    본문 수집 규칙:
    - p, li, blockquote, h2/h3/h4만 수집한다.
    - 메타데이터 섹션 제목이 나오면 중단한다.
    - 날짜만 있는 줄은 제외한다.
    - li는 접두사 "- "를 붙여 목록 구조를 보존한다.
    """
    content_container = soup.find("div", class_="field-docs-content")
    if content_container is None:
        return ""

    body_parts: list[str] = []

    for node in content_container.find_all(["p", "li", "blockquote", "h2", "h3", "h4"]):
        heading_text = clean_text(node.get_text(" ", strip=True)).lower()

        # "Filed Under", "Categories" 같은 메타데이터 섹션이 시작되면 본문 수집 종료
        if node.name in {"h2", "h3", "h4"} and heading_text.startswith(tuple(STOP_SECTION_TITLES)):
            break

        text = clean_text(node.get_text("\n", strip=True))
        if not text:
            continue

        # 본문 위쪽에 날짜가 다시 찍히는 경우가 있어 이 줄은 제외한다.
        if DATE_PATTERN.match(text):
            continue

        # 목록은 문단과 섞여도 구조를 잃지 않도록 접두사를 붙여 보존한다.
        if node.name == "li":
            text = f"- {text}"

        body_parts.append(text)

    # UCSB 본문은 마크업 영향으로 같은 문단이 반복 추출될 수 있어
    # 본문 조각 수준에서 한 번 더 중복 제거한다.
    deduped_parts: list[str] = []
    seen_parts: set[str] = set()
    for part in body_parts:
        if part in seen_parts:
            continue
        seen_parts.add(part)
        deduped_parts.append(part)

    return "\n\n".join(deduped_parts).strip()


def match_keywords(
    text: str,
    keyword_dictionary: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """
    제목+본문 텍스트에서 어떤 키워드 그룹이 매치되는지 찾는다.

    현재는 단순 substring 매칭을 사용한다.
    이 방식은:
    - 속도가 빠르고
    - 규칙이 명확하며
    - 왜 매치됐는지 설명하기 쉽다는 장점이 있다.
    """
    lowered_text = text.lower()
    matches: dict[str, list[str]] = {}

    for group_name, keywords in keyword_dictionary.items():
        group_matches = [keyword for keyword in keywords if keyword.lower() in lowered_text]
        if group_matches:
            matches[group_name] = sorted(set(group_matches), key=str.lower)

    return matches


def parse_article(
    metadata: Mapping[str, str],
    keyword_dictionary: Mapping[str, Sequence[str]],
) -> dict[str, Any] | None:
    """
    개별 문서 상세 페이지를 읽어 제목/본문/키워드 매칭 결과를 만든다.

    이 함수가 하는 일:
    1. 상세 페이지 요청
    2. 제목 추출
    3. 본문 추출
    4. 제목+본문 기준 키워드 매칭
    5. 매치된 문서만 표준 레코드로 반환

    키워드가 하나도 매치되지 않으면 None을 반환해서 최종 결과에서 제외한다.
    """
    url = metadata["url"]
    print(f"[ARTICLE] {url}")

    try:
        soup = fetch_page(url)
    except Exception as exc:
        print(f"  -> failed to fetch article: {exc}")
        return None

    title_tag = soup.find("h1")
    title = clean_text(title_tag.get_text(" ", strip=True)) if title_tag else metadata["title"]
    body = extract_article_body(soup)

    # 제목만으로 핵심 주제가 드러나는 문서도 많으므로 제목과 본문을 함께 매칭한다.
    combined_text = f"{title}\n{body}"
    matches = match_keywords(combined_text, keyword_dictionary)

    if not matches:
        print("  -> skipped: no keyword match")
        return None

    matched_keywords = sorted(
        {keyword for keywords in matches.values() for keyword in keywords},
        key=str.lower,
    )
    matched_groups = sorted(matches.keys(), key=str.lower)

    print(f"  -> kept: {', '.join(matched_groups)}")
    return {
        **metadata,
        "title": title,
        "body": body,
        "matched_keyword_groups": ", ".join(matched_groups),
        "matched_keywords": ", ".join(matched_keywords),
        "keyword_matches_json": json.dumps(matches, ensure_ascii=False),
    }


def crawl_ucsb_documents(
    keyword_dictionary: Mapping[str, Sequence[str]],
    start_date: str = DEFAULT_START_DATE,
    doc_types: Sequence[str] | None = None,
    sleep_sec: float = 0.5,
    output_csv: str = DEFAULT_OUTPUT_CSV,
) -> pd.DataFrame:
    """
    UCSB 크롤러의 메인 실행 함수.

    전체 흐름:
    1. 문서 유형별 목록 페이지를 날짜 기준으로 순회해 링크를 모은다.
    2. 각 문서 상세 페이지에서 제목/본문을 추출한다.
    3. 키워드가 매치된 문서만 결과에 남긴다.
    4. 최종 결과를 CSV로 저장한다.

    max-pages 방식 대신 start_date를 쓰는 이유는,
    기간별 문서량 편차가 있어도 항상 같은 시간 구간을 기준으로 결과를 만들 수 있기 때문이다.
    """
    selected_doc_types = list(doc_types or DOC_TYPE_URLS.keys())
    invalid_doc_types = [doc_type for doc_type in selected_doc_types if doc_type not in DOC_TYPE_URLS]
    if invalid_doc_types:
        raise ValueError(f"Unsupported doc types: {invalid_doc_types}")

    parsed_start_date = parse_start_date(start_date)
    results: list[dict[str, Any]] = []

    for doc_type in selected_doc_types:
        listing_items = crawl_listing(
            base_url=DOC_TYPE_URLS[doc_type],
            doc_type=doc_type,
            start_date=parsed_start_date,
            sleep_sec=sleep_sec,
        )
        print(f"[SUMMARY] {doc_type}: {len(listing_items)} listing items")

        for item in listing_items:
            article = parse_article(item, keyword_dictionary=keyword_dictionary)
            if article is not None:
                results.append(article)
            time.sleep(sleep_sec)

    df = pd.DataFrame(results)

    if not df.empty:
        # 후속 처리에서 쓰기 쉬운 순서로 컬럼을 정리하고 최신 날짜가 위로 오게 정렬한다.
        df = df[
            [
                "published_date",
                "category",
                "doc_type",
                "title",
                "url",
                "matched_keyword_groups",
                "matched_keywords",
                "keyword_matches_json",
                "body",
            ]
        ].sort_values(by=["published_date", "doc_type"], ascending=[False, True])

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] saved {len(df)} rows to {output_csv}")
    return df


def parse_args() -> argparse.Namespace:
    """
    CLI 실행 시 사용할 인자를 정의한다.
    """
    parser = argparse.ArgumentParser(description="UCSB Presidency Project crawler")
    parser.add_argument(
        "--keyword-config",
        default=str(DEFAULT_KEYWORD_CONFIG_PATH),
        help="Path to a JSON file with {group_name: [keywords...]}",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Collect documents published on or after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--doc-types",
        nargs="+",
        default=list(DOC_TYPE_URLS.keys()),
        choices=list(DOC_TYPE_URLS.keys()),
        help="Subset of document types to crawl",
    )
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    """
    CLI 인자를 읽어 UCSB 크롤러를 실행한다.
    """
    args = parse_args()
    keyword_dictionary = load_keyword_dictionary(args.keyword_config)
    crawl_ucsb_documents(
        keyword_dictionary=keyword_dictionary,
        start_date=args.start_date,
        doc_types=args.doc_types,
        sleep_sec=args.sleep_sec,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
