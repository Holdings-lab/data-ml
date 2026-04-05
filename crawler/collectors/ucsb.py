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
ITEMS_PER_PAGE = 20

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}

# UCSB에서 현재 수집 대상으로 삼는 문서 카테고리의 목록 페이지 URL
DOC_TYPE_URLS = {
    "Executive Orders": (
        f"{BASE_URL}/documents/app-categories/"
        "written-presidential-orders/presidential/executive-orders"
    ),
    "Press Conferences": f"{BASE_URL}/documents/app-categories/presidential/news-conferences",
    "Fact Sheets": f"{BASE_URL}/documents/app-attributes/fact-sheets",
}

# 본문 뒤쪽의 메타데이터/탐색 영역이 시작되는 제목들.
# 이런 구간이 나오면 기사 본문 수집을 멈춘다.
STOP_SECTION_TITLES = {
    "filed under",
    "categories",
    "simple search of our archives",
}

# UCSB 목록 페이지의 날짜 헤더 예: "April 5, 2026"
DATE_PATTERN = re.compile(r"^[A-Z][a-z]+ \d{1,2}, \d{4}$")


def clean_text(text: str) -> str:
    """
    연속 공백과 줄바꿈을 하나의 공백으로 정리한 뒤 양끝 공백을 제거한다.

    UCSB 페이지는 줄바꿈과 공백이 섞여 있는 경우가 많아서,
    대부분의 텍스트 비교와 저장 전에 이 정규화를 거친다.
    """
    return re.sub(r"\s+", " ", text).strip()


def parse_published_date(raw_value: str) -> str:
    """
    UCSB 목록 페이지의 날짜 문자열을 YYYY-MM-DD 형식으로 변환한다.
    """
    parsed = datetime.strptime(raw_value, "%B %d, %Y")
    return parsed.strftime("%Y-%m-%d")


def normalize_keyword_dictionary(
    keyword_dictionary: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """
    키워드 JSON을 내부에서 쓰기 쉬운 형태로 정리한다.

    - 그룹명과 키워드 문자열의 공백을 정리한다.
    - 빈 문자열은 제거한다.
    - 중복 키워드는 제거하고 소문자 기준으로 정렬한다.
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
    """
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def build_listing_url(base_url: str, page_number: int) -> str:
    """
    UCSB 카테고리 목록 URL에 페이지 번호와 페이지 크기 파라미터를 붙인다.

    이 사이트는 첫 페이지를 `page=1` 없이도 제공하므로,
    1페이지는 `items_per_page`만 붙이고 이후 페이지부터 `page=N`을 사용한다.
    """
    if page_number <= 1:
        return f"{base_url}?items_per_page={ITEMS_PER_PAGE}"
    return f"{base_url}?page={page_number}&items_per_page={ITEMS_PER_PAGE}"


def parse_listing_page(soup: BeautifulSoup, doc_type: str) -> list[dict[str, str]]:
    """
    UCSB 목록 페이지에서 문서 메타데이터를 추출한다.

    이 함수는 다음 정보를 만든다.
    - title
    - url
    - published_date
    - doc_type
    - category

    목록 페이지는 날짜 헤더(h4)와 문서 링크(a)가 섞여 있으므로,
    가장 최근에 본 날짜 헤더를 현재 링크의 게시일로 사용한다.
    """
    items: list[dict[str, str]] = []
    current_date = ""

    # 실제 목록 컨테이너는 view-content 안에 들어 있다.
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
        if not href.startswith("/documents/"):
            continue

        # 페이지네이션 UI 텍스트는 문자 인코딩에 따라 깨질 수 있으므로,
        # 완전일치 대신 대표 접두어 기준으로 제외한다.
        lowered_title = title.lower()
        if lowered_title.startswith("next") or lowered_title.startswith("last"):
            continue

        # 날짜 헤더를 만나기 전에 등장한 링크는 게시일을 알 수 없으므로 건너뛴다.
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

    # 목록 페이지 안에 같은 문서가 여러 번 보일 수 있어 URL 기준으로 중복 제거한다.
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
    max_pages: int,
    sleep_sec: float,
) -> list[dict[str, str]]:
    """
    한 문서 유형에 대해 여러 목록 페이지를 순회하며 문서 메타데이터를 수집한다.

    - 페이지마다 parse_listing_page()를 호출한다.
    - 중복 URL은 제거한다.
    - 어떤 페이지에서 결과가 더 이상 나오지 않으면 거기서 중단한다.
    """
    items: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for page_number in range(1, max_pages + 1):
        page_url = build_listing_url(base_url, page_number)
        print(f"[INDEX] {doc_type}: {page_url}")

        try:
            soup = fetch_page(page_url)
        except Exception as exc:
            print(f"  -> failed to fetch listing page: {exc}")
            continue

        page_items = parse_listing_page(soup, doc_type)
        if not page_items:
            print("  -> no listing items found, stopping")
            break

        added_count = 0
        for item in page_items:
            if item["url"] in seen_urls:
                continue
            seen_urls.add(item["url"])
            items.append(item)
            added_count += 1

        print(f"  -> found {added_count} new items")
        time.sleep(sleep_sec)

    return items


def extract_article_body(soup: BeautifulSoup) -> str:
    """
    문서 상세 페이지에서 실제 본문 텍스트만 추출한다.

    UCSB 문서 본문은 `div.field-docs-content` 안에 들어 있는 경우가 많다.
    여기서 문단/목록/인용문/소제목만 모으고,
    메타데이터 섹션으로 넘어가는 제목이 나오면 수집을 멈춘다.
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

        # 날짜만 있는 줄은 본문으로 쓰지 않는다.
        if DATE_PATTERN.match(text):
            continue

        if node.name == "li":
            text = f"- {text}"

        body_parts.append(text)

    # 동일한 텍스트 조각이 반복될 수 있어 한 번 더 중복 제거한다.
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

    반환값은 다음 형태다.
    {
        "group_name": ["matched keyword 1", "matched keyword 2", ...]
    }
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
    doc_types: Sequence[str] | None = None,
    max_pages: int = 5,
    sleep_sec: float = 0.5,
    output_csv: str = DEFAULT_OUTPUT_CSV,
) -> pd.DataFrame:
    """
    UCSB 크롤러의 메인 실행 함수.

    흐름은 다음과 같다.
    1. 문서 유형별 목록 페이지를 순회해 링크를 모은다.
    2. 각 문서 상세 페이지에서 제목/본문을 추출한다.
    3. 키워드가 매치된 문서만 결과에 남긴다.
    4. 최종 결과를 CSV로 저장한다.
    """
    selected_doc_types = list(doc_types or DOC_TYPE_URLS.keys())
    invalid_doc_types = [doc_type for doc_type in selected_doc_types if doc_type not in DOC_TYPE_URLS]
    if invalid_doc_types:
        raise ValueError(f"Unsupported doc types: {invalid_doc_types}")

    results: list[dict[str, Any]] = []

    for doc_type in selected_doc_types:
        listing_items = crawl_listing(
            base_url=DOC_TYPE_URLS[doc_type],
            doc_type=doc_type,
            max_pages=max_pages,
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
        # 후속 처리에서 쓰기 쉬운 순서로 컬럼을 정리하고 날짜 순으로 정렬한다.
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
        ].sort_values(by=["published_date", "doc_type"], ascending=[True, True])

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
        "--doc-types",
        nargs="+",
        default=list(DOC_TYPE_URLS.keys()),
        choices=list(DOC_TYPE_URLS.keys()),
        help="Subset of document types to crawl",
    )
    parser.add_argument("--max-pages", type=int, default=5)
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
        doc_types=args.doc_types,
        max_pages=args.max_pages,
        sleep_sec=args.sleep_sec,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
