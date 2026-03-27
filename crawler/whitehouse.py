import time
import re
from typing import List, Dict, Optional
from urllib.parse import urljoin
from datetime import datetime

import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://www.whitehouse.gov"
NEWS_URL = f"{BASE_URL}/news/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

# QQQ에 영향 줄 가능성이 큰 정책 키워드
QQQ_KEYWORDS = [
    "artificial intelligence", "ai", "semiconductor", "chip", "chips",
    "nvidia", "amd", "intel", "data center", "cloud", "cybersecurity",
    "antitrust", "competition", "big tech", "technology", "export control",
    "tariff", "trade", "china", "advanced computing", "software",
    "digital", "broadband", "quantum", "5g", "6g"
]

# 카테고리 추출용 후보
DOC_TYPE_CANDIDATES = {
    "Articles",
    "Briefings & Statements",
    "Fact Sheets",
    "Executive Orders",
    "Presidential Memoranda",
    "Proclamations",
    "Remarks",
    "Research",
    "Presidential Actions",
}


def clean_text(text: str) -> str:
    """공백 정리"""
    return re.sub(r"\s+", " ", text).strip()


def contains_qqq_keyword(text: str, keywords: List[str]) -> bool:
    """본문/제목에 QQQ 관련 키워드가 있는지 확인"""
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


def find_news_cards(soup: BeautifulSoup) -> List[BeautifulSoup]:
    """
    /news/ 페이지에서 개별 글 카드(목록 항목)를 찾는다.
    현재 화이트하우스 뉴스 페이지는 <h2> 안의 링크 형태가 많으므로,
    제목 링크를 기준으로 역추적한다.
    """
    cards = []

    # 일반적으로 목록 제목은 h2 / h3 / h4 내부 a 태그
    for tag in soup.find_all(["h2"]):
        a = tag.find("a", href=True)
        if not a:
            continue

        href = a["href"]

        if "whitehouse.gov" not in href:
            continue

        # 뉴스/발표성 문서만 남김
        if not any(part in href for part in ["/articles/", "/briefings-statements/", "/presidential-actions/"]):
            continue

        cards.append(tag)

    return cards


def parse_listing_item(tag: BeautifulSoup) -> Optional[Dict]:
    """
    목록 페이지의 카드에서
    title / url / doc_type / published_date 를 최대한 추출
    """
    a = tag.find("a", href=True)
    if not a:
        return None

    title = clean_text(a.get_text(" ", strip=True))
    url = a["href"]

    parent = tag.parent

    date_tag = parent.find("time")
    date = clean_text(date_tag.get_text(" ", strip=True)) if date_tag else ""

    type_tag = tag.parent.find("a", rel="tag")
    type = clean_text(type_tag.get_text(" ", strip=True)) if type_tag else ""

    # 날짜 추정
    date_match = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
        date
    )
    published_date = date_match.group(0) if date_match else None
    print
    try:
        dt_obj = datetime.strptime(published_date, "%B %d, %Y")
        published_date = dt_obj.strftime("%Y-%m-%d")
    except ValueError as e:
        print(f"날짜 변환 실패: {published_date} -> {e}")
        published_date = None

    # 카테고리 추정
    doc_type = None
    for candidate in DOC_TYPE_CANDIDATES:
        if candidate.lower() in type.lower():
            doc_type = candidate
            break

    return {
        "title": title,
        "url": url,
        "category": "White House",
        "doc_type": doc_type,
        "published_date": published_date,
    }


def fetch_page(url: str) -> BeautifulSoup:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def crawl_news_index(max_pages: int = 5, sleep_sec: float = 1.0) -> List[Dict]:
    """
    /news/ 와 /news/page/{n}/ 를 순회하면서
    글 목록 URL을 수집
    """
    items = []
    seen_urls = set()

    for page_num in range(1, max_pages + 1):
        if page_num == 1:
            url = NEWS_URL
        else:
            url = f"{NEWS_URL}page/{page_num}/"

        print(f"[INDEX] {url}")

        try:
            soup = fetch_page(url)
        except Exception as e:
            print(f"  -> 페이지 수집 실패: {e}")
            continue

        cards = find_news_cards(soup)
        print(f"  -> 발견 카드 수: {len(cards)}")

        for card in cards:
            item = parse_listing_item(card)
            if not item:
                continue
            if item["url"] in seen_urls:
                continue

            seen_urls.add(item["url"])
            items.append(item)

        time.sleep(sleep_sec)

    return items


def extract_article_body(soup: BeautifulSoup) -> str:
    """
    본문 추출
    - 기사형 페이지는 h1/title/date 아래에 p 태그가 이어지는 경우가 많음
    - footer, related, subscribe 영역 제외
    """
    # 너무 광범위하게 잡히는 걸 막기 위해 긴 p 태그만 우선 수집
    paragraphs = []

    for p in soup.find_all("p"):
        text = clean_text(p.get_text(" ", strip=True))

        if len(text) < 20:
            continue

        # 하단 공유/구독/저작권성 문구 제거
        lowered = text.lower()
        if any(bad in lowered for bad in [
            "subscribe", "click here", "follow on social media",
            "notifications", "privacy policy"
        ]):
            continue

        paragraphs.append(text)

    # 중복 제거
    deduped = []
    seen = set()
    for p in paragraphs:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    return "\n".join(deduped)


def parse_article(metadata: Optional[Dict]) -> Optional[Dict]:
    new_data = metadata.copy()
    
    url = metadata["url"]
    print(f"[ARTICLE] {url}")

    try:
        soup = fetch_page(url)
    except Exception as e:
        print(f"  -> 본문 수집 실패: {e}")
        return None
    
    body = extract_article_body(soup)
    new_data["body"] = body

    return new_data


def crawl_whitehouse_qqq_policy(
    max_pages: int = 10,
    sleep_sec: float = 1.0,
    output_csv: str = "whitehouse_qqq_policy.csv"
) -> pd.DataFrame:
    """
    1) 뉴스 인덱스 수집
    2) 개별 문서 본문 수집
    3) QQQ 관련 키워드 필터링
    4) CSV 저장
    """
    listing_items = crawl_news_index(max_pages=max_pages, sleep_sec=sleep_sec)

    print(f"\n총 목록 수집 개수: {len(listing_items)}")

    results = []

    for item in listing_items:
        article = parse_article(item)

        if not article:
            time.sleep(sleep_sec)
            continue

        combined_text = f"{article['title'] or ''}\n{article['body'] or ''}"

        if contains_qqq_keyword(combined_text, QQQ_KEYWORDS):
            matched_keywords = [
                kw for kw in QQQ_KEYWORDS
                if kw.lower() in combined_text.lower()
            ]

            article["matched_keywords"] = ", ".join(sorted(set(matched_keywords)))
            results.append(article)
            print(f"  -> QQQ 관련 문서 저장: {article['title']}")
        else:
            print("  -> 비관련 문서")

        time.sleep(sleep_sec)

    df = pd.DataFrame(results)

    if not df.empty:
        # 본문이 너무 길면 필요에 따라 일부 컬럼만 저장 가능
        df = df[[
            "published_date", "category", "doc_type", "title", "url",
            "matched_keywords", "body"
        ]].sort_values(by="published_date", ascending=False, na_position="last")

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {output_csv} / 건수: {len(df)}")

    return df


if __name__ == "__main__":
    df = crawl_whitehouse_qqq_policy(
        max_pages=160,       # 처음엔 3~5페이지 정도로 테스트
        sleep_sec=1.2,
        output_csv="whitehouse_qqq_policy.csv"
    )

    print(df.head(10))