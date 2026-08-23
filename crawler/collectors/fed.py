import time
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.support_legacy.data_paths import collected_csv_path

BASE_URL = "https://www.federalreserve.gov"
CALENDAR_URL ="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# 2017-2020 FOMC 캘린더 페이지 URL 맵핑 (과거 형식)
HISTORICAL_CALENDAR_URLS = {
    2017: f"{BASE_URL}/monetarypolicy/fomchistorical2017.htm",
    2018: f"{BASE_URL}/monetarypolicy/fomchistorical2018.htm",
    2019: f"{BASE_URL}/monetarypolicy/fomchistorical2019.htm",
    2020: f"{BASE_URL}/monetarypolicy/fomchistorical2020.htm",
}


def crawl_implementation_note(url: str) -> dict:
    """
    Implementation Note 상세 페이지에서
    공개일, 제목, 본문 텍스트를 추출한다.
    """
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # 페이지 상단에 노출된 공식 게시 날짜를 읽는다.
    date_tag = soup.find("p", class_="article__time")
    release_date = date_tag.get_text(" ", strip=True) if date_tag else ""

    # 제목은 보통 h3에 들어 있다.
    title_tag = soup.find("h3")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""

    # 같은 폭의 div가 여러 개 있을 수 있으므로,
    # heading 블록을 제외한 첫 번째 본문 컨테이너를 사용한다.
    divs = soup.find_all("div", class_="col-xs-12")

    article = None
    for div in divs:
        classes = div.get("class", [])
        if "heading" not in classes:
            article = div
            break

    contents = []

    if article:
        # 컨테이너 전체를 한 번에 텍스트로 뽑아 라인 단위로 처리한다.
        raw = article.get_text("\n", strip=True)
        for line in (ln.strip() for ln in raw.splitlines()):
            if not line:
                continue
            contents.append(line)

    body_text = " ".join(contents)

    return {
        "release_date": release_date,
        "title": title,
        "body": body_text
    }


def crawl_fomc_statement(url: str) -> dict:
    """
    FOMC statement 상세 페이지에서
    공개일, 제목, 본문 텍스트를 추출한다.
    """
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # 페이지 상단 게시 날짜
    date_tag = soup.find("p", class_="article__time")
    release_date = date_tag.get_text(" ", strip=True) if date_tag else ""

    # statement 제목
    title_tag = soup.find("h3")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""

    # statement 본문은 보통 col-sm-8 폭의 본문 영역에 들어 있다.
    divs = soup.find_all("div", class_="col-sm-8")

    article = None
    for div in divs:
        classes = div.get("class", [])
        if "heading" not in classes:
            article = div
            break

    contents = []
    stop_texts = ("for media inquiries", "implementation note issued")

    if article:
        # 본문 전체를 한 번에 가져와 라인 단위로 스톱 구간 전까지만 수집한다.
        raw = article.get_text("\n", strip=True)
        for line in (ln.strip() for ln in raw.splitlines()):
            if not line:
                continue

            lowered = line.lower()
            if lowered.startswith(stop_texts):
                break

            contents.append(line)

    body_text = " ".join(contents)

    return {
        "release_date": release_date,
        "title": title,
        "body": body_text
    }


def crawl_minutes(url: str) -> dict:
    """
    FOMC minutes 상세 페이지에서
    제목과 본문 텍스트를 추출한다.

    release_date는 상세 페이지가 아니라 캘린더 페이지의
    '(Released ...)' 문구에서 따로 채운다.
    """
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # minutes 전체 본문이 들어 있는 컨테이너
    article = soup.find("div", id="article")

    # 제목은 본문 컨테이너 안의 h3에서 읽는다.
    title_tag = article.find("h3") if article else None
    title = title_tag.get_text(" ", strip=True) if title_tag else ""

    contents = []

    if article:
        # 전체 텍스트를 라인 단위로 읽어 숫자 줄 제거, 'notation vote' 이전까지만 수집한다.
        raw = article.get_text("\n", strip=True)
        for line in (ln.strip() for ln in raw.splitlines()):
            if not line:
                continue

            contents.append(line)

    body_text = " ".join(contents)

    return {
        "release_date": None,
        "title": title,
        "body": body_text
    }


def crawl_latest_calendar(calendar_url: str = CALENDAR_URL) -> list:
    """최신 형식의 FOMC 캘린더를 크롤링해 레코드 리스트를 반환한다."""
    results = []
    try:
        response = requests.get(calendar_url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        sections = soup.find_all("div", class_="panel-default")

        for section in sections:
            heading = section.find("h4")
            if heading is None:
                continue

            heading_text = heading.get_text(" ", strip=True)
            match = re.match(r"(\d{4}) FOMC Meetings", heading_text)
            if not match:
                continue

            meetings = section.find_all("div", class_="fomc-meeting")

            for meeting in meetings:
                for link in meeting.find_all("a", href=True):
                    label = link.get_text(" ", strip=True).lower()
                    url = urljoin(BASE_URL, link["href"])

                    doc_type = None
                    article = None

                    if "implementation note" in label:
                        doc_type = "implementation_note"
                        try:
                            article = crawl_implementation_note(url)
                        except Exception:
                            continue

                    elif label == "html":
                        parent_strong = link.parent.find("strong") if link.parent else None
                        if parent_strong is None:
                            continue

                        parent_title = parent_strong.get_text(" ", strip=True).lower()

                        if "statement:" in parent_title:
                            doc_type = "statement"
                            try:
                                article = crawl_fomc_statement(url)
                            except Exception:
                                continue

                        elif "minutes:" in parent_title:
                            doc_type = "minutes"
                            try:
                                article = crawl_minutes(url)
                            except Exception:
                                continue

                            release_match = re.search(
                                r"Released ([A-Za-z]+ \d{1,2}, \d{4})",
                                link.parent.get_text(" ", strip=True)
                            )
                            release_date = release_match.group(1) if release_match else None
                            article["release_date"] = release_date

                    if doc_type and article:
                        results.append({
                            "release_date": article["release_date"],
                            "category": "FOMC",
                            "doc_type": doc_type,
                            "url": url,
                            "title": article["title"],
                            "body": article["body"]
                        })

                    time.sleep(0.5)

    except Exception as e:
        print(f"[WARN] Latest calendar crawl failed: {e}")

    return results


def crawl_historical_calendars(historical_urls: dict = HISTORICAL_CALENDAR_URLS) -> list:
    """과거(2017-2020) 형식의 FOMC 캘린더를 연도별로 크롤링해 레코드 리스트를 반환한다."""
    results = []

    for year, calendar_url in sorted(historical_urls.items()):
        print(f"\n[{year}] 크롤링 시작: {calendar_url}")
        try:
            response = requests.get(calendar_url, headers=HEADERS, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            meetings = soup.find_all("div", class_="panel-padded")
            year_count = 0

            for meeting in meetings:
                for link in meeting.find_all("a", href=True):
                    label = link.get_text(" ", strip=True).lower()
                    url = urljoin(BASE_URL, link["href"])

                    doc_type = None
                    article = None

                    if label == "statement":
                        doc_type = "statement"
                        try:
                            article = crawl_fomc_statement(url)
                        except Exception as e:
                            print(f"  [WARN] Statement 파싱 실패 ({url}): {e}")
                            continue

                    elif label == "html":
                        parent_title = link.parent.get_text(" ", strip=True) if link.parent else None
                        if not parent_title or not parent_title.startswith("Minutes"):
                            continue

                        doc_type = "minutes"
                        try:
                            article = crawl_minutes(url)
                            release_match = re.search(
                                r"Released ([A-Za-z]+ \d{1,2}, \d{4})",
                                link.parent.get_text(" ", strip=True)
                            )
                            release_date = release_match.group(1) if release_match else None
                            article["release_date"] = release_date
                        except Exception as e:
                            print(f"  [WARN] Minutes 파싱 실패 ({url}): {e}")
                            continue

                    if doc_type and article:
                        results.append({
                            "release_date": article["release_date"],
                            "category": "FOMC",
                            "doc_type": doc_type,
                            "url": url,
                            "title": article["title"],
                            "body": article["body"]
                        })
                        year_count += 1

                    time.sleep(0.5)

            print(f"[{year}] 크롤링 완료: {year_count}건 수집")

        except Exception as e:
            print(f"[{year}] 크롤링 실패: {e}")
            continue

    return results


def main() -> None:
    # 분리된 크롤러 함수를 호출해 결과를 수집
    results_latest = crawl_latest_calendar()
    results_hist = crawl_historical_calendars()

    results = []
    if results_latest:
        results.extend(results_latest)
    if results_hist:
        results.extend(results_hist)

    # 결과를 DataFrame으로 정리한다.
    new_df = pd.DataFrame(results)
    print(f"\n새로 수집된 데이터: {len(new_df)}건")

    # 기존 fed_fomc_links.csv가 있으면 읽어서 병합한다.
    existing_csv = collected_csv_path("fed_fomc_links.csv")
    existing_path = Path(existing_csv)

    if existing_path.exists():
        existing_df = pd.read_csv(existing_csv, encoding="utf-8-sig")
        print(f"기존 데이터: {len(existing_df)}건")
        merged_df = pd.concat([existing_df, new_df], ignore_index=True)
        print(f"병합 후 (중복 제거 전): {len(merged_df)}건")
        merged_df = merged_df.drop_duplicates(subset=["url"]).reset_index(drop=True)
        print(f"중복 제거 후: {len(merged_df)}건")
    else:
        print("기존 데이터 없음. 새 데이터만 저장합니다.")
        merged_df = new_df.drop_duplicates().reset_index(drop=True)

    print(f"\n문서 유형 분포:\n{merged_df['doc_type'].value_counts()}")
    print(f"\n샘플:\n{merged_df.head(10)}")

    # 최신순으로 정렬 (release_date 기준, 최신이 위)
    merged_df['_sort_date'] = pd.to_datetime(merged_df['release_date'], errors='coerce')
    merged_df = merged_df.sort_values(by='_sort_date', ascending=False, na_position='last').reset_index(drop=True)
    merged_df = merged_df.drop(columns=['_sort_date'])

    # 최종 결과를 fed_fomc_links.csv에 저장한다.
    output_csv = collected_csv_path("fed_fomc_links.csv")
    merged_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {output_csv}")


if __name__ == "__main__":
    main()
