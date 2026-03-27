import time
import re
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.federalreserve.gov"
CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# FOMC 캘린더에서 회의 월로 사용되는 값들
MONTHS = {
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Apr/May"
}


def crawl_implementation_note(url: str) -> dict:
    """
    FOMC Implementation Note 페이지에서
    날짜, 제목, 본문 텍스트를 추출한다.
    """
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # 페이지 상단에 표시된 공식 게시 날짜
    date_tag = soup.find("p", class_="article__time")
    release_date = date_tag.get_text(" ", strip=True) if date_tag else ""

    # 페이지 제목
    title_tag = soup.find("h3")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""

    # Implementation Note 본문이 들어 있는 div 탐색
    # col-xs-12 클래스를 가진 div들 중에서 heading 영역은 제외하고
    # 실제 본문이 들어 있는 첫 번째 div를 article로 사용
    divs = soup.find_all("div", class_="col-xs-12")

    article = None
    for div in divs:
        classes = div.get("class", [])
        if "heading" not in classes:
            article = div
            break

    contents = []

    if article:
        # 본문 안에서 문단(p), 목록(li), 인용(blockquote)만 수집
        for tag in article.find_all(["p", "li", "blockquote"]):
            text = tag.get_text(" ", strip=True)
            if not text:
                continue

            # 목록 항목은 나중에 구분하기 쉽도록 앞에 '-' 추가
            if tag.name == "li":
                text = f"- {text}"

            contents.append(text)

    # 줄바꿈 기준으로 하나의 긴 텍스트로 합침
    body_text = "\n".join(contents)

    return {
        "release_date": release_date,
        "release_time": None,
        "title": title,
        "body": body_text
    }


def crawl_fomc_statement(url: str) -> dict:
    """
    FOMC Statement 페이지에서
    날짜, 제목, 배포시각, 본문 텍스트를 추출한다.
    """
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # 페이지 상단 날짜
    date_tag = soup.find("p", class_="article__time")
    release_date = date_tag.get_text(" ", strip=True) if date_tag else ""

    # Statement 제목
    title_tag = soup.find("h3")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""

    # 제목 바로 아래 p 태그에 배포 시각이 있는 경우가 많음
    # 예: "For release at 2:00 p.m. EDT"
    release_time = ""
    if title_tag:
        next_p = title_tag.find_next("p")
        if next_p:
            release_time = next_p.get_text(" ", strip=True)

    paragraphs = []

    if title_tag:
        # 제목 이후에 나오는 태그들을 순서대로 탐색
        for tag in title_tag.find_all_next():

            # 다음 큰 섹션이 시작되면 본문 수집 종료
            if tag.name in ["hr", "h3", "h4"]:
                break

            # Statement 본문은 주로 p 태그에 들어 있으므로 p만 수집
            if tag.name != "p":
                continue

            text = tag.get_text(" ", strip=True)

            # 빈 문단은 제외
            if not text:
                continue

            # 배포 시각 문장은 본문이 아니므로 제외
            if text == release_time:
                continue

            lowered = text.lower()

            # 하단 연락처나 관련 링크 영역이 시작되면 종료
            if lowered.startswith("for media inquiries"):
                break
            if lowered.startswith("implementation note issued"):
                break

            paragraphs.append(text)

    body_text = "\n".join(paragraphs)

    return {
        "release_date": release_date,
        "release_time": release_time,
        "title": title,
        "body": body_text
    }


def crawl_minutes(url: str) -> dict:
    """
    FOMC Minutes 페이지에서
    제목과 본문 텍스트를 추출한다.
    release_date는 캘린더 페이지의 '(Released ...)' 문구에서 별도로 추출한다.
    """
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Minutes 전체 본문 영역
    article = soup.find("div", id="article")

    # Minutes 제목
    title_tag = soup.find("h3")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""

    contents = []

    if article:
        # 본문 안의 문단, 목록, 인용문 수집
        for tag in article.find_all(["p", "li", "blockquote"]):
            text = tag.get_text(" ", strip=True)

            if not text:
                continue

            # 숫자만 있는 경우는 각주 번호일 가능성이 높으므로 제거
            if text.isdigit():
                continue

            # 목록 항목 구분용 기호 추가
            if tag.name == "li":
                text = f"- {text}"

            contents.append(text)

    body_text = "\n".join(contents)

    return {
        "release_date": None,
        "release_time": None,
        "title": title,
        "body": body_text
    }


def main() -> None:
    # FOMC 캘린더 페이지 요청
    response = requests.get(CALENDAR_URL, headers=HEADERS, timeout=20)
    response.raise_for_status()

    # 캘린더 페이지 HTML 파싱
    soup = BeautifulSoup(response.text, "html.parser")

    results = []

    # 연도별 FOMC 섹션 탐색
    # 예: "2025 FOMC Meetings", "2024 FOMC Meetings"
    year_sections = soup.find_all("div", attrs={"class": "panel-heading"})

    for section in year_sections:
        heading = section.find("h4")
        if heading is None:
            continue

        # 연도 헤더에서 실제 FOMC 회의 연도인지 확인
        heading_text = heading.get_text(" ", strip=True)
        match = re.match(r"(\d{4}) FOMC Meetings", heading_text)
        if not match:
            continue

        # 현재 연도 섹션 아래의 형제 노드들을 순서대로 확인
        node = section.find_next_sibling()

        while node:

            # strong 태그가 있는 노드만 회의 정보 블록으로 간주
            if node.find("strong"):

                # 현재 노드의 첫 부분에서 월 이름 추출
                # 현재 사이트 구조상 node.contents[1]에 월 텍스트가 들어 있음
                text = node.contents[1].get_text(" ", strip=True)

                # 회의가 SEP인지 여부(월 매칭이 실패해도 기본값은 안전하게 유지)
                is_sep = False
                if text in MONTHS:

                    # 현재 회의 날짜 범위 추출
                    # 예: "27-28", "17-18*"
                    date_node = node.contents[3]
                    meeting_period = date_node.get_text(" ", strip=True)

                    # 별표(*)가 붙은 회의는 SEP 회의로 처리
                    is_sep = "*" in meeting_period

                # 현재 회의 블록 안의 모든 링크 순회
                for link in node.find_all("a", href=True):

                    label = link.get_text(" ", strip=True).lower()
                    url = urljoin(BASE_URL, link["href"])

                    doc_type = None
                    article = None

                    # Implementation Note 링크인 경우
                    if "implementation note" in label:
                        doc_type = "implementation_note"
                        article = crawl_implementation_note(url)

                    # HTML 링크인 경우 부모 strong 텍스트를 보고
                    # Statement인지 Minutes인지 구분
                    elif label == "html":

                        parent_title = link.parent.strong.get_text(" ", strip=True).lower()

                        if "statement:" in parent_title:
                            doc_type = "statement"
                            article = crawl_fomc_statement(url)

                        elif "minutes:" in parent_title:
                            doc_type = "minutes"
                            article = crawl_minutes(url)

                            # Minutes release 날짜는 개별 minutes 페이지가 아니라
                            # 캘린더 페이지의 '(Released ...)' 문구에서 추출
                            release_match = re.search(
                                r"Released ([A-Za-z]+ \d{1,2}, \d{4})",
                                node.get_text(" ", strip=True)
                            )

                            release_date = release_match.group(1) if release_match else None
                            article["release_date"] = release_date

                    # 정상적으로 문서 정보를 추출한 경우 결과 저장
                    if doc_type and article:
                        results.append({
                            "release_date": article["release_date"],
                            "release_time": article["release_time"],
                            "is_sep": is_sep,
                            "category": "FOMC",
                            "doc_type": doc_type,
                            "url": url,
                            "title": article["title"],
                            "body": article["body"]
                        })

            # 다음 회의 블록으로 이동
            time.sleep(0.5)
            node = node.find_next_sibling()

    # 결과를 데이터프레임으로 변환하고 중복 제거
    df = pd.DataFrame(results).drop_duplicates()

    print(df.head(20))

    # CSV 파일로 저장
    df.to_csv("fed_fomc_links.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()