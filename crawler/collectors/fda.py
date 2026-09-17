import time
import sys
import csv
import random
from pathlib import Path
from datetime import datetime, timedelta, timezone
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from curl_cffi import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.support_legacy.data_paths import collected_csv_path


BASE_URL = "https://www.fda.gov"
FDA_URL = "https://www.fda.gov/news-events/fda-newsroom/press-announcements"
FDA_URL_22 = "https://web.archive.org/web/20211007195908/https://www.fda.gov/news-events/fda-newsroom/press-announcements"
FDA_URL_18 = "https://wayback.archive-it.org/7993/20201229084636/https://www.fda.gov/news-events/fda-newsroom/press-announcements?ts=1606313529"
FDA_URL_20 = "https://web.archive.org/web/20210104225157/https://www.fda.gov/news-events/fda-newsroom/press-announcements"

HEADERS = {
    # 브라우저 식별 및 클라이언트 힌트
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Sec-Ch-Ua": '"Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    
    # 콘텐츠 협상 (압축 알고리즘 추가)
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
    
    # 보안 및 이동 컨텍스트 (목록 페이지를 넘길 때는 same-origin / same-site가 자연스러움)
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",  # 처음 진입이 아니면 same-origin이 차단율을 대폭 낮춤
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

# 세션 초기화 (Chrome 120 핑거프린트 모방)
session = requests.Session(impersonate="chrome120")

def _save_results(records, target_date):
    csv_path = Path(collected_csv_path(f"fda_{target_date}.csv"))
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with csv_path.open(
        "a" if file_exists else "w",
        encoding="utf-8" if file_exists else "utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=["release_date", "category", "doc_type", "url", "title", "body"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(records)

    return csv_path

def get_detail_content(url):
    """세부 뉴스 페이지에 접속하여 발행일, 제목, 본문 전체를 파싱합니다."""
    try:
        response = session.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # 1. 제목 (Title)
        title_tag = soup.select_one("h1.content-title") or soup.select_one("h1")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # 2. 발표일 (Release Date)
        date_tag = soup.select_one("time") or soup.find("dd", class_="cell-2_1")
        release_date = date_tag.get("datetime") or date_tag.get_text(strip=True)

        if release_date and not release_date.startswith("20"):
            release_date = datetime.strptime(release_date, "%B %d, %Y").replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


        # 3. 본문 (Body) - CSS Selector 문법 수정: div[role="main"] 또는 main
        body_container = soup.select_one('div[role="main"]') or soup.select_one("main") or soup.select_one("article")
        body_list = body_container.select("p, li") if body_container else []

        result_text_list = []
        for p in body_list:
            body_text = p.get_text(strip=True)
            if body_text.startswith("Media:") or body_text.startswith("###"):
                break  # 미디어 연락처나 문서 종료 표기 시 수집 중단
            if body_text:
                result_text_list.append(body_text)

        return {
            "release_date": release_date,
            "category": "FDA",
            "doc_type": "press-announcements",
            "url": url,
            "title": title,
            "body": " ".join(result_text_list).strip(),
        }

    except Exception as e:
        print(f"[Error fetching detail] {url}: {e}")
        return None


def crawl_fda_press_releases_24(target_date=None):
    """FDA Press Announcements 목록을 순회하며 target_date 이후의 데이터를 수집합니다."""
    page = 0
    results = []
    seen_urls = set()
    stop_crawling = False

    while not stop_crawling:
        # FDA Drupal 페이징 규칙: 첫 페이지 page=0 또는 파라미터 없음
        target_url = FDA_URL if page == 0 else f"{FDA_URL}?page={page}"
        print(f"\n[Scraping list page {page}] {target_url}")

        try:
            res = session.get(target_url, headers=HEADERS, timeout=15)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")

            # 기사 링크 태그 추출
            article_list = soup.find("div", class_="view-content")
            articles = article_list.find_all("a") if article_list else []

            if not articles:
                print("[Info] 더 이상 수집할 기사가 없습니다. 크롤링을 종료합니다.")
                break

            for a_tag in articles:
                href = a_tag.get("href")
                if not href or "news-events/press-announcements/" not in href:
                    continue

                # 날짜 태그 검증 및 datetime 변환
                time_tag = a_tag.find("time")
                release_date_str = time_tag.get("datetime") if time_tag else None

                if target_date and release_date_str:
                    try:
                        # "2026-05-01" 또는 ISO 형식 파싱
                        item_dt = datetime.fromisoformat(release_date_str.split("T")[0])
                        if item_dt < target_date:
                            print(f"[Info] 타겟 날짜 이전 기사에 도달함 ({item_dt.date()} < {target_date.date()}). 크롤링 종료.")
                            stop_crawling = True
                            break
                    except ValueError:
                        pass

                full_url = urljoin(BASE_URL, href)

                # 중복 수집 및 목록 링크 재진입 방지
                if full_url in seen_urls or full_url.rstrip("/") == FDA_URL.rstrip("/"):
                    continue
                seen_urls.add(full_url)

                print(f"  -> Extracting: {full_url}")
                data = get_detail_content(full_url)
                if data:
                    results.append(data)

                time.sleep(3)  # 서버 부하 방지용 딜레이

            # 다음 페이지로 증가 (루프 1회당 1페이지씩 증가)
            page += 1

            if page == 36:
                print("[Info] 2024년 이후 페이지 수집 완료. 크롤링 종료.")
                break

        except Exception as e:
            print(f"[Error fetching list page] {target_url}: {e}")
            break

    return results


def crawl_fda_press_releases_21(target_date=None, fda_url = FDA_URL_20):
    """FDA Press Announcements 목록을 순회하며 target_date 이후의 데이터를 수집합니다."""
    page = 0
    results = []
    seen_urls = set()
    stop_crawling = False

    while not stop_crawling:
        target_url = fda_url if page == 0 else f"{fda_url}?page={page}"
        print(f"\n[Scraping list page {page}] {target_url}")

        try:
            res = session.get(target_url, headers=HEADERS, timeout=30)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")

            # 기사 링크 태그 추출
            article_list = soup.find("div", class_="view-content")
            articles = article_list.find_all("a") if article_list else []

            if not articles:
                print("[Info] 더 이상 수집할 기사가 없습니다. 크롤링을 종료합니다.")
                break

            for a_tag in articles:
                href = a_tag.get("href")
                if not href:
                    continue

                # 날짜 태그 검증 및 datetime 변환
                time_tag = a_tag.find("time")
                release_date_str = time_tag.get_text().strip() if time_tag else None

                if target_date and release_date_str:
                    try:
                        # "2026-05-01" 또는 ISO 형식 파싱
                        item_dt = datetime.strptime(release_date_str, "%B %d, %Y")
                        if item_dt < target_date:
                            print(f"[Info] 타겟 날짜 이전 기사에 도달함 ({item_dt.date()} < {target_date.date()}). 크롤링 종료.")
                            stop_crawling = True
                            break
                    except ValueError:
                        pass

                if href.startswith("/web/"):
                    full_url = "https://web.archive.org" + href
                elif href.startswith("/news-events/press-announcements"):
                    full_url = "https://web.archive.org" + href
                else:
                    print(f"[Warning] 예상치 못한 href 형식: {href}")
                    continue

                # 중복 수집 및 목록 링크 재진입 방지
                if full_url in seen_urls:
                    continue
                seen_urls.add(full_url)

                print(f"  -> Extracting: {full_url}")
                data = get_detail_content(full_url)
                if data:
                    results.append(data)
            
                time.sleep(random.uniform(3, 7))  # 서버 부하 방지용 딜레이
    
            # 다음 페이지로 증가 (루프 1회당 1페이지씩 증가)
            page += 1

            if page == 1:
                print("[Info] 2021년 이후 페이지 수집 완료. 크롤링 종료.")
                break

            time.sleep(random.uniform(3, 7))  # 서버 부하 방지용 딜레이

        except Exception as e:
            print(f"[Error fetching list page] {target_url}: {e}")
            break

    return results


def crawl_fda_press_releases_18(target_date=None):
    """FDA Press Announcements 목록을 순회하며 target_date 이후의 데이터를 수집합니다."""
    seen_urls = set()
    stop_crawling = False

    for page in range(71, 100):  # 60부터 99까지 페이지 순회
        results = []
        target_url = f"{FDA_URL_18}&page={page}"
        print(f"\n[Scraping list page {page}] {target_url}")

        try:
            res = session.get(target_url, headers=HEADERS, timeout=30)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")

            # 기사 링크 태그 추출
            article_list = soup.find("div", class_="view-content")
            articles = article_list.find_all("a") if article_list else []
            print(articles)
            if not articles:
                print("[Info] 더 이상 수집할 기사가 없습니다. 크롤링을 종료합니다.")
                break

            for a_tag in articles:
                time.sleep(random.uniform(10, 20))  # 서버 부하 방지용 딜레이

                href = a_tag.get("href")
                if not href:
                    continue

                # 날짜 태그 검증 및 datetime 변환
                time_tag = a_tag.find("time")
                release_date_str = time_tag.get("datetime") if time_tag else None
                
                if target_date and release_date_str:
                    try:
                        # "2026-05-01" 또는 ISO 형식 파싱
                        item_dt = datetime.fromisoformat(release_date_str.split("T")[0])
                        if item_dt < target_date:
                            print(f"[Info] 타겟 날짜 이전 기사에 도달함 ({item_dt.date()} < {target_date.date()}). 크롤링 종료.")
                            stop_crawling = True
                            break
                    except ValueError:
                        pass

                if stop_crawling:
                    break
                
                if href.startswith("/7993/"):
                    full_url = "https://wayback.archive-it.org" + href
                else:
                    print(f"[Warning] 예상치 못한 href 형식: {href}")
                    continue

                # 중복 수집 및 목록 링크 재진입 방지
                if full_url in seen_urls:
                    continue
                seen_urls.add(full_url)

                print(f"  -> Extracting: {full_url}")
                data = get_detail_content(full_url)
                if data:
                    results.append(data)

            _save_results(results, target_date=target_date.date())
            if page == 100:
                print("[Info] 2018년 이후 페이지 수집 완료. 크롤링 종료.")
                break

            time.sleep(random.uniform(60, 90))  # 서버 부하 방지용 딜레이
        except Exception as e:
            print(f"[Error fetching list page] {target_url}: {e}")
            break

    return results


def scrape_news_sync(target_date):
    clean_dataset = []

    with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 800}
            )
            
            list_page = context.new_page()
            body_page = context.new_page()
            

            for i in range(100):
                print(f"\n[+] [{i}] FDA Press Announcements news-list loading...")
                url = f"{FDA_URL_18}?page={i}"
                list_page.goto(url, wait_until="domcontentloaded")
                list_page.wait_for_timeout(2000)

                # 스크롤 완료 후 전체 DOM 구조 확보
                html_content = list_page.content()
                soup = BeautifulSoup(html_content, "html.parser")

                article_list = soup.find("div", class_="view-content")
                articles = article_list.find_all("a") if article_list else []

                if not articles:
                    print("[Info] 더 이상 수집할 기사가 없습니다. 크롤링을 종료합니다.")
                    break

                for a_tag in articles:
                    href = a_tag.get("href")
                    if not href:
                        continue

                    # 날짜 태그 검증 및 datetime 변환
                    time_tag = a_tag.find("time")
                    release_date_str = time_tag.get("datetime") if time_tag else None
                    
                    if target_date and release_date_str:
                        try:
                            # "2026-05-01" 또는 ISO 형식 파싱
                            item_dt = datetime.fromisoformat(release_date_str.split("T")[0])
                            if item_dt < target_date:
                                print(f"[Info] 타겟 날짜 이전 기사에 도달함 ({item_dt.date()} < {target_date.date()}). 크롤링 종료.")
                                stop_crawling = True
                                break
                        except ValueError:
                            pass

                    if href.startswith("/7993/"):
                        full_url = "https://wayback.archive-it.org" + href
                    else:
                        print(f"[Warning] 예상치 못한 href 형식: {href}")
                        continue

                    try:
                        body_page.goto(full_url, wait_until="domcontentloaded")
                        body_page.wait_for_timeout(5000)
                        
                        soup_inner = BeautifulSoup(body_page.content(), "html.parser")

                        # 1. 제목 (Title)
                        title_tag = soup_inner.select_one("h1.content-title") or soup_inner.select_one("h1")
                        title = title_tag.get_text(strip=True) if title_tag else ""
                
                        # 2. 발표일 (Release Date)
                        date_tag = soup_inner.select_one("time")
                        release_date = date_tag.get("datetime") or date_tag.get_text(strip=True) if date_tag else ""
                
                        # 3. 본문 (Body) - CSS Selector 문법 수정: div[role="main"] 또는 main
                        body_container = soup_inner.select_one('div[role="main"]') or soup_inner.select_one("main") or soup_inner.select_one("article")
                        body_list = body_container.select("p, li") if body_container else []
                
                        result_text_list = []
                        for p in body_list:
                            body_text = p.get_text(strip=True)
                            if body_text.startswith("Media:") or body_text.startswith("###"):
                                break  # 미디어 연락처나 문서 종료 표기 시 수집 중단
                            if body_text:
                                result_text_list.append(body_text)
  
                        clean_dataset.append({
                            "release_date": release_date,
                            "category": "FDA",
                            "doc_type": "press-announcements",
                            "url": url,
                            "title": title,
                            "body": " ".join(result_text_list).strip(),
                        })
                        time.sleep(random.uniform(6, 10))  # 서버 부하 방지용 딜레이

                    except Exception as detail_err:
                        print(f"    [{i}] 상세 페이지 에러 패스: {detail_err}")
                        continue 

                time.sleep(random.uniform(6, 10))  # 서버 부하 방지용 딜레이
            
            browser.close()

    return clean_dataset








if __name__ == "__main__":

    target_dt = datetime.fromisoformat("2018-01-01")
    current_pivot_dt = datetime.fromisoformat("2019-06-25")
    count = 0
    result = crawl_fda_press_releases_18(target_date=target_dt)
    count += len(result)

    # seen_overall_urls = set()

    # if target_dt < datetime.fromisoformat("2024-01-01"):
    #     while current_pivot_dt and current_pivot_dt >= target_dt:
    #         date_str_formatted = current_pivot_dt.strftime("%Y%m%d")
    #         print(f"\n[Info] 아카이브 시점 조회 시작: {current_pivot_dt.date()} ({date_str_formatted})")
            
    #         # 아카이브 스크립트 오염 방지를 위해 id_ 플래그 사용 권장
    #         snapshot_url = (
    #             f"https://web.archive.org/web/{date_str_formatted}/"
    #             f"https://www.fda.gov/news-events/fda-newsroom/press-announcements"
    #         )

    #         data_21 = crawl_fda_press_releases_21(
    #             target_date=target_dt, 
    #             fda_url=snapshot_url
    #         )

    #         if not data_21:
    #             print(f"[Warning] {current_pivot_dt.date()} 시점에서 수집된 데이터가 없습니다. 3일 전으로 건너뜁니다.")
    #             current_pivot_dt -= timedelta(days=3)
    #             continue

    #         # 전체 중복 제거 및 저장
    #         new_records = []
    #         for item in data_21:
    #             if item["url"] not in seen_overall_urls:
    #                 seen_overall_urls.add(item["url"])
    #                 new_records.append(item)

    #         if new_records:
    #             _save_results(new_records, target_date=target_dt.date())
    #             count += len(new_records)
    #             print(f"[+] {len(new_records)}건 신규 저장 완료 (누적: {count}건)")

    #         # 마지막 기사의 날짜 파싱 -> 다음 루프의 기준 날짜 갱신
    #         last_date_str = data_21[-1].get("release_date")
    #         last_dt = datetime.fromisoformat(last_date_str[0:10])

    #         if last_dt:
    #             # 같은 날짜 중복 수집 방지를 위해 최소 하루 이전(-1 day)으로 점프
    #             if last_dt >= current_pivot_dt:
    #                 current_pivot_dt -= timedelta(days=1)
    #             else:
    #                 current_pivot_dt = last_dt - timedelta(days=1)
    #         else:
    #             print("[-] 날짜 파싱 불가로 안전하게 7일 전으로 이동합니다.")
    #             current_pivot_dt -= timedelta(days=7)

    print("\n==========================================")
    print(f"최종 수집 완료: 총 {count}건")
    print("==========================================")
        