from __future__ import annotations

import csv
import time
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.support_legacy.data_paths import collected_csv_path

# 미국 뉴욕 시간대 고정 (서머타임 자동 계산)
NY_TZ = ZoneInfo("America/New_York")

# 기본 타겟 데이트: 뉴욕 시간 기준 어제 (YYYY-MM-DD)
TARGET_DATE = (datetime.now(NY_TZ) - timedelta(days=1)).strftime("%Y-%m-%d")
TICKERS = ["QQQ", "XLF", "XLE"]


def _save_results(records, target_date):
    csv_path = Path(collected_csv_path(f"yahoo_market_news_{target_date}.csv"))

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sector", "title", "url", "release_date", "image", "body"])
        writer.writeheader()
        writer.writerows(records)

    return csv_path


def convert_to_iso_date(date_str):
    """
    "Mon, June 1, 2026 at 6:18 AM GMT+9" -> 시차 변환 후 "YYYY-MM-DD" 반환
    """
    try:
        # 1. " at " 제거
        cleaned = date_str.replace(" at ", " ")
        
        # 2. "GMT+9" 또는 "GMT-05:00" 형태를 "+0900" / "-0500" 형태로 정규화
        def repl_gmt(match):
            sign = match.group(1)
            hours = int(match.group(2))
            return f"{sign}{hours:02d}00"
        
        cleaned = re.sub(r'GMT([+-])(\d+)', repl_gmt, cleaned)
        # 예: "Thu, August 27, 2026 1:02 AM +0900"

        # 3. strptime으로 파싱
        dt = datetime.strptime(cleaned, "%a, %B %d, %Y %I:%M %p %z")
        
        # 4. ET 변환
        dt_et = dt.astimezone(NY_TZ)
        return dt_et.strftime("%Y-%m-%d")
        
    except Exception as e:
        print(f"[-] 파싱 실패: {e}")
        return "N/A"


def scrape_news_sync(target_date=TARGET_DATE, tickers=TICKERS):
    print(f"[YAHOO SUPPORT] Running synchronous playwright pipeline (target date: {target_date})")
    
    clean_dataset = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        
        list_page = context.new_page()
        body_page = context.new_page()
        
        for ticker in tickers:
            print(f"\n[+] [{ticker}] sector news-list loading...")
            url = f"https://finance.yahoo.com/quote/{ticker}/news/"
            list_page.goto(url, wait_until="domcontentloaded")
            list_page.wait_for_timeout(2000)

            try:
                list_page.locator(f"a[href='/quote/{ticker.upper()}/latest-news/']").first.click(timeout=5000)
                list_page.wait_for_load_state("domcontentloaded")
            except Exception:
                raise RuntimeError(f"[ERROR] [{ticker}] news link click failed. URL: {url}")

            for i in range(10):
                list_page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                # list_page.mouse.wheel(0, 5000)
                list_page.wait_for_timeout(2000)


            # 스크롤 완료 후 전체 DOM 구조 확보
            html_content = list_page.content()
            soup = BeautifulSoup(html_content, "html.parser")
            
            # 다양한 Yahoo Finance 최신 Layout CSS Selector 대응
            news_items = soup.select("li.stream-item") 
        
            print(f"[YAHOO SUPPORT] 최종 포착된 {ticker} 뉴스 후보: 총 {len(news_items)}개")
            
            out_of_date_count = 0
            MAX_OUT_OF_DATE = 3  # 연속 과거 기사 감지 시 조기 탈출 기준

            for i, item in enumerate(news_items):
                if out_of_date_count >= MAX_OUT_OF_DATE:
                    print(f"    [{i}] 연속 {MAX_OUT_OF_DATE}회 기준일({target_date}) 이전 과거 기사 감지. {ticker} 수집 종료.")
                    break

                a_tag = item.find("a", class_=["subtitle-link", "titles"])
                if not (a_tag and a_tag.get_text()):
                    print(f"    [{i}] 뉴스 링크 추출 실패")
                    continue

                title = a_tag.get_text(strip=True)
                link = a_tag.get("href", "")
                    
                # 노이즈 기사 필터링
                if any(kw in title for kw in ["Sector Update", "Exchange-Traded"]):
                    print(f"    [{i}] Premium 기사 필터링: {title[:60]}...")
                    continue
                
                if "finance.yahoo.com/" not in link:
                    print(f"    [{i}] Yahoo Finance 외부 링크(광고) 필터링: {link}")
                    continue
                    
                print(f"    [{i}] [-> 상세 수집] {title[:30]}...")
                
                try:
                    body_page.goto(link, wait_until="domcontentloaded")
                    body_page.wait_for_selector("time, .caas-body, .bodyItems-wrapper", timeout=5000)
                    
                    soup_inner = BeautifulSoup(body_page.content(), "html.parser")
                    
                    # 1. 날짜 추출 및 변환
                    date_str = "N/A"
                    time_tag = soup_inner.find("time", class_="byline-attr-meta-time") or soup_inner.find("time")
                    if time_tag:
                        raw_date = time_tag.get_text(strip=True)
                        date_str = convert_to_iso_date(raw_date)

                    if date_str == "N/A":
                        print(f"    [{i}] 날짜 추출 실패: {link}")
                        continue

                    # 2. 날짜 검증 (target_date보다 과거면 카운트 증가)
                    if date_str != "N/A":
                        if date_str < target_date:
                            print(f"    [{i}] 과거 기사 발견 ({date_str})")
                            out_of_date_count += 1
                            continue
                        elif date_str > target_date:
                            print(f"    [{i}] {target_date} 이후 기사 발견 ({date_str})")
                            out_of_date_count = 0
                        else:
                            out_of_date_count = 0
                    
                    # 3. 본문 추출
                    body_tag = soup_inner.find("div", class_="bodyItems-wrapper") or soup_inner.find(class_="caas-body")
                    if not body_tag:
                        print(f"    [{i}] 본문 추출 실패")
                        continue

                    paragraphs = [p.get_text(strip=True) for p in body_tag.find_all("p")]
                    full_body = " ".join(paragraphs)

                    if not full_body:
                        print(f"    [{i}] 본문 추출 실패: {link}")
                        continue

                    og_image = soup_inner.find("meta", property="og:image")
                    image = og_image.get("content") if og_image else ""
                    
                    clean_dataset.append({
                        "sector": ticker,
                        "title": title,
                        "url": link,
                        "release_date": date_str,
                        "image": image,
                        "body": full_body,
                        
                    })

                except Exception as detail_err:
                    print(f"    [{i}] 상세 페이지 에러 패스: {detail_err}")
                    continue 

                time.sleep(0.5)
        
        browser.close()

    print(clean_dataset)
    return clean_dataset


def main():
    target_date = TARGET_DATE
    news_results = scrape_news_sync(target_date=target_date, tickers=TICKERS)

    if news_results:
        
        output_csv_path = _save_results(news_results, target_date)
        print(
            f"\n[✔] 배치가 완벽히 완료되었습니다. 총 {len(news_results)}개 적재 완료"
            f"\n    - CSV : {output_csv_path}"
        )
    else:
        print("\n[!] 저장할 데이터가 없어 출력 파일 생성을 건너뜁니다.")
    

if __name__ == "__main__":
    main()