from __future__ import annotations

import csv
import time
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from pytz import timezone
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.support_legacy.data_paths import collected_csv_path

# 미국 뉴욕 시간대 고정 (서머타임 자동 계산)
NY_TZ = timezone('America/New_York')
# 기본 타겟 데이트: 뉴욕 시간 기준 어제 (YYYY-MM-DD)
TARGET_DATE = (datetime.now(NY_TZ) - timedelta(days=1)).strftime("%Y-%m-%d")
TICKERS = ["QQQ", "XLF", "XLE"]


def _save_results(records, target_date):
    csv_path = Path(collected_csv_path(f"yahoo_market_news_{target_date}.csv"))

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sector", "title", "url", "release_date", "body"])
        writer.writeheader()
        writer.writerows(records)

    return csv_path


def convert_to_iso_date(date_str):
    """
    1) "Mon, June 1, 2026 at 6:18 AM GMT+9" -> 시차 변환 후 "YYYY-MM-DD" 반환
    2) "July 30, 2026" -> "YYYY-MM-DD" 반환
    """
    if not date_str:
        return "N/A"
        
    date_str = date_str.strip()
    
    try:
        # 💡 케이스 1: "July 30, 2026" 같은 단순 날짜 형태
        # (시간 및 타임존 정보가 없는 경우)
        if not re.search(r'\b(AM|PM|GMT|UTC|[+-]\d{2})\b', date_str, re.IGNORECASE):
            # "July 30, 2026" 또는 "Jul 30, 2026" 대응
            for fmt in ("%B %d, %Y", "%b %d, %Y"):
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime("%Y-%m-%d")
                except ValueError:
                    continue

        # 💡 케이스 2: 기존의 상세 날짜/시간/타임존 형태
        # 1. 'at ' 글자 제거
        cleaned_str = re.sub(r'\s+at\s+', ' ', date_str, flags=re.IGNORECASE)
        
        # 2. 타임존 텍스트(GMT+9 등) 분리 및 획득
        match = re.search(r'(GMT)([+-]\d+)$', cleaned_str, re.IGNORECASE)
        
        if match:
            tz_offset = match.group(2) # "+9"
            cleaned_str = re.sub(r'\s+GMT[+-]\d+$', '', cleaned_str, flags=re.IGNORECASE).strip()
        else:
            tz_offset = "+9"
            
        # 3. datetime 객체 생성 (요일 유무 모두 대응)
        # 예: "Mon, June 1, 2026 6:18 AM" 또는 "June 1, 2026 6:18 AM"
        naive_dt = None
        formats_to_try = [
            "%a, %B %d, %Y %I:%M %p",  # Mon, June 1, 2026 6:18 AM
            "%a, %b %d, %Y %I:%M %p",  # Mon, Jun 1, 2026 6:18 AM
            "%B %d, %Y %I:%M %p",       # June 1, 2026 6:18 AM
            "%b %d, %Y %I:%M %p"        # Jun 1, 2026 6:18 AM
        ]
        
        for fmt in formats_to_try:
            try:
                naive_dt = datetime.strptime(cleaned_str, fmt)
                break
            except ValueError:
                continue

        if not naive_dt:
            raise ValueError(f"지원하지 않는 날짜 포맷입니다: {date_str}")
        
        # 4. 원본 시간 타임존 부여
        if tz_offset == "+9":
            origin_tz = timezone('Asia/Seoul')
        else:
            origin_tz = timezone('Asia/Seoul')
            
        localized_dt = origin_tz.localize(naive_dt)
        
        # 5. 미국 뉴욕 시간대로 시차 변환
        ny_tz = timezone('America/New_York')
        ny_dt = localized_dt.astimezone(ny_tz)
        
        # 6. YYYY-MM-DD 반환
        return ny_dt.strftime("%Y-%m-%d")
        
    except Exception as e:
        print(f"[-] 날짜 타임존 변환 실패 ('{date_str}'): {e}")
        return "N/A"


def scrape_news_sync(target_date=TARGET_DATE, tickers=TICKERS):
    print(f"[+] 동기식 Playwright 파이프라인 가동 (기준일자: {target_date})")
    
    clean_dataset = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        
        page = context.new_page()
        detail_page = context.new_page()
        
        for ticker in tickers:
            print(f"\n[+] [{ticker}] 섹션 뉴스 목록 로드 중...")
            url = f"https://finance.yahoo.com/quote/{ticker}/news/"
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(2000)

            try:
                page.locator(f"a[href='/quote/{ticker.upper()}/latest-news/']").first.click(timeout=5000)
                page.wait_for_load_state("domcontentloaded")
            except Exception:
                raise RuntimeError(f"[-] [{ticker}] 뉴스 섹션 링크 클릭 실패. URL: {url}")

            for i in range(20):
                page.mouse.wheel(0, 5000)
                page.keyboard.press("PageDown")
                page.wait_for_timeout(1500)


            # 스크롤 완료 후 전체 DOM 구조 확보
            html_content = page.content()
            soup = BeautifulSoup(html_content, "html.parser")
            
            # 다양한 Yahoo Finance 최신 Layout CSS Selector 대응
            news_items = soup.select("li.stream-item") 
        
            print(f"[*] 최종 포착된 {ticker} 뉴스 후보: 총 {len(news_items)}개")
            
            out_of_date_count = 0
            MAX_OUT_OF_DATE = 3  # 연속 과거 기사 감지 시 조기 탈출 기준

            for i, item in enumerate(news_items):
                if out_of_date_count >= MAX_OUT_OF_DATE:
                    print(f"    [{i}] 연속 {MAX_OUT_OF_DATE}회 기준일({target_date}) 이전 과거 기사 감지. {ticker} 수집 종료.")
                    break

                a_tag = item.find("a", class_=["subtitle-link", "titles"]) or item.find("a")
                if not (a_tag and a_tag.get_text()):
                    print(f"    [{i}] 뉴스 링크 추출 실패")
                    continue

                title = a_tag.get_text(strip=True)
                link = a_tag.get("href", "")
                
                if link.startswith("/"):
                    link = "https://finance.yahoo.com" + link
                    
                # 노이즈 기사 필터링
                if any(kw in title for kw in ["Sector Update", "Exchange-Traded"]):
                    print(f"    [{i}] Premium 기사 필터링: {title[:60]}...")
                    continue
                
                if "finance.yahoo.com/" not in link:
                    print(f"    [{i}] Yahoo Finance 외부 링크(광고) 필터링: {link[:60]}...")
                    continue
                    
                print(f"    [{i}] [-> 상세 수집] {title[:30]}...")
                
                try:
                    detail_page.goto(link, wait_until="domcontentloaded")
                    detail_page.wait_for_selector("time, .caas-body, .bodyItems-wrapper", timeout=5000)
                    
                    soup_inner = BeautifulSoup(detail_page.content(), "html.parser")
                    
                    # 1. 날짜 추출 및 변환
                    date_str = "N/A"
                    time_tag = soup_inner.find("time", class_="byline-attr-meta-time") or soup_inner.find("time")
                    if time_tag:
                        raw_date = time_tag.get_text(strip=True)
                        date_str = convert_to_iso_date(raw_date)

                    if date_str == "N/A":
                        print(f"    [{i}] 날짜 추출 실패: {link[:60]}...")
                        continue

                    # 2. 날짜 검증 (target_date보다 과거면 카운트 증가)
                    if date_str != "N/A":
                        if date_str < target_date:
                            print(f"    [{i}] 과거 기사 발견 ({date_str})")
                            out_of_date_count += 1
                            continue
                        elif date_str >= target_date:
                            out_of_date_count = 0
                    
                    # 3. 본문 추출
                    body_tag = soup_inner.find("div", class_="bodyItems-wrapper") or soup_inner.find(class_="caas-body")
                    if not body_tag:
                        print(f"    [{i}] 본문 추출 실패")
                        continue

                    paragraphs = [p.get_text(strip=True) for p in body_tag.find_all("p")]
                    full_body = " ".join(paragraphs)

                    if not full_body:
                        print(f"    [{i}] 본문 추출 실패: {link[:60]}...")
                        continue

                    # image_tag = soup_inner.find("img", class_="yf-lf8hkhu")
                    # image = image_tag.get("src") if image_tag else "N/A"
                    
                    clean_dataset.append({
                        "sector": ticker,
                        "title": title,
                        "url": link,
                        "release_date": date_str,
                        # "image": image,
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
    target_date = "2026-07-15"
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