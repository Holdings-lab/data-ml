import requests
import pandas as pd
from pathlib import Path
import argparse
import time
import re
from urllib3.util.retry import Retry
import logging
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crawler.collectors.collection_status import report_failure
from crawler.collectors.common import create_retry_session

BASE_URL = "https://fraser.stlouisfed.org/api"
API_KEY = "9e92168d5b63454b89ef1d4ff149ca69" 

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# module logger
logger = logging.getLogger(__name__)


START_DATE = "2017-01-01"  
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "crawler" / "collected"

DEFAULT_KEYWORDS = [
    '"monetary policy" "interest rate"',
    '"stress test" capital',                    
    '"financial stability" risk',               
    '"yield curve" inversion'                   
]


def _build_session(retries=3, backoff_factor=1.0, status_forcelist=(429, 500, 502, 504)):
    retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff_factor,
                  status_forcelist=status_forcelist, allowed_methods=frozenset(["GET"]))
    return create_retry_session(headers, retry)


def _get_with_backoff(session, url, params=None, max_attempts=5, base_delay=1.5):
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            if attempt == max_attempts: return None, str(exc)
            time.sleep(base_delay * attempt)
            continue

        if resp.status_code == 429:
            if attempt == max_attempts: return None, "too many 429 error responses"
            retry_after = resp.headers.get("Retry-After")
            delay = max(1, int(retry_after)) if (retry_after and retry_after.isdigit()) else base_delay * attempt
            logger.warning("rate limited (429). retrying in %.1fs (attempt %d/%d)", delay, attempt, max_attempts)
            time.sleep(delay)
            continue

        if not resp.ok and resp.status_code in (500, 502, 503, 504):
            if attempt == max_attempts: return None, f"status {resp.status_code}"
            time.sleep(base_delay * attempt)
            continue

        return resp, None
    return None, "unknown request failure"


def _extract_release_date(record):
    origin_info = record.get("originInfo", {}) if isinstance(record, dict) else {}
    return origin_info.get("sortDate") or origin_info.get("dateIssued") or ""


def search_fraser(session, keyword, start_date=None, per_page=100, max_results=None, page_delay=0.5):
    base = f"{BASE_URL}/search"
    requested_fields = "titleInfo!originInfo!location!abstract!recordInfo!physicalDescription"
    params = {
        "q": keyword,
        "limit": per_page,
        "page": 1,
        "sort": "sort_date_text desc",
        "fields": requested_fields,
    }

    resp, err = _get_with_backoff(session, base, params)
    if err or not resp.ok:
        report_failure(code="UPSTREAM_REQUEST_FAILED")
        logger.error("FRASER API 연결 실패 (%s)", err if err else resp.status_code)
        return None

    aggregate = resp.json()
    if not isinstance(aggregate, dict) or "records" not in aggregate or "total" not in aggregate:
        raise ValueError("Invalid FRASER search response")
    total = aggregate.get("total", 0)
    records = aggregate.get("records", [])

    logger.info("검색 결과 - 총 매칭: %d건 / 1페이지 다운로드: %d건", total, len(records))

    start_dt = pd.to_datetime(start_date, errors='coerce') if start_date else pd.NaT

    if not pd.isna(start_dt) and records:
        first_date = _extract_release_date(records[0])
        first_dt = pd.to_datetime(first_date, errors='coerce') if first_date else pd.NaT
        if not pd.isna(first_dt) and first_dt < start_dt:
            logger.info("첫 문서 날짜 %s 가 start_date %s 이전입니다. 추가 페이지 요청을 중단합니다.", first_date, start_date)
            aggregate["records"] = []
            return aggregate

    if total <= len(records):
        if max_results: records = records[:max_results]
        aggregate["records"] = records
        return aggregate

    per = aggregate.get("limit", per_page) or per_page
    total_pages = (total + per - 1) // per

    for page in range(2, total_pages + 1):
        params["page"] = page

        resp, err = _get_with_backoff(session, base, params)
        if err or not resp.ok:
            report_failure(code="UPSTREAM_REQUEST_FAILED")
            logger.warning("page %d 수집 실패: %s", page, err if err else resp.status_code)
            break

        page_json = resp.json()
        page_records = page_json.get("records", [])
        if not page_records:
            logger.info("page %d 수집 결과 없음", page)
            break

        page_first = page_records[0]
        page_first_date = _extract_release_date(page_first)
        page_first_dt = pd.to_datetime(page_first_date, errors='coerce') if page_first_date else pd.NaT
        if not pd.isna(start_dt) and not pd.isna(page_first_dt) and page_first_dt < start_dt:
            logger.info("page %d 첫 문서 날짜 %s 가 start_date %s 이전입니다. 추가 페이지 요청을 중단합니다.", page, page_first_date, start_date)
            break

        page_last = page_records[-1]
        page_last_date = _extract_release_date(page_last)
        logger.info(
            "page %d JSON 수집 완료: %d건 (date %s ~ %s)",
            page,
            len(page_records),
            page_first_date or "?",
            page_last_date or "?",
        )

        records.extend(page_records)
        if max_results and len(records) >= max_results:
            records = records[:max_results]
            break

        if page_delay > 0:
            time.sleep(page_delay)

    aggregate["records"] = records
    return aggregate


def fetch_text_body(session, text_url, max_chars=5000):
    if not text_url or text_url == "Text 없음": return ""
    resp, err = _get_with_backoff(session, text_url)
    if err or not resp or not resp.ok:
        report_failure(code="UPSTREAM_REQUEST_FAILED")
        return ""
    
    raw_text = resp.text.strip()
    
    # 🔥 [개선 1] 웹 URL, 페이지 번호(1/11), 비정상 특수문자 전면 청소 (Noise Scrubbing)
    cleaned_text = re.sub(r'https?://\S+', '', raw_text)  # URL 제거
    cleaned_text = re.sub(r'\b\d+/\d+\b', '', cleaned_text)  # 1/11, 2/11 같은 페이지 기호 제거
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # 연속 공백 및 줄바꿈을 단일 공백으로 통합
    
    # 🔥 [개선 2] 목차(TOC) 무조건 건너뛰기 메커니즘 (TOC Bypass)
    # 점선(....)이 연속되거나 목차용 단어가 밀집한 초반 구역(약 3000자)은 검색 대상에서 제외
    search_zone = cleaned_text
    toc_match = re.search(r'(\.{4,}|table\s+of\s+contents)', cleaned_text[:4000], re.IGNORECASE)
    if toc_match:
        # 목차 기호가 발견되면 그 지점부터 2000자 뒤를 진짜 본문 검색 시작점으로 강제 점프
        search_zone = cleaned_text[toc_match.end() + 2000:]
    
    # 🔥 [개선 3] 정교한 섹션 정규식 앵커링 (공백 통합 버전)
    patterns = [
        r'\bexecutive\s+summary\b',
        r'\bintroduction\b',
        r'\bfinance\s+overview\b'
    ]
    
    start_idx = 0
    for pattern in patterns:
        match = re.search(pattern, search_zone, re.IGNORECASE)
        if match:
            # 앵커를 찾았다면 전체 원본 clean 텍스트에서의 절대 위치를 계산
            start_idx = cleaned_text.find(search_zone) + match.end()
            break
            
    # 💡 [개선 4] 문장 잘림 방지 (Sentence-Boundary Preservation)
    # 단순히 5000자로 자르면 문장 중간이 툭 끊깁니다. 5000자 근처의 마침표(.)를 찾아 문장 단위로 마감합니다.
    sub_body = cleaned_text[start_idx : start_idx + max_chars]
    
    # 뒤에서부터 가장 가까운 마침표, 온점 위치 탐색
    last_period = max(sub_body.rfind('. '), sub_body.rfind('? '), sub_body.rfind('! '))
    if last_period != -1 and last_period > 4000:
        final_body = sub_body[:last_period + 1].strip()
    else:
        final_body = sub_body.strip()
        
    # 만약 앞단 단어 유실로 인해 시작이 이상하다면(예: "ation. In fact") 첫 문장 버림 처리
    if final_body.startswith(('ation.', 'ect.', 'fact.')):
        final_body = re.sub(r'^[^.]*\.\s*', '', final_body)
        
    return final_body


def parse_fraser_results(session, search_json, keyword="fraser", start_date=None, doc_delay=0.3):
    if not search_json: return []
    records = search_json.get('records', [])
    parsed_list = []

    start_dt = pd.to_datetime(start_date or START_DATE, errors='coerce')

    for idx, record in enumerate(records, start=1):
        title_info = record.get("titleInfo", [])
        title = title_info[0].get("title", "") if title_info else ""

        release_date = _extract_release_date(record)

        release_dt = pd.to_datetime(release_date, errors='coerce') if release_date else pd.NaT
        if not pd.isna(start_dt) and not pd.isna(release_dt) and release_dt < start_dt:
            break

        location_info = record.get("location", {})
        text_urls = location_info.get("textUrl", [])
        text_url = text_urls[0] if text_urls else ""

        body = fetch_text_body(session, text_url, max_chars=5000)
        if not body: body = f"Document Title Summary: {title}"

        title_preview = title.replace("\n", " ").strip()
        if len(title_preview) > 80: title_preview = title_preview[:77] + "..."
            
        logger.info("[DOC %d] keyword='%s' date='%s' body_chars=%d title='%s'", idx, keyword, release_date, len(body), title_preview)

        parsed_list.append({
            "category": "FRASER",
            "doc_type": keyword,
            "title": title,
            "url": text_url if text_url else "URL 없음",
            "release_date": release_date,
            "body": body,
        })
        if doc_delay > 0:
            time.sleep(doc_delay)

    return parsed_list


def collect_fraser_documents(keywords, start_date=None, per_page=100, max_results=None, page_delay=0.5, doc_delay=0.3):
    all_rows = []
    session = _build_session()

    for keyword in keywords:
        api_response = search_fraser(
            session=session, keyword=keyword, start_date=start_date,
            per_page=per_page, max_results=max_results, page_delay=page_delay,
        )
        if not api_response: continue

        rows = parse_fraser_results(session, api_response, keyword=keyword, start_date=start_date, doc_delay=doc_delay)
        all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame(columns=["category", "doc_type", "title", "url", "release_date", "body"])

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["url", "title"], keep="first")
    df["release_date_dt"] = pd.to_datetime(df["release_date"], errors="coerce")

    df = df.sort_values(by="release_date_dt", ascending=False)
    return df


def save_results_csv(df, output_dir=OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "fraser_results.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords", "-k", nargs="*", default=DEFAULT_KEYWORDS, help="검색 키워드 목록")
    parser.add_argument("--start-date", "-s", default=START_DATE, help="시작 날짜 (YYYY-MM-DD)")
    parser.add_argument("--per-page", type=int, default=100, help="페이지당 요청 수")
    parser.add_argument("--max-results", type=int, default=None, help="최대 문서 수 (없음=전체)")
    parser.add_argument("--page-delay", type=float, default=0.5, help="페이지 요청 간 대기 시간(초)")
    parser.add_argument("--doc-delay", type=float, default=0.3, help="개별 본문 텍스트 요청 간 대기 시간(초)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    df = collect_fraser_documents(
        args.keywords, start_date=args.start_date, per_page=args.per_page,
        max_results=args.max_results, page_delay=args.page_delay, doc_delay=args.doc_delay
    )

    if not df.empty:
        final_df = df[['category', 'doc_type', 'title', 'url', 'release_date', 'body']]
        save_results_csv(final_df)