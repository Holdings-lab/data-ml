"""Search fact-extractor JSON with Brave Search.

PowerShell:
    $env:BRAVE_SEARCH_API_KEY = "your-key"
    python crawler/validating/evidence_search.py --input facts.json --output search_results.json

Python:
    search_results = search_news_facts(extract_news_facts(body))
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


def search_news_facts(
    extracted: str | dict[str, Any], *, api_key: str | None = None,
    count: int = 10, session: Any = None, search_mode: str = "as_of",
    lookback_days: int | None = None, expand_search: bool = False,
) -> str:
    """Return JSON containing each claim, query and raw Brave API response.

    Accept the JSON string returned by extract_news_facts or its decoded dict.
    Preserve the original article's published_date in the output (null if absent).
    Each claim is searched separately; count applies per claim (default: 10).
    HTTP/network errors propagate. Empty facts return an empty facts list
    without needing an API key or making a request.
    Search from one calendar month before published_date through published_date.
    If published_date is absent or null, use today in Asia/Seoul instead.
    If the previous month has no matching day, use its last day.
    Brave uses the page's published or last modified date.
    """
    if search_mode not in {"as_of", "current"}:
        raise ValueError("search_mode must be as_of or current")
    if lookback_days is not None and (type(lookback_days) is not int or not 1 <= lookback_days <= 3650):
        raise ValueError("lookback_days must be between 1 and 3650")
    data = json.loads(extracted) if isinstance(extracted, str) else extracted
    if not isinstance(data, dict) or not isinstance(data.get("facts"), list):
        raise ValueError("입력 JSON에는 facts 배열이 있어야 합니다.")
    facts = data["facts"]
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 20:
        raise ValueError("count는 1~20 사이의 정수여야 합니다.")
    for fact in facts:
        if not isinstance(fact, dict) or not isinstance(fact.get("claim"), str) or not fact["claim"].strip():
            raise ValueError("각 facts 항목에는 비어 있지 않은 claim이 있어야 합니다.")
        queries = fact.get("search_queries")
        if (not isinstance(queries, list) or len(queries) != 1
                or not isinstance(queries[0], str) or not queries[0].strip()):
            raise ValueError("search_queries에는 검색어가 정확히 1개 있어야 합니다.")
        query = queries[0].strip()
        if len(query) > 600 or len(query.split()) > 75:
            raise ValueError("검색어는 600자 및 75단어 이하여야 합니다.")
    if not facts:
        return json.dumps({"published_date": data.get("published_date"), "facts": []},
                          ensure_ascii=False, indent=2)
    published_date = data.get("published_date")
    if published_date is None:
        end = datetime.now(timezone(timedelta(hours=9))).date()
    else:
        try:
            end = date.fromisoformat(published_date)
            if end.isoformat() != published_date:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError("published_date는 YYYY-MM-DD 형식의 유효한 날짜여야 합니다.") from exc
    article_end = end
    if search_mode == "current":
        end = datetime.now(timezone(timedelta(hours=9))).date()
        if article_end > end:
            raise ValueError("Publication date is in the future")
    year = article_end.year if article_end.month > 1 else article_end.year - 1
    month = article_end.month - 1 if article_end.month > 1 else 12
    day = min(article_end.day, calendar.monthrange(year, month)[1])
    start = article_end.replace(year=year, month=month, day=day)
    if lookback_days is not None:
        start = article_end - timedelta(days=lookback_days)
    key = api_key if api_key is not None else os.getenv("BRAVE_SEARCH_API_KEY")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("BRAVE_SEARCH_API_KEY 환경 변수를 설정하세요.")
    if session is None:
        with requests.Session() as http:
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            http.mount("https://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])))
            return search_news_facts(data, api_key=key, count=count, session=http, search_mode=search_mode, lookback_days=lookback_days, expand_search=expand_search)
    results = []
    for fact in facts:
        query = fact["search_queries"][0].strip()
        queries = [query]
        if expand_search:
            queries += [" ".join(query.split()[:65])[:540] + " official announcement source",
                        " ".join(query.split()[:65])[:540] + " correction clarification"]
        responses = []
        for search_query in queries:
            params = {"q": search_query, "count": count,
                      "freshness": f"{start.isoformat()}to{end.isoformat()}"}
            response = session.get(
                BRAVE_SEARCH_URL,
                headers={"Accept": "application/json", "X-Subscription-Token": key.strip()},
                params=params, timeout=(5, 30))
            response.raise_for_status()
            search_response = response.json()
            if not isinstance(search_response, dict):
                raise ValueError("Brave API 응답이 JSON 객체가 아닙니다.")
            responses.append(search_response)
        merged = dict(responses[0])
        if len(responses) > 1:
            for section in ("web", "news"):
                merged[section] = {"results": [item for payload in responses for item in payload.get(section, {}).get("results", [])]}
        results.append({"claim": fact["claim"], "search_query": query,
                        "search_response": merged, "search_queries_used": queries,
                        "search_window": {"start": start.isoformat(), "end": end.isoformat(), "mode": search_mode}})
    return json.dumps({"published_date": data.get("published_date"), "facts": results},
                      ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="추출 결과의 검색어를 Brave Search API로 검색")
    parser.add_argument("--input", type=Path, help="추출 결과 JSON 파일 (생략 시 표준 입력)")
    parser.add_argument("--output", type=Path, help="검색 결과 JSON 파일 (생략 시 표준 출력)")
    parser.add_argument("--count", type=int, choices=range(1, 21), default=10,
                        help="주장별 검색 결과 수 (기본 10, 최대 20)")
    parser.add_argument("--search-mode", choices=["as_of", "current"], default="as_of")
    parser.add_argument("--lookback-days", type=int)
    parser.add_argument("--expand-search", action="store_true")
    args = parser.parse_args()
    try:
        extracted = args.input.read_text(encoding="utf-8-sig") if args.input else sys.stdin.read()
        result = search_news_facts(extracted, count=args.count, search_mode=args.search_mode, lookback_days=args.lookback_days, expand_search=args.expand_search) + "\n"
        if args.output:
            args.output.write_text(result, encoding="utf-8")
        else:
            print(result, end="")
    except Exception as exc:
        parser.exit(1, f"검색 실패: {exc}\n")


if __name__ == "__main__":
    main()
