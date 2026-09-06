from __future__ import annotations

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import json
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.support_legacy.data_paths import feature_csv_path

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
NY_TZ = ZoneInfo("America/New_York")
MAX_ARTICLES_FOR_PROMPT = 20
MAX_BODY_CHARS_PER_ARTICLE = 1200


def _normalize_to_iso_date(series: pd.Series) -> pd.Series:
    """
    시리즈의 날짜 문자열을 ISO 형식(YYYY-MM-DD)으로 변환한다.
    비어 있는 값은 NaN으로 처리한다.
    """
    raw = series.fillna("").astype(str).str.strip()
    parsed = pd.to_datetime(raw.where(raw != "", other=pd.NA), errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def _build_news_payload(df: pd.DataFrame) -> tuple[str, str]:
    """
    뉴스 DataFrame을 Claude 프롬프트에 넣을 수 있는 문자열로 변환한다.
    MAX_ARTICLES_FOR_PROMPT 개수까지만 포함하며, 각 뉴스의 body는 MAX_BODY_CHARS_PER_ARTICLE 글자까지만 포함한다.
    """
    rows: list[str] = []
    image = ""
    for i, row in enumerate(df.head(MAX_ARTICLES_FOR_PROMPT).itertuples(index=False), start=1):
        row_dict = row._asdict()
        title = str(row_dict.get("title", "")).strip()
        body_summary = str(row_dict.get("body_summary", "")).strip()
        body_summary = body_summary[:MAX_BODY_CHARS_PER_ARTICLE]
        if not title or not body_summary:
            continue

        image = str(row_dict.get("image", "")).strip() or image
        rows.append(f"[{i}] title: {title}\nbody: {body_summary}")
    return "\n\n".join(rows), image


def _call_claude_summary(prompt: str, model: str) -> str:
    """
    Claude 모델을 호출하여 요약을 생성한다.
    주의사항 : 환경변수에 ANTHROPIC_API_KEY가 설정되어 있어야 한다.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic 패키지가 필요합니다. `pip install anthropic` 후 다시 시도하세요."
        ) from exc

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    if not getattr(response, "content", None):
        raise ValueError("[ERROR] Claude response is empty.")

    first_block = response.content[0]
    text = getattr(first_block, "text", "")
    return str(text or "").strip()


def summarize_news_for_date(
    target_sector: str,
    target_date: datetime,
    window: int,
    news_df: pd.DataFrame
) -> str:
    """
    타겟 날짜의 뉴스들을 하나로 요약해 json 형식으로 반환한다.

    Args:
        target_date: 타겟 날짜 (datetime 객체)
        window: 타겟 날짜 이전으로 범위 지정 (일 단위)
        news_df: 원본 뉴스 DataFrame (title, body, 날짜 컬럼 필요)

    Returns:
        {"title": "...", "content": "..."}
    """
    if news_df is None or news_df.empty:
        raise ValueError("news_df가 비어 있습니다.")
    if "title" not in news_df.columns:
        raise ValueError(f"'title' 컬럼이 필요합니다. 현재 컬럼: {list(news_df.columns)}")
    if "body_summary" not in news_df.columns:
        raise ValueError(f"'body_summary' 컬럼이 필요합니다. 현재 컬럼: {list(news_df.columns)}")

    date_col = "release_date"

    target_date = pd.to_datetime(target_date)
    if target_date.tzinfo is not None:
        target_date = target_date.tz_localize(None)
    target_date = target_date.normalize()

    start_date = target_date - pd.Timedelta(days=window)
    end_date = target_date

    work_df = news_df.copy()
    work_df[date_col] = _normalize_to_iso_date(work_df[date_col])
    work_df[date_col] = pd.to_datetime(work_df[date_col], errors="coerce")
    
    filtered = work_df.loc[(work_df[date_col] >= start_date) & (work_df[date_col] <= end_date) & (work_df["sector"] == target_sector)].copy()
    if filtered.empty:
        raise ValueError(f"{target_date.strftime('%Y-%m-%d')} 날짜에 해당하는 뉴스가 없습니다.")

    filtered["title"] = filtered["title"].fillna("").astype(str)
    filtered["body_summary"] = filtered["body_summary"].fillna("").astype(str)
    filtered = filtered.loc[(filtered["title"].str.strip() != "") | (filtered["body_summary"].str.strip() != "")]
    if filtered.empty:
        raise ValueError("타겟 날짜 뉴스에 요약 가능한 title/body 텍스트가 없습니다.")

    news_payload, image = _build_news_payload(filtered)

    FIXED_PROMPT = (
        "아래 뉴스 묶음을 하나의 일간 뉴스로 통합 요약하세요.\n"
        "반드시 JSON 객체만 출력하세요.\n"
        "출력 스키마: {\"title\": \"...\", \"content\": \"...\"}\n\n"
        "엄격한 제약:\n"
        "1) title은 기사 전체를 대표하는 단일 제목이어야 합니다.\n"
        "2) content는 200자 이상 300자 이하 한국어 문장으로 작성하세요.\n"
        "3) 사실 기반으로만 작성하고, 제공된 뉴스에 없는 내용을 추론하지 마세요.\n"
        "4) 불릿/번호/마크다운/코드블록을 사용하지 마세요.\n"
        "5) JSON 외 텍스트를 절대 출력하지 마세요.\n\n"
        f"타겟 섹터: {target_sector}\n"
        f"타겟 날짜: {target_date.strftime('%Y-%m-%d')}\n"
        f"뉴스 개수: {len(filtered)}\n\n"
        "뉴스 원문:\n"
        f"{news_payload}"
    )

    result = _call_claude_summary(prompt=FIXED_PROMPT, model=DEFAULT_MODEL)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", result.strip(), flags=re.DOTALL)
    json_result = json.loads(cleaned)
    json_result["sector"] = target_sector
    json_result["release_date"] = target_date.strftime("%Y-%m-%d")
    json_result["image"] = image
    result = json.dumps(json_result, ensure_ascii=False, indent=2)
   
    return result


def main(ticker: str, target_date: datetime, days: int, csv_filename: str):
    data = pd.read_csv(csv_filename)
    result = summarize_news_for_date(ticker, target_date, days, data)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize news for a specific date and ticker.")

    parser.add_argument(
        "--ticker",
        type=str,
        default="qqq",
        help="종목 티커 (기본값: qqq)"
    )
    parser.add_argument(
        "--date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=datetime.now(NY_TZ) - timedelta(days=1),
        help="기준 날짜 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="조회 일수 (기본값: 1)"
    )
    parser.add_argument(
        "--file",
        type=str,
        default= feature_csv_path("policy_updates_features.csv"),
        help="CSV 파일명"
    )

    args = parser.parse_args()

    main(
        ticker=args.ticker,
        target_date=args.date,
        days=args.days,
        csv_filename=args.file
    )