from __future__ import annotations

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.support_legacy.data_paths import feature_csv_path

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_ARTICLES_FOR_PROMPT = 20
MAX_BODY_CHARS_PER_ARTICLE = 1200


def _parse_iso_date(target_date_iso: str) -> datetime:
    """
    ISO 형식(YYYY-MM-DD) 문자열을 datetime 객체로 변환한다.
    """
    try:
        return datetime.fromisoformat(str(target_date_iso).strip())
    except ValueError as exc:
        raise ValueError("target_date는 ISO 형식(YYYY-MM-DD)이어야 합니다.") from exc


def _normalize_to_iso_date(series: pd.Series) -> pd.Series:
    raw = series.fillna("").astype(str).str.strip()
    parsed = pd.to_datetime(raw.where(raw != "", other=pd.NA), errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def _build_news_payload(df: pd.DataFrame) -> str:
    """
    뉴스 DataFrame을 Claude 프롬프트에 넣을 수 있는 문자열로 변환한다.
    MAX_ARTICLES_FOR_PROMPT 개수까지만 포함하며, 각 뉴스의 body는 MAX_BODY_CHARS_PER_ARTICLE 글자까지만 포함한다.
    """
    rows: list[str] = []
    for i, row in enumerate(df.head(MAX_ARTICLES_FOR_PROMPT).itertuples(index=False), start=1):
        row_dict = row._asdict()
        title = str(row_dict.get("title", "")).strip() or "(No Title)"
        body_summary = str(row_dict.get("body_summary", "")).strip()
        body_summary = body_summary[:MAX_BODY_CHARS_PER_ARTICLE]
        rows.append(f"[{i}] title: {title}\nbody: {body_summary}")
    return "\n\n".join(rows)


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
        temperature=0.5,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    if not getattr(response, "content", None):
        raise ValueError("Claude 응답이 비어 있습니다.")

    first_block = response.content[0]
    text = getattr(first_block, "text", "")
    return str(text or "").strip()


def summarize_news_for_date(
    target_sector: str,
    target_date: datetime,
    window: int,
    news_df: pd.DataFrame,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    타겟 날짜의 뉴스들을 하나로 요약해 json 형식으로 반환한다.

    Args:
        news_df: 원본 뉴스 DataFrame (title, body, 날짜 컬럼 필요)
        target_date: 타겟 날짜 (datetime 객체)
        window: 뉴스를 가져올 기간 (일 단위)
        model: Claude 모델명

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

    start_date = pd.to_datetime(target_date) - timedelta(days=window)
    end_date = pd.to_datetime(target_date)

    work_df = news_df.copy()
    work_df[date_col] = _normalize_to_iso_date(work_df[date_col])
    work_df[date_col] = pd.to_datetime(work_df[date_col], errors="coerce")
    
    filtered = work_df.loc[(work_df[date_col] >= start_date) & (work_df[date_col] <= end_date) & (work_df["sector"] == target_sector)].copy()
    if filtered.empty:
        raise ValueError(f"{target_date.isoformat()} 날짜에 해당하는 뉴스가 없습니다.")

    filtered["title"] = filtered["title"].fillna("").astype(str)
    filtered["body_summary"] = filtered["body_summary"].fillna("").astype(str)
    filtered = filtered.loc[(filtered["title"].str.strip() != "") | (filtered["body_summary"].str.strip() != "")]
    if filtered.empty:
        raise ValueError("타겟 날짜 뉴스에 요약 가능한 title/body 텍스트가 없습니다.")

    news_payload = _build_news_payload(filtered)

    FIXED_PROMPT = (
        "아래 뉴스 묶음을 하나의 일간 뉴스로 통합 요약하세요.\n"
        "반드시 JSON 객체만 출력하세요.\n"
        "출력 스키마: {\"sector\": \"...\", \"release_date\": \"YYYY-MM-DD\", \"title\": \"...\", \"content\": \"...\"}\n\n"
        "엄격한 제약:\n"
        "1) title은 기사 전체를 대표하는 단일 제목이어야 합니다.\n"
        "2) content는 200자 이상 300자 이하 한국어 문장으로 작성하세요.\n"
        "3) 사실 기반으로만 작성하고, 제공된 뉴스에 없는 내용을 추론하지 마세요.\n"
        "4) 불릿/번호/마크다운/코드블록을 사용하지 마세요.\n"
        "5) JSON 외 텍스트를 절대 출력하지 마세요.\n\n"
        f"타겟 섹터: {target_sector}\n"
        f"타겟 날짜: {target_date.isoformat()}\n"
        f"뉴스 개수: {len(filtered)}\n\n"
        "뉴스 원문:\n"
        f"{news_payload}"
    )

    print(FIXED_PROMPT)  # 디버깅용 출력

    result = _call_claude_summary(prompt=FIXED_PROMPT, model=model)
    return result


def main(ticker: str, target_date: datetime, days: int, csv_filename: str):
    data = pd.read_csv(feature_csv_path(csv_filename))
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
        default=datetime.now() - timedelta(days=1),
        help="기준 날짜 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="조회 일수 (기본값: 5)"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="policy_updates_features_20days_revision_sorted.csv",
        help="CSV 파일명"
    )

    args = parser.parse_args()

    main(
        ticker=args.ticker,
        target_date=args.date,
        days=args.days,
        csv_filename=args.file
    )