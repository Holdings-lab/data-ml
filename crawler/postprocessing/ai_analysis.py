from __future__ import annotations

import sys
import pandas as pd
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.support_legacy.data_paths import feature_csv_path


DEFAULT_MODEL = "claude-haiku-4-5-20251001"

FIXED_PROMPT = """
당신은 머신러닝 기반 금융시장 예측 결과를 브리핑 뉴스로 설명하는 AI 브리핑 작성자다.

입력으로 다음 정보가 제공된다.

- 예측 대상 자산
- 예측 기간
- 머신러닝의 상승·하락 확률
- 예상 변동률
- 시장 변동성 및 주요 금융 지표
- 예측 시점 이전 5일간의 뉴스

머신러닝 예측 결과를 기반으로 하고, 최근 뉴스 중 해당 자산과 직접 관련성이 높으며 예측 방향을 설명할 수 있는 핵심 요인 2개를 선택해 사용자용 브리핑을 작성하라.

분석 규칙:

1. 뉴스와 자산의 관련성을 가장 중요하게 고려한다.
2. 관련성이 낮은 뉴스는 브리핑 작성에 사용하지 않는다.
3. 예측 방향과 일치하는 뉴스를 우선적으로 사용한다.
4. 뉴스가 예측 방향과 일치하지 않으면 상반된 신호가 있다고 설명한다.
5. 뉴스가 시장 움직임의 직접 원인이라고 단정하지 않는다.
6. 입력에 없는 사건이나 수치를 만들지 않는다.
7. 분량은 2~3문장으로 작성한다.
8. 수치 값이 아닌 자연어로 완곡하게 풀어서 작성한다. 표현 기준은 아래 '확률 표현 기준'을 참고한다.

확률 표현 기준: 
- 45% 초과 55% 미만: 뚜렷한 방향을 판단하기 어려움
- 55% 이상 65% 미만: 해당 방향의 가능성이 조금 더 높음
- 65% 이상: 해당 방향의 가능성이 비교적 높음

다음 JSON 형식으로만 출력하라.

{
"title": "AI는 이렇게 판단했어요",
"headline": "예측 방향 한 문장",
"reason": "최근 뉴스와 금융 지표를 이용한 2~3문장 설명",
"alignment": "SUPPORTIVE | MIXED | CONTRADICTORY | INSUFFICIENT",
"used_news_url": ["사용한 뉴스 URL"],
"disclaimer": "본 내용은 투자 판단의 근거가 아닙니다."
}

- SUPPORTIVE: 뉴스와 시장 데이터가 예측 방향을 대체로 지지함
- MIXED: 지지 요인과 반대 요인이 함께 존재함
- CONTRADICTORY: 주요 뉴스가 예측 방향과 상반됨
- INSUFFICIENT: 관련 근거가 부족함

입력 : \n
"""


TEST_JSON = """
{
  "asset":"QQQ",
  "prediction_date":"2026-08-01",
  "news_window_days":5,
  "horizon":"5거래일",
  "prediction": {
    "direction":"UP",
    "up_probability":0.62,
    "expected_return_pct":1.3
  },
  "market_data": {
    "vix":19.4,
    "vix_change_5d_pct":3.8,
    "qqq_return_5d_pct":1.1
  }
}
"""


def _get_news(target_sector: str, target_date: datetime, window: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    df 에서 target_date 기준, 이전 window 일 만큼의 뉴스를 가져온다.
    """
    if "release_date" not in df.columns:
        print("[Error] release_date column is not exist.")
        return pd.DataFrame()

    # 원본 데이터 보호를 위해 복사본 사용 및 datetime 타입 변환
    df_copy = df.copy()
    df_copy["release_date"] = pd.to_datetime(df_copy["release_date"])

    # 기간 계산 (target_date 기준 window 일 전부터 target_date 까지)
    start_date = pd.to_datetime(target_date) - timedelta(days=window)
    end_date = pd.to_datetime(target_date)

    # 기간 조건 필터링
    mask = (df_copy["release_date"] >= start_date) & (df_copy["release_date"] <= end_date) & (df_copy["sector"].str.lower() == target_sector.lower())
    result_df = df_copy[mask].sort_values(by="release_date", ascending=False)
    result_df = result_df[["sector", "release_date", "title", "body_summary", "url"]]

    return result_df

    
def insert_news_to_payload(base_data: str, news_data: pd.DataFrame) -> str:
    """
    예측 결과 json에 필터링된 뉴스 목록을 'news' 필드로 삽입하여 반환합니다.
    """
    # 1. base_data 파싱
    if isinstance(base_data, str):
      payload = json.loads(base_data)
    else:
      raise ValueError("base_data must be a JSON string.")

    # 2. news_data 형식 변환
    if isinstance(news_data, pd.DataFrame):
        formatted_news = news_data.to_dict(orient="records")
    else:
        raise ValueError("news_data must be a pandas DataFrame or a list of dicts.")

    # 3. news 키 삽입
    payload["news"] = formatted_news

    # 4. JSON 문자열로 직렬화 (한글 유니코드 유지)
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _call_claude_summary(message: str, model: str) -> str:
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
        messages=[{"role": "user", "content": message}],
    )

    if not getattr(response, "content", None):
        raise ValueError("Claude 응답이 비어 있습니다.")

    first_block = response.content[0]
    text = getattr(first_block, "text", "")
    return str(text or "").strip()


def main(
    ticker: str,
    target_date: datetime,
    days: int,
    csv_filename: str,
    prediction_json: str,
):
    news_df = pd.read_csv(csv_filename)
    news_df = _get_news(ticker, target_date, days, news_df)
    prediction_json_text = Path(prediction_json).read_text(encoding="utf-8")
    result_json = insert_news_to_payload(prediction_json_text, news_df)
    briefing = _call_claude_summary(FIXED_PROMPT + result_json, model=DEFAULT_MODEL)
    print(briefing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate news briefing summary.")
    
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
        default="policy_updates_features_20days.csv",
        help="CSV 파일명"
    )
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="예측 결과 JSON 파일 경로"
    )

    args = parser.parse_args()

    main(
        ticker=args.ticker,
        target_date=args.date,
        days=args.days,
        csv_filename=args.file,
        prediction_json=args.json
    )