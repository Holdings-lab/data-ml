# Data Layout

`data/` 아래의 산출물은 역할 기준으로 아래처럼 정리.

## `data/crawler`

- `collected/`: 사이트에서 방금 수집한 원문 CSV
- `summarized/`: 요약이 끝난 문서 CSV
- `features/`: 병합, 시간 피처, 감성 점수 등 학습용 뉴스 피처 CSV

## `data/training`

- `market_only/`: 가격 피처만 사용한 XGBoost 실험 산출물
- `market_news/`: 가격 + 뉴스 피처를 함께 사용한 XGBoost 실험 산출물
- `comparison/`: 두 실험의 성능 비교표와 메타데이터

이 구조를 유지하면 "이 파일이 수집 결과인지, 전처리 결과인지, 최종 모델 산출물인지"를 판단할 수 있음.
