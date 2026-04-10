# Data-ML Pipeline

이 저장소는 정책/거시 뉴스 수집부터 뉴스 후처리, 시장 데이터 피처 생성, XGBoost 학습과 비교 실험까지 한 번에 관리하는 데이터 파이프라인 프로젝트.

현재 코드 기준으로는 `shared/` 아래 파이프라인이 메인 실행 경로이고, `training/`과 `crawler/support_legacy/`는 초기 실험 또는 레거시 호환 코드로 분리함.

## 프로젝트 목적(데모)

- 정책/거시 이벤트 문서를 크롤링해 학습 가능한 형태로 정리.
- QQQ와 거시 자산 데이터를 함께 사용해 가격 예측용 피처 생성.
- `market_only`와 `market_news` 두 실험을 같은 절차로 학습해 성능을 비교.
- 여기에 `market_news`도 추가. (뉴스데이터와, 주가데이터 날짜 일치)

## 프로젝트 개요(대략적인 프로젝트 파이프라인)

1. Crawler/collectors/fed.py, crawler/collectors/whitehouse.py, crawler/collectors/bis.py 실행해서 원문 문서 모음
    * 이 단계에서는  FOMC 문서, White House 정책 문서, BIS 보도자료를 CSV 형태로 저장하는것 -> 결과는 data/crawler/collected/ 아래에 쌓임.


2. 수집한 문서를 학습 가능한 형태로 정리하는 단계
    * crawler/postprocessing/text_summarizer.py -> 너무 긴 본문을 Ollama를 이용해서 요약해서 길이를 줄이는 역할
    * crawler/postprocessing/proprocessing.py -> 여러 수집 결과 CSV를 하나로 합치면서 날짜, 카테고리, 문서 타입, 본문 길이 같은 컬럼을 정리
    * crawler/postprocessing/sentiment_score.py -> FinBERT를 사용해 제목과 본문에 감성 점수를 붙여 최종적으로 merged_finbert.csv 생성
* 이 과정을 거쳐서 모델이 읽을 수 있는 수치화된 뉴스데이터로 변환


3. 학습단계
    * shared/run_market_news_training.py: merged_finbert.csv를 입력으로 받아 전체 파이프라인 실행
    * 맹점은 모델이 문서 한건한건을 직접 읽는 것이 아니라, 하루 단위로 압축된 뉴스 신호를 사용한다는 것.
        * 예를 들어
        * 어떤 날짜에는 뉴스가 몇 건 있었는지
        * 부정 뉴스 비율이 높았는지
        * FOMC 관련 문서가 있었는지
        * 최근 3일과 5일 평균 감성이 어땠는지 같은 값으로 변환한 뒤 시장 데이터와 합친다. 


4. 정리하면
    * 뉴스 피처가 이미 준비된 상태에서 모델 성능만 보고 싶으면 : python shared/run_market_news_training.py만 실행
    * 데이터부터 새로 만들고 싶으면 : collectors -> text_summarizer -> proprocessing -> sentiment_score -> run_market_news_training 순서대로 실행
    * 다만 현재 코드 기준으로 text_summarizer.py는 기본 입력이 BIS 파일 쪽에 맞춰져 있어서, FOMC나 White House 요약까지 자동으로 한 번에 돌리는 구조는 아님. 


## 핵심 실행 흐름

1. `crawler/collectors/`
   외부 사이트에서 원문 문서를 수집.
2. `crawler/postprocessing/`
   긴 문서를 요약하고, 소스별 CSV를 병합하고, 각 소스별 감정 점수 추가.
3. `shared/news/`
   문서 단위 뉴스 데이터를 날짜별 숫자 피처로 집계.
4. `shared/market/`
   QQQ와 거시 자산 가격 데이터를 내려받아 시장 피처를 생성.
5. `shared/training/`
   horizon 선택, 피처 선택, Optuna 튜닝, XGBoost 학습과 평가를 수행.
6. `data/`
   중간 산출물과 최종 모델, 메타데이터, 비교 결과를 저장.

## 디렉터리 구조

```text
data-ml/
├─ crawler/
│  ├─ collectors/
│  │  ├─ fed.py
│  │  ├─ bis.py
│  │  └─ whitehouse.py
│  ├─ postprocessing/
│  │  ├─ text_summarizer.py
│  │  ├─ proprocessing.py
│  │  └─ sentiment_score.py
│  └─ support_legacy/
│     ├─ data_paths.py
│     ├─ pipeline.py
│     ├─ run_crawler.py
│     ├─ scraper.py
│     └─ crawling_test.py
├─ data/
│  ├─ crawler/
│  │  ├─ collected/
│  │  ├─ summarized/
│  │  └─ features/
│  └─ training/
│     ├─ market_only/
│     ├─ market_news/
│     └─ comparison/
├─ shared/
│  ├─ common/
│  ├─ config/
│  ├─ market/
│  ├─ news/
│  ├─ pipelines/
│  ├─ training/
│  └─ run_market_news_training.py
├─ training/
├─ requirements.txt
└─ README.md
```

## 주요 파일 가이드

### 메인 파이프라인

- `shared/run_market_news_training.py`
  가장 먼저 실행하면 되는 메인 CLI 엔트리포인트.(그냥 실행해도 되고, 명령어로 실행해도됨 -> 밑에 설명)
- `shared/pipelines/market_news.py`
  뉴스 로드, 시장 피처 생성, 두 실험 학습, 비교 저장까지의 전체 순서를 관리.
- `shared/training/xgboost_pipeline.py`
  horizon 선택, 피처 선택, Optuna 튜닝, 최종 모델 학습과 평가 수행.

### 뉴스 수집

- `crawler/collectors/fed.py`
  FOMC statement, minutes, implementation note를 수집.
- `crawler/collectors/bis.py`
  BIS 보도자료 목록을 Selenium으로 탐색하고 상세 본문을 수집.
- `crawler/collectors/whitehouse.py`
  White House 문서를 수집한 뒤 QQQ 관련 키워드가 포함된 정책 문서만 남김.

### 뉴스 후처리

- `crawler/postprocessing/text_summarizer.py`
  긴 본문을 Ollama 기반 로컬 LLM으로 요약.
- `crawler/postprocessing/proprocessing.py`
  수집 결과를 표준 컬럼으로 병합하고 카테고리/시간 피처를 추가.
- `crawler/postprocessing/sentiment_score.py`
  FinBERT로 제목/본문 감성 점수를 계산해 최종 뉴스 피처 CSV를 생성.

### 레거시/실험 코드

- `training/train_regression.py`
  초기 단일 회귀 실험 코드.
- `training/dataset.py`
  QQQ 단일 종목 기반 분류 실험 코드.
- `crawler/support_legacy/`
  경로 유틸과 예전 실행 진입점, 간단한 테스트 케이스들을 포함.

## 산출물 저장 위치

### 뉴스 관련

- `data/crawler/collected/`
  크롤러 원문 수집 결과 CSV
- `data/crawler/summarized/`
  요약이 적용된 문서 CSV
- `data/crawler/features/`
  병합, 시간 피처, 감성 점수까지 포함된 학습용 뉴스 CSV

### 학습 관련

- `data/training/market_only/`
  시장 피처만 사용한 실험 결과
- `data/training/market_news/`
  시장 + 뉴스 피처를 사용한 실험 결과
- `data/training/comparison/`
  두 실험의 성능 비교 CSV/JSON

## 코드 실행 가이드

아래 명령은 모두 프로젝트 루트(`data-ml/`)에서 실행하는 것을 기준으로 작성.

### 1. 가상환경 및 기본 패키지 설치(Mac OS 사용시 추천)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 추가 패키지 설치

크롤러/후처리까지 모두 실행하려면 아래 패키지를 추가로 설치해야 함.

```bash
pip install selenium transformers certifi
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

참고:

- `crawler/collectors/bis.py`는 Selenium과 로컬 Chrome/Chromium 환경이 필요.
- `crawler/postprocessing/text_summarizer.py`는 로컬 Ollama 서버가 실행 중이어야 함.
- `crawler/postprocessing/sentiment_score.py`는 `torch`와 `transformers`가 필요함.

## 빠른 실행 시나리오

### A. 이미 뉴스 피처 CSV가 있을 때 학습만 바로 실행

`data/crawler/features/merged_finbert.csv`가 이미 준비되어 있다면 아래 한 줄로 메인 학습 파이프라인 실행 가능.

```bash
python shared/run_market_news_training.py
```

실행이 끝나면 기본적으로 아래 산출물들이 생성됨.(현재 data 폴더안에 생성되어있음)

- `data/training/market_only/qqq_market_only_xgboost_model.json`
- `data/training/market_only/qqq_market_only_metadata.json`
- `data/training/market_news/qqq_market_news_xgboost_model.json`
- `data/training/market_news/qqq_market_news_metadata.json`
- `data/training/comparison/qqq_market_model_comparison.csv`
- `data/training/comparison/qqq_market_model_comparison.json`

### B. 뉴스 수집부터 학습까지 전체 파이프라인 실행

#### Step 1. 뉴스 원문 수집

```bash
python crawler/collectors/fed.py
python crawler/collectors/whitehouse.py
python crawler/collectors/bis.py --max-pages 9
```

기본 출력 위치:

- `data/crawler/collected/fed_fomc_links.csv`
- `data/crawler/collected/whitehouse_qqq_policy.csv`
- `data/crawler/collected/bis_press_releases.csv`

#### Step 2. 긴 문서 요약

```bash
python crawler/postprocessing/text_summarizer.py
```

현재 구현 기준 주의사항:

- `text_summarizer.py`의 기본 `INPUT_CSV`와 `OUTPUT_CSV`는 BIS 파일 기준으로 고정되어 있음.
- `proprocessing.py`는 아래 세 파일이 모두 준비되어 있다고 가정.
  - `data/crawler/summarized/fed_fomc_links_summarized.csv`
  - `data/crawler/summarized/whitehouse_qqq_policy_summarized.csv`
  - `data/crawler/summarized/bis_press_releases_summarized.csv`
- 따라서 FOMC/White House 쪽도 요약 산출물이 필요하면 스크립트 상단 상수를 바꿔 같은 방식으로 다시 실행해야 함.

#### Step 3. 수집 결과 병합 및 시간 피처 생성

```bash
python crawler/postprocessing/proprocessing.py
```

생성 파일:

- `data/crawler/features/merged_table_sorted.csv`
- `data/crawler/features/merged_table_sorted_encoded.csv`
- `data/crawler/features/merged_table_sorted_time_features.csv`

#### Step 4. FinBERT 감성 점수 계산

```bash
python crawler/postprocessing/sentiment_score.py
```

생성 파일:

- `data/crawler/features/merged_finbert.csv`

#### Step 5. 메인 학습 파이프라인 실행

```bash
python shared/run_market_news_training.py
```

## 학습 파이프라인 옵션 예시

기본값 대신 일부 설정을 바꿔 실행할 수도 있음.(종목, 날짜값 등 변경 가능하게)

```bash
python shared/run_market_news_training.py \
  --target-ticker QQQ \
  --start-date 2016-01-01 \
  --end-date 2026-01-01 \
  --horizons 5,10,15,20 \
  --optuna-trials 30 \
  --top-feature-count 20
```

주요 옵션:

- `--target-ticker`
  예측 대상 티커
- `--start-date`, `--end-date`
  시장 데이터 다운로드 구간
- `--news-input`
  입력 뉴스 피처 CSV 경로
- `--horizons`
  비교할 예측 horizon 후보
- `--optuna-trials`
  하이퍼파라미터 탐색 횟수
- `--top-feature-count`
  최종 후보로 남길 상위 중요 피처 개수

## 추천 읽기 순서

처음 프로젝트를 파악할 때는 아래 순서로 읽는 것을 추천.

1. `shared/run_market_news_training.py`
2. `shared/pipelines/market_news.py`
3. `shared/market/data.py`
4. `shared/news/features.py`
5. `shared/news/merge.py`
6. `shared/training/xgboost_pipeline.py`

## 현재 코드 기준 메모

- 메인 학습 파이프라인은 `shared/` 아래에 정리.
- `training/` 폴더는 실험, 테스트용 코드로 사용. 새로운 작업은 가급적 `shared/` 기준으로 진행하는 것이 좋을듯.
- 결과 CSV는 `.gitignore`에 의해 기본적으로 Git 추적 대상에서 제외.

## `train_regression.py` 코드가 `shared/`에 반영된 방식

현재 `shared/` 메인 학습은 `training/train_regression.py`를 가능한 한 그 코드 그대로 가져와서 모듈화한 버전으로 보면 됨.

쉽게 말하면:

- `training/train_regression.py`
  한 파일 안에서 데이터 다운로드 -> 피처 생성 -> 뉴스 병합 -> 학습 -> 평가까지 한 번에 처리하는 원본 실험 코드
- `shared/`
  위 흐름을 파일별로 나눠서 유지보수하기 쉽게 만든 구조
  대신 메인 실험 설정은 원본 스크립트와 최대한 같게 맞춰 둠

### 1. 메인 실험 자체를 `train_regression.py`처럼 고정 설정으로 돌림

 메인 실험 기준으로 아래처럼 바뀜.

- `market_only`
  `train_regression.py`에서 쓰는 시장 정예 피처만 사용
- `market_news`
  위 시장 정예 피처 + 뉴스 감성 정예 피처만 사용
- 기본 horizon
  `15일` 고정


참고:

- 이 고정 horizon 값은 `shared/config/schema.py`의 `regression_style_fixed_horizon = 15`
- CLI에서는 `--regression-style-fixed-horizon`으로 바꿀 수 있음

### 2. 시장 피처는 `train_regression.py`의 정예 피처 기준으로 맞춤

`shared/market/data.py`에는 원래 다양한 시장 피처가 많지만, 실제 메인 학습에서 사용하는 피처는 `train_regression.py` 기준 정예 목록으로 제한함.

현재 메인 학습에 쓰는 시장 피처:

- `ret_5`
- `ret_accel`
- `dist_to_ma5`
- `bb_pos`
- `rsi_14`
- `vol_shock`
- `vix_z_score_5`
- `drawdown`
- `vol_ratio`
- `rel_strength_5`
- `uup_shock_5`
- `tlt_shock_5`
- `hyg_ret`
- `target_spy_rel_ret`

특히 아래 계산식은 원본에 맞춰 반영함.

- `ret_accel = ret_1 - ret_5`
- `vol_shock = vol_5 / (vol_20 + 1e-9)`
- `dist_to_ma5 = price / MA(5) - 1`
- `rel_strength_5 = QQQ 5일 수익률 - SPY 5일 수익률`
- `uup_shock_5 = UUP 5일 변화율`
- `tlt_shock_5 = TLT 5일 변화율`
- `vix_z_score_5 = 5일 기준 VIX z-score`

즉 `shared` 안에 다른 피처가 더 남아 있더라도, 메인 실험이 실제로 보는 핵심 시장 피처는 `train_regression.py`와 거의 같은 세트라고 보면 됨.

### 3. 뉴스 일자 집계도 `train_regression.py` 흐름으로 맞춤

`train_regression.py`에서는 `merged_finbert.csv`를 읽은 뒤:

1. 필요한 뉴스 컬럼만 선택
2. 주말 뉴스를 다음 월요일로 이동
3. 같은 날짜 뉴스는 평균을 내어 하루 1행으로 압축

현재 `shared/news/features.py`도 같은 생각으로 동작함.

반영된 규칙:

- 주말 뉴스는 다음 영업일(월요일)로 이동
- 같은 날짜 뉴스는 평균값으로 압축
- 주요 입력 컬럼은 아래와 같은 `train_regression.py` 스타일 컬럼
  - `category_BIS`
  - `category_FOMC`
  - `category_UCSB`
  - `day_of_week_sin`, `day_of_week_cos`
  - `month_sin`, `month_cos`
  - `is_weekend`
  - `title_positive_prob`, `title_negative_prob`, `title_neutral_prob`
  - `title_sentiment_score`
  - `body_positive_prob`, `body_negative_prob`, `body_neutral_prob`
  - `body_sentiment_score`
  - `body_n_chunks`

차이점이 있다면, `shared`는 이 작업을 `load_news_source_table()`과 `build_daily_news_feature_table()` 두 단계로 나눠둔 것뿐임.

### 4. 뉴스 병합과 결측 처리 순서도 최대한 그대로 맞춤

`shared/news/merge.py`는 `train_regression.py`의 병합 흐름을 따름.

현재 순서:

1. 시장 데이터와 뉴스 일자 테이블을 날짜 기준 `left join`
2. `title_neutral_prob`, `body_neutral_prob`는 기본값 `1.0`으로 채움
3. 주요 뉴스/감성 컬럼은 **`ffill` 없이 `0.0`으로 채움** (뉴스 없는 날 = 무신호)
4. `days_since_news` 계산: 마지막 뉴스 이후 경과 거래일 수 (최대 30일)
5. 전체 프레임 `ffill()` + `0.0` (시장 피처용, 뉴스 컬럼은 이미 채워진 상태)

`ffill`을 쓰지 않는 이유:

- `ffill`을 쓰면 며칠 전 뉴스 감성이 아무 뉴스도 없는 날까지 그대로 전파됨
- "뉴스 없음(0)"과 "예전 뉴스의 잔존 영향"이 섞여 신호가 왜곡될 수 있음
- 대신 `days_since_news`와 `body_sentiment_decay_3d`를 통해 오래된 뉴스의 감쇠된 영향을 모델이 별도로 학습하게 함

### 5. 뉴스 파생 피처는 원본 스크립트 기준 핵심 6개

현재 메인 `market_news` 실험에서 쓰는 뉴스 피처는 아래 6개임.

| 피처 | 설명 |
| --- | --- |
| `sentiment_gap` | 제목 긍정 확률 - 부정 확률 |
| `body_sentiment_gap` | 본문 긍정 확률 - 부정 확률 |
| `sentiment_shock` | `sentiment_gap`의 최근 5일 평균 대비 변화량 |
| `body_sentiment_score` | 본문 감성 점수 자체 |
| `days_since_news` | 마지막 뉴스 이후 경과 거래일 수 (최대 30) |
| `body_sentiment_decay_3d` | `body_sentiment_score × 0.5^(days_since_news / 3)` — 반감기 3일 감쇠 적용 |

`days_since_news`와 `body_sentiment_decay_3d`를 추가한 이유:

- 뉴스가 없는 날 감성값을 `0`으로 채우면 "오늘 뉴스가 있어서 0점"과 "뉴스 자체가 없어서 0점"을 구분 못 함
- `days_since_news`로 경과 일수를 직접 제공하면 모델이 "최근 뉴스"와 "며칠 지난 뉴스"를 구분해서 학습 가능
- `body_sentiment_decay_3d`는 같은 감성 점수라도 오래된 뉴스일수록 영향력이 작아지도록 반감기 감쇠를 적용한 것

추가로 `shared`에서는 aligned comparison 시작일 계산을 위해 `news_count_lag1` 보조 컬럼도 남겨 둠.
이 컬럼은 메인 뉴스 피처라기보다 비교 구간을 자르는 데 쓰는 운영용 컬럼이라고 보면 됨.

### 6. 학습 타깃과 Optuna 목적함수도 `train_regression.py` 기준

`shared/training/xgboost_pipeline.py`에 반영된 핵심은 아래와 같음.

- 타깃 로그수익률을 `* 100` 스케일로 학습
- 미래 가격 복원 시 `exp(pred_logret / 100.0)` 사용
- Optuna 탐색 범위:
  - `n_estimators: 100 ~ 500`
  - `max_depth: 4 ~ 6`
  - `learning_rate: 0.01 ~ 0.1`
  - `subsample: 0.5 ~ 0.9`
  - `colsample_bytree: 0.5 ~ 0.9`
- 목적함수:
  - `direction_accuracy - rmse * 0.1`

즉 지금 `shared`의 메인 학습은 모델 튜닝 관점에서도 `train_regression.py`와 거의 같은 기준으로 움직임.

### 7. 아직 `shared/`에만 남겨둔 구조적 차이

완전히 똑같이 복붙한 것은 아님.
차이는 "실험 구조" 쪽에만 남겨 둔 상태.

- `shared`는 `market_only`와 `market_news`를 같은 실행에서 같이 돌림
- 결과를 `data/training/market_only/`, `market_news/`, `comparison/`에 나눠 저장
- aligned comparison을 따로 만들어 공정 비교를 계속 볼 수 있게 함

중요한 점:

- 메인 두 실험은 `train_regression.py` 스타일 고정 horizon/고정 피처를 사용
- 반면 aligned comparison은 `--horizons`에 들어온 후보 horizon들에 대해 같은 피처 세트로 다시 비교함

즉 현재 구조를 한 문장으로 정리하면:

- 메인 모델 학습 로직은 `train_regression.py`를 거의 그대로 따르고
- shared는 그 위에 비교 실험과 저장 구조만 얹어 둔 상태라고 보면 됨.

### 8. 아직 옮기지 않은 부분

아래 요소들은 아직 `shared` 메인 파이프라인에는 넣지 않음.

- importance plot 시각화
- threshold별 전략 곡선
- 고확신 구간 상승/하락 정밀 분석
- 산점도, 누적수익률, 에러 분포 시각화

즉 "모델을 학습하고 비교하는 코어 로직"은 대부분 옮겼고,
"실험 분석용 시각화/리포트 코드"는 아직 `train_regression.py` 쪽에 더 많이 남아 있음.

그래서:

- 최종모델 결과물 생성과 비교는 `shared`에 저장하고
- 모델 수정하면서 진행하는 분석은 `training/train_regression.py`

이렇게 역할을 나눠서 작업해보면 될듯.


## 최신 실험 결과(2026.04.01)

아래 결과는 프로젝트 루트에서 다음 명령으로 실행한 최신 산출물 기준.

```bash
./venv/bin/python shared/run_market_news_training.py --horizons 10,20 --optuna-trials 10
```

### 1. 기본 비교 결과

출력 파일:

- `data/training/comparison/qqq_market_model_comparison.csv`
- `data/training/comparison/qqq_market_model_comparison.json`

결과 요약:

| experiment_name | best_horizon | direction_accuracy | rmse | mae | r2_score | mape |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| market_only | 10 | 60.98% | 17.3639 | 13.3083 | 0.9285 | 2.7178 |
| market_news | 20 | 71.29% | 22.9058 | 17.7965 | 0.8746 | 3.6617 |
| market_news - market_only | +10 | +10.31%p | +5.5418 | +4.4882 | -0.0539 | +0.9440 |

해석:

- 이 비교만 보면 `market_news`가 방향 정확도는 더 높게 나옴.
- 하지만 `market_only`는 10거래일 예측, `market_news`는 20거래일 예측이라 완전한 동일 조건 비교는 아님.
- 따라서 위 표는 참고용으로 보고, 실제 결론은 아래 aligned comparison 기준으로 판단하는 것이 좋음.

### 2. 공정 비교 결과 (Aligned Comparison)

출력 파일:

- `data/training/comparison/qqq_market_model_comparison_aligned.csv`
- `data/training/comparison/qqq_market_model_comparison_aligned.json`

aligned comparison은 다음 조건으로 비교:

- 뉴스가 실제로 존재하는 구간만 사용
- 시작일: `2019-03-22`
- 같은 거래일끼리 `market_only`와 `market_news`를 직접 비교

결과 요약:

| shared_horizon | market_only direction | market_news direction | direction delta | rmse delta | mae delta | r2 delta | mape delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 60.18% | 57.23% | -2.95%p | +0.2294 | +0.2947 | -0.0031 | +0.0538 |
| 20 | 67.95% | 66.77% | -1.19%p | +0.2384 | +0.1286 | -0.0044 | +0.0264 |

해석:

- `10일`, `20일` 모두에서 현재 `market_news`가 `market_only`보다 약간 성능이 낮음.
- 방향 정확도도 소폭 낮고, RMSE/MAE/MAPE도 모두 조금 더 큼.
- 즉 현재 데이터와 피처 구성에서는 뉴스 피처가 추가적인 예측력으로 연결됐다고 보기 어려움.
- aligned comparison의 요약 기준으로 보면 `20일 horizon`이 `10일 horizon`보다 덜 나쁘지만, 그래도 개선은 아님.

### 3. 현재 실험에서 볼 수 있는 결론

- 지금 시점의 기준 모델은 `market_only`로 보는 것이 안전함.
- 뉴스 피처는 일부 중요 피처로 선택되긴 했지만, 공정 비교 기준 성능 개선까지는 이어지지 못함.


### 4. 날짜 범위 문제에 대한 설명

- 시장 데이터 시작일은 기본적으로 `2015-01-01`로 설정되어 있음.
- 뉴스 피처는 현재 저장된 파일 기준 `2019-03-21`부터 존재함.
- 뉴스를 시장 프레임에 병합할 때는 날짜 기준 `left join` 이후 빈 뉴스 값을 `0.0`으로 채우는 구조라, 모델 입장에서는 아래 두 경우를 구분하지 못함.
  - 진짜로 그날 뉴스가 0건인 경우
  - 아직 그 시기 뉴스 데이터를 아예 수집하지 못한 경우

현재 생성된 `market_news` 학습 프레임 기준으로 보면:

- 전체 학습 프레임 시작일: `2015-06-25`
- `news_count_lag1`이 처음 0이 아닌 날짜: `2019-03-22`
- 전체 행 수: `2626`
- `news_count_lag1 = 0`인 행 수: `2139`
- train 구간 행 수: `2100`
- train 구간에서 실제로 뉴스가 있는 행 수: `214`

해석:

- 즉 `market_news` 모델이라 해도 학습 초반의 긴 구간은 사실상 뉴스 없이 학습되고 있음.
- 이런 상태에서는 `market_news`가 실제 뉴스 신호를 얼마나 잘 활용하는지보다, 오랫동안 `market_only`처럼 학습한 효과가 섞여 들어가게 됨.
- 그래서 기본 비교 결과보다 aligned comparison 결과를 더 중요하게 봐야 함.

소스별 현실적인 제약도 있음.

- White House는 정권이 바뀌면 HTML 구조와 문서 분류 방식이 달라질 수 있음.
- 과거 아카이브는 구조가 일정하지 않아 장기 백필이 어렵고, 기사 품질도 시기별로 들쭉날쭉할 수 있음.
- 이런 상황에서 과거 구간을 무리하게 `0`으로 채워 넣으면 White House 관련 피처 의미가 희석될 수 있음.

정리하면:

- `0`은 "그날 뉴스가 없었다"는 값으로는 쓸 수 있지만, "그 시기 뉴스 데이터가 아직 없다"는 결측 표현으로는 위험함.
- 현재 실험 결과를 해석할 때는 반드시 이 점을 염두에 두어야 하고, 앞으로도 동일 horizon + 동일 기간의 aligned comparison을 기준 지표로 삼는 것이 좋음.

추후 진행할 것:

1. 뉴스 결측과 실제 0건을 더 명확히 분리
2. 감성 평균 외에 이벤트성 피처를 더 정교하게 설계
3. aligned comparison 결과를 기준 지표로 계속 발전시키는 게 나을듯 한데
