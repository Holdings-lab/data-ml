# Data-ML Pipeline

최종 점검일: 2026-05-26. 이 README는 현재 코드와 `data/training/**/metadata.json`, 비교 JSON, 클러스터 리포트를 다시 읽고 맞춘 기준이다.

이 저장소는 정책/거시 뉴스 수집, 뉴스 후처리, 시장 가격 피처 생성, XGBoost 회귀 학습, market-only와 market-news 비교, 예측 수익률 profile 리포트까지 관리하는 실험 파이프라인이다. 현재 메인 실행 경로는 `shared/` 아래 코드이고, `crawler/support_legacy/`와 `TEST/`는 보조/레거시 성격이 강하다.

## 프로젝트 목적

이 프로젝트의 목적은 뉴스 문서 자체를 모델에 직접 넣는 것이 아니라, 정책/거시 문서에서 날짜별 신호를 뽑아 시장 데이터와 결합한 뒤 가격 방향성과 미래 가격을 예측해 보는 것이다.

현재 파이프라인은 크게 세 가지 질문에 답하도록 구성되어 있다.

- 가격/거시 피처만 사용한 `market_only` 모델은 어느 정도 성능을 내는가?
- 정책/거시 뉴스 신호와 본문 임베딩을 추가한 `market_news` 모델은 같은 조건에서 나아지는가?
- `market_news` 모델이 상승/하락을 예측하는 구간은 어떤 시장/뉴스 profile을 갖는가?

그래서 최종 산출물은 단순히 모델 파일 하나가 아니라, 학습 metadata, 예측 결과, baseline 비교, aligned comparison, 예측 수익률 profile 리포트까지 함께 남긴다. 성능 숫자만 보는 것보다 “어떤 구간에서 어떤 신호를 보고 예측했는지”를 같이 추적하기 위한 구조다.

## 지금 기준 핵심

- 메인 엔트리포인트는 `shared/run_market_news_training.py`.
- 기본 실행은 `--target-ticker QQQ --ticker-preset auto`와 같다.
- 티커별 자동 프리셋은 `shared/config/ticker_presets.py`에서 관리한다.
- 뉴스 입력 기본 경로는 `data/crawler/features/{ticker}/merged_finbert_with_embeddings.csv`.
- 뉴스 입력에는 `body_summary_embedding` 컬럼이 필요하고, 현재 학습 코드는 이를 `body_emb_0~29`로 펼친 뒤 train split에서만 `StandardScaler + PCA(5)`를 fit한다.
- `market_news` 학습 피처는 "티커 프리셋 시장 피처 + 스칼라 뉴스 피처 12개 + 임베딩 PCA 5개" 구조다.
- 예측 수익률 profile은 KMeans가 아니다. `market_news` 회귀 모델의 예측 미래 가격을 현재가와 비교해 수익률 구간 label을 만들고, label별 시장/뉴스 profile centroid를 요약한다.
- 최신 `market_model_comparison.json`은 `--market-news-only` 실행으로 덮여서 market-only baseline이 빠져 있다. baseline 비교는 현재 `market_model_comparison_aligned.json` 또는 각 `metadata.json`을 같이 봐야 한다.

전체 흐름을 한 줄로 쓰면 아래와 같다.

```text
raw policy/news documents
-> summarized/normalized news table
-> FinBERT sentiment + body summary embedding
-> daily news feature table
-> market feature frame
-> XGBoost market_only / market_news experiments
-> comparison + predicted return profile report
```

중요한 점은 `market_news` 모델이 개별 문서 한 건을 직접 읽는 구조가 아니라는 것이다. 예를 들어 “오늘 FOMC 문서가 있었는가”, “최근 5일 뉴스 수가 늘었는가”, “본문 감성이 최근 며칠 동안 어떤 방향으로 움직였는가”, “본문 임베딩이 어떤 PCA 좌표에 가까운가” 같은 숫자형 신호로 압축한 뒤 시장 가격 피처와 합친다.

## 티커 프리셋

| 대상 | auto preset | 시장 피처 | 매크로/보조 티커 | horizon 후보 | Optuna |
| --- | --- | ---: | --- | --- | ---: |
| QQQ | `qqq_growth_tech` | 34 | SPY, ^VIX, TLT, HYG, UUP, XLK, SOXX, IWM | 5, 7, 10, 15 | 200 |
| XLE | `xle_energy` | 25 | SPY, ^VIX, TLT, HYG, UUP, USO, XOP, OIH, XLB | 3, 5, 10, 20 | 30 |
| XLF | `xlf_financials` | 37 | SPY, ^VIX, TLT, HYG, UUP, KBE, KRE, KIE, IAI | 3, 5, 10, 20 | 30 |
| 기타 | `default` | 20 | SPY, ^VIX, TLT, HYG, UUP | 5, 7, 10, 15 | 200 |

QQQ의 34개 시장 피처는 base 20개, 성장/기술 섹터용 5개, XLK/SOXX/IWM 보충 피처 9개로 구성된다. XLE의 현재 코드상 auto preset은 에너지 섹터 피처 25개를 고정으로 사용하고, XLF는 금융 섹터 피처 37개를 사용한다.

프리셋은 단순히 이름만 바꾸는 옵션이 아니다. `target_ticker`, 매크로 티커 목록, 시장 피처 목록, horizon 후보, Optuna trial 수, random seed를 함께 정해 준다. 따라서 QQQ, XLE, XLF를 같은 CLI로 실행해도 실제로 쓰는 시장 문맥은 다르다.

예를 들어 QQQ는 성장/기술주 문맥을 더 보기 위해 XLK, SOXX, IWM 보충 피처를 붙인다. XLE는 에너지 ETF라서 USO, XOP, OIH, XLB 쪽 피처가 들어가고, XLF는 금융 ETF라서 KBE, KRE, KIE, IAI 쪽 피처가 들어간다. 이런 차이를 `shared/config/ticker_presets.py`에 모아 둔 이유는 새 티커를 추가할 때 실험 설정과 출력 경로가 서로 섞이지 않게 하기 위해서다.

## 실행 흐름

1. `crawler/collectors/`
   FOMC, BIS, UCSB Presidency Project 문서를 수집한다. 정책 문서 쪽은 현재 `ucsb.py`와 `policy_monitor.py`가 담당한다.
2. `crawler/postprocessing/`
   문서 병합, 요약, FinBERT 감성, sentence-transformer 임베딩, PCA 축소를 처리한다. 현재 병합 스크립트는 `preprocessing.py`다.
3. `shared/news/`
   문서 단위 뉴스를 거래일 단위 숫자 피처로 집계하고 시장 프레임에 붙인다.
4. `shared/market/`
   yfinance로 타깃/매크로 가격을 내려받고 시장 피처를 만든다.
5. `shared/training/`
   지도학습 프레임 생성, XGBoost 튜닝, 평가, metadata/predictions/model 저장을 처리한다.
6. `shared/cluster/`
   `market_news` 예측 수익률을 5개 label로 나누고 profile 리포트와 PNG 시각화를 만든다.

## 주요 파일 가이드

처음 코드를 읽을 때는 모든 파일을 한 번에 보려고 하기보다, 실행 진입점부터 아래 순서로 내려가는 편이 편하다.

- `shared/run_market_news_training.py`
  CLI 인자를 config로 바꾸고 전체 파이프라인을 실행한다. `--target-ticker`, `--ticker-preset`, `--market-news-only` 같은 사용자-facing 옵션은 여기서 확인하면 된다.
- `shared/config/ticker_presets.py`
  티커별 시장 피처와 매크로 티커 구성을 관리한다. 현재 QQQ/XLE/XLF 차이가 여기서 갈린다.
- `shared/config/schema.py`
  경로와 학습 기본값을 담은 `MarketNewsTrainingConfig`가 있다. 티커별 출력 경로도 여기서 생성된다.
- `shared/pipelines/market_news.py`
  뉴스 로드, 시장 피처 생성, market-only 학습, market-news 학습, aligned comparison, profile 리포트 생성을 연결하는 오케스트레이션 레이어다.
- `shared/market/data.py`
  yfinance 데이터 다운로드와 수익률/변동성/기술적 지표/상대강도/보충 티커 피처 생성을 담당한다.
- `shared/news/features.py`
  원본 뉴스 CSV를 읽고 `body_summary_embedding`을 `body_emb_*` 컬럼으로 펼친 뒤 날짜별로 집계한다.
- `shared/news/merge.py`
  시장 프레임과 일자별 뉴스 피처를 붙이고, 뉴스 결측/감쇠/rolling 파생 피처를 만든다.
- `shared/training/xgboost_pipeline.py`
  supervised frame 생성, train/test split, Optuna 튜닝, XGBoost 학습, 평가 지표와 predictions 저장을 처리한다.
- `shared/cluster/model.py`, `shared/cluster/visualize.py`
  예측 수익률 label별 profile 데이터셋, 대표 뉴스, feature ranking, PNG 시각화를 만든다.

## 빠른 실행

이미 티커별 뉴스 임베딩 CSV가 준비되어 있다면 프로젝트 루트에서 실행한다.

```bash
python shared/run_market_news_training.py --target-ticker QQQ --ticker-preset auto
python shared/run_market_news_training.py --target-ticker XLE --ticker-preset auto
python shared/run_market_news_training.py --target-ticker XLF --ticker-preset auto
```

market-news 모델만 빠르게 다시 돌릴 때:

```bash
python shared/run_market_news_training.py --target-ticker QQQ --market-news-only
python shared/run_market_news_training.py --target-ticker XLE --market-news-only
python shared/run_market_news_training.py --target-ticker XLF --market-news-only
```

주요 옵션:

- `--target-ticker`: 예측 대상 티커. 예: `QQQ`, `XLE`, `XLF`
- `--ticker-preset`: `auto`, `none`, `default`, `qqq_legacy`, `qqq_growth_tech`, `xle_energy`, `xlf_financials`
- `--news-input`: 기본값 대신 사용할 뉴스 임베딩 CSV
- `--horizons`: 쉼표 구분 horizon 후보
- `--training-embedding-pca-components`: 학습용 `body_emb_*` PCA 차원. 기본값 5
- `--market-news-only`: market-only와 aligned comparison을 건너뛰고 market-news와 cluster 산출물만 갱신

`--market-news-only`는 피처 수정이나 임베딩 처리 수정처럼 `market_news` 쪽만 빠르게 반복 확인할 때 유용하다. 대신 이 모드에서는 market-only baseline과 aligned comparison이 새로 만들어지지 않으므로, 두 모델을 공정하게 비교해야 하는 최종 확인 단계에서는 플래그 없이 다시 실행하는 것이 맞다.

기본 실행이 끝나면 티커별로 대략 아래 파일들이 생긴다.

- `data/training/{ticker}/market_only/metadata.json`
- `data/training/{ticker}/market_only/predictions.csv`
- `data/training/{ticker}/market_news/metadata.json`
- `data/training/{ticker}/market_news/predictions.csv`
- `data/training/{ticker}/comparison/market_model_comparison_aligned.json`
- `data/training/{ticker}/comparison/volatility_cluster_report.json`
- `data/training/{ticker}/comparison/cluster_visualization.png`

## 수집/후처리 실행

개별 수집:

```bash
python crawler/collectors/fed.py
python crawler/collectors/ucsb.py --start-date 2017-01-01
python crawler/collectors/bis.py --max-pages 10
```

하루 단위 모니터링 수집과 통합 후처리:

```bash
python crawler/collectors/policy_monitor.py --max-cycles 1 --interval-sec 0
```

주의: 현재 `policy_monitor.py`의 `run_monitor()` 안에는 테스트용 `target_date_value = datetime(2026, 3, 18).date()` 고정값이 남아 있다. 실제 매일 모니터링으로 쓰려면 `_target_policy_news_date()` 경로로 되돌려야 한다.

레거시 배치 후처리:

```bash
python crawler/postprocessing/preprocessing.py
python crawler/postprocessing/sentiment_score.py
python crawler/postprocessing/sentence_transformer.py
```

추가 의존성:

```bash
pip install selenium transformers sentence-transformers certifi joblib
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`requirements.txt`에는 메인 학습에 필요한 기본 패키지만 들어 있다. BIS 수집은 Selenium/Chrome 환경, 요약은 로컬 Ollama, 감성과 임베딩은 PyTorch/HuggingFace 계열 패키지가 필요하다.

후처리 쪽은 두 갈래가 남아 있다.

- 기존 배치 흐름:
  `preprocessing.py -> sentiment_score.py -> sentence_transformer.py` 순서로 CSV를 단계별로 만든다.
- 통합 모니터링 흐름:
  `policy_monitor.py`가 새 문서를 수집한 뒤 `crawler/postprocessing/unified_pipeline.py`를 호출해 요약, 인코딩, 감성, 임베딩을 한 번에 적용한다.

메인 학습 관점에서 최종적으로 필요한 것은 티커별 `data/crawler/features/{ticker}/merged_finbert_with_embeddings.csv`다. 이 파일에 `body_summary_embedding`이 없거나, 값이 비어 있거나, 행마다 임베딩 차원이 다르면 `shared/news/features.py`에서 바로 오류를 내도록 되어 있다.

## 학습 피처

### 시장 피처

시장 피처는 타깃 ETF 가격 자체의 최근 움직임과 거시 자산 문맥을 같이 담는다.

기본 20개 피처는 `ret_3`, `ret_5`, `ret_accel`, `price_to_ma_5`, `slope_5`, `bb_pos_5`, `bb_width_5`, `macd_hist`, `rsi_14`, `vol_5`, `vol_shock`, `vix_z_score_5`, `drawdown`, `vol_ratio_5`, `rel_strength_5`, `uup_ret_5`, `tlt_shock_5`, `hyg_ret_5`, `target_spy_rel_ret_5`, `target_tlt_rel_ret_5`다.

QQQ auto preset은 여기에 `spy_ret_20`, `vix_speed`, `tlt_ret_20`, `uup_shock_5`, `target_spy_ratio_20`을 더하고, XLK/SOXX/IWM의 `ret_5`, `ret_20`, `shock_5`를 보충 피처로 붙인다.

XLE auto preset은 현재 코드 기준 `ret_1`, `ret_3`, `ret_5`, `ret_accel`, `price_to_ma_5`, `bb_pos_5`, `bb_width_5`, `vol_5`, `vol_10`, `drawdown`, `vol_ratio_5`, `rel_strength_5`, `spy_ret_5`, `vix_ret_5`, `vix_z_score_5`, `hyg_ret_5`, `uup_ret_5`, `target_spy_rel_ret_5`, `uso_ret_5`, `uso_shock_5`, `xop_ret_5`, `xop_shock_5`, `oih_ret_5`, `oih_shock_5`, `xlb_shock_5`를 사용한다.

XLF auto preset은 금융 섹터 문맥을 위해 금리/채권 민감도(`tlt_ret_5`, `tlt_ret_20`, `target_tlt_rel_ret_5`, `target_tlt_ratio_20`), 신용/리스크 문맥(`hyg_ret_5`, `hyg_z_score`, `vix_z_score_5`)과 KBE/KRE/KIE/IAI 보조 ETF의 `ret_5`, `shock_5` 피처를 함께 사용한다.

### 뉴스 피처

스칼라 뉴스 피처 12개:

| 피처 | 의미 |
| --- | --- |
| `news_count_5d` | 최근 5거래일 뉴스 수 합 |
| `days_since_news` | 마지막 뉴스 이후 경과 거래일 수, 최대 30 |
| `sentiment_gap` | 제목 긍정 확률 - 부정 확률 |
| `body_sentiment_gap` | 본문 긍정 확률 - 부정 확률 |
| `sentiment_shock` | 제목 감성 gap의 최근 5일 평균 대비 변화 |
| `body_sentiment_5d_mean` | 본문 감성 5일 평균 |
| `title_sentiment_5d_mean` | 제목 감성 5일 평균 |
| `negative_news_spike_5d` | 본문 부정 확률의 최근 5일 평균 대비 비율 |
| `body_sentiment_decay_3d` | 마지막 실제 뉴스 감성을 3일 반감기로 감쇠 |
| `fomc_sentiment` | 본문 감성 x FOMC 여부 |
| `fomc_recent_5d` | 최근 5일 FOMC 문서 존재 여부 |
| `sentiment_divergence` | 제목 감성과 본문 감성의 절대 차이 |

결측 처리 핵심:

- 일반 뉴스/감성 컬럼은 뉴스가 없는 날 0으로 채운다.
- neutral 확률은 기본값 1.0으로 둔다.
- `days_since_news`는 마지막 뉴스 이후 경과일을 별도 피처로 만든다.
- `body_sentiment_decay_*d`와 `body_emb_*`는 마지막 실제 뉴스를 감쇠해 반영한다.
- 임베딩은 무한정 forward-fill하지 않고 최대 5일까지만 감쇠한다.

### 임베딩 PCA

`merged_finbert_with_embeddings.csv`의 `body_summary_embedding`은 먼저 `body_emb_0~29` 30차원으로 펼쳐진다. 이 raw 임베딩 30개를 그대로 학습 피처에 모두 넣지는 않고, train 구간에서만 `StandardScaler + PCA(5)`를 fit한 뒤 test 구간에는 transform만 적용한다. 이렇게 하는 이유는 test 정보가 PCA 좌표계에 섞이는 것을 막고, market-news 모델의 피처 수를 과하게 늘리지 않기 위해서다.

학습 결과 metadata에는 `embedding_pca.source_columns`, `embedding_pca.feature_columns`, `explained_variance_ratio` 등이 저장된다. 따라서 나중에 모델이 어떤 임베딩 축을 사용했는지 재확인할 수 있다.

## 예측 수익률 Profile

`shared/cluster/model.py` 기준 label은 5개다.

| label | 5거래일 예측 수익률 |
| --- | ---: |
| `fall_strong` | -0.3% 미만 |
| `fall` | -0.3% 이상, 0% 미만 |
| `neutral` | 0% 이상, +0.3% 미만 |
| `rise` | +0.3% 이상, +0.6% 미만 |
| `rise_strong` | +0.6% 이상 |

profile 피처는 기본 시장/뉴스 피처 14개와 설명용 임베딩 PCA 10개, 총 24개다. 학습용 PCA는 5개, profile/대표뉴스용 PCA는 10개라 목적이 다르다.

이 profile 리포트는 별도의 예측 모델을 새로 학습하는 것이 아니다. 이미 만들어진 `market_news` 회귀 모델의 test predictions를 읽고, 각 row의 `Pred_Future_Price / Current_Price - 1`로 예측 수익률을 계산한다. 그 수익률을 위 label로 나눈 뒤, 각 label에 속한 날짜들의 시장/뉴스 피처 평균과 대표 뉴스를 정리한다.

`volatility_cluster_report.json`에서 특히 볼 만한 항목은 아래다.

- `source_model_metrics`: profile을 만든 원본 `market_news` 모델의 평가 지표
- `predicted_groups[].count`: 각 label에 속한 test row 수
- `predicted_groups[].average_predicted_return_pct`: 해당 label의 평균 예측 수익률
- `predicted_groups[].average_actual_return_pct`: 같은 row들의 실제 평균 수익률
- `predicted_groups[].direction_accuracy`: label 방향과 실제 방향이 맞은 비율
- `predicted_groups[].profile_feature_ranking`: 전체 평균 대비 가장 많이 다른 profile 피처
- `predicted_groups[].representative_embedding_news`: label centroid와 임베딩상 가까운 실제 뉴스 문서

대표 뉴스는 해당 test 날짜 근처의 뉴스만 고르는 방식이 아니라, centroid를 원본 임베딩 공간으로 복원한 뒤 전체 source news 중 cosine similarity가 높은 문서를 찾는 방식이다. 그래서 “이 label의 의미를 설명하는 비슷한 문서 예시”로 보는 것이 맞고, 특정 예측일의 직접 원인이라고 해석하면 안 된다.

## 저장된 결과 스냅샷

파일 수정 시각 기준 2026-05-24 산출물이다. `market_news` 결과는 나중에 `--market-news-only`로 다시 실행되어 market-only 비교 파일보다 더 최신이다.

| 티커 | 실험 | preset | 피처 수 | 테스트 구간 | RMSE | 방향성 정확도 |
| --- | --- | --- | ---: | --- | ---: | ---: |
| QQQ | market-only | `qqq_growth_tech` | 34 | 2024-06-26 ~ 2026-04-24 | 14.5613 | 58.61% |
| QQQ | market-news | `qqq_growth_tech` | 51 | 2024-06-13 ~ 2026-04-24 | 14.4487 | 57.82% |
| XLE | market-only | `xle_energy` | 40 | 2024-06-26 ~ 2026-04-24 | 1.5820 | 46.19% |
| XLE | market-news | `xle_energy` | 42 | 2024-06-13 ~ 2026-04-24 | 1.5403 | 49.04% |

XLF는 `data/crawler/features/xlf/merged_finbert_with_embeddings.csv`와 `xlf_financials` 프리셋이 준비된 상태이며, 학습 산출물은 첫 실행 후 `data/training/xlf/` 아래에 생성된다.

해석 주의:

- QQQ market-news는 RMSE가 약간 좋아졌지만 방향성 정확도는 market-only보다 낮다.
- XLE market-news는 저장된 최신 단독 결과 기준 RMSE와 방향성 정확도가 모두 좋아졌다.
- XLE market-only 40피처 결과는 현재 코드의 25피처 `xle_energy` 프리셋보다 이전 산출물이다. 최신 코드와 완전히 맞는 baseline 비교가 필요하면 XLE를 `--market-news-only` 없이 다시 실행해야 한다.

Aligned comparison 스냅샷:

| 티커 | aligned 시작일 | best shared horizon | 방향성 delta | RMSE delta | 비고 |
| --- | --- | ---: | ---: | ---: | --- |
| QQQ | 2017-01-13 | 10 | +0.66%p | +0.1198 | 방향성은 개선, RMSE는 악화 |
| XLE | 2017-01-13 | 5 | +3.70%p | -0.0268 | 이전 XLE 58피처 market-news 기준 |

이 표에서 delta는 `market_news - market_only`다. 방향성 delta는 높을수록 좋고, RMSE delta는 낮을수록 좋다. 예를 들어 QQQ의 aligned best horizon 10은 방향성은 +0.66%p 개선됐지만 RMSE는 +0.1198로 악화됐다. 반대로 XLE의 aligned horizon 5는 방향성과 RMSE가 모두 개선된 결과다. 다만 XLE aligned 비교는 이후 생성된 42피처 market-news 결과보다 이전 산출물이므로 해석할 때 이 점을 같이 봐야 한다.

예측 수익률 profile 스냅샷:

| 티커 | label | count | 평균 예측 수익률 | 평균 실제 수익률 | 방향성 정확도 |
| --- | --- | ---: | ---: | ---: | ---: |
| QQQ | `fall_strong` | 20 | -0.472% | +0.271% | 45.0% |
| QQQ | `fall` | 15 | -0.186% | +0.808% | 46.7% |
| QQQ | `neutral` | 73 | +0.219% | +0.223% | 50.7% |
| QQQ | `rise` | 299 | +0.416% | +0.378% | 60.5% |
| QQQ | `rise_strong` | 60 | +0.867% | +0.649% | 60.0% |
| XLE | `fall_strong` | 47 | -1.391% | +0.277% | 36.2% |
| XLE | `fall` | 151 | -0.129% | +0.811% | 38.4% |
| XLE | `neutral` | 173 | +0.116% | +0.236% | 56.6% |
| XLE | `rise` | 51 | +0.404% | -0.432% | 52.9% |
| XLE | `rise_strong` | 45 | +1.005% | +0.913% | 64.4% |

## 산출물 위치

```text
data/
├─ crawler/
│  └─ features/
│     ├─ qqq/
│     │  ├─ merged_finbert_with_embeddings.csv
│     │  └─ daily_news_features.csv
│     ├─ xle/
│     │  ├─ merged_finbert_with_embeddings.csv
│     │  └─ daily_news_features.csv
│     └─ xlf/
│        ├─ merged_finbert_with_embeddings.csv
│        └─ daily_news_features.csv
└─ training/
   ├─ qqq/
   │  ├─ market_only/
   │  ├─ market_news/
   │  └─ comparison/
   ├─ xle/
   │  ├─ market_only/
   │  ├─ market_news/
   │  └─ comparison/
   └─ xlf/
      ├─ market_only/
      ├─ market_news/
      └─ comparison/
```

각 실험 폴더에는 보통 `training_frame.csv`, `predictions.csv`, `xgboost_model.json`, `metadata.json`이 생성된다. `comparison/`에는 `market_model_comparison*`, `volatility_cluster_*`, `cluster_visualization.png`가 저장된다.

## 결과를 볼 때 기준

성능을 볼 때는 단일 숫자 하나보다 아래 순서로 확인하는 편이 안전하다.

1. 각 모델의 `metadata.json`
   현재 실행의 피처 수, 테스트 구간, horizon, RMSE, 방향성 정확도를 확인한다.
2. `market_model_comparison_aligned.json`
   같은 시작일과 같은 horizon에서 market-only와 market-news를 비교했는지 확인한다.
3. `predictions.csv`
   특정 기간에서 예측이 한쪽 방향으로 치우쳤는지, 고확신 long/short 샘플 수가 너무 적지 않은지 확인한다.
4. `volatility_cluster_report.json`
   모델이 상승/하락 구간을 어떤 profile로 나누고 있는지 확인한다.

특히 `high_conf_short_count`처럼 샘플 수가 매우 작은 지표는 정확도가 높거나 낮아도 그대로 일반화하면 위험하다. QQQ market-only의 short 고확신 정확도는 100%지만 샘플 수가 3개뿐이다.

## 현재 주의점

- `market_news_only` 실행은 `market_model_comparison.json`을 최신 market-news 중심 payload로 덮는다. market-only baseline까지 같은 시점으로 보려면 플래그 없이 다시 실행한다.
- XLE의 저장된 market-only 결과는 현재 코드의 XLE 25피처 프리셋과 완전히 같은 기준이 아니다.
- yfinance와 크롤러는 네트워크 의존성이 크므로 재실행 시 데이터 종료일과 외부 사이트 상태가 달라질 수 있다.
- 결과 CSV는 `.gitignore` 정책상 일부만 추적된다. 티커별 feature CSV는 예외적으로 추적 가능하도록 열려 있다.

아직 남아 있는 개선 후보도 있다.

- XLE를 플래그 없이 다시 실행해 최신 25피처 프리셋 기준의 aligned comparison을 새로 만들기
- `policy_monitor.py`의 테스트용 고정 날짜를 실제 운영용 날짜 계산으로 되돌리기
- `requirements.txt`에 크롤러/후처리 확장 의존성을 별도 extra처럼 정리하기
- profile 대표 뉴스를 전체 source news가 아니라 label/test 기간 근처 뉴스로 제한하는 옵션 추가하기
