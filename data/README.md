# Data Layout

최종 점검일: 2026-05-26. 이 문서는 현재 `data/` 폴더의 티커별 산출물과 저장된 metadata 기준 결과를 요약한다.

`data/`는 크게 두 종류의 파일을 담는다. `data/crawler/`는 학습 전에 만들어지는 뉴스 입력과 후처리 산출물이고, `data/training/`은 XGBoost 학습 이후 만들어지는 모델/예측/비교 산출물이다. 파일이 많아질수록 이 둘을 섞지 않는 것이 중요하다.

## 폴더 구조

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
│     ├─ xlf/
│     │  ├─ merged_finbert_with_embeddings.csv
│     │  └─ daily_news_features.csv
│     ├─ merged_finbert_pca.pkl
│     └─ policy_updates_pca.pkl
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

## `data/crawler`

- `features/{ticker}/merged_finbert_with_embeddings.csv`: 메인 학습 입력 뉴스 CSV. `date`, `category`, `doc_type`, `title`, `body`, 감성 컬럼, `body_summary_embedding`이 필요하다.
- `features/{ticker}/daily_news_features.csv`: `shared/news/features.py`가 만든 일 단위 뉴스 피처.
- `features/*.pkl`: 임베딩 PCA/후처리 모델 산출물.

현재 메인 코드는 티커별 nested 경로를 우선 사용한다. 예: `data/crawler/features/qqq/merged_finbert_with_embeddings.csv`.

티커별 `merged_finbert_with_embeddings.csv`는 학습 파이프라인의 출발점이다. 이 파일이 준비되어 있으면 크롤러를 다시 돌리지 않고도 `shared/run_market_news_training.py`만 실행해서 모델을 재학습할 수 있다. 반대로 이 파일의 임베딩 컬럼이 비어 있거나 차원이 맞지 않으면, 학습 이전 단계에서 바로 실패한다.

## `data/training`

각 티커는 같은 구조를 가진다.

- `{ticker}/market_only/`: 시장 피처만 사용한 XGBoost 결과.
- `{ticker}/market_news/`: 시장 + 뉴스 + 임베딩 PCA 피처를 사용한 XGBoost 결과.
- `{ticker}/comparison/`: 비교 JSON/CSV, 예측 수익률 profile 모델, 리포트, 시각화.

주요 파일:

- `metadata.json`: 설정, 선택 피처, train/test 구간, 평가 지표.
- `predictions.csv`: 테스트 구간 예측 테이블.
- `training_frame.csv`: 지도학습 프레임.
- `xgboost_model.json`: 저장된 XGBoost 모델.
- `market_model_comparison_aligned.json`: 같은 기간/같은 horizon 기준 비교.
- `volatility_cluster_report.json`: 예측 수익률 label별 profile 요약.

결과를 확인할 때는 보통 `metadata.json`을 먼저 본다. 이 파일에는 실행 시점의 config, 선택된 피처, train/test 기간, RMSE, 방향성 정확도, 고확신 구간 지표가 들어 있다. 그 다음 두 모델 비교가 필요하면 `comparison/market_model_comparison_aligned.json`을 보고, 예측 구간의 특징을 보고 싶으면 `comparison/volatility_cluster_report.json`을 본다.

## 최신 저장 결과

파일 수정 시각 기준 2026-05-24 산출물이다.

| 티커 | 실험 | metadata 시각 | preset | 피처 수 | 테스트 구간 | RMSE | 방향성 정확도 |
| --- | --- | --- | --- | ---: | --- | ---: | ---: |
| QQQ | market-only | 2026-05-24 11:31 | `qqq_growth_tech` | 34 | 2024-06-26 ~ 2026-04-24 | 14.5613 | 58.61% |
| QQQ | market-news | 2026-05-24 17:16 | `qqq_growth_tech` | 51 | 2024-06-13 ~ 2026-04-24 | 14.4487 | 57.82% |
| XLE | market-only | 2026-05-24 12:05 | `xle_energy` | 40 | 2024-06-26 ~ 2026-04-24 | 1.5820 | 46.19% |
| XLE | market-news | 2026-05-24 17:26 | `xle_energy` | 42 | 2024-06-13 ~ 2026-04-24 | 1.5403 | 49.04% |

XLF는 입력 CSV가 `data/crawler/features/xlf/merged_finbert_with_embeddings.csv`에 추가되어 있고, 학습 결과는 아직 저장되지 않았다. `shared/run_market_news_training.py --target-ticker XLF --ticker-preset auto` 실행 후 이 표에 같은 형식으로 추가하면 된다.

고확신 구간:

| 티커 | 실험 | Long 정확도 | Long n | Short 정확도 | Short n |
| --- | --- | ---: | ---: | ---: | ---: |
| QQQ | market-only | 66.67% | 135 | 100.00% | 3 |
| QQQ | market-news | 65.38% | 130 | 50.00% | 10 |
| XLE | market-only | 56.78% | 118 | 5.00% | 20 |
| XLE | market-news | 59.14% | 93 | 36.17% | 47 |

## Aligned Comparison

`market_model_comparison_aligned.json`은 같은 시작일과 shared horizon으로 market-only와 market-news를 비교한다.

| 티커 | aligned 시작일 | best shared horizon | 방향성 delta | RMSE delta | 참고 |
| --- | --- | ---: | ---: | ---: | --- |
| QQQ | 2017-01-13 | 10 | +0.66%p | +0.1198 | QQQ 최신 피처 수와 일치 |
| XLE | 2017-01-13 | 5 | +3.70%p | -0.0268 | 이전 XLE 58피처 market-news 기준 |

XLE는 이후 `--market-news-only`로 42피처 market-news 결과가 다시 생성되었다. 따라서 XLE의 완전한 최신 fair comparison이 필요하면 `--market-news-only` 없이 다시 실행해야 한다.

## 예측 수익률 Profile

label 기준:

| label | 예측 수익률 구간 |
| --- | ---: |
| `fall_strong` | -0.3% 미만 |
| `fall` | -0.3% 이상, 0% 미만 |
| `neutral` | 0% 이상, +0.3% 미만 |
| `rise` | +0.3% 이상, +0.6% 미만 |
| `rise_strong` | +0.6% 이상 |

저장된 `volatility_cluster_report.json` 요약:

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

## 일관성 메모

- `market_model_comparison.json`은 가장 최근 `--market-news-only` 실행으로 market-news와 cluster 중심 payload가 저장되어 있다.
- baseline 비교는 `market_model_comparison_aligned.json` 또는 각 `market_only/metadata.json`, `market_news/metadata.json`을 함께 확인한다.
- 현재 코드 기준 XLE auto preset은 25개 시장 피처다. 저장된 XLE market-only 40피처 metadata는 이전 프리셋/결과이므로 최신 코드와 1:1 비교하려면 재실행이 필요하다.
- 티커별 폴더 구조를 유지하면 QQQ, XLE, XLF 산출물이 서로 덮어써지지 않는다.

결과를 갱신할 때는 `--market-news-only` 여부를 꼭 확인한다. 이 플래그를 쓰면 빠르게 market-news와 profile 산출물을 갱신할 수 있지만, market-only baseline과 aligned comparison은 같이 갱신되지 않는다. 최종 비교표를 만들 때는 플래그 없이 실행하는 편이 안전하다.
