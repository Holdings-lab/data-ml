# 5일 선행 예측 수익률 profile 요약

`shared/cluster/` 패키지는 이름은 아직 cluster를 쓰지만, 현재 기본 동작은 비지도 KMeans도 아니고 별도 변동성 classifier도 아니다.

현재 흐름은 다음과 같다.

1. `market_news` XGBoost 회귀 모델이 test 구간에서 5거래일 뒤 가격을 예측한다.
2. `Pred_Future_Price / Current_Price - 1`로 오늘 대비 예측 수익률을 계산한다.
3. 이 예측 수익률을 4개 상승/하락 구간으로 나눈다.
4. 각 예측 label에 속한 test row의 직전 5일 뉴스/시장 피처 벡터를 묶는다.
5. label별 count, 평균 예측 수익률, 실제 수익률, 방향성 적중률, 평균 피처 profile을 저장한다.

따라서 `data/training/{ticker}/comparison/volatility_cluster_report.json`의 label별 수치는 "실제로 그 label이었던 과거 샘플 평균"이 아니라, **회귀 모델이 그 수익률 구간으로 예측한 test 샘플들의 profile 요약**이다.

## Label

target은 모델이 예측한 5거래일 뒤 가격의 현재가 대비 수익률이다.

| label | predicted 5-day return |
| --- | ---: |
| `fall_strong` | `< -0.3%` |
| `fall` | `-0.3% ~ 0%` |
| `neutral` | `0% ~ +0.3%` |
| `rise` | `+0.3% ~ +0.6%` |
| `rise_strong` | `>= +0.6%` |

## Feature Vector

기본 피처 14개와 임베딩 PCA 피처 10개를 합쳐 총 24개 피처를 사용한다.

- 기본 피처: `CLUSTER_BASE_FEATURE_COLS`
- 임베딩 입력: `body_emb_0~29`
- 임베딩 압축: train 구간에서만 `StandardScaler + PCA(10)` fit, test 구간은 transform만 적용
- 최종 임베딩 피처: `body_emb_cluster_pc1~10`

## Outputs

- `data/training/{ticker}/comparison/volatility_cluster_model.json`
  - `source_model`: 예측 수익률을 만든 `market_news` 회귀 모델 정보
  - `embedding_pca`: 임베딩 PCA 복원 정보
  - `centroids`: test 구간 예측 label별 profile centroid
  - `fixed_thresholds`: 예측 수익률 label 경계값

- `data/training/{ticker}/comparison/volatility_cluster_report.json`
  - `predicted_groups`: 예측 label별 count, 평균 예측 수익률, 실제 수익률, 방향성 적중률, 평균 피처 profile
  - `predicted_groups[].profile_feature_ranking`: label centroid가 전체 평균 대비 가장 크게 다른 상위 profile 피처
  - `predicted_groups[].representative_embedding_news`: label centroid의 PCA 임베딩을 원본 30차원으로 복원한 뒤, cosine similarity가 높은 실제 뉴스 문서
  - `source_model_metrics`: 원래 `market_news` 회귀 모델의 test 평가 지표
  - `fixed_thresholds`: label 경계값

- `data/training/{ticker}/comparison/cluster_visualization.png`
  - test 구간 예측 벡터를 LDA 2D로 우선 투영하고, 불가능하면 PCA 2D로 fallback한다.
  - 예측 label별 profile centroid와 상위 profile 피처를 표시한다.
