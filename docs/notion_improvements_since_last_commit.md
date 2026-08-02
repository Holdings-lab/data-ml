# 마지막 커밋 이후 개선 사항

기준 커밋: `8eade17 feat(crawler): policy monitoring pipeline 개선 및 후처리 임베딩 PCA 적용`

## 요약

이번 변경은 뉴스 임베딩을 단순히 학습 피처로 넣는 단계에서 한 단계 더 나아가, `market_news` 회귀 모델의 5거래일 뒤 예측 결과를 설명 가능한 profile 리포트로 정리하는 방향이다.

## 주요 변경점

- `market_news` 학습 피처를 정리함.
  - 시장 피처 20개 + 스칼라 뉴스 피처 12개 + 임베딩 PCA 5개 구조로 고정.
  - `body_emb_0~29`는 train 구간에서만 `StandardScaler + PCA(5)`를 fit하고, test 구간은 transform만 적용.
  - CLI 옵션 `--training-embedding-pca-components`로 학습용 PCA 차원을 조절할 수 있게 함.

- 클러스터링 의미를 재정의함.
  - 기존처럼 실제 미래 수익률을 기준으로 비지도 군집화하는 방식이 아님.
  - `market_news` 모델이 예측한 `Pred_Future_Price`를 `Current_Price`와 비교해 예측 수익률을 계산.
  - 예측 수익률을 `fall`, `neutral`, `rise`, `rise_strong` 4개 구간으로 나눠 test 기간의 profile을 요약.

- 클러스터 profile 피처를 정리함.
  - 기본 시장/뉴스 피처 14개 사용.
  - 설명/클러스터용 임베딩은 별도 `PCA(10)` 유지.
  - 최종 profile 벡터는 14개 기본 피처 + `body_emb_cluster_pc1~10` = 24개.

- `cluster feature ranking`을 추가함.
  - 각 label centroid가 전체 평균 대비 얼마나 다른지 `z_diff`로 계산.
  - label별 상위 10개 피처를 `profile_feature_ranking`으로 `qqq_volatility_cluster_report.json`에 저장.
  - PNG 시각화 오른쪽에도 label별 상위 피처 ranking을 표시.

- label별 대표 뉴스를 추가함.
  - label centroid의 `body_emb_cluster_pc*` 값을 원본 30차원 `body_emb_*` 공간으로 복원.
  - `data/crawler/features/merged_finbert_with_embeddings.csv`의 뉴스 임베딩과 cosine similarity를 계산.
  - label별 가장 가까운 뉴스 top 5를 `representative_embedding_news`로 리포트에 저장.

- 클러스터 시각화를 개선함.
  - label 분리가 더 잘 보이도록 LDA 2D projection을 우선 사용.
  - LDA 조건이 맞지 않으면 PCA 2D projection으로 fallback.
  - centroid heatmap과 feature ranking 패널을 함께 저장.

- 출력 리포트 구조를 확장함.
  - `qqq_volatility_cluster_model.json`: profile centroid, scaler, PCA 복원 정보, source model 정보 저장.
  - `qqq_volatility_cluster_report.json`: label별 예측 수익률 요약, 실제 수익률, 방향성 적중률, 피처 평균, feature ranking, 대표 뉴스 저장.
  - `qqq_market_model_comparison.json`에도 예측 수익률 profile 리포트를 함께 포함.

## 해석할 때 주의할 점

- 여기서 말하는 cluster는 KMeans 결과가 아니라, 회귀 모델의 예측 수익률 구간별 profile 요약이다.
- 대표 뉴스는 현재 전체 원본 뉴스 파일에서 cosine similarity로 찾는다. 따라서 반드시 해당 test 날짜 근처 뉴스라는 뜻은 아니다.
- 학습용 PCA는 5차원이고, 설명/클러스터용 PCA는 10차원이다. 두 PCA는 목적이 다르다.

## 다음 개선 후보

- 대표 뉴스를 전체 뉴스가 아니라 해당 label의 test 기간 또는 주변 날짜로 제한해 비교.
- 학습용 PCA feature 이름과 클러스터용 PCA feature 이름을 분리해 리포트 가독성 개선.
- label threshold를 고정값으로 둘지, 예측 분포 기반 분위수로 둘지 비교.
- `profile_feature_ranking`과 대표 뉴스를 LLM 설명 입력 포맷으로 바로 변환하는 별도 요약 파일 생성.
