"""
shared 패키지.

크롤러와 학습 코드를 직접 뒤섞기보다는, 두 영역을 이어주는 공통 브리지 로직을
여기에 배치해서 실행 흐름을 한눈에 볼 수 있도록 한다.

주요 모듈 가이드:
- `shared.pipelines.market_news`: shared 파이프라인의 최상위 실행 흐름
- `shared.market.data`: 시장 가격 다운로드와 가격 피처 생성
- `shared.news.features`: 뉴스 원본 로드와 일자별 집계
- `shared.news.merge`: 뉴스 피처를 시장 프레임에 병합
- `shared.training.xgboost_pipeline`: XGBoost 실험, 평가, 비교 결과 생성
"""
