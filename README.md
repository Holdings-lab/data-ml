# 🚀 Data-ML Pipeline Project

본 레포지토리는 웹 크롤링을 통한 **데이터 수집(Crawler)** 부터 수집된 데이터를 활용한 **머신러닝 모델 학습 및 추론(Training)** 까지 전체 파이프라인을 관리하는 통합 프로젝트입니다.

## 🗂️ 프로젝트 구조 (Directory Structure)

프로젝트는 크게 데이터 수집, 모델 학습, 그리고 두 도메인 간의 공통 영역으로 나뉘어 있습니다. 
초기 개발 속도와 가독성을 위해 직관적인 파일 단위 구조를 채택했습니다.

```text
data-ml/
├─ crawler/             # 🌿 [데이터 수집 로직]
│  ├─ scraper.py        # 외부 API 호출 및 응답 파싱
│  ├─ pipeline.py       # 데이터 검증, 중복제거 및 저장
│  └─ run_crawler.py    # 크롤러 실행 엔트리포인트
│
├─ training/            # 🌿 [모델 학습 및 추론 로직]
│  ├─ dataset.py        # 데이터 전처리 및 Dataset/DataLoader 구성
│  ├─ model.py          # 모델 아키텍처 정의 및 Loss 함수
│  ├─ train.py          # 모델 학습 루프 및 평가 지표 계산
│  └─ inference.py      # 학습된 모델 기반 추론 로직
│
├─ shared/              # 🌿 [공통 사용 영역]
│  ├─ schema.py         # ⚠️ Crawler와 Training 간의 데이터 포맷 약속 (매우 중요)
│  └─ utils.py          # 공통 설정 로더, 로깅, 범용 유틸
│
├─ configs/             # 환경 설정 파일 (.yaml, .json 등)
├─ tests/               # 단위 테스트 및 통합 테스트 코드
└─ README.md
```
