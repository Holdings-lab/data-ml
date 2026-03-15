Markdown
# 🚀 Data-ML Pipeline Project

본 레포지토리는 웹 크롤링을 통한 **데이터 수집(Crawler)**부터 수집된 데이터를 활용한 **머신러닝 모델 학습 및 추론(Training)**까지 전체 파이프라인을 관리하는 통합 프로젝트입니다.

---

## 🗂️ 프로젝트 구조 (Directory Structure)

프로젝트는 크게 데이터 수집, 모델 학습, 그리고 두 도메인 간의 공통 영역으로 나뉘어 있습니다. 초기 개발 속도와 가독성을 위해 직관적인 파일 단위 구조를 채택했습니다.

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
🌿 브랜치 전략 (Branching Strategy)
안정적인 파이프라인 운영과 도메인(Crawler, Training) 간의 독립적인 개발을 위해 간소화된 Git Flow 전략을 사용.

📌 브랜치 종류

main: 실제 서비스나 프로덕션 환경에 배포되는 최종 안정화 브랜치. (직접 커밋 불가, PR을 통해서만 병합)

dev: 다음 릴리즈를 위해 각 feature 브랜치들의 기능이 모이는 통합 개발 브랜치.

crawler: 데이터 수집 파이프라인 기능 개발 및 수정 시 사용하는 브랜치.

(커밋메시지 예시: feature/crawler-naver-news, feature/crawler-pipeline-fix)

training: 모델 구조 변경, 전처리, 학습 로직 개선 등 ML 관련 작업 시 사용하는 브랜치

(커밋메시지 예시: feature/training-xgboost, feature/training-loss-tuning)

fix: dev나 main 브랜치에서 발생한 버그를 긴급하게 수정할 때 사용.

(커밋메시지 예시: fix/crawler-parsing-error)

🔄 작업 워크플로우 (Workflow)

기능 개발이 필요하면 최신 dev 브랜치에서 새로운 브랜치를 생성. (dev 브랜치는 최신화 유지)

Bash
git checkout dev
git pull origin dev
git checkout -b feature/crawler-add-scraper
작업 완료 후, 로컬에서 테스트를 마친 뒤 Github(또는 GitLab)에 Push.

dev 브랜치로 Pull Request (PR) 를 생성하고, 팀원간의 코드 리뷰 후 머지. PR 생성후에는 작업상황 카톡으로 공유

📝 커밋 규칙 (Commit Convention)

📌 커밋 메시지 구조

Plaintext
<타입>(<스코프>): <제목>

<본문> (선택 사항 - 변경 이유, 상세 설명 등)
🏷️ 커밋 타입 (Type)

태그	설명
feat: 새로운 기능 추가 (예: 모델 추가, 크롤러 API 연동)
fix:	버그 수정 (예: 파싱 에러 수정, loss 함수 오류 수정)
docs: 문서 추가 및 수정 (예: README.md, 주석 업데이트)
style: 코드 포맷팅, 세미콜론 누락, 오타 수정 (로직 변경 없음)
refactor: 코드 리팩토링 (기능 변화 없이 구조 개선)
test: 테스트 코드 추가 및 수정
chore: 빌드 설정, 패키지 매니저, 환경 설정 수정 (예: requirements.txt)
💡 커밋 메시지 작성 예시

Crawler 도메인 작업 시:

Plaintext
feat(crawler): 네이버 뉴스 데이터 파싱 로직 추가

- BeautifulSoup을 활용하여 뉴스 제목 및 본문 추출 함수 구현
- shared/schema.py의 NewsData 모델 포맷에 맞게 반환되도록 수정
Training 도메인 작업 시:

Plaintext
fix(training): DataLoader 메모리 누수 버그 수정

- num_workers 설정을 4에서 2로 낮추고 pin_memory 옵션 해제
- 대용량 텍스트 학습 시 OOM 발생하는 문제 해결
공통 도메인 작업 시:

Plaintext
chore(shared): PyTorch 및 Pydantic 라이브러리 버전 업데이트
⚠️ 협업 시 주의사항
데이터 계약 (shared/schema.py) 준수: crawler 브랜치 담당자와 training 브랜치 담당자는 주고받는 데이터의 구조가 변경될 경우, 반드시 사전에 협의하여 shared/schema.py를 업데이트.

의존성 분리: 가급적 crawler/ 폴더 내의 코드가 training/ 폴더 내의 코드를 직접 참조하지 않도록 유의. (필요한 경우 shared/ 활용)
