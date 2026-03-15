# data-ml

데이터 수집(`crawler`)과 모델 학습(`training`)을 함께 관리하는 저장소입니다.

이 문서는 프로젝트 소개보다 **협업 규칙과 코드 구조 기준**에 집중합니다.
실행 방법이나 모델 상세는 각 파트 문서에서 관리합니다.

- `crawler/README.md`: 수집 대상, 실행 방법, 파서 규칙
- `training/README.md`: 데이터셋 구성, 학습/평가 방법

---

## 1. 운영 원칙

- `crawler`와 `training`은 같은 저장소에서 관리하되 디렉터리를 분리한다.
- 두 파트는 직접 import로 연결하지 않고 `shared/schemas`와 데이터 포맷으로 연결한다.
- 공통 코드는 실제로 양쪽에서 재사용되고 안정적인 경우에만 `shared/`로 올린다.
- 원본 데이터, 대용량 산출물, 모델 가중치, 비밀 정보는 Git에 커밋하지 않는다.
- 장기 운영 브랜치(`crawler`, `training`)를 만들지 않고 기능 단위 브랜치만 사용한다.

---

## 2. 권장 디렉터리 구조

```text
data-ml/
├─ crawler/
│  ├─ clients/      # 외부 요청, API 호출
│  ├─ parsers/      # 응답 파싱(HTML/JSON -> 내부 포맷)
│  ├─ pipelines/    # 수집 흐름 연결, retry/dedup/save
│  ├─ validators/   # 필드/스키마 검증
│  ├─ jobs/         # 실행 entrypoint
│  └─ README.md
│
├─ training/
│  ├─ datasets/     # Dataset/Dataloader/split
│  ├─ preprocess/   # 전처리, feature 생성
│  ├─ models/       # 모델 정의
│  ├─ losses/       # loss 함수
│  ├─ metrics/      # 평가 지표
│  ├─ trainers/     # 학습/검증 루프, checkpoint
│  ├─ inference/    # 추론 로직
│  └─ README.md
│
├─ shared/
│  ├─ schemas/      # crawler <-> training 데이터 계약
│  ├─ utils/        # 범용 유틸
│  ├─ config/       # 공통 설정 로더
│  └─ logging/      # 공통 로깅 설정
│
├─ configs/
├─ scripts/
├─ tests/
│  ├─ crawler/
│  ├─ training/
│  └─ shared/
├─ docs/
├─ .gitignore
└─ README.md
```

---

## 3. 브랜치 전략

### 기본 원칙
- `main`은 항상 안정 상태를 유지한다.
- 모든 작업은 feature 브랜치에서 진행한다.
- `main` 직접 push는 금지하고 PR로만 병합한다.

### 브랜치 이름
```text
feat/crawler-...
feat/training-...
fix/crawler-...
fix/training-...
refactor/shared-...
chore/config-...
```

예시:
```text
feat/crawler-hospital-detail-parser
feat/training-add-kfold-loader
fix/shared-schema-date-format
```

### 금지
- 파트 전용 장기 브랜치 운영
- unrelated 변경을 한 브랜치에 누적

---

## 4. 커밋 규칙

커밋 메시지 형식:

```text
type(scope): summary
```

예시:
```text
feat(crawler): 병원 상세 페이지 파서 추가
fix(training): validation split 누수 수정
refactor(shared): 날짜 변환 유틸 분리
docs(repo): 브랜치 규칙 정리
```

### type
- `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`, `data`, `exp`

`data`는 원본 데이터 업로드 용도가 아니라 샘플/fixture/스키마 예시에만 사용한다.

### 좋은 커밋 기준
- 하나의 논리적 변경만 담는다.
- 제목만 봐도 의도가 드러난다.
- 되돌리기 쉽다.
- 포맷팅/rename/로직 변경은 가능하면 분리한다.

### 커밋 전 확인
- 디버그 코드와 임시 주석 제거
- 사용하지 않는 import 제거
- 로컬 절대경로 제거
- 비밀 정보 포함 여부 확인
- raw data/weight/cache 포함 여부 확인

---

## 5. 코드 분리 기준

- 한 파일은 한 책임만 가진다.
- I/O와 순수 로직을 분리한다.
- 실행 entrypoint와 재사용 로직을 분리한다.
- `crawler`와 `training`은 서로 직접 의존하지 않는다.

### 같이 두지 않는 조합
- 요청 코드 + 파싱 로직
- 파싱 로직 + DB 저장 로직
- 데이터 로딩 + 모델 정의
- 모델 정의 + 학습 루프
- 학습 루프 + 지표 계산 + 시각화

### `shared/` 이동 기준
아래 3가지 모두 만족할 때만 이동한다.
1. `crawler`, `training` 양쪽에서 실제 사용
2. 특정 파트 문맥에 종속되지 않음
3. 규칙이 안정적이고 장기 유지 가능

---

## 6. 노트북/설정/테스트 규칙

### 노트북
- 탐색/실험/시각화 용도로만 사용
- 운영 로직은 `.py`로 이동
- 커밋 전 output 정리

### 설정
- 하이퍼파라미터/경로/토글은 `configs/`로 분리
- 비밀값은 `.env` 또는 비공개 설정으로 관리

### 테스트
- 테스트 구조는 실제 코드 구조를 따른다.
- 새 모듈 추가 시 대응 테스트 파일을 함께 작성한다.

예시:
```text
crawler/parsers/review_parser.py
-> tests/crawler/test_review_parser.py
```

---

## 7. 데이터/산출물 관리

### 커밋 금지
- raw 데이터
- 대용량 processed 데이터
- 모델 가중치(`.pt`, `.bin`, `.ckpt`, `.onnx`)
- cache, 로그 덤프
- API key, 쿠키, 토큰, 계정 정보

### 커밋 허용
- 작은 샘플 데이터
- fixture
- 스키마 예시 JSON
- 재현용 최소 설정 예시

---

## 8. PR 규칙

- PR 하나는 하나의 목적만 담는다.
- 변경 목적/범위/스키마 영향/테스트 방법/리뷰 포인트를 본문에 적는다.
- 병합 전 아래 항목을 확인한다.

체크리스트:
- [ ] `main` 기준 충돌 해결
- [ ] 불필요 파일 제거
- [ ] raw data/weight 미포함 확인
- [ ] 테스트 또는 수동 검증 완료
- [ ] 문서 업데이트 필요 시 반영

---

## 9. 파트 간 의존 규칙

허용:
- `crawler -> shared`
- `training -> shared`

금지:
- `crawler -> training`
- `training -> crawler`

두 파트는 공통 스키마와 저장 포맷으로만 연결한다.

---

## 10. 권장 작업 흐름

1. 작업 목적 정리(이슈 또는 메모)
2. feature 브랜치 생성
3. 작은 단위로 커밋
4. PR 생성 및 리뷰 반영
5. `main` 병합
6. 관련 문서 업데이트

예시:

```bash
git checkout -b feat/crawler-review-parser
# 작업 및 테스트

git add .
git commit -m "feat(crawler): 리뷰 목록 parser 추가"
git commit -m "test(crawler): parser 샘플 응답 테스트 추가"
git push origin feat/crawler-review-parser
```

---

## 11. 최소 규칙 요약

1. `main` 직접 push 금지
2. 장기 브랜치 운영 금지
3. 커밋 하나에 하나의 논리만 담기
4. `crawler` / `training` / `shared` 책임 섞지 않기
5. raw data / weight / secret 커밋 금지
6. notebook에 핵심 로직 고정 금지

---

## 12. 한 줄 원칙

**같이 변해야 하는 코드는 가깝게 두고, 다른 이유로 변하는 코드는 분리한다.**
