# validating v2.5

## v2.5 보충 검색 제거·형식 조건 완화 (2026-11-19)

이 절은 아래 이전 버전 설명보다 우선한다.

- 통합 파이프라인에서 보충 검색 단계를 제거했다. 근거 부족 시에도 최초 주장별 검색 결과만 사용하며, 추가 검색어 확장·재수집·재선별을 실행하지 않는다. 기본 최대 주장 1개에 대해 Brave 검색 요청은 정상 응답 기준 1회다. 통신 오류에 대한 HTTP 재시도는 별개다.
- `supplemental_search`의 기본값은 false이며 true 지정은 API 호출 전에 거부한다. 기존 CLI의 `--no-supplemental-search`는 호환용으로 유지하며, 지정 여부와 관계없이 보충 검색은 실행하지 않는다.
- 모델이 checks 일부를 누락하면 해당 항목을 unresolved로 보완한다. 동일한 중복은 합치고, 상충하는 중복은 unresolved로 처리한다. 알 수 없는 항목 ID는 제외한다. 변경 전 checks와 처리 사유를 결과에 보존하며 이 보정에 추가 API 호출은 없다.
- 비핵심 항목 누락만으로 중단하거나 지지 판정을 보류하지 않는다. 핵심 항목 미확인, 실제 수치 차이, 근거 없는 인용은 여전히 통과하지 않는다.
- 단위의 단복수와 텍스트 항목의 동등한 숫자 표기(1,000 / 1000.0)를 허용한다. 텍스트는 모델의 의미 동등성 판단과 실제 인용 검사를 함께 거친다.
- 검색 20건, 기간 90일, 관련성 하한 0, 최대 근거 6개를 유지한다. 통과율 50%는 목표이며 달성을 보장하지 않는다.

## v2.4 추가 완화와 통과율 목표 (2026-11-19)

이 절은 아래 이전 설정 설명보다 우선한다. 목표는 검증한 주장 중 `supported` 비율 50%이며, 목표값은 판정을 바꾸는 임계값이 아니다.

- 통합 실행의 관련성 하한을 0.15에서 0으로 낮추고, 발행일 이전 검색 범위를 90일로 늘렸다. 검색 종료일은 발행일을 유지한다(`as_of`).
- 최종 근거가 설정된 최대 개수(기본 6개)에 못 미치면 보충 검색을 한 번 수행한다. 검색 결과 수 20건과 최종 근거 최대 6개는 유지한다.
- 인용문의 괄호·통화 기호 등 기호 주변 공백 차이를 허용한다. 단어·숫자·부호의 순서나 내용은 변경할 수 없다.
- `run_summary.json`과 라벨 기반 평가 결과에 `supported_rate`, `target_supported_rate`, `target_met`를 추가했다. 실행 요약의 분모는 해당 실행의 주장 수다. 주장이 없으면 실행 요약의 통과율과 목표 달성 여부는 null이다. 전체 기사 배치 통과율이나 정확도를 뜻하지 않는다.
- 13건을 대상으로 50% 이상을 달성하려면 최소 7건이 통과해야 한다. 이전 제한적 오프라인 시뮬레이션은 3/13이며, 이번 검색 설정의 성능 측정값은 아니다.
- 이번 변경 시 Anthropic 인증은 401로 실패하여 실검증을 재실행하지 못했다. 따라서 50% 달성은 아직 확인되지 않았다. 검색 확대에 따른 처리 시간·API 비용 증가가 가능하다.

## v2.3 완화 설정 (2026-11-19)

이 절의 설정은 아래 과거 버전 설명보다 우선한다.

- 통합 파이프라인 기본 검색 결과 수는 20건, SBERT 관련성 하한은 0.15, 최종 근거는 최대 6개다. `--count 10 --min-similarity 0.3 --max-evidence 3`으로 이전 검색·선별 설정을 사용할 수 있다.
- 숫자 관측값의 `19%`, `$2.10 billion` 같은 표기를 허용하고, `%`/`percent`, `USD`/`dollars`, 단위 단어 순서를 정규화한다. 수치 자체는 일치해야 하며 million/billion 차이나 실제 통화 차이는 허용하지 않는다. 인용문에서 해당 숫자에 연결된 단위를 확인한다.
- 주장 추출은 하나의 핵심 사실에 집중하도록 하고, 해당 주장 밖의 부가 정보를 검증 요구사항으로 추가하지 않도록 했다. 핵심 결론을 바꾸지 않는 세부 정보는 비핵심으로 처리한다.
- 직접 근거 한 개로도 핵심 사실이 확인되면 지지 판정이 가능함을 명시했다. 실제 인용 부재, 핵심 사실 미확인, 수치 불일치는 여전히 근거 부족으로 처리한다.
- 검색·근거 확대는 처리 시간과 API 입력량을 늘릴 수 있다. 새 임계값은 실뉴스 정답 데이터로 보정한 값이 아니다.
- 기존 검증 산출물은 수정하지 않는다. 저장 근거를 이용한 제한적 오프라인 비교는 `docs/validation_relaxation_replay_2026-11-19.json`에 있으며, 검색·모델 재실행 결과가 아니다.

## 정책 모니터 연결

`crawler/collectors/policy_monitor.py`는 통합 후처리의 요약 직후 `validation_pipeline.validate_yahoo_summaries`를 호출한다. 실행 순서는 **수집 → 요약 → Yahoo 검증 → 인코딩 → 감성분석 → 임베딩 → 저장**이다. 이 설명은 아래의 과거 'policy_monitor 미연결' 설명보다 우선한다.

- 대상은 `source=YAHOO`인 행이다. Yahoo 보조 수집기 결과도 동일 태그이므로 포함된다. 다른 출처는 검증 API를 호출하지 않는다.
- 검증 입력은 `body_summary`이고 재배포 제외에는 요약 전 `body`와 원 URL을 사용한다. 발행일은 원 행의 `release_date`를 전달한다.
- 현재 validation 기본값(최대 주장 1개, 검색 10건, as_of, 보충 검색 사용)을 따른다.
- 상세 결과는 `data/validation_monitor/cycle_*/article_*/pipeline/`에 저장한다. 원 요약문은 해당 기사 폴더의 `summary.txt`에 보존한다.
- 모니터 결과에 `validation_status`, `validation_verdict`, `validation_claim_count`, `validation_result_path`, `validation_error`가 추가된다. 비대상은 not_applicable, 검증 성공은 completed, API·입력 오류는 failed다. 주장이 없으면 verdict는 no_claims로 기록하며 supported로 취급하지 않는다.
- 실패·반박·근거 부족 기사를 자동 삭제하지 않는다. 기사별 오류를 기록하고 다음 기사 및 후속 후처리를 계속한다. 필요 환경변수는 ANTHROPIC_API_KEY, BRAVE_SEARCH_API_KEY다.
- 공통 `apply_unified_pipeline`은 선택적 `post_summary_hook`을 제공한다. 모니터 이외 호출자는 훅을 지정하지 않으면 기존 흐름을 유지한다.

## v2.2 변경 사항 (아래 v2 설명보다 우선)

- 텍스트 항목의 완전한 문자열 일치 요구를 완화했다. 모델이 `relation`(exact/equivalent/different/unresolved)과 `comparison_reason`을 반환하며, 실제 근거 인용이 있으면 약칭·정식명칭·괄호 설명·동의 표현의 의미 동등성을 인정한다. 수치·단위 불일치, 인용 부재 및 명시적인 사실 차이는 계속 보류한다. 인용의 따옴표 모양과 보이지 않는 서식 문자는 정규화한다.
- 모델의 동등성 판단에 의존하는 부분이 늘었으므로, 텍스트 판정의 의미적 정확성을 코드가 보증하는 것은 아니다. 구형 응답은 어휘 기반 비교를 사용한다.
- 출처 그룹화에 본문 포함률을 추가했다. 40개 이상의 5단어 shingles에서 Jaccard 0.9 또는 포함률 0.9를 사용한다. 재배포 출처 표시가 일치하면 포함률 0.65를 적용한다. 출처 표기만 같고 본문이 다르면 묶지 않는다.
- 통합 실행은 원 기사 본문도 중복 비교에 전달하여 원문 재배포를 제외한다. 직접 함수 호출 시 `source_url`과 `source_text`를 함께 지정해야 한다.
- 구조화 추출의 형식 오류는 원 응답과 오류를 모델에 전달해 최대 1회 교정 요청한다. 코드는 숫자와 단위를 임의로 분리하거나 값을 추정하지 않는다. 재검증에 실패하면 중단한다. `facts.json`에 교정 횟수와 총 토큰 사용량을 기록한다.

뉴스 본문 → 구조화 주장 추출 → 기간별 Brave 검색 → 근거 수집 → 출처 그룹별 선별 → 항목별 Claude 검증 및 코드 일관성 검사.

## 설치 및 실행

프로젝트 requirements.txt와 함께 다음 추가 의존성을 설치한다.

```powershell
.\venv\Scripts\python.exe -m pip install -r crawler/validating/requirements.txt
```

ANTHROPIC_API_KEY와 BRAVE_SEARCH_API_KEY가 설정되어 있어야 한다. 모델 최초 사용 시 다운로드가 필요할 수 있다.

```powershell
.\venv\Scripts\python.exe -m crawler.validating.validation_pipeline `
  --input C:\path\article.txt `
  --published-date 2026-11-09 `
  --source-url https://publisher.example/article `
  --max-facts 3
```

기본 결과는 data/validation 아래 새 폴더에 저장된다. 운영 CSV를 읽거나 쓰지 않는다. 지정 output-dir에 기존 단계 파일 또는 run_summary.json이 있으면 중단한다.

## 우선순위별 변경

1. **상태 기록**: run_summary.json에 실행 상태, 단계별 시간·오류, 설정·버전, API 토큰 사용량, 주장별 URL 상태 및 최종 근거 그룹 수를 기록한다. 단계 실패 시 이전 JSON을 보존한다. HTTP GET은 일시적 오류에 최대 2회 재시도한다. 개별 페이지 실패는 해당 URL 상태에 남긴다.
2. **구조화 검증**: 추출 단계의 structure에 subject/predicate/object/reference_period/claim_type와 requirements를 저장한다. 각 requirement는 고유 ID, 확인할 항목, 기대값, 단위, 중요도, number/text 구분을 갖는다. 검증 단계는 각 항목의 관측값, 판정, 근거 ID와 실제 인용을 반환한다. quality.py가 인용 존재·수치·단위·텍스트 불일치와 핵심 미확인을 검사한다. 모순이 있으면 insufficient_evidence로 낮추고 model_verdict와 이유를 남긴다. 선택적 항목 오류는 한계로 기록한다. 판정을 자동으로 상향하지 않는다.
3. **출처 중복 제거**: 추적 파라미터·fragment 제거, 최종 리다이렉트 URL 반영, --source-url 제외, 본문 해시와 5단어 shingle Jaccard 유사도(0.9)로 재배포 후보 그룹화. 서로 다른 기사 식별용 query parameter는 유지한다. 정확 중복 외 유사 본문 비교는 최소 40 shingles에만 적용한다.
4. **수집 및 선별**: article/main 우선, 메뉴·광고 후보 태그 제거, 제목·문단·목록·표의 행을 후보로 사용한다. 키워드 미일치 문단도 의미 검색 후보로 남긴다. HTML 문맥은 제목과 인접 블록으로 제한하며 최대 6,000자다. 문서당 후보 최대 200개, 응답 최대 5MB. PDF는 페이지 위치를 보존하고 최대 100페이지까지 텍스트 추출한다. 스캔·암호화 PDF는 명시적 오류로 기록한다. --render-dynamic을 지정하면 본문 후보가 없는 HTML만 Playwright로 재조회한다.
5. **평가**: evaluate.py로 ID가 일치하는 라벨·판정 파일을 비교한다. 정확도, supported precision, 잘못된 supported 건수·전체 대비 비율, 근거 부족률, 혼동행렬을 출력한다.

## 검색·선별 옵션

| 옵션 | 의미 |
| --- | --- |
| --search-mode as_of | 기본. 기사 발행일 이전 한 달부터 발행일까지 |
| --search-mode current | 기사 발행일 이전 한 달부터 한국 시간 오늘까지. 후속 정정 검색 가능 |
| --lookback-days N | 기사 발행일 이전 범위를 N일로 지정 |
| --no-supplemental-search | 부족한 주장에 대한 추가 검색 비활성화 |
| --min-similarity 0.3 | SBERT 후보 하한. 초기 운영값이며 아직 실뉴스 데이터로 보정하지 않음 |
| --max-evidence 3 | 주장당 최종 근거 그룹 한도, 1~6 |
| --reranker-model MODEL | 선택적 CrossEncoder 재선별. SBERT 상위 30개까지 문맥과 함께 평가 |
| --render-dynamic | 본문 후보가 없는 HTML의 브라우저 재조회 |

최종 근거가 2개 미만인 주장만 1회 보충 검색한다. 기본 검색어 및 official announcement source / correction clarification 조합을 사용한다. 주장당 Brave 요청은 기본 1회, 보충 시 3회 추가로 최대 4회다. 기존 시도와 보충 결과를 모두 보존한다. 공식 출처를 검색하는 것은 실제 공식 출처임을 보증하지 않는다. 한 개의 직접적인 원문도 판정에 사용될 수 있다.

## 산출물

- facts.json / search_results.json / article_sentences.json / top_sentences.json / verified_claims.json
- run_summary.json: 실행 및 수집 품질 상태, 경고, 설정, 소요 시간, API 토큰 사용량
- 보충 시 supplemental_search.json / supplemental_articles.json / combined_articles.json / supplemental_top_sentences.json

보충 검색을 했다면 verified_claims.json은 supplemental_top_sentences.json을 사용한다. 최초 top_sentences.json을 덮어쓰지 않는다. 근거에는 URL·날짜·수집 시각·본문 해시·추출 방식·위치가 가능한 범위에서 보존된다. 검색 날짜는 검색 서비스가 제공하는 발행/수정일에 의존한다.

## 호환성

통합 실행과 추출·수집 CLI는 강화 경로를 기본 사용한다. 기존 함수 호출자는 extract_news_facts(structured=True), extract_article_paragraphs(enhanced=True)를 명시한다. 기본 Python 함수 계약은 기존 호출자를 위해 유지한다. 구형 JSON은 읽을 수 있으나 structure가 없는 주장은 새 항목별 검사를 적용할 수 없다. 추출/수집 CLI의 --legacy는 이전 동작을 사용한다.

## 한계와 운영 조건

- 코드의 일관성 검사도 모델이 추출한 요구사항과 항목 판단에 의존한다. 중요한 요구사항 자체를 모델이 누락하는 경우까지 보장하지 않는다.
- 텍스트 확인은 보수적으로 대소문자·공백 정규화 후 일치하는 관측값과 실제 인용을 요구한다. 별칭·번역·의역 때문에 근거 부족이 늘 수 있다. 자동 환율·단위 변환이나 임의 수치 허용 오차는 적용하지 않는다.
- 본문 유사도는 재배포 후보를 묶는 휴리스틱이며 편집 독립성 또는 공신력을 인증하지 않는다. 원본 URL은 호출자가 제공해야 정확히 제외할 수 있다.
- SBERT는 토큰 한도 때문에 긴 문단을 잘라 읽을 수 있다. CrossEncoder는 선택 기능이며 실제 모델의 점수 의미에 따라 별도 평가가 필요하다.
- OCR 실행, 모든 사이트의 렌더링 성공, 유료벽 우회는 지원하지 않는다. 스캔 PDF는 OCR 필요 상태를 알린다.
- 실뉴스 전문가 라벨 평가, 임계값 보정, 실제 API/브라우저 재검증은 별도 단계다. 합성 테스트 성적을 사실검증 정확도로 해석하지 않는다.
- policy_monitor 연결 및 기사 자동 삭제·학습 제외는 적용하지 않았다.
