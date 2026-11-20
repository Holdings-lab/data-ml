"""Run extraction, search, collection, ranking, and verification in one command.

Set ANTHROPIC_API_KEY and BRAVE_SEARCH_API_KEY before running.
Each successful stage saves its JSON; failures stop subsequent stages.
"""
from __future__ import annotations

import argparse
import json
import time
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crawler.validating.validation_rules import VERSION, retrieval_summary
from crawler.monitor_paths import VALIDATION_DIR
from crawler.validating.claim_extractor import extract_news_facts
from crawler.validating.evidence_search import search_news_facts
from crawler.validating.evidence_collector import extract_article_paragraphs
from crawler.validating.evidence_ranker import DEFAULT_MODEL as DEFAULT_SBERT_MODEL, rank_claim_sentences
from crawler.validating.claim_verifier import verify_ranked_claims

OUTPUT_FILES = (
    "facts.json", "search_results.json", "article_sentences.json",
    "top_sentences.json", "verified_claims.json",
)


def run_validation_pipeline(
    input_path: str | Path = "C:/example.txt", *,
    published_date: str | None = None, max_facts: int = 1, count: int = 20,
    output_dir: str | Path | None = None, model: str | None = None,
    sbert_model: str = DEFAULT_SBERT_MODEL, source_url: str | None = None,
    search_mode: str = "as_of", lookback_days: int | None = 90,
    min_similarity: float = .0, max_evidence: int = 6,
    supplemental_search: bool = False, render_dynamic: bool = False,
    reranker_model: str | None = None, original_source_text: str | None = None,
) -> Path:
    """Save all five stages and return the absolute final JSON path.

    Default output is a new directory under data/validation for each run.
    Explicit output directories must not contain any of the five output files.
    Missing published_date retains the search stage's Korean-today fallback.
    model overrides both Claude stages; sbert_model controls ranking only.
    """
    if supplemental_search:
        raise ValueError("Supplemental search is disabled to limit Brave API costs")
    if search_mode not in {"as_of", "current"}:
        raise ValueError("Invalid search mode")
    if lookback_days is not None and (type(lookback_days) is not int or not 1 <= lookback_days <= 3650):
        raise ValueError("Invalid lookback_days")
    if not -1 <= min_similarity <= 1 or type(max_evidence) is not int or not 1 <= max_evidence <= 6:
        raise ValueError("Invalid ranking options")
    if isinstance(max_facts, bool) or not isinstance(max_facts, int) or max_facts < 1:
        raise ValueError("max_facts는 1 이상의 정수여야 합니다.")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 20:
        raise ValueError("count는 1~20 사이의 정수여야 합니다.")
    if published_date is not None:
        try:
            parsed = date.fromisoformat(published_date)
            if parsed.isoformat() != published_date:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError("published_date는 YYYY-MM-DD 형식이어야 합니다.") from exc
    body = Path(input_path).read_text(encoding="utf-8-sig")
    if not body.strip():
        raise ValueError("뉴스 본문이 비어 있습니다.")

    if output_dir is None:
        root = PROJECT_ROOT / "data" / "validation"
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S_")
        destination = Path(tempfile.mkdtemp(prefix=stamp, dir=root)).resolve()
    else:
        destination = Path(output_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        if any((destination / name).exists() for name in (*OUTPUT_FILES, "run_summary.json")):
            raise FileExistsError("기존 단계별 결과가 있습니다. 새로운 --output-dir을 지정하세요.")
    print(f"결과 폴더: {destination}", file=sys.stderr)

    summary = {
        "version": VERSION, "execution_status": "running", "stages": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "settings": {"model": model or os.getenv("ANTHROPIC_MODEL") or "claude-haiku-4-5-20251001",
                     "sbert_model": sbert_model, "reranker_model": reranker_model,
                     "search_mode": search_mode, "published_date": published_date,
                     "lookback_days": lookback_days, "min_similarity": min_similarity,
                     "max_evidence": max_evidence, "max_facts": max_facts, "count": count,
                     "source_url": source_url, "supplemental_search": supplemental_search,
                     "render_dynamic": render_dynamic},
        "warnings": [],
    }
    def save_summary():
        (destination / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_summary()

    def stage(number, label, operation, filename=None):
        print(f"[{number}/5] {label}", file=sys.stderr)
        entry = {"name": label, "status": "running"}
        summary["stages"].append(entry)
        started = time.monotonic()
        save_summary()
        try:
            result = operation()
            json.loads(result)
            with (destination / (filename or OUTPUT_FILES[number - 1])).open("x", encoding="utf-8") as output:
                output.write(result + "\n")
            entry["status"] = "completed"
            return result
        except Exception as exc:
            entry.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            summary["execution_status"] = "failed"
            raise RuntimeError(f"{label} 실패: {exc}\n완료된 단계의 결과 폴더: {destination}") from exc
        finally:
            entry["elapsed_seconds"] = round(time.monotonic() - started, 3)
            save_summary()

    extracted = stage(1, "핵심 주장 추출", lambda: extract_news_facts(
        body, published_date=published_date, max_facts=max_facts, model=model, structured=True))
    searched = stage(2, "주장별 Brave 검색", lambda: search_news_facts(
        extracted, count=count, search_mode=search_mode, lookback_days=lookback_days))
    collected = stage(3, "문단 및 부모 태그 텍스트 수집", lambda: extract_article_paragraphs(
        extracted, searched, enhanced=True, source_url=source_url, render_dynamic=render_dynamic, source_text=(original_source_text if original_source_text is not None else body) if source_url else None))
    ranked = stage(4, "출처 중복 없이 근거 선별", lambda: rank_claim_sentences(
        collected, model_name=sbert_model, min_similarity=min_similarity, max_evidence=max_evidence,
        reranker_name=reranker_model))
    summary["retrieval"] = retrieval_summary(json.loads(collected), json.loads(ranked))
    save_summary()
    verified = stage(5, "Claude 최종 검증", lambda: verify_ranked_claims(ranked, model=model))
    judgments = json.loads(verified).get("facts", [])
    summary["api_usage"] = {"extraction": json.loads(extracted).get("api_usage", {}),
                            "verification": json.loads(verified).get("api_usage", {})}
    if summary["retrieval"]["status"] != "available":
        summary["warnings"].append("근거 확보가 제한적임: retrieval 통계 및 URL별 상태 확인 필요")
    if any(f.get("verdict") == "insufficient_evidence" for f in judgments):
        summary["warnings"].append("근거 부족 판정이 포함됨")
    summary["verdict_counts"] = {v: sum(f.get("verdict") == v for f in judgments)
                                  for v in ("supported", "refuted", "insufficient_evidence")}
    summary["supported_rate"] = summary["verdict_counts"]["supported"] / len(judgments) if judgments else None
    summary["target_supported_rate"] = 0.5
    summary["target_met"] = summary["supported_rate"] >= 0.5 if judgments else None
    summary["execution_status"] = "completed_with_warnings" if summary["warnings"] else "completed"
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    save_summary()
    return destination / OUTPUT_FILES[-1]


DEFAULT_MONITOR_OUTPUT_ROOT = VALIDATION_DIR


def _text(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def validate_yahoo_summaries(
    frame: pd.DataFrame, *, source_records: pd.DataFrame,
    output_root: str | Path = DEFAULT_MONITOR_OUTPUT_ROOT,
) -> pd.DataFrame:
    result = frame.copy()
    defaults = {"validation_status": "not_applicable", "validation_verdict": "",
                "validation_claim_count": 0, "validation_result_path": "",
                "validation_error": ""}
    for column, value in defaults.items():
        result[column] = value
    if result.empty or "source" not in result:
        return result
    originals = {}
    for _, row in source_records.iterrows():
        key = (_text(row.get("sector", "")), _text(row.get("url", "")))
        originals[key] = _text(row.get("body", ""))
    run_dir = None
    # Positional writes also support frames with duplicated index labels.
    for position, (_, row) in enumerate(result.iterrows()):
        if _text(row["source"]).upper() != "YAHOO":
            continue
        values = dict(defaults, validation_status="failed")
        try:
            summary = _text(row.get("body_summary", ""))
            if not summary:
                raise ValueError("Yahoo summary is empty")
            published = _text(row.get("release_date", ""))
            if date.fromisoformat(published).isoformat() != published:
                raise ValueError("Yahoo release_date must be YYYY-MM-DD")
            url = _text(row.get("url", ""))
            if not url:
                raise ValueError("Yahoo source URL is missing")
            if run_dir is None:
                root = Path(output_root)
                root.mkdir(parents=True, exist_ok=True)
                run_dir = Path(tempfile.mkdtemp(prefix="cycle_", dir=root)).resolve()
            article_dir = run_dir / f"article_{position:05d}"
            article_dir.mkdir()
            input_path = article_dir / "summary.txt"
            input_path.write_text(summary, encoding="utf-8")
            values["validation_result_path"] = str(article_dir)
            original = originals.get((_text(row.get("sector", "")), url), "")
            path = run_validation_pipeline(
                input_path, published_date=published, source_url=url,
                original_source_text=original or summary,
                output_dir=article_dir / "pipeline",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            facts = payload["facts"]
            verdicts = [fact["verdict"] for fact in facts]
            if any(v not in {"supported", "refuted", "insufficient_evidence"} for v in verdicts):
                raise ValueError("Unexpected validation verdict")
            verdict = ("no_claims" if not facts else "refuted" if "refuted" in verdicts
                       else "insufficient_evidence" if "insufficient_evidence" in verdicts else "supported")
            values.update(validation_status="completed", validation_verdict=verdict,
                          validation_claim_count=len(facts), validation_result_path=str(path))
        except Exception as exc:
            values["validation_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[VALIDATION] Yahoo row={position} failed: {exc}")
        for column, value in values.items():
            result.iat[position, result.columns.get_loc(column)] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="뉴스 사실검증 5단계 통합 파이프라인")
    parser.add_argument("--input", type=Path, default=Path("C:/example.txt"),
                        help="UTF-8 뉴스 본문 (기본: C:/example.txt)")
    parser.add_argument("--published-date", help="기사 발행일 YYYY-MM-DD (생략 시 검색은 한국 시간 오늘 기준)")
    parser.add_argument("--max-facts", type=int, default=1, help="최대 주장 수 (기본: 1)")
    parser.add_argument("--count", type=int, choices=range(1, 21), default=20,
                        help="주장별 검색 결과 요청 수 (기본: 20)")
    parser.add_argument("--output-dir", type=Path, help="결과 폴더 (기본: data/validation 아래 새 실행 폴더)")
    parser.add_argument("--model", help="추출·검증 Claude 모델 (생략 시 ANTHROPIC_MODEL 또는 기본 모델)")
    parser.add_argument("--sbert-model", default=DEFAULT_SBERT_MODEL, help="SBERT 모델 이름 또는 로컬 경로")
    parser.add_argument("--source-url", help="원 기사 URL: 독립 근거에서 제외")
    parser.add_argument("--search-mode", choices=["as_of", "current"], default="as_of")
    parser.add_argument("--lookback-days", type=int, default=90, help="기사 발행일 이전 검색 기간 (기본: 90일)")
    parser.add_argument("--min-similarity", type=float, default=.0, help="임시 관련성 하한; 평가 데이터로 조정 필요")
    parser.add_argument("--max-evidence", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--no-supplemental-search", action="store_true",
                        help="호환용 옵션: 보충 검색은 항상 비활성화됩니다")
    parser.add_argument("--render-dynamic", action="store_true", help="본문이 없는 HTML만 브라우저로 재시도")
    parser.add_argument("--reranker-model", help="선택적 CrossEncoder 모델명 또는 로컬 경로")
    args = parser.parse_args()
    try:
        result = run_validation_pipeline(
            args.input, published_date=args.published_date, max_facts=args.max_facts,
            count=args.count, output_dir=args.output_dir, model=args.model,
            sbert_model=args.sbert_model, source_url=args.source_url,
            search_mode=args.search_mode, lookback_days=args.lookback_days,
            min_similarity=args.min_similarity, max_evidence=args.max_evidence,
            supplemental_search=False, render_dynamic=args.render_dynamic,
            reranker_model=args.reranker_model,
        )
        print(f"검증 완료: {result}")
    except Exception as exc:
        parser.exit(1, f"통합 파이프라인 실패: {exc}\n")


if __name__ == "__main__":
    main()
