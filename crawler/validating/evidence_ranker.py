"""Rank collected paragraph entries against each claim with SBERT.

python crawler/validating/evidence_ranker.py --input article_sentences.json \
    --output top_sentences.json

The original sentences entries (whole paragraphs) are preserved in the output.
The model may truncate text exceeding its token limit. On first use, the model
is downloaded unless already cached. Similarity does not establish truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"


def rank_claim_sentences(
    collected: str | dict[str, Any], *, model_name: str = DEFAULT_MODEL,
    model: Any = None, min_similarity: float = -1.0, max_evidence: int = 3, reranker_name: str | None = None, reranker: Any = None,
) -> str:
    """Return up to three entries from distinct source URLs for each claim.

    Compute cosine similarity on normalized SBERT embeddings. Ties retain
    input order. Keep only the highest-scoring paragraph per exact source URL,
    then take the top three sources (or all sources if fewer than three).
    Identical entries with the same context from the same source are deduplicated.
    Optional parent_context text is preserved but is not embedded or scored.
    Candidates and deduplication are scoped to each claim; claim order is preserved.
    Claims without candidates return an empty top_sentences list.
    Preserve the original article's published_date separately from each source date.
    """
    if not -1 <= min_similarity <= 1 or isinstance(max_evidence, bool) or not isinstance(max_evidence, int) or not 1 <= max_evidence <= 6:
        raise ValueError("Invalid ranking threshold or evidence budget")
    try:
        from .validation_rules import matching_fingerprints, attribution_matches
    except ImportError:
        from validation_rules import matching_fingerprints, attribution_matches
    data = json.loads(collected) if isinstance(collected, str) else collected
    if not isinstance(data, dict) or not isinstance(data.get("facts"), list):
        raise ValueError("입력 JSON에는 facts 배열이 있어야 합니다.")
    prepared = []
    for fact in data["facts"]:
        if (not isinstance(fact, dict) or not isinstance(fact.get("claim"), str)
                or not fact["claim"].strip() or not isinstance(fact.get("articles"), list)):
            raise ValueError("각 facts 항목에는 claim과 articles 배열이 있어야 합니다.")
        candidates = []
        seen = set()
        fingerprints = []
        for article in fact["articles"]:
            if not isinstance(article, dict) or not isinstance(article.get("sentences"), list):
                raise ValueError("각 기사에는 sentences 배열이 있어야 합니다.")
            source = article.get("canonical_url", article.get("source_url"))
            if not isinstance(source, str) or not source.strip():
                raise ValueError("각 기사에는 source_url이 있어야 합니다.")
            group = article.get("source_group")
            fingerprint = set(article.get("body_fingerprint", []))
            if group:
                for known_source, known_hash, known_fingerprint, known_group, known_credits in fingerprints:
                    near = matching_fingerprints(fingerprint, known_fingerprint, attribution_matches(source, known_source, article.get("publication_credits", []), known_credits))
                    if source == known_source or (article.get("body_hash") and article["body_hash"] == known_hash) or near:
                        group = known_group
                        break
                fingerprints.append((source, article.get("body_hash"), fingerprint, group, article.get("publication_credits", [])))
            contexts = article.get("sentence_contexts")
            if "sentence_contexts" in article:
                if not isinstance(contexts, list) or len(contexts) != len(article["sentences"]):
                    raise ValueError("sentence_contexts는 sentences와 길이가 같은 배열이어야 합니다.")
                for sentence, context in zip(article["sentences"], contexts):
                    if (not isinstance(context, dict) or context.get("sentence") != sentence
                            or "parent_context" not in context
                            or (context["parent_context"] is not None
                                and not isinstance(context["parent_context"], str))):
                        raise ValueError("sentence_contexts에는 일치하는 문단과 parent_context가 필요합니다. 문단 수집을 다시 실행하세요.")
            for index, sentence in enumerate(article["sentences"]):
                if not isinstance(sentence, str):
                    raise ValueError("sentences 항목은 문자열이어야 합니다.")
                if not sentence.strip():
                    continue
                context = ({"parent_context": contexts[index]["parent_context"]}
                           if contexts is not None else {})
                identity = (source, sentence, context.get("parent_context"))
                if identity in seen:
                    continue
                seen.add(identity)
                candidates.append({"date": article.get("date"), "source": source,
                                   "sentence": sentence, **context,
                                   **{k: article[k] for k in ("fetched_at", "body_hash", "extraction_method", "date_source", "title") if k in article},
                                   **({"source_group": group} if group else {}),
                                   **({"location": contexts[index]["location"]} if contexts and "location" in contexts[index] else {})})
        prepared.append((fact["claim"], candidates))
    if model is None and any(candidates for _, candidates in prepared):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
    if reranker is None and reranker_name and any(c for _, c in prepared):
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(reranker_name)
    output = []
    for claim, candidates in prepared:
        selected = []
        if candidates:
            texts = [claim] + [entry["sentence"] for entry in candidates]
            embeddings = np.asarray(model.encode(
                texts, batch_size=32, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            ), dtype=np.float64)
            if (embeddings.ndim != 2 or embeddings.shape[0] != len(texts)
                    or not np.isfinite(embeddings).all()):
                raise ValueError("SBERT 임베딩 형식이 잘못되었습니다.")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("SBERT가 길이가 0인 임베딩을 반환했습니다.")
            embeddings = embeddings / norms
            scores = embeddings[1:] @ embeddings[0]
            order = np.argsort(-scores, kind="stable")
            if reranker is not None:
                # Bound cross-encoder work; use surrounding context for direct relevance.
                shortlist = [int(i) for i in order if scores[int(i)] >= min_similarity][:30]
                if shortlist:
                    relevance = np.asarray(reranker.predict([(claim, candidates[i]["sentence"] + " " + (candidates[i].get("parent_context") or "")) for i in shortlist]))
                    if relevance.shape != (len(shortlist),) or not np.isfinite(relevance).all():
                        raise ValueError("Reranker must return one finite relevance score per candidate")
                    order = [shortlist[int(i)] for i in np.argsort(-relevance, kind="stable")]
            selected_sources = set()
            for index in order:
                candidate = candidates[int(index)]
                if scores[int(index)] < min_similarity:
                    continue
                identity = candidate.get("source_group", candidate["source"])
                if identity in selected_sources:
                    continue
                selected_sources.add(identity)
                selected.append({**candidate,
                                 "similarity": float(np.clip(scores[int(index)], -1.0, 1.0))})
                if len(selected) == max_evidence:
                    break
        fact = data["facts"][len(output)]
        output.append({"claim": claim, "top_sentences": selected,
                       **({"structure": fact["structure"]} if "structure" in fact else {})})
    return json.dumps({"published_date": data.get("published_date"), "facts": output},
                      ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="SBERT로 출처 URL 중복 없이 주장별 상위 3개 문단 저장")
    parser.add_argument("--input", type=Path, required=True, help="기사 문장 수집 결과 JSON")
    parser.add_argument("--output", type=Path, help="결과 JSON (생략 시 표준 출력)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SBERT 모델 이름 또는 로컬 경로")
    parser.add_argument("--min-similarity", type=float, default=.3)
    parser.add_argument("--max-evidence", type=int, choices=range(1, 7), default=3)
    parser.add_argument("--reranker-model")
    args = parser.parse_args()
    try:
        result = rank_claim_sentences(args.input.read_text(encoding="utf-8-sig"),
                                      model_name=args.model, min_similarity=args.min_similarity, max_evidence=args.max_evidence, reranker_name=args.reranker_model) + "\n"
        if args.output:
            args.output.write_text(result, encoding="utf-8")
        else:
            print(result, end="")
    except Exception as exc:
        parser.exit(1, f"문장 선출 실패: {exc}\n")


if __name__ == "__main__":
    main()
