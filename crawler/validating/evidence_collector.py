"""Collect keyword-matching paragraph text and sources from Brave results.

python crawler/validating/evidence_collector.py --facts facts.json \
    --search-results search_results.json --output article_sentences.json

Each sentences entry is the entire text of one matching <p>, without splitting
it into individual sentences. Aligned sentence_contexts preserve the text of
each matching paragraph's immediate parent. Only server-returned HTML is inspected.
"""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup


def _load(value: str | dict[str, Any]) -> dict[str, Any]:
    data = json.loads(value) if isinstance(value, str) else value
    if not isinstance(data, dict) or not isinstance(data.get("facts"), list):
        raise ValueError("입력 JSON에는 facts 배열이 있어야 합니다.")
    return data


def matching_paragraphs(html: str | bytes, keywords: list[str]) -> list[str]:
    """Match any keyword as a case-insensitive substring; normalize whitespace."""
    return [entry["sentence"] for entry in matching_paragraph_contexts(html, keywords)]


def matching_paragraph_contexts(
    html: str | bytes, keywords: list[str],
) -> list[dict[str, str | None]]:
    """Keep the immediate parent's full text, including the matching paragraph.

    Parent text may include headings, lists, or unrelated page elements.
    Text is whitespace-normalized, not truncated or synthesized.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "template"]):
        tag.decompose()
    needles = [" ".join(keyword.split()).casefold() for keyword in keywords]
    matches = []
    for tag in soup.find_all("p"):
        text = " ".join(tag.get_text(" ", strip=True).split())
        if text and any(needle in text.casefold() for needle in needles):
            parent_text = (" ".join(tag.parent.get_text(" ", strip=True).split())
                           if tag.parent is not None else None)
            matches.append({"sentence": text, "parent_context": parent_text})
    return matches


def extract_article_paragraphs(
    extracted: str | dict[str, Any], search_results: str | dict[str, Any],
    *, session: Any = None, enhanced: bool = False, source_url: str | None = None, render_dynamic: bool = False, source_text: str | None = None,
) -> str:
    """Return JSON with sentences grouped by source, plus per-URL errors.

    Inputs are the outputs of extract_news_facts and search_news_facts.
    Match unique claims by exact text, independent of search result order.
    Preserve extracted claim order and filter paragraphs with each claim's keywords.
    Non-HTML documents (including PDFs) are recorded as skipped.
    date preserves Brave's page_age (published or modified date), falling back
    to article.date. date_source identifies the field; missing dates are null.
    """
    extracted_data = _load(extracted)
    search_data = _load(search_results)
    published_date = extracted_data.get("published_date") or search_data.get("published_date")
    facts = extracted_data["facts"]
    searches = search_data["facts"]
    if not facts and not searches:
        return json.dumps({"published_date": published_date, "facts": []},
                          ensure_ascii=False, indent=2)
    def index_claims(entries: list[Any]) -> dict[str, dict[str, Any]]:
        indexed = {}
        for entry in entries:
            if (not isinstance(entry, dict) or not isinstance(entry.get("claim"), str)
                    or not entry["claim"].strip()):
                raise ValueError("각 항목에는 비어 있지 않은 claim이 있어야 합니다.")
            claim = entry["claim"]
            if claim in indexed:
                raise ValueError("중복 claim은 허용하지 않습니다.")
            indexed[claim] = entry
        return indexed

    fact_map = index_claims(facts)
    search_map = index_claims(searches)
    if fact_map.keys() != search_map.keys():
        raise ValueError("두 입력의 claim이 일치해야 합니다.")
    prepared = []
    for claim, fact in fact_map.items():
        search = search_map[claim]
        keywords = fact.get("keywords")
        if (not isinstance(keywords, list) or not keywords
                or any(not isinstance(k, str) or not k.strip() for k in keywords)):
            raise ValueError("keywords에는 비어 있지 않은 문자열이 있어야 합니다.")
        response_data = search.get("search_response")
        if not isinstance(response_data, dict):
            raise ValueError("search_response는 JSON 객체여야 합니다.")
        results = []
        for section in ("web", "news"):
            group = response_data.get(section)
            if group is None:
                continue
            if not isinstance(group, dict) or not isinstance(group.get("results"), list):
                raise ValueError(f"search_response.{section}.results는 배열이어야 합니다.")
            results.extend(group["results"])
        if any(not isinstance(result, dict) or not isinstance(result.get("url"), str)
               for result in results):
            raise ValueError("검색 결과에는 url 문자열이 있어야 합니다.")
        prepared.append((fact, keywords, results))
    if session is None:
        with requests.Session() as http:
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            adapter = HTTPAdapter(max_retries=Retry(total=2, backoff_factor=.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"]))
            http.mount("https://", adapter)
            http.mount("http://", adapter)
            return extract_article_paragraphs(extracted, search_results, session=http, enhanced=enhanced, source_url=source_url, render_dynamic=render_dynamic, source_text=source_text)
    try:
        from .validation_rules import canonical_url, content_hash, same_document, shingles, publication_credits, attribution_matches, matching_fingerprints
        from .document_parser import html_evidence, pdf_evidence, render_html, MAX_BYTES
    except ImportError:
        from validation_rules import canonical_url, content_hash, same_document, shingles, publication_credits, attribution_matches, matching_fingerprints
        from document_parser import html_evidence, pdf_evidence, render_html, MAX_BYTES
    output = []
    for fact, keywords, results in prepared:
        articles = []
        seen = set()
        documents = []
        if enhanced and source_url and source_text:
            documents.append((source_text, "original", canonical_url(source_url), []))
        for result in results:
            url = result["url"]
            identity = canonical_url(url) if enhanced else url
            if identity in seen:
                continue
            seen.add(identity)
            article_date = result.get("page_age")
            date_source = "page_age"
            if not isinstance(article_date, str) or not article_date.strip():
                metadata = result.get("article")
                article_date = metadata.get("date") if isinstance(metadata, dict) else None
                date_source = "article.date"
            if not isinstance(article_date, str) or not article_date.strip():
                article_date = None
                date_source = None
            article = {"source_url": url, "title": result.get("title", ""),
                       "date": article_date, "date_source": date_source,
                       "sentences": [], "sentence_contexts": [], "status": "ok"}
            articles.append(article)
            response = None
            try:
                if enhanced and source_url and identity == canonical_url(source_url):
                    article.update(status="excluded_original", error="Original article is not independent evidence")
                    continue
                parsed = urlsplit(url)
                if parsed.scheme not in ("http", "https") or not parsed.hostname:
                    raise ValueError("HTTP(S) URL만 지원합니다.")
                response = session.get(url, headers={"User-Agent": "Mozilla/5.0"},
                                       timeout=(5, 30), **({"stream": True} if enhanced else {}))
                response.raise_for_status()
                article["resolved_url"] = response.url
                if enhanced:
                    article["fetched_at"] = datetime.now(timezone.utc).isoformat()
                    article["canonical_url"] = canonical_url(response.url)
                    if source_url and article["canonical_url"] == canonical_url(source_url):
                        article.update(status="excluded_original", error="Redirect resolves to original article")
                        continue
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                if enhanced:
                    if hasattr(response, "iter_content"):
                        chunks, size = [], 0
                        try:
                            for chunk in response.iter_content(65536):
                                size += len(chunk)
                                if size > MAX_BYTES:
                                    raise ValueError("Document exceeds 5 MB extraction budget")
                                chunks.append(chunk)
                            content = b"".join(chunks)
                        finally:
                            response.close()
                    else:
                        content = response.content
                    if len(content) > MAX_BYTES:
                        raise ValueError("Document exceeds 5 MB extraction budget")
                    if content_type == "application/pdf":
                        contexts, document = pdf_evidence(content, keywords)
                        article["extraction_method"] = "pdf_text"
                    elif content_type in ("text/html", "application/xhtml+xml"):
                        contexts, document = html_evidence(content, keywords)
                        article["extraction_method"] = "html_main"
                        if not contexts and render_dynamic:
                            rendered, resolved = render_html(response.url)
                            contexts, document = html_evidence(rendered, keywords)
                            article["resolved_url"] = resolved
                            article["canonical_url"] = canonical_url(resolved)
                            article["extraction_method"] = "browser"
                            if source_url and article["canonical_url"] == canonical_url(source_url):
                                article.update(status="excluded_original", error="Browser resolves to original article")
                                continue
                    else:
                        article.update(status="skipped", error=f"Unsupported content type: {content_type}")
                        continue
                    if not document.strip():
                        article.update(status="no_match", error="No extractable body")
                        continue
                    credits = publication_credits(document)
                    article["publication_credits"] = credits
                    group = next((g for text, g, u, known_credits in documents
                                  if u == article["canonical_url"] or same_document(text, document)
                                  or matching_fingerprints(shingles(text), shingles(document),
                                      attribution_matches(u, article["canonical_url"], known_credits, credits))), None)
                    article["body_hash"] = content_hash(document)
                    article["body_fingerprint"] = sorted(content_hash(part)[:16] for part in shingles(document))
                    article["source_group"] = group or article["body_hash"]
                    if group:
                        article.update(status="excluded_original_republication" if group == "original" else "duplicate", error="Original republication" if group == "original" else "Duplicate or syndicated document")
                        continue
                    documents.append((document, article["source_group"], article["canonical_url"], credits))
                    article["sentence_contexts"] = contexts
                    article["sentences"] = [entry["sentence"] for entry in contexts]
                    article["candidate_count"] = len(contexts)
                    article["selection_method"] = "keyword_and_semantic_candidates"
                    continue
                if content_type not in ("text/html", "application/xhtml+xml"):
                    article.update(status="skipped", error=f"HTML이 아닌 응답: {content_type or 'unknown'}")
                    continue
                contexts = matching_paragraph_contexts(response.content, keywords)
                article["sentence_contexts"] = contexts
                article["sentences"] = [entry["sentence"] for entry in contexts]
                if not article["sentences"]:
                    article["status"] = "no_match"
            except Exception as exc:
                article.update(status="error", error=str(exc))
            finally:
                if "response" in locals() and enhanced and hasattr(response, "close"):
                    response.close()
        output.append({"claim": fact["claim"], "keywords": keywords, "articles": articles,
                       **({"structure": fact["structure"]} if "structure" in fact else {})})
    return json.dumps({"published_date": published_date, "facts": output},
                      ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="기사의 p 태그에서 키워드 포함 텍스트와 출처 저장")
    parser.add_argument("--facts", type=Path, required=True, help="뉴스 추출 결과 JSON")
    parser.add_argument("--search-results", type=Path, required=True, help="Brave 검색 결과 JSON")
    parser.add_argument("--output", type=Path, help="저장할 JSON 파일 (생략 시 표준 출력)")
    parser.add_argument("--legacy", action="store_true", help="기존 p 태그 키워드 방식 사용")
    parser.add_argument("--source-url")
    parser.add_argument("--render-dynamic", action="store_true")
    args = parser.parse_args()
    try:
        result = extract_article_paragraphs(
            args.facts.read_text(encoding="utf-8-sig"),
            args.search_results.read_text(encoding="utf-8-sig"),
            enhanced=not args.legacy, source_url=args.source_url, render_dynamic=args.render_dynamic,
        ) + "\n"
        if args.output:
            args.output.write_text(result, encoding="utf-8")
        else:
            print(result, end="")
    except Exception as exc:
        parser.exit(1, f"기사 추출 실패: {exc}\n")


if __name__ == "__main__":
    main()
