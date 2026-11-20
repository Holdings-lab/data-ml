"""Bounded evidence extraction with reproducible document locations."""
from __future__ import annotations

from io import BytesIO
import re
from bs4 import BeautifulSoup

MAX_BYTES = 5 * 1024 * 1024
MAX_CANDIDATES = 200


def html_evidence(content: bytes | str, keywords: list[str]) -> tuple[list[dict], str]:
    soup = BeautifulSoup(content, "html.parser")
    for node in soup.select("script, style, template, nav, footer, aside, form"):
        node.decompose()
    root = soup.select_one("article") or soup.select_one("main") or soup.body or soup
    nodes = root.find_all(["h1", "h2", "h3", "p", "li", "tr"])
    blocks = []
    for node in nodes:
        if node.name != "tr" and node.find_parent("tr") is not None:
            continue
        text = " ".join(node.get_text(" ", strip=True).split())
        if text:
            blocks.append((node.name, text))
    full_text = "\n".join(text for _, text in blocks)
    needles = [" ".join(k.split()).casefold() for k in keywords]
    heading, candidates = "", []
    # Include non-keyword paragraphs as semantic candidates, within an explicit budget.
    for i, (kind, text) in enumerate(blocks):
        if kind.startswith("h"):
            heading = text
        context = " ".join(dict.fromkeys([heading] + [x[1] for x in blocks[max(0, i-1):i+2]]))
        candidates.append({"sentence": text, "parent_context": context[:6000],
                           "location": f"{kind}[{i}]", "keyword_match": any(k in text.casefold() for k in needles)})
    # Stable keyword-first budget; preserve original document locations.
    candidates.sort(key=lambda c: not c["keyword_match"])
    return candidates[:MAX_CANDIDATES], full_text


def pdf_evidence(content: bytes, keywords: list[str]) -> tuple[list[dict], str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError("PDF dependency missing: install pypdf") from exc
    reader = PdfReader(BytesIO(content))
    if reader.is_encrypted:
        raise ValueError("Encrypted PDF is not supported")
    if len(reader.pages) > 100:
        raise ValueError("PDF exceeds 100-page extraction budget")
    entries, pages = [], []
    for page_no, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        pages.append(text)
        for block in re.split(r"\n\s*\n", text):
            # Bound long unbroken PDF pages; keep the page provenance.
            for start in range(0, len(block), 2500):
                paragraph = " ".join(block[start:start+2500].split())
                if paragraph:
                    entries.append({"sentence": paragraph, "parent_context": paragraph,
                                    "location": f"page:{page_no}",
                                    "keyword_match": any(k.casefold() in paragraph.casefold() for k in keywords)})
    if not any(p.strip() for p in pages):
        raise ValueError("PDF has no extractable text; OCR required")
    entries.sort(key=lambda c: not c["keyword_match"])
    return entries[:MAX_CANDIDATES], "\n".join(pages)


def render_html(url: str) -> tuple[str, str]:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(1500)
            return page.content(), page.url
        finally:
            browser.close()
