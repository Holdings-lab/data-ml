"""Deterministic provenance, retrieval statistics, and judgment consistency checks."""
from __future__ import annotations

from collections import Counter
import copy
from decimal import Decimal, InvalidOperation
import hashlib
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

VERSION = "2.5"


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid", "msclkid"}]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/",
                       urlencode(sorted(query)), ""))


def normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def quote_text(text: str) -> str:
    # Typography and invisible formatting are not factual differences.
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    return normalized(text.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})))


def content_hash(text: str) -> str:
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def shingles(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", normalized(text))
    return {" ".join(words[i:i + 5]) for i in range(max(0, len(words) - 4))}


def same_document(left: str, right: str) -> bool:
    if normalized(left) == normalized(right):
        return True
    a, b = shingles(left), shingles(right)
    return min(len(a), len(b)) >= 40 and len(a & b) / len(a | b) >= .9


# Explicit, reviewable aliases; do not infer arbitrary entity equivalence.
TEXT_ALIASES = {
    "u.s. food and drug administration": "fda",
    "us food and drug administration": "fda",
    "food and drug administration": "fda",
    "transient ischemic attack": "tia",
}
PROTECTED = re.compile(r"\b(not|no|without|except|unless|if|only|pending|planned|proposed|may|must)\b", re.I)


def comparison_text(text: str) -> str:
    value = " ".join(text.split())
    for long, short in TEXT_ALIASES.items():
        value = re.sub(r"\b" + re.escape(long) + r"\b", short, value, flags=re.I)
    def parenthesis(match):
        inner = match.group(1)
        # Keep qualifiers, negation and numeric restrictions in parentheses.
        acronym = bool(re.fullmatch(r"[A-Za-z.]{2,10}", inner)) and sum(c.isupper() for c in inner) >= 2
        explanation = bool(re.match(r"(?:a|an)\s", inner, re.I))
        removable = (acronym or explanation) and not PROTECTED.search(inner) and not re.search(r"\d", inner)
        return " " if removable else " " + inner + " "
    value = re.sub(r"\(([^()]*)\)", parenthesis, value)
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def phrase_supported(expected: str, observed: str, quote: str) -> bool:
    wanted, actual, cited = map(comparison_text, (expected, observed, quote))
    if not wanted or not actual:
        return False
    # Allow inserted descriptive wording, but never negation/conditions/numbers.
    def contained(needle, haystack):
        tokens, words = needle.split(), haystack.split()
        for start, word in enumerate(words):
            if word != tokens[0]:
                continue
            cursor, gaps = 1, []
            for next_word in words[start+1:]:
                if cursor == len(tokens):
                    break
                if next_word == tokens[cursor]:
                    cursor += 1
                else:
                    gaps.append(next_word)
            if cursor == len(tokens) and not any(PROTECTED.fullmatch(x) or any(c.isdigit() for c in x) for x in gaps):
                # Nearby negation/qualification is significant even before the phrase.
                if not any(PROTECTED.fullmatch(x) for x in words[max(0,start-3):start]):
                    return True
        return False
    return (wanted == actual or contained(wanted, actual) or contained(actual, wanted)) and contained(actual, cited)


def publisher_key(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    return re.sub(r"[^a-z0-9]", "", host.split(".")[0])


def publication_credits(text: str) -> list[str]:
    credits = []
    for match in re.finditer(r"(?:originally published (?:on|by)|information source\s*:|republished from)\s*([^.!\n]{2,80})", text, re.I):
        credits.append(re.sub(r"[^a-z0-9]", "", match.group(1).casefold().strip()))
    if re.search(r"\(Reuters\)", text, re.I):
        credits.append("reuters")
    return sorted(set(credits))


def attribution_matches(left_url, right_url, left_credits, right_credits) -> bool:
    left, right = set(left_credits), set(right_credits)
    return bool(left & right or publisher_key(left_url) in right or publisher_key(right_url) in left)


def matching_fingerprints(a: set, b: set, attributed: bool = False) -> bool:
    if min(len(a), len(b)) < 40:
        return False
    intersection = len(a & b)
    containment = intersection / min(len(a), len(b))
    return intersection / len(a | b) >= .9 or containment >= .9 or (attributed and containment >= .65)


def retrieval_summary(collected: dict, ranked: dict | None = None) -> dict:
    per_claim = []
    for fact in collected.get("facts", []):
        articles = fact.get("articles", [])
        counts = Counter(a.get("status", "unknown") for a in articles)
        per_claim.append({"claim": fact["claim"], "requested_urls": len(articles),
                          "status_counts": dict(counts),
                          "fetched_urls": sum(bool(a.get("resolved_url")) for a in articles),
                          "matched_documents": sum(bool(a.get("sentences")) for a in articles)})
    selected = [f.get("top_sentences", []) for f in (ranked or {}).get("facts", [])]
    warning = any(any(c["status_counts"].get(k, 0) for k in ("error", "skipped", "no_match")) for c in per_claim)
    return {"claims": per_claim, "selected_groups_per_claim": [len(s) for s in selected],
            "status": "no_evidence" if not any(selected) else
                      "limited" if warning or any(not s for s in selected) else "available",
            "has_collection_errors": any(c["status_counts"].get("error", 0) for c in per_claim)}


def validate_structure(structure: dict) -> None:
    keys = {"subject", "predicate", "object", "reference_period", "claim_type", "requirements"}
    if not isinstance(structure, dict) or set(structure) != keys:
        raise ValueError("Invalid claim structure")
    if any(not isinstance(structure[k], str) for k in keys - {"requirements"}):
        raise ValueError("Claim structure fields must be strings")
    requirements = structure["requirements"]
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("Structured claims need requirements")
    seen = set()
    for r in requirements:
        if not isinstance(r, dict) or set(r) != {"id", "field", "expected", "unit", "critical", "kind"}:
            raise ValueError("Invalid requirement")
        if any(not isinstance(r[k], str) for k in ("id", "field", "expected", "unit", "kind")):
            raise ValueError("Invalid requirement text")
        if not r["id"].strip() or not r["expected"].strip() or r["id"] in seen or type(r["critical"]) is not bool:
            raise ValueError("Invalid or duplicate requirement ID")
        if r["kind"] not in {"number", "text"}:
            raise ValueError("Invalid requirement kind")
        if r["kind"] == "number":
            number(r["expected"])
        seen.add(r["id"])
    if not any(r["critical"] for r in requirements):
        raise ValueError("At least one critical requirement is required")


def number(value: str) -> Decimal:
    try:
        result = Decimal(value.replace(",", "").strip())
    except (InvalidOperation, AttributeError) as exc:
        raise ValueError("Expected a numeric value without units") from exc
    if not result.is_finite():
        raise ValueError("Numeric value must be finite")
    return result


def _unit_tokens(value: str) -> set[str]:
    value = normalized(value).replace("$", " usd ").replace("%", " percent ")
    value = re.sub(r"\b(?:us dollars?|dollars?)\b", "usd", value)
    value = re.sub(r"\b(?:percentage|per cent)\b", "percent", value)
    value = re.sub(r"\bbasis points?\b", "bps", value)
    aliases = {"shares": "share", "positions": "position", "purchases": "purchase",
               "days": "day", "months": "month", "years": "year", "bps": "bps"}
    return {aliases.get(token, token) for token in re.findall(r"[a-z]+", value)}


def numeric_check_matches(requirement: dict, check: dict) -> bool:
    """Accept formatted amounts and unit aliases, without tolerating value changes."""
    pattern = r"[-+]?\d[\d,]*(?:\.\d+)?"
    observed = re.findall(pattern, check["observed"])
    if len(observed) != 1:
        return False
    try:
        expected = number(requirement["expected"])
        if number(observed[0]) != expected:
            return False
    except ValueError:
        return False
    expected_units = _unit_tokens(requirement["unit"])
    declared_units = _unit_tokens(check["unit"])
    if expected_units != declared_units:
        return False
    embedded_units = _unit_tokens(re.sub(pattern, "", check["observed"]))
    # Some historical requirements specify only a scale (e.g. billion).
    allowed = expected_units | ({"usd"} if expected_units & {"million", "billion", "trillion"} else set())
    if not embedded_units <= allowed:
        return False
    quote = quote_text(check["quote"])
    for match in re.finditer(pattern, quote):
        if number(match.group()) != expected:
            continue
        # Associate the unit with this number, not another number in the quote.
        tail = re.split(pattern, quote[match.end():], maxsplit=1)[0]
        prefix = quote[max(0, match.start() - 5):match.start()]
        units = _unit_tokens(tail)
        if re.search(r"(?:\$|usd)\s*$", prefix):
            units.add("usd")
        scales = {"million", "billion", "trillion"}
        currencies = {"usd", "eur", "gbp"}
        if units & scales != expected_units & scales:
            continue
        if units & currencies and expected_units & currencies and units & currencies != expected_units & currencies:
            continue
        if expected_units <= units:
            return True
    return False


def grounded_quote(quote: str, text: str) -> bool:
    """Ignore punctuation spacing only; preserve words, numbers and their order."""
    quote, text = quote_text(quote), quote_text(text)
    if not quote:
        return False
    if quote in text:
        return True
    tokens = lambda value: re.findall(r"\w+|[^\w\s]", value)
    needle, haystack = tokens(quote), tokens(text)
    return bool(needle) and any(haystack[i:i + len(needle)] == needle
                                for i in range(len(haystack) - len(needle) + 1))


def guard_judgment(judgment: dict, structure: dict, evidence: list[dict]) -> dict:
    """Downgrade contradictory model judgments; never upgrade a judgment."""
    validate_structure(structure)
    checks = judgment.get("checks")
    requirements = {r["id"]: r for r in structure["requirements"]}
    if not isinstance(checks, list):
        raise ValueError("Requirement checks must be a list")
    grouped = {rid: [] for rid in requirements}
    notices = []
    for check in checks:
        if not isinstance(check, dict) or not isinstance(check.get("requirement_id"), str):
            notices.append("Ignored check without a valid requirement ID")
            continue
        rid = check["requirement_id"]
        if rid not in grouped:
            notices.append("Ignored unknown requirement ID: " + rid)
        else:
            grouped[rid].append(check)
    aligned = []
    for rid, entries in grouped.items():
        if entries and all(entry == entries[0] for entry in entries):
            aligned.append(entries[0])
            if len(entries) > 1:
                notices.append("Collapsed identical duplicate checks: " + rid)
        else:
            notices.append(("Conflicting duplicate checks: " if entries else "Missing check: ") + rid)
            aligned.append({"requirement_id": rid, "status": "unresolved", "observed": "",
                            "unit": "", "evidence_ids": [], "quote": ""})
    if notices:
        judgment["original_checks"] = copy.deepcopy(checks)
        judgment["limitations"].extend(notices)
    judgment["checks"] = checks = aligned
    seen, problems, critical = set(), [], []
    for check in checks:
        basic_keys = {"requirement_id", "status", "observed", "unit", "evidence_ids", "quote"}
        if not isinstance(check, dict) or set(check) not in (basic_keys, basic_keys | {"relation", "comparison_reason"}):
            raise ValueError("Invalid requirement check")
        rid = check["requirement_id"]
        if not isinstance(rid, str) or rid not in requirements or rid in seen:
            raise ValueError("Unknown or duplicate requirement check")
        seen.add(rid)
        r = requirements[rid]
        problem_start = len(problems)
        status = check["status"]
        if status not in {"supported", "refuted", "unresolved"}:
            raise ValueError("Invalid check status")
        if any(not isinstance(check[k], str) for k in ("observed", "unit", "quote")):
            raise ValueError("Invalid check text")
        if "relation" in check:
            if check["relation"] not in {"exact", "equivalent", "different", "unresolved"} or not isinstance(check["comparison_reason"], str) or not check["comparison_reason"].strip():
                raise ValueError("Invalid semantic comparison")
        ids = check["evidence_ids"]
        if not isinstance(ids, list) or any(type(i) is not int or not 1 <= i <= len(evidence) for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Invalid check evidence IDs")
        if status != "unresolved":
            quoted = quote_text(check["quote"])
            if not ids or not quoted or not any(grounded_quote(quoted, evidence[i-1]["sentence"] + " " + (evidence[i-1].get("parent_context") or "") + " " + (evidence[i-1].get("title") or "")) for i in ids):
                problems.append(rid + ": ungrounded quotation")
                status = "unresolved"
            if status == "supported" and check.get("relation") in {"different", "unresolved"}:
                problems.append(rid + ": semantic comparison is not equivalent")
                status = "unresolved"
            if status == "supported" and r["kind"] == "number":
                try:
                    matches = numeric_check_matches(r, check)
                except ValueError:
                    matches = False
                if not matches:
                    problems.append(rid + ": numeric value/unit mismatch")
                    status = "unresolved"
            if status == "supported" and r["kind"] == "text":
                semantic_match = check.get("relation") in {"exact", "equivalent"}
                numeric_pattern = r"[-+]?\d[\d,]*(?:\.\d+)?"
                expected_numbers = [number(n) for n in re.findall(numeric_pattern, r["expected"])]
                observed_numbers = [number(n) for n in re.findall(numeric_pattern, check["observed"])]
                if expected_numbers != observed_numbers:
                    semantic_match = False
                lexical_match = phrase_supported(r["expected"], check["observed"], check["quote"])
                if check.get("relation") in {"different", "unresolved"}:
                    lexical_match = False
                if not (lexical_match or semantic_match):
                    problems.append(rid + ": text value not explicitly grounded")
                    status = "unresolved"
        check["status"] = status
        if r["critical"]:
            critical.append(status)
        else:
            judgment["limitations"].extend(problems[problem_start:])
            del problems[problem_start:]
    if judgment["verdict"] == "supported" and any(s != "supported" for s in critical):
        problems.append("핵심 항목이 모두 확인되지 않음")
    if judgment["verdict"] == "refuted" and ("refuted" not in critical or "unresolved" in critical):
        problems.append("핵심 항목의 직접 반박 또는 비교 조건이 확인되지 않음")
    if problems:
        judgment["model_verdict"] = judgment["verdict"]
        judgment["verdict"] = "insufficient_evidence"
        judgment["reason"] = "일관성 검사에서 판정 보류: " + "; ".join(problems) + ". 모델 설명: " + judgment["reason"]
        judgment["limitations"].extend(problems)
    judgment["critical_missing_fields"] = [r["field"] for r in requirements.values()
        if r["critical"] and next(c for c in checks if c["requirement_id"] == r["id"])["status"] == "unresolved"]
    return judgment
