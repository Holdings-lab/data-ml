import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests

try:
    from transformers import AutoTokenizer
except ImportError as exc:
    raise ImportError(
        "transformers가 현재 환경에 설치되어 있지 않습니다. "
        "text_summarizer.py는 tokenizer 기반 청크 분할을 위해 transformers가 필요합니다."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.support_legacy.data_paths import collected_csv_path, summarized_csv_path


INPUT_CSV = collected_csv_path("fed_fomc_links.csv")
OUTPUT_CSV = summarized_csv_path("fed_fomc_links_summarized.csv")
BODY_COL = "body"
ORIGINAL_LENGTH_COL = "body_original_length"
MAX_CHARS = 10_000
SLEEP_BETWEEN_CALLS_SEC = 0.5

# Ollama 설정은 코드에 고정한다.
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"

# 청크 분할에 사용할 tokenizer와 token 기준 설정도 코드에 고정한다.
TOKENIZER_NAME = "hf-internal-testing/llama-tokenizer"
CHUNK_TOKENS = 2000
CHUNK_OVERLAP_TOKENS = 200
SLEEP_BETWEEN_CHUNK_CALLS_SEC = 0.5

LOG_FILE = os.path.join(os.path.dirname(__file__), "ollama_calls.log")
VERBOSE_LOG = False

OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _check_ollama() -> None:
    """Ollama 서버에 연결 가능한지 확인한다."""
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=10)
        _log(f"[OLLAMA] /api/tags status={resp.status_code}")
        try:
            data = resp.json()
            if isinstance(data, dict):
                models = data.get("models") or []
                _log(f"[OLLAMA] models_found={len(models)}")
        except Exception as e:
            _log(f"[OLLAMA] /api/tags json_parse_failed: {e}")
    except Exception as e:
        _log(f"[OLLAMA][ERROR] cannot reach Ollama: {e}")
        raise


def _ollama_generate(payload: Dict[str, Any], timeout_sec: int = 180) -> str:
    model = payload.get("model", "")
    prompt = payload.get("prompt", "")
    _log(f"[OLLAMA] generate start model={model} prompt_len={len(prompt)} timeout={timeout_sec}s")

    resp = requests.post(
        OLLAMA_GENERATE_URL,
        json=payload,
        timeout=timeout_sec,
    )
    _log(f"[OLLAMA] generate http_status={resp.status_code} resp_text_len={len(resp.text or '')}")
    resp.raise_for_status()

    try:
        data = resp.json()
    except Exception as e:
        preview = (resp.text or "")[:500]
        _log(f"[OLLAMA][ERROR] json_parse_failed: {e} resp_preview={preview!r}")
        raise

    response_text = (data.get("response") or "").strip()
    if VERBOSE_LOG:
        preview = response_text[:500]
        _log(f"[OLLAMA] generate done response_len={len(response_text)} preview={preview!r}")
    else:
        _log(f"[OLLAMA] generate done response_len={len(response_text)}")

    return response_text


def _build_tokenizer() -> AutoTokenizer:
    """청크 분할에 사용할 tokenizer를 준비한다."""
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


TOKENIZER = _build_tokenizer()
TOKENIZER.model_max_length = 10**9


def _encode_text(text: str) -> List[int]:
    """텍스트를 special token 없이 token ID 목록으로 인코딩한다."""
    return TOKENIZER.encode(text or "", add_special_tokens=False)


def _decode_tokens(token_ids: List[int]) -> str:
    """token ID 목록을 다시 텍스트로 복원한다."""
    return TOKENIZER.decode(token_ids, skip_special_tokens=True).strip()


def _chunk_text(text: str, chunk_tokens: int, overlap_tokens: int) -> List[str]:
    """
    문자 수가 아니라 tokenizer 기준 token 수로 텍스트를 chunk로 나눈다.
    chunk 경계에서 문맥 손실이 줄어들도록 overlap을 유지한다.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be greater than 0.")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative.")

    if overlap_tokens >= chunk_tokens:
        overlap_tokens = max(0, chunk_tokens - 1)

    text = text or ""
    token_ids = _encode_text(text)
    n_tokens = len(token_ids)
    if n_tokens <= chunk_tokens:
        return [text]

    chunks: List[str] = []
    step = max(1, chunk_tokens - overlap_tokens)
    start_idx = 0
    while start_idx < n_tokens:
        end_idx = min(n_tokens, start_idx + chunk_tokens)
        chunk_text = _decode_tokens(token_ids[start_idx:end_idx])
        if chunk_text:
            chunks.append(chunk_text)
        if end_idx >= n_tokens:
            break
        start_idx += step

    return chunks


def summarize_to_under_limit(text: str, limit_chars: int = MAX_CHARS) -> str:
    """
    입력 텍스트를 요약해 최종 결과가 문자 수 제한을 넘지 않도록 만든다.
    긴 입력은 먼저 tokenizer 기준 token 수로 나눈 뒤 단계적으로 요약한다.
    """
    text = text or ""
    if not text.strip():
        return ""

    input_tokens = len(_encode_text(text))
    if input_tokens <= CHUNK_TOKENS:
        prompt = (
            "You are a careful summarization assistant.\n\n"

            "Summarize the following document in English.\n"
            f"Maximum length: {limit_chars} characters (strict limit).\n\n"

            "Requirements:\n"
            "- Preserve only information that is explicitly stated in the document\n"
            "- Start with the main action, decision, or claim in the document\n"
            "- Include important names, actions, dates, and numbers if they are clearly stated\n"
            "- Keep the summary fact-based and neutral\n\n"

            "Rules:\n"
            "- Do not add interpretation, classification, or commentary\n"
            "- Do not infer policy stance, sentiment, intent, or implications unless explicitly stated\n"
            "- Do not include phrases such as 'Here is the summary', 'Here is the final summary', "
            "'Here is the combined summary', 'Note:', or any other meta commentary\n"
            "- Do not mention that you are summarizing or combining text\n"
            "- No bullet points\n"
            "- No headings\n"
            "- No quotation marks around the whole summary\n\n"

            "Output only the summary text.\n\n"

            "Document:\n"
            f"{text}"
        )
        summary = _ollama_generate(
            {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            }
        )
        _log(f"[SUM] single_call input_len={len(text)} input_tokens={input_tokens} output_len={len(summary)}")
        return summary[:limit_chars].rstrip()

    chunks = _chunk_text(text, chunk_tokens=CHUNK_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS)
    num_chunks = max(1, len(chunks))
    _log(
        f"[SUM] chunked input_len={len(text)} input_tokens={input_tokens} "
        f"num_chunks={num_chunks} chunk_tokens={CHUNK_TOKENS} overlap_tokens={CHUNK_OVERLAP_TOKENS}"
    )

    chunk_summary_target = int((limit_chars / num_chunks) * 0.8)
    chunk_summary_target = max(300, chunk_summary_target)
    chunk_summary_target = min(chunk_summary_target, 1200)

    chunk_summaries: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        prompt = (
            "You are a careful summarization assistant.\n\n"

            "Summarize the following text chunk in English.\n"
            f"Maximum length: {chunk_summary_target} characters (strict limit).\n\n"

            "This chunk is part of a larger source document.\n"
            "Keep only information that is explicitly stated in this chunk and is important for the full document.\n\n"

            "Requirements:\n"
            "- Capture the key action, decision, claim, or event in this chunk\n"
            "- Include important names, actions, dates, and numbers if clearly stated\n"
            "- Keep the summary factual and neutral\n\n"

            "Rules:\n"
            "- Do not add interpretation, classification, commentary, or implications\n"
            "- Do not infer policy stance, sentiment, or intent\n"
            "- Do not include phrases such as 'Here is the summary', 'Here is the final summary', "
            "'Here is the combined summary', 'Note:', or any meta explanation\n"
            "- Do not mention that this is a chunk\n"
            "- No repetition\n"
            "- No bullet points\n"
            "- No headings\n\n"

            "Output only the summary text.\n\n"

            f"Text:\n{chunk}"
        )
        summary = _ollama_generate(
            {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            }
        )
        chunk_summaries.append(summary[:chunk_summary_target].rstrip())
        _log(
            f"[SUM] chunk {idx}/{num_chunks} chunk_len={len(chunk)} "
            f"chunk_tokens={len(_encode_text(chunk))} chunk_summary_len={len(summary)} "
            f"target={chunk_summary_target}"
        )
        if idx < num_chunks:
            time.sleep(SLEEP_BETWEEN_CHUNK_CALLS_SEC)

    combined = "\n\n".join(chunk_summaries).strip()

    final_prompt = (
        "You are a careful summarization assistant.\n\n"

        "The following text consists of partial summaries from one source document.\n"
        "Write one final English summary of the source document.\n"
        f"Maximum length: {limit_chars} characters (strict limit).\n\n"

        "Requirements:\n"
        "- Preserve only information supported by the partial summaries\n"
        "- Start with the document's main action, decision, claim, or event\n"
        "- Include important names, actions, dates, and numbers if clearly present\n"
        "- Keep the summary factual, neutral, and concise\n"
        "- Merge overlapping points without adding new interpretation\n\n"

        "Rules:\n"
        "- Do not add interpretation, classification, commentary, or implications\n"
        "- Do not infer policy stance, sentiment, or intent\n"
        "- Do not include phrases such as 'Here is the summary', 'Here is the final summary', "
        "'Here is the combined summary', 'Note:', or any meta explanation\n"
        "- Do not mention chunks, combining, or summarization process\n"
        "- No bullet points\n"
        "- No headings\n"
        "- Output only the final summary text\n\n"

        "Partial summaries:\n"
        f"{combined}"
    )

    final_summary = _ollama_generate(
        {
            "model": OLLAMA_MODEL,
            "prompt": final_prompt,
            "stream": False,
        }
    )
    final_summary = final_summary[:limit_chars].rstrip()
    _log(f"[SUM] final_combined output_len={len(final_summary)} limit={limit_chars}")
    return final_summary


def main() -> None:
    _check_ollama()
    input_path = INPUT_CSV
    output_path = OUTPUT_CSV

    df = pd.read_csv(input_path, encoding="utf-8-sig")

    if BODY_COL not in df.columns:
        raise ValueError(f"Column '{BODY_COL}' not found. Available columns: {list(df.columns)}")

    df[BODY_COL] = df[BODY_COL].fillna("").astype(str)
    lengths = df[BODY_COL].str.len()
    df[ORIGINAL_LENGTH_COL] = lengths
    need_summary_mask = lengths >= MAX_CHARS

    indices = df.index[need_summary_mask].tolist()
    print(f"[INFO] {len(df)} rows, rows to summarize: {len(indices)} (>= {MAX_CHARS} chars)")

    for i, idx in enumerate(indices, start=1):
        text = df.at[idx, BODY_COL]
        _log(f"[MAIN] processing document {i}/{len(indices)} row={idx} body_len={len(text)}")

        try:
            summary = summarize_to_under_limit(text, limit_chars=MAX_CHARS)
        except Exception as e:
            print(f"[WARN] summarize failed row={idx}: {e} -> truncating to {MAX_CHARS} chars")
            summary = text[:MAX_CHARS].rstrip()

        df.at[idx, BODY_COL] = summary

        if i < len(indices):
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] saved: {output_path}")


if __name__ == "__main__":
    main()
