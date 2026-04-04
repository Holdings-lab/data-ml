import os
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.support_legacy.data_paths import collected_csv_path, summarized_csv_path


INPUT_CSV = collected_csv_path("bis_press_releases.csv")
OUTPUT_CSV = summarized_csv_path("bis_press_releases_summarized.csv")
BODY_COL = "body"
ORIGINAL_LENGTH_COL = "body_original_length"
MAX_CHARS = 10_000
SLEEP_BETWEEN_CALLS_SEC = 0.5

# Ollama (로컬) 설정: 환경변수로 바꿀 수 있게 해둠
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"

LOG_FILE = os.getenv(
    "OLLAMA_LOG_FILE",
    os.path.join(os.path.dirname(__file__), "ollama_calls.log"),
)
VERBOSE_LOG = os.getenv("OLLAMA_VERBOSE_LOG", "0").strip() in {"1", "true", "True", "YES", "yes"}


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # 로그 파일 기록 실패는 요약 실행에 영향을 주지 않음
        pass


def _check_ollama() -> None:
    """
    Ollama가 실제로 떠 있는지 최소한의 endpoint로 확인한다.
    """
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=10)
        _log(f"[OLLAMA] /api/tags status={resp.status_code}")
        # json 파싱 실패해도 상태코드만은 의미가 있으므로 따로 처리
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

# 긴 본문이 한 번에 들어가면 요청이 실패할 수 있어 chunk로 나눠 호출한다.
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "3500"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
SLEEP_BETWEEN_CHUNK_CALLS_SEC = float(
    os.getenv("SLEEP_BETWEEN_CHUNK_CALLS_SEC", str(SLEEP_BETWEEN_CALLS_SEC))
)


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

    # Ollama는 보통 {"response": "...", "done": true, ...} 형태로 옴
    response_text = (data.get("response") or "").strip()
    if VERBOSE_LOG:
        preview = response_text[:500]
        _log(f"[OLLAMA] generate done response_len={len(response_text)} preview={preview!r}")
    else:
        _log(f"[OLLAMA] generate done response_len={len(response_text)}")

    return response_text


def _chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    """
    문자를 기준으로 chunk를 자른다.
    overlap을 둬서 청크 경계에서 의미가 끊기는 것을 완화한다.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars는 0보다 커야 합니다.")
    if overlap_chars < 0:
        raise ValueError("overlap_chars는 0보다 작을 수 없습니다.")

    if overlap_chars >= chunk_chars:
        overlap_chars = max(0, chunk_chars - 1)

    text = text or ""
    n = len(text)
    if n <= chunk_chars:
        return [text]

    chunks: List[str] = []
    step = max(1, chunk_chars - overlap_chars)
    start = 0
    while start < n:
        end = min(n, start + chunk_chars)
        chunks.append(text[start:end])
        if end >= n:
            break
        start += step

    return chunks


def summarize_to_under_limit(text: str, limit_chars: int = MAX_CHARS) -> str:
    """
    input text를 요약해서 limit_chars 이하로 만든다.
    - 입력이 너무 길면 chunk로 나눠서 요청한다.
    - 최종 출력이 길어지면 마지막에 안전하게 자른다.
    """
    text = text or ""
    if not text.strip():
        return ""

    # 입력 길이가 작으면 단일 호출
    if len(text) <= CHUNK_CHARS:
        prompt = (
            "다음 문서를 영어로 핵심 위주로 요약해. "
            f"최종 결과는 '문자 수 {limit_chars}자 이하'가 되도록 작성해.\n\n"
            "가능하면 다음을 포함해:\n"
            "- 핵심 주장/결론\n"
            "- 중요한 수치(있다면)\n"
            "- 문서의 목적과 범위\n\n"
            "요약 대상 문서:\n"
            f"{text}"
        )
        summary = _ollama_generate(
            {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            }
        )
        _log(f"[SUM] single_call input_len={len(text)} output_len={len(summary)}")
        return summary[:limit_chars].rstrip()

    # 입력이 길면 chunk로 나눠 단계적으로 요약
    chunks = _chunk_text(text, chunk_chars=CHUNK_CHARS, overlap_chars=CHUNK_OVERLAP_CHARS)
    num_chunks = max(1, len(chunks))
    _log(f"[SUM] chunked input_len={len(text)} num_chunks={num_chunks} chunk_chars={CHUNK_CHARS} overlap_chars={CHUNK_OVERLAP_CHARS}")
    # 청크 요약 길이를 동적으로 낮춰서, 전체 요약이 지나치게 커지지 않게 한다.
    chunk_summary_target = max(200, int(limit_chars / num_chunks * 1.2))
    # 사용자가 환경변수로 더 강하게 제한하고 싶으면 상한을 걸어줄 수도 있게 한다.
    chunk_summary_target = min(chunk_summary_target, 2500)

    chunk_summaries: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        prompt = (
            f"다음 텍스트 청크를 영어로 핵심 위주로 요약해. "
            f"최종 결과는 '문자 수 {chunk_summary_target}자 이하'가 되도록 작성해.\n\n"
            "가능하면 다음을 포함해:\n"
            "- 핵심 주장/결론\n"
            "- 중요한 수치(있다면)\n\n"
            f"청크 {idx}/{num_chunks}:\n"
            f"{chunk}"
        )
        summary = _ollama_generate(
            {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            }
        )
        chunk_summaries.append(summary[:chunk_summary_target].rstrip())
        _log(f"[SUM] chunk {idx}/{num_chunks} chunk_len={len(chunk)} chunk_summary_len={len(summary)} target={chunk_summary_target}")
        if idx < num_chunks:
            time.sleep(SLEEP_BETWEEN_CHUNK_CALLS_SEC)

    combined = "\n\n".join(chunk_summaries).strip()

    final_prompt = (
        "아래는 원문을 chunk로 나눠서 얻은 여러 청크 요약이야. "
        f"이 내용을 종합해서 영어로 최종 요약을 작성해. "
        f"최종 결과는 '문자 수 {limit_chars}자 이하'가 되도록 작성해.\n\n"
        "가능하면 다음을 포함해:\n"
        "- 핵심 주장/결론\n"
        "- 중요한 수치(있다면)\n"
        "- 문서의 목적과 범위\n\n"
        "청크 요약 모음:\n"
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


def _resolve_input_path(filename: str) -> str:
    """
    입력 파일을 1) 현재 작업 디렉토리, 2) 스크립트가 있는 디렉토리 순서로 찾는다.
    """
    candidates = [filename]

    for c in candidates:
        if os.path.exists(c):
            return c

    raise FileNotFoundError(
        f"입력 CSV을 찾을 수 없습니다: '{filename}'. "
        f"현재 위치 또는 스크립트 폴더에 있어야 합니다. "
        f"스크립트 폴더: '{os.path.dirname(__file__)}'"
    )


def main() -> None:
    _check_ollama()
    input_path = _resolve_input_path(INPUT_CSV)
    output_path = OUTPUT_CSV

    df = pd.read_csv(input_path, encoding="utf-8-sig")

    if BODY_COL not in df.columns:
        raise ValueError(f"'{BODY_COL}' 컬럼을 찾을 수 없습니다. 현재 컬럼: {list(df.columns)}")

    df[BODY_COL] = df[BODY_COL].fillna("").astype(str)
    lengths = df[BODY_COL].str.len()
    df[ORIGINAL_LENGTH_COL] = lengths
    need_summary_mask = lengths >= MAX_CHARS

    indices = df.index[need_summary_mask].tolist()
    print(f"[INFO] {len(df)} rows, 요약 대상: {len(indices)} rows (>= {MAX_CHARS} chars)")

    for i, idx in enumerate(indices, start=1):
        text = df.at[idx, BODY_COL]

        try:
            summary = summarize_to_under_limit(text, limit_chars=MAX_CHARS)
        except Exception as e:
            # LLM 호출 실패 시: 최소한 길이 제한만 보장(트레이닝용으로는 품질이 낮을 수 있음)
            print(f"[WARN] 요약 실패 row={idx}: {e} -> 원문을 {MAX_CHARS}자로 절단")
            summary = text[:MAX_CHARS].rstrip()

        df.at[idx, BODY_COL] = summary

        if i < len(indices):
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
