import re
import sys
from pathlib import Path
import pandas as pd
from transformers import pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.support_legacy.data_paths import feature_csv_path

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "PyTorch is not installed in the current environment. "
        "Install it first, for example with "
        "`pip install torch --index-url https://download.pytorch.org/whl/cpu` "
        "or the CUDA build that matches your system."
    ) from exc

# =========================
# Config
# =========================
INPUT_CSV = feature_csv_path("xlf_merged_table_sorted_encoded.csv")
OUTPUT_CSV = feature_csv_path("xlf_merged_finbert.csv")

TITLE_COL = "title"
BODY_SUMMARY_COL = "body_summary"

MODEL_NAME = "ProsusAI/finbert"

# 본문이 길면 모델 입력 한도를 넘길 수 있으므로
# 청크 길이를 보수적으로 제한한다.
MAX_CHARS_PER_CHUNK = 800

# 배치 크기 기본값.
BATCH_SIZE = 8

# 장치 설정: GPU 사용 불가하므로 CPU로 고정
DEVICE = -1

# 내부 캐시된 pipeline 객체
_CLASSIFIER = None


def get_classifier():
    """
    Return a cached transformers pipeline configured to run on CPU.

    HuggingFace `pipeline` accepts `device` as an int (cuda device id) or -1 for CPU.
    We cache the pipeline to avoid re-loading the model repeatedly.
    """
    global _CLASSIFIER
    if _CLASSIFIER is not None:
        return _CLASSIFIER

    device_arg = DEVICE
    _CLASSIFIER = pipeline(
        task="text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=device_arg,
        top_k=None,
    )
    return _CLASSIFIER


# =========================
# Helpers
# =========================
def clean_text(text: str) -> str:
    """
    입력값을 문자열로 통일하고 공백을 정리한다.
    CSV에서 읽은 NaN도 빈 문자열로 바꿔 후속 로직을 단순화한다.
    """
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_text_into_chunks(text: str, max_chars: int = 800) -> list[str]:
    """
    긴 본문을 문장 경계 기준으로 나눠 청크 리스트를 만든다.
    문장 하나가 너무 길면 max_chars 단위로 강제 분할한다.
    """
    text = clean_text(text)
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        if len(sent) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            for i in range(0, len(sent), max_chars):
                piece = sent[i:i + max_chars].strip()
                if piece:
                    chunks.append(piece)
            continue

        if len(current_chunk) + len(sent) + 1 <= max_chars:
            current_chunk = f"{current_chunk} {sent}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sent

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def empty_scores() -> dict:
    """
    텍스트가 비어 있거나 추론 결과를 만들 수 없을 때 쓰는 기본값.
    """
    return {
        "positive_prob": None,
        "negative_prob": None,
        "neutral_prob": None,
        "sentiment_score": None,
    }


def weighted_average_scores(score_dicts: list[dict], weights: list[int]) -> dict:
    """
    청크별 확률을 길이 가중 평균으로 합친다.
    짧은 청크가 긴 청크와 같은 영향력을 갖지 않도록 하기 위한 단계다.
    """
    if not score_dicts or not weights or len(score_dicts) != len(weights):
        return empty_scores()

    total_weight = sum(weights)
    if total_weight <= 0:
        return empty_scores()

    pos = sum(d["positive_prob"] * w for d, w in zip(score_dicts, weights)) / total_weight
    neg = sum(d["negative_prob"] * w for d, w in zip(score_dicts, weights)) / total_weight
    neu = sum(d["neutral_prob"] * w for d, w in zip(score_dicts, weights)) / total_weight

    return {
        "positive_prob": pos,
        "negative_prob": neg,
        "neutral_prob": neu,
        "sentiment_score": pos - neg,
    }


def extract_probs_from_output(output_one_text) -> dict:
    """
    transformers pipeline의 결과를 고정된 dict 형태로 바꾼다.
    """
    score_map = {}

    if not output_one_text:
        return empty_scores()

    # 1. 단일 dict로 들어온 경우 예외 처리 (예: {'label': 'positive', 'score': 0.95})
    if isinstance(output_one_text, dict):
        label = str(output_one_text.get("label", "")).lower()
        score = float(output_one_text.get("score", 0.0))
        if label:
            score_map[label] = score

    # 2. 정상적인 list[dict] 형태인 경우
    elif isinstance(output_one_text, list):
        for item in output_one_text:
            if isinstance(item, dict):
                label = str(item.get("label", "")).lower()
                score = float(item.get("score", 0.0))
                if label:
                    score_map[label] = score

    pos = score_map.get("positive", 0.0)
    neg = score_map.get("negative", 0.0)
    neu = score_map.get("neutral", 0.0)

    return {
        "positive_prob": pos,
        "negative_prob": neg,
        "neutral_prob": neu,
        "sentiment_score": pos - neg,
    }

def classify_texts(text_list: list[str], batch_size: int = 8) -> list[dict]:
    """
    여러 텍스트를 한 번에 배치 추론한다.
    빈 문자열은 모델에 보내지 않고 None 형태의 기본값으로 채운다.
    """
    cleaned = [clean_text(x) for x in text_list]
    non_empty_indices = [i for i, x in enumerate(cleaned) if x]
    results = [None] * len(cleaned)

    if non_empty_indices:
        # 모델 호출은 비어 있지 않은 텍스트에 대해서만 수행하되,
        # 메모리/성능 이슈를 줄이기 위해 입력을 스트리밍 배치로 나눠 처리한다.
        non_empty_texts = [cleaned[i] for i in non_empty_indices]
        clf = get_classifier()

        total = len(non_empty_texts)
        for start in range(0, total, batch_size):
            end = start + batch_size
            batch_texts = non_empty_texts[start:end]
            outputs_batch = clf(
                batch_texts,
                return_all_scores=True,
                batch_size=len(batch_texts),
                truncation=True,
            )

            # map outputs back to original indices
            for rel_idx, out in enumerate(outputs_batch):
                orig_idx = non_empty_indices[start + rel_idx]
                results[orig_idx] = extract_probs_from_output(out)

    # 빈 입력이나 실패한 항목에는 기본값을 채운다.
    for i, result in enumerate(results):
        if result is None:
            results[i] = empty_scores()

    return results


def format_title_result(result: dict) -> dict:
    """
    제목 점수를 최종 컬럼 이름에 맞춰 변환한다.
    """
    return {
        "title_positive_prob": result["positive_prob"],
        "title_negative_prob": result["negative_prob"],
        "title_neutral_prob": result["neutral_prob"],
        "title_sentiment_score": result["sentiment_score"],
    }


def analyze_titles(titles: list[str], batch_size: int = 8) -> list[dict]:
    """
    제목 전체를 한 번에 배치 추론한다.
    행별로 모델을 반복 호출하지 않아 병목을 크게 줄인다.
    """
    title_scores = classify_texts(titles, batch_size=batch_size)
    return [format_title_result(result) for result in title_scores]


def empty_body_result() -> dict:
    """
    본문이 비어 있을 때 반환할 기본 결과.
    """
    return {
        "body_positive_prob": None,
        "body_negative_prob": None,
        "body_neutral_prob": None,
        "body_sentiment_score": None,
        "body_n_chunks": 0,
    }


def analyze_bodies(bodies: list[str], max_chars: int = 800, batch_size: int = 8) -> list[dict]:
    """
    본문 전체를 청크 단위로 펼친 뒤 한 번에 배치 추론한다.
    기사별로 pipeline을 반복 호출하지 않고, 모든 청크를 모아 처리한 후
    다시 원래 행 단위 결과로 묶는다.
    """
    cleaned_bodies = [clean_text(body) for body in bodies]
    chunks_per_body = []

    # 모든 청크 문자열을 한 번에 모으지 않고, 각 기사별로 청크 정보를
    # 보관한 뒤 classify_texts에 스트리밍 배치로 전달한다.
    # 이를 위해 먼저 각 기사별 청크 리스트와 전체 청크 개수를 산정한다.
    total_chunks = 0
    for body in cleaned_bodies:
        chunks = split_text_into_chunks(body, max_chars=max_chars) if body else []
        chunks_per_body.append(chunks)
        total_chunks += len(chunks)

    results = []

    # 모든 청크를 순차적으로 처리하되 classify_texts가 내부적으로 배치 처리하므로
    # 여기서는 청크 문자열 목록을 작은 메모리로 묶어 보내는 역할만 한다.
    # 먼저 전체 non-empty chunk list를 만들되, 이 리스트는 문자열 참조이므로
    # 보통 원본 텍스트보다 훨씬 작다. 그래도 매우 큰 데이터셋이라면 추가 스트리밍을 고려할 수 있다.
    all_chunks = []
    for chunks in chunks_per_body:
        all_chunks.extend(chunks)

    all_chunk_scores = classify_texts(all_chunks, batch_size=batch_size) if all_chunks else []

    score_start = 0
    for body, chunks in zip(cleaned_bodies, chunks_per_body):
        if not body or not chunks:
            results.append(empty_body_result())
            continue

        score_end = score_start + len(chunks)
        chunk_scores = all_chunk_scores[score_start:score_end]
        score_start = score_end

        avg = weighted_average_scores(
            chunk_scores,
            [len(chunk) for chunk in chunks],
        )

        results.append({
            "body_positive_prob": avg["positive_prob"],
            "body_negative_prob": avg["negative_prob"],
            "body_neutral_prob": avg["neutral_prob"],
            "body_sentiment_score": avg["sentiment_score"],
            "body_n_chunks": len(chunks),
        })

    return results


# =========================
# Main
# =========================
def main():
    """
    CSV를 읽고 제목/본문 감성 분석 결과를 추가한 뒤 저장한다.
    """
    df = pd.read_csv(INPUT_CSV)

    required_cols = [TITLE_COL, BODY_SUMMARY_COL]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df[TITLE_COL] = df[TITLE_COL].fillna("").astype(str)
    df[BODY_SUMMARY_COL] = df[BODY_SUMMARY_COL].fillna("").astype(str)

    print(f"[INFO] Total rows: {len(df)}")
    # GPU 사용 불가 환경이므로 CPU로 고정하고 배치 크기도 하드코딩한다.
    print("[INFO] Forcing CPU inference. Batch size fixed to 8.")

    # 제목은 짧기 때문에 전체를 한 번에 배치 처리한다.
    print("[INFO] Starting title sentiment analysis")
    title_results = pd.DataFrame(
        analyze_titles(df[TITLE_COL].tolist(), batch_size=BATCH_SIZE)
    )
    df = pd.concat([df, title_results], axis=1)

    # 본문은 길 수 있으므로 청크로 나눈 뒤, 모든 청크를 배치 추론한다.
    print("[INFO] Starting body sentiment analysis")
    body_results = pd.DataFrame(
        analyze_bodies(
            df[BODY_SUMMARY_COL].tolist(),
            max_chars=MAX_CHARS_PER_CHUNK,
            batch_size=BATCH_SIZE,
        )
    )
    df = pd.concat([df, body_results], axis=1)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved: {OUTPUT_CSV}")

    preview_cols = [
        TITLE_COL,
        "title_sentiment_score",
        "body_sentiment_score",
        "body_n_chunks",
    ]
    print(df[preview_cols].head(10))


if __name__ == "__main__":
    main()
