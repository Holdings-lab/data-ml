from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    import torch
except ImportError:  # pragma: no cover - sentence-transformers usually pulls torch in.
    torch = None

from sklearn.decomposition import PCA
import joblib

# 이 스크립트는 파일 경로로 직접 실행될 수 있으므로,
# 프로젝트 루트를 import 검색 경로에 먼저 넣어야 `crawler.*` 모듈을 안정적으로 불러올 수 있다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crawler.support_legacy.data_paths import feature_csv_path


INPUT_CSV = feature_csv_path("xlv_merged_finbert.csv")
OUTPUT_CSV = feature_csv_path("xlv_merged_finbert_with_embeddings.csv")
OUTPUT_PCA = feature_csv_path("xlv_merged_finbert_pca.pkl")

SUMMARY_COL = "body_summary"
EMBEDDING_COL = "body_summary_embedding"
MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 32
PCA_DIM = 30


def clean_text(value: object) -> str:
    """
    CSV에서 읽은 값이 NaN이거나 공백만 있는 경우를 안전하게 정리한다.

    임베딩 모델은 빈 문자열도 처리할 수 있지만,
    입력값을 먼저 문자열로 통일해 두면 후속 로직이 단순해지고 예외 가능성도 줄어든다.
    """
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_feature_table(csv_path: str | Path = INPUT_CSV) -> pd.DataFrame:
    """
    `merged_finbert.csv`를 읽고, 임베딩 대상 컬럼이 있는지 확인한다.

    이 단계에서 컬럼 존재 여부를 먼저 검사하면,
    나중에 모델 로딩이나 배치 추론까지 진행한 뒤에야 실패하는 상황을 막을 수 있다.
    """
    df = pd.read_csv(csv_path)
    if SUMMARY_COL not in df.columns:
        raise KeyError(f"{SUMMARY_COL} 컬럼을 찾을 수 없습니다: {csv_path}")
    return df


def get_device() -> str:
    """
    가능한 경우 GPU를 우선 사용하고, 없으면 CPU로 돌아간다.

    sentence-transformers는 device 인자를 지원하므로,
    여기서 장치를 한 번만 결정해 두면 전체 파이프라인이 같은 장치를 일관되게 사용한다.
    """
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def tokenize_to_ids(tokenizer: object, text: str) -> list[int]:
    """
    토크나이저 구현 차이를 흡수해 토큰 ID 목록을 얻는다.

    fast tokenizer가 있으면 내부 Rust 토크나이저를 우선 사용하고,
    없으면 일반 encode 경로로 대체한다.
    """
    fast_tokenizer = getattr(tokenizer, "_tokenizer", None)
    if fast_tokenizer is not None:
        return list(fast_tokenizer.encode(text).ids)

    return list(tokenizer.encode(text, add_special_tokens=False))


def chunk_text_by_tokens(model: SentenceTransformer, text: str, max_tokens: int) -> list[str]:
    """
    긴 텍스트를 토큰 수 기준으로 잘라 모델 입력 한도를 넘지 않게 만든다.

    문자 수 기준 분할은 모델 한도를 정확히 제어하기 어렵기 때문에,
    실제 모델이 보는 기준인 토큰 단위로 잘라야 안정적으로 동작한다.
    """
    tokenizer = model.tokenizer
    token_ids = tokenize_to_ids(tokenizer, text)
    if not token_ids:
        return [""]

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    chunks: list[str] = []
    for start in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[start:start + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks or [""]


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    평균낸 벡터를 다시 정규화한다.

    긴 문장을 여러 청크로 나눠 임베딩한 뒤 평균을 내면 벡터 길이가 달라질 수 있으므로,
    마지막에 단위 벡터 형태로 맞춰 두면 샘플 간 비교가 더 일관된다.
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def encode_one_summary(model: SentenceTransformer, text: str) -> np.ndarray:
    """
    요약문 한 건을 임베딩한다.

    1. 비어 있으면 0 벡터를 반환한다.
    2. 토큰 길이가 짧으면 그대로 한 번에 임베딩한다.
    3. 토큰 길이가 길면 여러 청크로 나눈 뒤 청크 임베딩의 평균을 반환한다.
    """
    cleaned_text = clean_text(text)
    if not cleaned_text:
        # 빈 텍스트는 모델에 넣지 않고, 차원만 맞는 0 벡터로 채운다.
        return np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)

    # 모델의 최대 입력 길이보다 여유를 조금 둬서 특수 토큰을 고려한다.
    max_tokens = max(1, int(getattr(model, "max_seq_length", 512) or 512) - 2)
    token_ids = tokenize_to_ids(model.tokenizer, cleaned_text)

    if len(token_ids) <= max_tokens:
        # 짧은 문장은 청크 분할 없이 바로 임베딩한다.
        return model.encode(
            cleaned_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=get_device(),
        )

    # 긴 문장은 토큰 기준으로 여러 조각으로 나눈 뒤 각 청크를 임베딩한다.
    chunk_texts = chunk_text_by_tokens(model, cleaned_text, max_tokens=max_tokens)
    chunk_embeddings = model.encode(
        chunk_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=get_device(),
    )

    # 청크별 임베딩을 평균 내어 하나의 문서 벡터로 합친다.
    averaged = chunk_embeddings.mean(axis=0)
    return normalize_embedding(averaged.astype(np.float32))


def encode_summaries(summaries: Iterable[str], model_name: str = MODEL_NAME) -> np.ndarray:
    """
    여러 요약문을 순차적으로 임베딩한다.

    모델은 한 번만 로드하고, 각 행은 `encode_one_summary`에서 처리한다.
    이렇게 하면 긴 문장 처리 로직과 GPU 선택 로직이 모든 샘플에 동일하게 적용된다.
    """
    model = SentenceTransformer(model_name, device=get_device())
    texts = [clean_text(text) for text in summaries]
    embeddings = [encode_one_summary(model, text) for text in texts]
    return np.asarray(embeddings, dtype=np.float32)


def embed_summary_column(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    데이터프레임의 `body_summary` 컬럼을 임베딩해 결과 벡터를 만든다.

    CSV에 벡터를 직접 길게 펼치기보다는,
    원본 테이블에는 결과를 표시용 컬럼으로 남기고 실제 벡터는 별도 NPY 파일에 저장한다.
    """
    summaries = df[SUMMARY_COL].map(clean_text).tolist()
    embeddings = encode_summaries(summaries)
    df = df.copy()
    # CSV에는 사람이 읽기 쉬운 형태의 식별자만 남기고,
    # 실제 수치 벡터는 NPY로 저장한다.
    df[EMBEDDING_COL] = embeddings.tolist()
    return df, embeddings


def save_embedding_outputs(df: pd.DataFrame, csv_path: str | Path = OUTPUT_CSV) -> None:
    """
    임베딩 결과를 CSV로 저장한다.
    """
    df.to_csv(csv_path, index=False)
    print(f"CSV 저장: {csv_path}")


def load_or_fit_pca(embeddings: np.ndarray, n_components: int = PCA_DIM, pca_path: str | Path = None):
    """
    기존 PCA 모델이 있으면 로드하고, 없으면 새로 fit한다.
    
    이를 통해 모든 임베딩이 동일한 주성분 기저(basis)로 변환되어 일관성을 유지한다.
    
    Args:
        embeddings: 임베딩 배열 (fit할 때만 사용)
        n_components: PCA 차원
        pca_path: PCA 모델 경로 (기본값: OUTPUT_PCA)
    
    반환값: (pca_model, is_loaded) - 로드 여부 플래그 포함
    """
    if pca_path is None:
        pca_path = OUTPUT_PCA
    
    pca_path = Path(pca_path)
    
    # 기존 PCA 모델이 있으면 로드
    if pca_path.exists():
        pca = joblib.load(pca_path)
        return pca, True
    
    # 없으면 새로 fit
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D array")
    
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(embeddings)
    return pca, False


def reduce_embeddings(embeddings: np.ndarray, pca_model, n_components: int = PCA_DIM):
    """
    PCA로 임베딩 차원을 축소한다. (fit이 아닌 transform만 수행)

    반환값: reduced_embeddings (np.ndarray, float32)
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D array")

    reduced = pca_model.transform(embeddings)
    return reduced.astype(np.float32)


def main() -> None:
    """
    스크립트 진입점.

    입력 CSV를 읽고, 요약문 임베딩을 만들고, 결과를 디스크에 저장한다.
    기존 PCA 모델이 있으면 로드하고, 없으면 새로 fit한다.
    """
    df = load_feature_table(INPUT_CSV)
    embedded_df, embeddings = embed_summary_column(df)

    # 기존 PCA 모델 로드 또는 새로 fit
    pca_model, is_loaded = load_or_fit_pca(embeddings, n_components=PCA_DIM, pca_path=OUTPUT_PCA)
    if is_loaded:
        print(f"[INFO] 기존 PCA 모델 로드: {OUTPUT_PCA}")
    else:
        print(f"[INFO] 새로운 PCA 모델 생성 및 fit")

    # PCA로 차원 축소 (모든 임베딩이 동일한 주성분으로 변환됨)
    reduced_embeddings = reduce_embeddings(embeddings, pca_model, n_components=PCA_DIM)

    # CSV에는 축소된 벡터를 넣고 저장
    embedded_df = embedded_df.copy()
    embedded_df[EMBEDDING_COL] = reduced_embeddings.tolist()
    save_embedding_outputs(embedded_df, OUTPUT_CSV)

    # PCA 모델 저장 (첫 실행 시에는 필수, 이후는 덮어씀)
    if not is_loaded:
        joblib.dump(pca_model, OUTPUT_PCA)
        print(f"[INFO] PCA 모델 저장: {OUTPUT_PCA}")
    
    print(f"[INFO] 처리 완료: {len(df)}개 요약문 임베딩(축소)됨")


if __name__ == "__main__":
    main()