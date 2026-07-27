"""
통합 데이터 파이프라인: 정책 모니터링 데이터를 수집 후처리부터 감성분석, 임베딩까지 한 번에 처리.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import anthropic
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.support_legacy.data_paths import collected_csv_path, feature_csv_path
from crawler.postprocessing.text_summarizer import summarize_to_under_limit as ollama_summarize
from crawler.postprocessing.sentiment_score import analyze_titles, analyze_bodies
from crawler.postprocessing.sentence_transformer import encode_summaries
from crawler.postprocessing.preprocessing import one_hot_encode_category

# 후처리 단계 설정
TITLE_COL = "title"
BODY_COL = "body"
BODY_SUMMARY_COL = "body_summary"
EMBEDDING_COL = f"{BODY_SUMMARY_COL}_embedding"
MAX_SUMMARY_CHARS = 2000
SLEEP_BETWEEN_SUMMARIZE_SEC = 0.5

PCA_DIM = 30
EXPECTED_CATEGORY_VALUES = ["BIS", "EIA", "FOMC", "FRASER", "UCSB", "YAHOO"]

def _sector_pca_model_path(sector: str | None) -> str:
    """섹터별 PCA 모델 경로를 반환한다."""
    sector_key = str(sector or "").strip().lower()
    if not sector_key:
        return feature_csv_path("policy_updates_pca.pkl")
    return feature_csv_path(f"{sector_key}_merged_finbert_pca.pkl")


def _resolve_pca_components(embeddings: object, requested_components: int = PCA_DIM) -> int:
    """PCA에 사용할 실제 차원 수를 데이터 크기에 맞게 제한한다."""
    if not hasattr(embeddings, "shape"):
        return max(1, requested_components)

    n_samples, n_features = embeddings.shape
    return max(1, min(requested_components, n_samples, n_features))


def _load_or_fit_pca(embeddings, requested_components: int = PCA_DIM, pca_path: str | None = None):
    """기존 PCA 모델이 있으면 재사용하고, 없으면 새로 학습한다."""
    if pca_path is None:
        pca_path = _sector_pca_model_path(None)

    pca_file = Path(pca_path)
    if pca_file.exists():
        return joblib.load(pca_file), True

    n_components = _resolve_pca_components(embeddings, requested_components=requested_components)
    pca_model = PCA(n_components=n_components, random_state=42)
    pca_model.fit(embeddings)
    return pca_model, False


def _reduce_embeddings(embeddings, pca_model):
    """학습된 PCA 모델로 임베딩 차원을 축소한다."""
    return pca_model.transform(embeddings).astype("float32")


def apply_text_summarization(df: pd.DataFrame, max_chars: int = MAX_SUMMARY_CHARS, sleep_sec: float = SLEEP_BETWEEN_SUMMARIZE_SEC) -> pd.DataFrame:
    """
    긴 본문을 요약한다.
    
    Args:
        df: 입력 데이터프레임
        max_chars: 요약 결과 최대 문자 수 제한
        sleep_sec: 요약 API 호출 간 대기 시간
    
    Returns:
        body 컬럼을 body_summary 컬럼으로 변경한 데이터프레임
    """
    df = df.copy()
    df[BODY_COL] = df[BODY_COL].fillna("").astype(str)
    
    lengths = df[BODY_COL].str.len()
    df["body_original_length"] = lengths
    
    indices = df.index.tolist()
    print(f"[SUMMARIZE] 요약할 행의 총 개수 : {len(df)} (본문 길이와 상관없이 모든 행을 처리)")
    
    if indices:
        client = anthropic.Anthropic()  # 환경 변수 사용 시 api_key 생략 가능

        for i, idx in enumerate(indices, start=1):
            text = df.at[idx, BODY_COL]

            # 본문이 비어있는 경우 LLM 호출 없이 빈 문자열 처리
            if not text.strip():
                print(f"[SUMMARIZE] 요약 스킵 {i}/{len(indices)} row={idx} (본문 비어있음)")
                continue
            print(f"[SUMMARIZE] 요약 시작 {i}/{len(indices)} row={idx} body_len={len(text)}")

            if df.at[idx, "category"] == "YAHOO":
                # system prompt: 모델의 역할, 절대적 규칙, 금지 사항 명시
                system_prompt = """
                You are a document summarization engine.

                Rules:

                - Produce an abstractive summary.
                - Rewrite everything in your own words.
                - Never copy complete sentences from the source.
                - Never reproduce long phrases from the source except proper nouns, dates, or exact numbers.
                - Compress the content substantially while preserving the factual meaning.
                - Remove redundant wording, examples, advertisements, opinions, and repetition.
                - Preserve only factual information explicitly stated.
                - Output only the summary text.
                """.strip()

                # user content: 동적 변수를 활용한 개별 요약 요청

                user_content = f"""
                Summarize the following document in English.

                The summary MUST:
                - Be significantly shorter than the original document.
                - Rewrite the content using new wording.
                - Not copy complete sentences from the source.
                - Preserve only essential factual information.
                - Begin immediately with the main action or decision.
                - Stay under {max_chars} characters.

                Document:
                {text}
                """.strip()

                try:
                    summary = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=500,
                        temperature=0.0,
                        system=system_prompt,
                        messages=[
                            {
                                "role": "user",
                                "content": user_content
                            }
                        ]
                    )
                    df.at[idx, BODY_COL] = summary.content[0].text
                except Exception as e:
                    print(f"[SUMMARIZE] 경고: 요약 실패 row={idx}: {e} -> truncating")
                    df.at[idx, BODY_COL] = text[:max_chars].rstrip()
            else:
                try:
                    summary = ollama_summarize(text, limit_chars=max_chars)
                    df.at[idx, BODY_COL] = summary
                except Exception as e:
                    print(f"[SUMMARIZE] 경고: 요약 실패 row={idx}: {e} -> truncating")
                    df.at[idx, BODY_COL] = text[:max_chars].rstrip()
            
            if i < len(indices):
                time.sleep(sleep_sec)
        
        print(f"[SUMMARIZE] 요약 완료: {len(indices)}개 행 요약됨")
    else:
        print(f"[SUMMARIZE] 요약할 행이 존재하지 않음")

    df.rename(columns={BODY_COL: BODY_SUMMARY_COL}, inplace=True)
    return df


def apply_one_hot_encoding(df: pd.DataFrame, category_col: str = "category") -> pd.DataFrame:
    """
    카테고리 컬럼을 one-hot 인코딩한다.
    """
    df = df.copy()
    try:
        df = one_hot_encode_category(
            df,
            prefix=f"{category_col}",
            expected_categories=EXPECTED_CATEGORY_VALUES,
        )
        print(f"[ONEHOT] One-hot Encoding: applied to {category_col}")
    except Exception as e:
        print(f"[ONEHOT] WARN: one-hot encoding failed: {e}")

    return df


def apply_sentiment_analysis(df: pd.DataFrame, batch_size: int = 8) -> pd.DataFrame:
    """
    제목과 본문에 감성 분석을 적용한다.
    """
    df = df.copy()
    
    if TITLE_COL not in df.columns:
        print(f"[SENTIMENT] WARN: title column '{TITLE_COL}' not found, skipping sentiment analysis")
        return df
    
    if BODY_SUMMARY_COL not in df.columns:
        print(f"[SENTIMENT] WARN: body_summary column '{BODY_SUMMARY_COL}' not found, skipping sentiment analysis")
        return df
    
    df[TITLE_COL] = df[TITLE_COL].fillna("").astype(str)
    df[BODY_SUMMARY_COL] = df[BODY_SUMMARY_COL].fillna("").astype(str)
    
    print(f"[SENTIMENT] Sentiment Analysis: analyzing {len(df)} rows...")
    
    # 제목 분석
    title_results = analyze_titles(df[TITLE_COL].tolist(), batch_size=batch_size)
    for col, values in zip(["title_positive_prob", "title_negative_prob", "title_neutral_prob", "title_sentiment_score"], zip(*[r.values() for r in title_results])):
        df[col] = list(values)
    
    # 본문 분석
    body_results = analyze_bodies(df[BODY_SUMMARY_COL].tolist(), max_chars=800, batch_size=batch_size)
    for col in ["body_positive_prob", "body_negative_prob", "body_neutral_prob", "body_sentiment_score", "body_n_chunks"]:
        df[col] = [r[col] for r in body_results]
    
    print(f"[SENTIMENT] Sentiment Analysis: complete")
    
    return df


def apply_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    본문 요약에 대해 임베딩을 생성한다.
    """
    df = df.copy()
    
    if BODY_SUMMARY_COL not in df.columns:
        print(f"[EMBEDDING] WARN: body_summary column '{BODY_SUMMARY_COL}' not found, skipping embeddings")
        return df
    
    print(f"[EMBEDDING] Embeddings: encoding {len(df)} rows...")

    if "sector" not in df.columns:
        df["sector"] = ""

    df["sector"] = df["sector"].fillna("").astype(str)
    df[EMBEDDING_COL] = None

    for sector_value, sector_df in df.groupby("sector", dropna=False, sort=False):
        sector_label = str(sector_value or "").strip()
        pca_path = _sector_pca_model_path(sector_label)

        summaries = sector_df[BODY_SUMMARY_COL].fillna("").astype(str).tolist()
        embeddings = encode_summaries(summaries)

        pca_model, is_loaded = _load_or_fit_pca(embeddings, requested_components=PCA_DIM, pca_path=pca_path)
        reduced_embeddings = _reduce_embeddings(embeddings, pca_model)

        if not is_loaded:
            joblib.dump(pca_model, pca_path)
            print(f"[UNIFIED] PCA: saved new model to {pca_path}")

        reduced_vectors = reduced_embeddings.tolist()
        for row_index, vector in zip(sector_df.index, reduced_vectors):
            df.at[row_index, EMBEDDING_COL] = vector
        print(
            f"[EMBEDDING] Embeddings: sector={sector_label or 'default'} "
            f"shape={embeddings.shape} -> PCA shape={reduced_embeddings.shape}"
        )
    
    return df


def apply_unified_pipeline(
    df: pd.DataFrame,
    include_summarization: bool = True,
    include_encoding: bool = True,
    include_sentiment: bool = True,
    include_embeddings: bool = True,
) -> pd.DataFrame:
    """데이터프레임에 통합 후처리 파이프라인을 적용합니다.

    인자:
        df: 처리할 DataFrame.
        include_*: 각 처리 단계(include_summarization, include_encoding, include_sentiment, include_embeddings)를 활성화하는 플래그.

    반환:
        처리된 DataFrame. 이 함수는 파일을 저장하지 않으며,
        결과 저장은 호출자가 담당합니다.
    """
    if df is None:
        print("[UNIFIED] ERROR: input dataframe is None")
        return pd.DataFrame()

    df = df.copy()
    print(f"[UNIFIED] Loaded dataframe: {len(df)} rows, {len(df.columns)} columns")
    
    # 단계별 처리
    if include_summarization:
        print("[UNIFIED] Step 1/4: Text Summarization")
        df = apply_text_summarization(df)
    
    if include_encoding:
        print("[UNIFIED] Step 2/4: One-hot Encoding")
        df = apply_one_hot_encoding(df)
    
    if include_sentiment:
        print("[UNIFIED] Step 3/4: Sentiment Analysis")
        df = apply_sentiment_analysis(df)
    
    if include_embeddings:
        print("[UNIFIED] Step 4/4: Embeddings")
        df = apply_embeddings(df)
    
    return df


def main() -> None:
    """CLI 진입점: 기본값으로 전체 파이프라인 실행."""
    print("[UNIFIED] Starting unified postprocessing pipeline...")
    
    try:
        input_path = collected_csv_path("policy_updates_monitor.csv")
        if not Path(input_path).exists():
            print(f"[UNIFIED] ERROR: input file not found: {input_path}")
            return

        df = pd.read_csv(input_path, encoding="utf-8-sig")
        df = apply_unified_pipeline(df=df)
        print(f"[UNIFIED] SUCCESS: processed {len(df)} rows")
    except Exception as e:
        print(f"[UNIFIED] FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
