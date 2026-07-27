from __future__ import annotations

"""
market_news 회귀 모델의 5거래일 선행 가격 예측을 수익률 regime으로 요약한다.

학습 흐름:
  1. market_news XGBoost 회귀 모델이 T+5 미래 가격을 예측한다.
  2. 예측 미래 가격을 현재 가격과 비교해 예측 수익률을 계산한다.
  3. 예측 수익률을 고정 구간으로 나눈다.
  4. 각 예측 구간별로 직전 5일 뉴스/시장 profile centroid를 요약한다.

추론 흐름:
  - 회귀 모델 예측값으로 먼저 상승/하락 수익률 구간을 정한다.
  - centroid는 그 예측 구간의 뉴스/시장 profile을 설명하는 용도로 사용한다.
"""

import ast

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


CLUSTER_BASE_FEATURE_COLS: list[str] = [
    # Market context features (kept minimal so 뉴스 signal dominates)
    "ret_5",
    "vol_5",
    "vol_ratio_5",
    "drawdown",
    "vix_z_score_5",
    "vol_shock",

    # News-derived scalar features that help label separation
    "days_since_news",
    "news_count_lag1",
    "negative_count_ratio_5d",
    "title_sentiment_3d_mean",
    "title_sentiment_5d_mean",
    "body_sentiment_decay_5d",
    "fomc_recent_5d",
    "fomc_sentiment_shock",
]

CLUSTER_EMBEDDING_PCA_COMPONENTS = 10
CLUSTER_FEATURE_COLS: list[str] = CLUSTER_BASE_FEATURE_COLS + [
    f"body_emb_cluster_pc{i}" for i in range(1, CLUSTER_EMBEDDING_PCA_COMPONENTS + 1)
]

VOLATILITY_LABELS: list[str] = [
    "fall_strong",
    "fall",
    "neutral",
    "rise",
    "rise_strong",
]

# 5-trading-day predicted simple-return thresholds:
# fall_strong < -0.3%, fall -0.3~0%, neutral 0~+0.3%,
# rise +0.3~+0.6%, rise_strong >= +0.6%
# Symmetric around 0 for fall/rise pairs; calibrated against the
# market_news model's actual prediction distribution.
FIXED_THRESHOLDS: tuple[float, ...] = (
    -0.003,
    0.0,
    0.003,
    0.006,
)


def _assign_label_fixed(predicted_return: float) -> str:
    if predicted_return >= 0.006:
        return "rise_strong"
    if predicted_return >= 0.003:
        return "rise"
    if predicted_return >= 0.0:
        return "neutral"
    if predicted_return >= -0.003:
        return "fall"
    return "fall_strong"


def _predicted_return_from_row(row: pd.Series) -> float:
    if "Pred_Future_Price" in row and "Current_Price" in row:
        return float(row["Pred_Future_Price"] / row["Current_Price"] - 1.0)
    if "Pred_LogRet" in row:
        return float(np.exp(row["Pred_LogRet"] / 100.0) - 1.0)
    raise ValueError(
        "Prediction rows must include Pred_Future_Price + Current_Price or Pred_LogRet."
    )


def _actual_return_from_row(row: pd.Series) -> float | None:
    if "Actual_Future_Price" in row and "Current_Price" in row:
        return float(row["Actual_Future_Price"] / row["Current_Price"] - 1.0)
    if "Actual_LogRet" in row:
        return float(np.exp(row["Actual_LogRet"] / 100.0) - 1.0)
    return None


def _return_direction(value: float, flat_band: float = 0.000001) -> str:
    if value > flat_band:
        return "up"
    if value < -flat_band:
        return "down"
    return "flat"


def _label_indices(labels: list[str], target: str) -> list[int]:
    return [i for i, l in enumerate(labels) if l == target]


def _aggregate_news_window(
    anchor_date: pd.Timestamp,
    news_indexed: pd.DataFrame,
    window_days: int,
    feature_columns: list[str],
) -> np.ndarray | None:
    """anchor_date 직전 window_days 달력일의 뉴스 피처를 평균 벡터로 집계한다.

    news_indexed 는 date 를 인덱스로 가진 DataFrame 이어야 한다.
    """
    cutoff = anchor_date - pd.Timedelta(days=window_days)
    upper = anchor_date - pd.Timedelta(days=1)
    try:
        window = news_indexed.loc[cutoff:upper, feature_columns]
    except KeyError:
        return None
    if window.empty:
        return None
    return window.mean(axis=0).to_numpy(dtype=float)


def _build_window_feature_dataset(
    market_df: pd.DataFrame,
    daily_news_df: pd.DataFrame,
    window_days: int = 5,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, list[pd.Timestamp]]:
    """각 anchor 거래일 직전 window_days의 feature 평균 벡터를 만든다."""
    df = market_df[["Date"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df = df.sort_values("Date").dropna(subset=["Date"]).reset_index(drop=True)

    news = daily_news_df.copy()
    resolved_feature_columns = CLUSTER_FEATURE_COLS if feature_columns is None else feature_columns
    date_column = "date" if "date" in news.columns else "Date"
    news[date_column] = pd.to_datetime(news[date_column], errors="coerce").dt.tz_localize(None)
    news_indexed = news.sort_values(date_column).set_index(date_column).sort_index()

    vectors: list[np.ndarray] = []
    dates: list[pd.Timestamp] = []

    for row in df.itertuples(index=False):
        vec = _aggregate_news_window(
            row.Date,
            news_indexed,
            window_days,
            resolved_feature_columns,
        )
        if vec is None:
            continue
        vectors.append(vec)
        dates.append(row.Date)

    return np.array(vectors, dtype=float), dates


def infer_embedding_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return body embedding columns in numeric suffix order."""
    embedding_columns = [column for column in df.columns if column.startswith("body_emb_")]

    def sort_key(column: str) -> tuple[int, str]:
        suffix = column.removeprefix("body_emb_")
        return (int(suffix), column) if suffix.isdigit() else (10**9, column)

    return sorted(embedding_columns, key=sort_key)


def transform_embedding_pca_features(
    embedding_vectors: np.ndarray,
    embedding_pca: dict,
) -> np.ndarray:
    """Transform raw embedding window vectors with a saved cluster PCA payload."""
    scaler_mean = np.array(embedding_pca["scaler_mean"], dtype=float)
    scaler_scale = np.array(embedding_pca["scaler_scale"], dtype=float)
    pca_mean = np.array(embedding_pca["pca_mean"], dtype=float)
    pca_components = np.array(embedding_pca["pca_components"], dtype=float)

    scaled = (embedding_vectors - scaler_mean) / scaler_scale
    return (scaled - pca_mean) @ pca_components.T


def inverse_transform_embedding_pca_features(
    embedding_pc_vectors: np.ndarray,
    embedding_pca: dict,
) -> np.ndarray:
    """Reconstruct raw body_emb_* coordinates from saved cluster PCA coordinates."""
    scaler_mean = np.array(embedding_pca["scaler_mean"], dtype=float)
    scaler_scale = np.array(embedding_pca["scaler_scale"], dtype=float)
    pca_mean = np.array(embedding_pca["pca_mean"], dtype=float)
    pca_components = np.array(embedding_pca["pca_components"], dtype=float)

    scaled = embedding_pc_vectors @ pca_components + pca_mean
    return scaled * scaler_scale + scaler_mean


def fit_embedding_pca_features(
    embedding_vectors: np.ndarray,
    source_columns: list[str],
    n_components: int = CLUSTER_EMBEDDING_PCA_COMPONENTS,
) -> tuple[np.ndarray, dict]:
    """Fit the cluster-only embedding PCA and return transformed PC features."""
    if embedding_vectors.ndim != 2:
        raise ValueError("Embedding vectors must be a 2D array.")
    if embedding_vectors.shape[1] == 0:
        raise ValueError("Embedding PCA requires at least one source embedding column.")

    resolved_components = min(n_components, embedding_vectors.shape[1])
    scaler = StandardScaler()
    embedding_scaled = scaler.fit_transform(embedding_vectors)
    pca = PCA(n_components=resolved_components, random_state=42)
    embedding_pc_vectors = pca.fit_transform(embedding_scaled)
    feature_columns = [f"body_emb_cluster_pc{i}" for i in range(1, resolved_components + 1)]

    payload: dict = {
        "source_columns": source_columns,
        "feature_columns": feature_columns,
        "n_components": resolved_components,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "pca_mean": pca.mean_.tolist(),
        "pca_components": pca.components_.tolist(),
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "singular_values": pca.singular_values_.tolist(),
    }
    return embedding_pc_vectors, payload


def build_predicted_return_cluster_dataset(
    market_df: pd.DataFrame,
    daily_news_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    window_days: int = 5,
    base_feature_columns: list[str] | None = None,
    embedding_feature_columns: list[str] | None = None,
    n_components: int = CLUSTER_EMBEDDING_PCA_COMPONENTS,
    embedding_pca: dict | None = None,
    pca_fit_ratio: float | None = None,
) -> tuple[np.ndarray, list[str], list[pd.Timestamp], list[str], dict, list[dict]]:
    """회귀 모델의 예측 수익률 label별 profile clustering 입력을 만든다."""
    resolved_base_columns = (
        CLUSTER_BASE_FEATURE_COLS if base_feature_columns is None else base_feature_columns
    )
    resolved_embedding_columns = (
        embedding_pca.get("source_columns", [])
        if embedding_pca is not None and embedding_feature_columns is None
        else embedding_feature_columns
    )
    if resolved_embedding_columns is None:
        resolved_embedding_columns = infer_embedding_feature_columns(daily_news_df)
    if not resolved_embedding_columns:
        raise ValueError("No body_emb_* columns are available for cluster embedding PCA.")

    base_vectors, dates = _build_window_feature_dataset(
        market_df,
        daily_news_df,
        window_days=window_days,
        feature_columns=resolved_base_columns,
    )
    embedding_vectors, embedding_dates = _build_window_feature_dataset(
        market_df,
        daily_news_df,
        window_days=window_days,
        feature_columns=resolved_embedding_columns,
    )
    if dates != embedding_dates:
        raise ValueError("Base cluster vectors and embedding vectors are not aligned.")

    if embedding_pca is None:
        if pca_fit_ratio is None:
            embedding_pc_vectors, resolved_embedding_pca = fit_embedding_pca_features(
                embedding_vectors,
                source_columns=resolved_embedding_columns,
                n_components=n_components,
            )
        else:
            pca_fit_rows = int(len(embedding_vectors) * pca_fit_ratio)
            if pca_fit_rows <= 0 or pca_fit_rows >= len(embedding_vectors):
                raise ValueError("Embedding PCA fit split is empty.")
            _, resolved_embedding_pca = fit_embedding_pca_features(
                embedding_vectors[:pca_fit_rows],
                source_columns=resolved_embedding_columns,
                n_components=n_components,
            )
            embedding_pc_vectors = transform_embedding_pca_features(
                embedding_vectors,
                resolved_embedding_pca,
            )
    else:
        embedding_pc_vectors = transform_embedding_pca_features(
            embedding_vectors,
            embedding_pca,
        )
        resolved_embedding_pca = embedding_pca

    feature_columns = resolved_base_columns + list(resolved_embedding_pca["feature_columns"])
    all_vectors = np.hstack([base_vectors, embedding_pc_vectors])
    vector_by_date = {
        pd.Timestamp(date).normalize(): all_vectors[i]
        for i, date in enumerate(dates)
    }

    predictions = prediction_df.copy()
    date_column = "Current_Date" if "Current_Date" in predictions.columns else "Date"
    predictions[date_column] = pd.to_datetime(
        predictions[date_column],
        errors="coerce",
    ).dt.tz_localize(None)
    predictions = predictions.dropna(subset=[date_column]).sort_values(date_column)

    vectors: list[np.ndarray] = []
    labels: list[str] = []
    matched_dates: list[pd.Timestamp] = []
    prediction_records: list[dict] = []

    for _, row in predictions.iterrows():
        anchor_date = pd.Timestamp(row[date_column]).normalize()
        vec = vector_by_date.get(anchor_date)
        if vec is None:
            continue

        predicted_return = _predicted_return_from_row(row)
        actual_return = _actual_return_from_row(row)
        label = _assign_label_fixed(predicted_return)

        vectors.append(vec)
        labels.append(label)
        matched_dates.append(anchor_date)
        prediction_records.append(
            {
                "date": str(anchor_date.date()),
                "target_date": (
                    None
                    if "Target_Date" not in row or pd.isna(row["Target_Date"])
                    else str(pd.Timestamp(row["Target_Date"]).date())
                ),
                "current_price": (
                    None if "Current_Price" not in row else float(row["Current_Price"])
                ),
                "predicted_future_price": (
                    None
                    if "Pred_Future_Price" not in row
                    else float(row["Pred_Future_Price"])
                ),
                "actual_future_price": (
                    None
                    if "Actual_Future_Price" not in row
                    else float(row["Actual_Future_Price"])
                ),
                "predicted_return": float(predicted_return),
                "predicted_return_pct": float(predicted_return * 100.0),
                "predicted_direction": _return_direction(predicted_return),
                "actual_return": None if actual_return is None else float(actual_return),
                "actual_return_pct": (
                    None if actual_return is None else float(actual_return * 100.0)
                ),
                "actual_direction": (
                    None if actual_return is None else _return_direction(actual_return)
                ),
                "label": label,
            }
        )

    if not vectors:
        raise ValueError("No prediction rows could be matched to cluster feature vectors.")

    return (
        np.array(vectors, dtype=float),
        labels,
        matched_dates,
        feature_columns,
        resolved_embedding_pca,
        prediction_records,
    )


def fit_news_centroids(
    vectors: np.ndarray,
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """각 레이블 그룹의 평균 스케일 벡터를 중심점으로 계산한다.

    Returns
    -------
    centroids : (n_labels, n_features) — VOLATILITY_LABELS 순서
    counts    : (n_labels,) — 각 그룹 샘플 수
    scaler    : 전체 데이터로 fit 된 StandardScaler
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(vectors)

    n_features = scaled.shape[1]
    centroids = np.zeros((len(VOLATILITY_LABELS), n_features))
    counts = np.zeros(len(VOLATILITY_LABELS), dtype=int)

    for i, label in enumerate(VOLATILITY_LABELS):
        idx = _label_indices(labels, label)
        counts[i] = len(idx)
        if idx:
            centroids[i] = scaled[np.array(idx)].mean(axis=0)

    return centroids, counts, scaler


def build_predicted_return_cluster_summary(
    dates: list[pd.Timestamp],
    labels: list[str],
    prediction_records: list[dict],
    vectors: np.ndarray,
    feature_columns: list[str],
) -> list[dict]:
    """market_news 예측 수익률 label별 count, 수익률, profile 평균을 요약한다."""
    summary: list[dict] = []
    for label in VOLATILITY_LABELS:
        idx = [i for i, predicted in enumerate(labels) if predicted == label]
        if idx:
            selected_dates = [dates[i] for i in idx]
            date_range: dict = {
                "first": str(min(selected_dates).date()),
                "last": str(max(selected_dates).date()),
            }
            selected_records = [prediction_records[i] for i in idx]
            predicted_returns = np.array(
                [record["predicted_return"] for record in selected_records],
                dtype=float,
            )
            actual_returns = np.array(
                [
                    record["actual_return"]
                    for record in selected_records
                    if record.get("actual_return") is not None
                ],
                dtype=float,
            )
            direction_hits = [
                record["predicted_direction"] == record.get("actual_direction")
                for record in selected_records
                if record.get("actual_direction") is not None
            ]
            feature_means = vectors[idx].mean(axis=0)
        else:
            date_range = {}
            predicted_returns = np.array([], dtype=float)
            actual_returns = np.array([], dtype=float)
            direction_hits = []
            feature_means = np.zeros(len(feature_columns), dtype=float)

        summary.append(
            {
                "label": label,
                "count": int(len(idx)),
                "date_range": date_range,
                "average_predicted_return_pct": (
                    None if len(predicted_returns) == 0 else float(predicted_returns.mean() * 100.0)
                ),
                "min_predicted_return_pct": (
                    None if len(predicted_returns) == 0 else float(predicted_returns.min() * 100.0)
                ),
                "max_predicted_return_pct": (
                    None if len(predicted_returns) == 0 else float(predicted_returns.max() * 100.0)
                ),
                "average_actual_return_pct": (
                    None if len(actual_returns) == 0 else float(actual_returns.mean() * 100.0)
                ),
                "direction_accuracy": (
                    None if not direction_hits else float(np.mean(direction_hits))
                ),
                "profile_feature_means": {
                    col: round(float(feature_means[j]), 4)
                    for j, col in enumerate(feature_columns)
                },
            }
        )
    return summary


def _parse_embedding_value(value: object) -> list[float] | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = ast.literal_eval(str(value))
    except (TypeError, ValueError, SyntaxError):
        return None
    if not isinstance(parsed, (list, tuple)):
        return None
    try:
        return [float(item) for item in parsed]
    except (TypeError, ValueError):
        return None


def _news_embedding_matrix(
    news_df: pd.DataFrame,
    source_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    if all(column in news_df.columns for column in source_columns):
        matrix = news_df[source_columns].to_numpy(dtype=float)
        valid_mask = np.isfinite(matrix).all(axis=1)
        return news_df.loc[valid_mask].copy(), matrix[valid_mask]

    if "body_summary_embedding" not in news_df.columns:
        raise ValueError(
            "Representative news lookup requires body_emb_* columns or "
            "body_summary_embedding in the source news dataframe."
        )

    parsed = news_df["body_summary_embedding"].map(_parse_embedding_value)
    expected_dim = len(source_columns)
    valid_mask = parsed.map(
        lambda value: value is not None and len(value) == expected_dim
    ).to_numpy()
    valid_news = news_df.loc[valid_mask].copy()
    matrix = np.array(parsed.loc[valid_mask].tolist(), dtype=float)
    return valid_news, matrix


def _safe_optional_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _safe_optional_text(value: object) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def build_representative_embedding_news(
    centroids: np.ndarray,
    counts: np.ndarray,
    scaler: StandardScaler,
    feature_columns: list[str],
    embedding_pca: dict,
    source_news_df: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, list[dict]]:
    """Find source news documents closest to each label's reconstructed embedding centroid."""
    if top_n <= 0:
        return {label: [] for label in VOLATILITY_LABELS}

    embedding_feature_columns = list(embedding_pca.get("feature_columns", []))
    if not embedding_feature_columns:
        return {label: [] for label in VOLATILITY_LABELS}

    missing_pc_columns = [
        column for column in embedding_feature_columns if column not in feature_columns
    ]
    if missing_pc_columns:
        raise ValueError(f"Cluster feature columns are missing PCA columns: {missing_pc_columns}")

    pc_indices = [feature_columns.index(column) for column in embedding_feature_columns]
    centroid_original = scaler.inverse_transform(centroids)
    centroid_pc_vectors = centroid_original[:, pc_indices]
    centroid_raw_embeddings = inverse_transform_embedding_pca_features(
        centroid_pc_vectors,
        embedding_pca,
    )

    valid_news, news_embeddings = _news_embedding_matrix(
        source_news_df,
        list(embedding_pca.get("source_columns", [])),
    )
    if len(valid_news) == 0:
        return {label: [] for label in VOLATILITY_LABELS}

    news_norm = np.linalg.norm(news_embeddings, axis=1)
    news_norm = np.where(news_norm == 0.0, 1.0, news_norm)

    date_column = "date" if "date" in valid_news.columns else "Date"
    if date_column in valid_news.columns:
        valid_news[date_column] = pd.to_datetime(
            valid_news[date_column],
            errors="coerce",
        ).dt.tz_localize(None)

    representative: dict[str, list[dict]] = {}
    for label_index, label in enumerate(VOLATILITY_LABELS):
        if label_index < len(counts) and int(counts[label_index]) == 0:
            representative[label] = []
            continue

        target = centroid_raw_embeddings[label_index]
        target_norm = np.linalg.norm(target)
        if target_norm == 0.0:
            representative[label] = []
            continue

        cosine_similarity = news_embeddings @ target / (news_norm * target_norm)
        nearest_indices = np.argsort(-cosine_similarity)[:top_n]

        label_news: list[dict] = []
        for rank, news_index in enumerate(nearest_indices, start=1):
            row = valid_news.iloc[int(news_index)]
            news_date = row.get(date_column)
            label_news.append(
                {
                    "rank": rank,
                    "similarity": round(float(cosine_similarity[news_index]), 6),
                    "date": (
                        None
                        if pd.isna(news_date)
                        else str(pd.Timestamp(news_date).date())
                    ),
                    "category": _safe_optional_text(row.get("category")),
                    "doc_type": _safe_optional_text(row.get("doc_type")),
                    "title": _safe_optional_text(row.get("title")),
                    "body_sentiment_score": _safe_optional_float(
                        row.get("body_sentiment_score")
                    ),
                    "title_sentiment_score": _safe_optional_float(
                        row.get("title_sentiment_score")
                    ),
                    "link": _safe_optional_text(row.get("link")),
                }
            )
        representative[label] = label_news

    return representative


def rank_cluster_features(
    centroids: np.ndarray,
    scaler: StandardScaler,
    feature_columns: list[str] | None = None,
    top_n: int = 10,
    reference: str = "global_mean",
) -> dict[str, list[dict[str, float]]]:
    """각 클러스터별로 기준 대비 가장 차이가 큰 피처 순서를 반환한다.

    Parameters
    ----------
    centroids : (n_labels, n_features) float 배열
    scaler : 전체 벡터에 fit 된 StandardScaler
    feature_columns : 출력 피처 이름 목록
    top_n : 반환할 상위 피처 개수
    reference : 기준값. 현재는 "global_mean"만 지원한다.
    """
    resolved_feature_columns = CLUSTER_FEATURE_COLS if feature_columns is None else feature_columns
    centroid_original = scaler.inverse_transform(centroids)

    if reference == "global_mean":
        reference_vector = scaler.mean_
    else:
        raise ValueError(f"Unsupported reference: {reference}")

    feature_diff = (centroid_original - reference_vector) / (scaler.scale_ + 1e-9)
    rankings: dict[str, list[dict[str, float]]] = {}
    for i, label in enumerate(VOLATILITY_LABELS):
        order = np.argsort(-np.abs(feature_diff[i]))
        rankings[label] = [
            {
                "feature": resolved_feature_columns[j],
                "centroid_value": float(centroid_original[i, j]),
                "z_diff": float(feature_diff[i, j]),
                "abs_z": float(abs(feature_diff[i, j])),
            }
            for j in order[:top_n]
        ]
    return rankings


def load_cluster_model(
    model_dict: dict,
) -> tuple[np.ndarray, StandardScaler]:
    """저장된 클러스터 모델 딕셔너리에서 centroids, scaler 를 복원한다."""
    centroids = np.array(model_dict["centroids"], dtype=float)
    scaler = StandardScaler()
    scaler.mean_ = np.array(model_dict["scaler_mean"], dtype=float)
    scaler.scale_ = np.array(model_dict["scaler_scale"], dtype=float)
    scaler.n_features_in_ = len(scaler.mean_)
    return centroids, scaler
