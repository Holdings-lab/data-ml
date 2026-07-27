from __future__ import annotations

"""
뉴스 클러스터 시각화.

파이프라인에서 직접 호출하거나 단독으로 실행할 수 있다.

- 파이프라인 통합 (설정된 뉴스 창 실제 학습 벡터 사용):
      save_cluster_visualization(vectors, labels, counts, centroids, scaler, ...)

- 단독 실행 (저장된 training frame + predictions 사용):
      python shared/cluster/visualize.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless 환경 — pyplot import 전에 선언해야 한다
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.cluster.model import (
    CLUSTER_BASE_FEATURE_COLS,
    CLUSTER_FEATURE_COLS,
    VOLATILITY_LABELS,
    build_predicted_return_cluster_dataset,
    load_cluster_model,
    rank_cluster_features,
)


_LABEL_COLORS: dict[str, str] = {
    "fall_strong": "#a50026",
    "fall": "#d73027",
    "neutral": "#9e9e9e",
    "rise": "#91cf60",
    "rise_strong": "#1a9850",
}


def _assign_nearest_labels(
    points_scaled: np.ndarray,
    centroids_scaled: np.ndarray,
) -> list[str]:
    labels = []
    for pt in points_scaled:
        dists = np.linalg.norm(centroids_scaled - pt, axis=1)
        labels.append(VOLATILITY_LABELS[int(np.argmin(dists))])
    return labels


def _plot_scatter(
    ax: plt.Axes,
    pts_2d: np.ndarray,
    point_labels: list[str],
    cen_2d: np.ndarray,
    projection_name: str,
    variance_ratio: np.ndarray | None = None,
) -> None:
    for label in VOLATILITY_LABELS:
        mask = np.array([l == label for l in point_labels])
        if not mask.any():
            continue
        ax.scatter(
            pts_2d[mask, 0],
            pts_2d[mask, 1],
            c=_LABEL_COLORS.get(label, "#666666"),
            label=f"{label}  (n={int(mask.sum())})",
            alpha=0.4,
            s=18,
            linewidths=0,
        )

    for i, label in enumerate(VOLATILITY_LABELS):
        ax.scatter(
            cen_2d[i, 0],
            cen_2d[i, 1],
            c=_LABEL_COLORS.get(label, "#666666"),
            marker="*",
            s=420,
            edgecolors="black",
            linewidths=0.7,
            zorder=10,
        )
        ax.annotate(
            label,
            xy=(cen_2d[i, 0], cen_2d[i, 1]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color=_LABEL_COLORS.get(label, "#666666"),
        )

    if variance_ratio is not None and len(variance_ratio) >= 2:
        var1 = variance_ratio[0] * 100
        var2 = variance_ratio[1] * 100
        ratio_label = "variance explained" if projection_name == "PC" else "separation explained"
        ax.set_xlabel(f"{projection_name}1  ({var1:.1f}% {ratio_label})", fontsize=10)
        ax.set_ylabel(f"{projection_name}2  ({var2:.1f}% {ratio_label})", fontsize=10)
    else:
        ax.set_xlabel(f"{projection_name}1", fontsize=10)
        ax.set_ylabel(f"{projection_name}2", fontsize=10)
    ax.set_title("Predicted Return Profiles + Centroids", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.75, borderpad=0.6)
    ax.grid(True, alpha=0.22)


def _project_for_label_separation(
    vectors_scaled: np.ndarray,
    labels: list[str],
    centroids_scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray | None]:
    """Project vectors for visualization, preferring label-discriminative LDA."""
    observed_labels = [label for label in VOLATILITY_LABELS if label in set(labels)]
    if len(observed_labels) >= 3:
        try:
            lda = LinearDiscriminantAnalysis(n_components=2)
            pts_2d = lda.fit_transform(vectors_scaled, labels)
            cen_2d = lda.transform(centroids_scaled)
            return pts_2d, cen_2d, "LD", getattr(lda, "explained_variance_ratio_", None)
        except (ValueError, np.linalg.LinAlgError):
            pass

    pca = PCA(n_components=2, random_state=42)
    pca.fit(np.vstack([vectors_scaled, centroids_scaled]))
    return (
        pca.transform(vectors_scaled),
        pca.transform(centroids_scaled),
        "PC",
        pca.explained_variance_ratio_,
    )


def _plot_heatmap(
    ax: plt.Axes,
    centroids_scaled: np.ndarray,
    counts: np.ndarray,
    scaler: StandardScaler,
    feature_columns: list[str],
) -> None:
    centroid_orig = scaler.inverse_transform(centroids_scaled)
    col_min = centroid_orig.min(axis=0)
    col_max = centroid_orig.max(axis=0)
    centroid_norm = (centroid_orig - col_min) / (col_max - col_min + 1e-9)

    im = ax.imshow(centroid_norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(feature_columns)))
    ax.set_xticklabels(feature_columns, rotation=48, ha="right", fontsize=7)

    y_labels = [f"{lbl}  (n={cnt})" for lbl, cnt in zip(VOLATILITY_LABELS, counts)]
    ax.set_yticks(range(len(VOLATILITY_LABELS)))
    ax.set_yticklabels(y_labels, fontsize=8.5)

    for r in range(len(VOLATILITY_LABELS)):
        for c in range(len(feature_columns)):
            ax.text(
                c, r,
                f"{centroid_orig[r, c]:.2f}",
                ha="center", va="center",
                fontsize=5.5,
                color="black",
            )

    plt.colorbar(im, ax=ax, fraction=0.034, label="min-max normalized (per feature)")
    ax.set_title("Centroid Feature Profile", fontsize=10.5)


def _plot_rank_summary(
    ax: plt.Axes,
    rankings: dict[str, list[dict[str, float]]],
    top_n: int,
) -> None:
    ax.axis("off")

    header_lines = [
        f"Top {top_n} profile features per predicted label",
        "(+ means feature above global mean; - means below)",
        "value = centroid mean in original feature scale",
        "",
    ]
    lines = header_lines
    for label in VOLATILITY_LABELS:
        lines.append(f"{label}:")
        for item in rankings[label][:top_n]:
            sign = "+" if item["z_diff"] >= 0 else "-"
            lines.append(
                f" {item['feature'][:20]:20s} {sign}{abs(item['z_diff']):.2f}  v={item['centroid_value']:.3f}"
            )
        lines.append("")

    ax.text(
        0,
        1,
        "\n".join(lines),
        fontsize=6.8,
        fontfamily="monospace",
        va="top",
        ha="left",
    )
    ax.set_title("Cluster feature ranking", fontsize=10.5)


def save_cluster_visualization(
    vectors: np.ndarray,
    labels: list[str],
    counts: np.ndarray,
    centroids: np.ndarray,
    scaler: StandardScaler,
    output_path: Path,
    horizon: int = 5,
    window_days: int = 5,
    feature_columns: list[str] | None = None,
    top_ranked_features: int = 10,
) -> dict[str, list[dict[str, float]]]:
    """학습에 사용한 뉴스 창 벡터와 label별 중심점을 PCA 2D로 투영해 PNG로 저장한다."""
    vectors_scaled = scaler.transform(vectors)
    resolved_feature_columns = CLUSTER_FEATURE_COLS if feature_columns is None else feature_columns

    pts_2d, cen_2d, projection_name, variance_ratio = _project_for_label_separation(
        vectors_scaled,
        labels,
        centroids,
    )

    fig = plt.figure(figsize=(22, 8))
    fig.suptitle(
        f"Predicted Forward Return Regimes - {projection_name}A 2D Projection\n"
        f"horizon={horizon}d · window={window_days}d · total vectors={len(vectors)}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    gs = fig.add_gridspec(1, 3, width_ratios=[3, 2, 1.5], wspace=0.32)
    _plot_scatter(
        fig.add_subplot(gs[0]),
        pts_2d,
        labels,
        cen_2d,
        projection_name,
        variance_ratio,
    )
    _plot_heatmap(fig.add_subplot(gs[1]), centroids, counts, scaler, resolved_feature_columns)

    rankings = rank_cluster_features(
        centroids=centroids,
        scaler=scaler,
        feature_columns=resolved_feature_columns,
        top_n=top_ranked_features,
    )
    _plot_rank_summary(fig.add_subplot(gs[2]), rankings, top_ranked_features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Cluster visualization saved: {output_path}")
    return rankings


def main() -> None:
    """단독 실행 모드 - 저장된 QQQ profile cluster 산출물로 시각화를 재생성한다."""
    from shared.common.utils import training_data_path

    model_path = training_data_path("qqq", "comparison", "volatility_cluster_model.json")
    training_frame_path = training_data_path("qqq", "market_news", "training_frame.csv")
    predictions_path = training_data_path("qqq", "market_news", "predictions.csv")
    out_path = training_data_path("qqq", "comparison", "cluster_visualization.png")

    with open(model_path, encoding="utf-8") as f:
        model_dict = json.load(f)

    centroids, scaler = load_cluster_model(model_dict)
    feature_columns = model_dict.get("feature_columns", CLUSTER_FEATURE_COLS)

    training_frame = pd.read_csv(training_frame_path, encoding="utf-8-sig")
    predictions = pd.read_csv(predictions_path, encoding="utf-8-sig")
    embedding_pca = model_dict.get("embedding_pca")
    if embedding_pca is not None:
        X_raw, point_labels, _dates, feature_columns, _embedding_pca, _records = (
            build_predicted_return_cluster_dataset(
                training_frame,
                training_frame,
                predictions,
                window_days=model_dict.get("window_days", 5),
                base_feature_columns=model_dict.get(
                    "base_feature_columns",
                    CLUSTER_BASE_FEATURE_COLS,
                ),
                embedding_feature_columns=list(embedding_pca.get("source_columns", [])),
                embedding_pca=embedding_pca,
            )
        )
        counts = np.array(
            [sum(1 for l in point_labels if l == lbl) for lbl in VOLATILITY_LABELS],
            dtype=int,
        )
    else:
        X_raw = training_frame[feature_columns].dropna().to_numpy(dtype=float)
        X_scaled = scaler.transform(X_raw)

        point_labels = _assign_nearest_labels(X_scaled, centroids)
        counts = np.array(
            [sum(1 for l in point_labels if l == lbl) for lbl in VOLATILITY_LABELS],
            dtype=int,
        )

    save_cluster_visualization(
        vectors=X_raw,
        labels=point_labels,
        counts=counts,
        centroids=centroids,
        scaler=scaler,
        output_path=out_path,
        horizon=model_dict.get("horizon", 5),
        window_days=model_dict.get("window_days", 5),
        feature_columns=feature_columns,
    )


if __name__ == "__main__":
    main()
