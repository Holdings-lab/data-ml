from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CRAWLER_DATA_DIR = DATA_DIR / "crawler"
COLLECTED_DIR = CRAWLER_DATA_DIR / "collected"
SUMMARIZED_DIR = CRAWLER_DATA_DIR / "summarized"
FEATURES_DIR = CRAWLER_DATA_DIR / "features"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_data_dir() -> Path:
    return _ensure_dir(DATA_DIR)


def collected_csv_path(filename: str) -> str:
    """
    수집기(raw collector output)가 저장되는 위치.
    """
    return str(_ensure_dir(COLLECTED_DIR) / filename)


def summarized_csv_path(filename: str) -> str:
    """
    요약이 끝난 크롤링 산출물이 저장되는 위치.
    """
    return str(_ensure_dir(SUMMARIZED_DIR) / filename)


def feature_csv_path(filename: str) -> str:
    """
    병합, 피처 엔지니어링, 감성 점수화가 끝난 학습용 CSV가 저장되는 위치.
    """
    return str(_ensure_dir(FEATURES_DIR) / filename)


def csv_path(filename: str) -> str:
    """
    레거시 코드 호환용 경로 헬퍼.

    기존 파일들이 `csv_path("...")`를 널리 사용하고 있어, 전면 수정 전에도
    새 폴더 구조로 자연스럽게 매핑되도록 분기 로직을 둔다.
    """
    collected_files = {
        "fed_fomc_links.csv",
        "whitehouse_qqq_policy.csv",
        "bis_press_releases.csv",
    }
    summarized_files = {
        "fed_fomc_links_summarized.csv",
        "whitehouse_qqq_policy_summarized.csv",
        "bis_press_releases_summarized.csv",
    }
    feature_files = {
        "merged_table_sorted.csv",
        "merged_table_sorted_encoded.csv",
        "merged_table_sorted_time_features.csv",
        "merged_finbert.csv",
        "daily_news_features.csv",
    }

    if filename in collected_files:
        return collected_csv_path(filename)

    if filename in summarized_files:
        return summarized_csv_path(filename)

    if filename in feature_files:
        return feature_csv_path(filename)

    return str(ensure_data_dir() / filename)
