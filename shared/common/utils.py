from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping


def project_root() -> Path:
    """
    프로젝트 루트 경로를 반환한다.

    shared/ 아래의 실행 파일을 직접 돌릴 때도 항상 동일한 루트를 가리키도록
    기준점을 하나로 고정해 두는 역할을 한다.
    """
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """
    data/ 디렉터리를 반환하고, 없으면 미리 생성한다.

    크롤러 산출물과 모델 학습 산출물을 한곳에 모아두는 역할.
    """
    root = project_root()
    output_dir = root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def data_path(*parts: str) -> Path:
    """
    data/ 아래에 저장할 파일 경로를 일관된 방식으로 만든다.
    """
    return data_dir().joinpath(*parts)


def crawler_data_path(*parts: str) -> Path:
    """
    크롤러 산출물을 data/crawler/... 아래에 저장할 때 사용하는 헬퍼.
    """
    path = data_path("crawler", *parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def training_data_path(*parts: str) -> Path:
    """
    학습 결과물을 data/training/... 아래에 저장할 때 사용하는 헬퍼.
    """
    path = data_path("training", *parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_project_root_on_path() -> Path:
    """
    프로젝트 루트를 sys.path에 추가한다.

    `python shared/run_market_news_training.py`처럼 파일 경로로 직접 실행해도
    `shared.*`, `crawler.*` import가 안정적으로 동작하도록 돕는다.
    """
    root = project_root()
    root_str = str(root)

    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    return root


def write_json(payload: Mapping[str, Any], output_path: Path) -> None:
    """
    dict 형태의 메타데이터를 UTF-8 JSON으로 저장한다.

    모델 설정값, 선택된 피처, 평가 지표를 사람이 읽기 좋은 형태로 남겨두면
    나중에 결과를 재검증하거나 회귀를 추적할 때 큰 도움이 된다.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
