"""
shared 공통 유틸 패키지.

여기에는 여러 하위 기능에서 함께 사용하는 경로/직렬화 보조 함수를 둔다.
"""

from shared.common.utils import (
    crawler_data_path,
    data_dir,
    data_path,
    ensure_project_root_on_path,
    project_root,
    training_data_path,
    write_json,
)

__all__ = [
    "crawler_data_path",
    "data_dir",
    "data_path",
    "ensure_project_root_on_path",
    "project_root",
    "training_data_path",
    "write_json",
]
