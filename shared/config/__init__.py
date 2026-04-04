"""
shared 설정/스키마 패키지.

실행 설정 dataclass와 공통 TypedDict 스키마를 관리한다.
"""

from shared.config.schema import DailyNewsFeatureRow, FedDocument, MarketNewsTrainingConfig, StandardNewsDocument

__all__ = [
    "DailyNewsFeatureRow",
    "FedDocument",
    "MarketNewsTrainingConfig",
    "StandardNewsDocument",
]
