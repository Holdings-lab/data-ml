"""
shared 설정/스키마 패키지.

실행 설정 dataclass와 공통 TypedDict 스키마를 관리한다.
"""

from shared.config.schema import (
    DailyNewsFeatureRow,
    FedDocument,
    MarketNewsTrainingConfig,
    StandardNewsDocument,
    make_training_config,
)
from shared.config.ticker_presets import (
    BASE_MARKET_FEATURE_COLUMNS,
    FIXED_MACRO_TICKERS,
    TickerTrainingPreset,
    available_ticker_presets,
    get_ticker_training_preset,
)

__all__ = [
    "BASE_MARKET_FEATURE_COLUMNS",
    "DailyNewsFeatureRow",
    "FedDocument",
    "FIXED_MACRO_TICKERS",
    "MarketNewsTrainingConfig",
    "StandardNewsDocument",
    "TickerTrainingPreset",
    "available_ticker_presets",
    "get_ticker_training_preset",
    "make_training_config",
]
