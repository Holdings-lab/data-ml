from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


FIXED_MACRO_TICKERS: tuple[str, ...] = ("SPY", "^VIX", "TLT", "HYG", "UUP")
DEFAULT_SUPPLEMENTARY_TICKER_FEATURE_SUFFIXES: tuple[str, ...] = (
    "ret_5",
    "ret_20",
    "shock_5",
)

BASE_MARKET_FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_3",
    "ret_5",
    "ret_accel",
    "price_to_ma_5",
    "slope_5",
    "bb_pos_5",
    "bb_width_5",
    "macd_hist",
    "rsi_14",
    "vol_5",
    "vol_shock",
    "vix_z_score_5",
    "drawdown",
    "vol_ratio_5",
    "rel_strength_5",
    "uup_ret_5",
    "tlt_shock_5",
    "hyg_ret_5",
    "target_spy_rel_ret_5",
    "target_tlt_rel_ret_5",
)

QQQ_GROWTH_TECH_MARKET_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(
        BASE_MARKET_FEATURE_COLUMNS
        + (
            "spy_ret_20",
            "vix_speed",
            "tlt_ret_20",
            "uup_shock_5",
            "target_spy_ratio_20",
        )
    )
)

XLE_MARKET_FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_accel",
    "price_to_ma_5",
    "bb_pos_5",
    "bb_width_5",
    "vol_5",
    "vol_10",
    "drawdown",
    "vol_ratio_5",
    "rel_strength_5",
    "spy_ret_5",
    "vix_ret_5",
    "vix_z_score_5",
    "hyg_ret_5",
    "uup_ret_5",
    "target_spy_rel_ret_5",
    "uso_ret_5",
    "uso_shock_5",
    "xop_ret_5",
    "xop_shock_5",
    "oih_ret_5",
    "oih_shock_5",
    "xlb_shock_5",
)

XLF_MARKET_FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_accel",
    "price_to_ma_5",
    "price_to_ma_20",
    "slope_5",
    "bb_pos_5",
    "bb_width_5",
    "macd_hist",
    "rsi_14",
    "vol_5",
    "vol_10",
    "vol_shock",
    "drawdown",
    "vol_ratio_5",
    "rel_strength_5",
    "spy_ret_5",
    "vix_ret_5",
    "vix_z_score_5",
    "hyg_ret_5",
    "hyg_z_score",
    "uup_ret_5",
    "tlt_ret_5",
    "tlt_ret_20",
    "target_spy_rel_ret_5",
    "target_tlt_rel_ret_5",
    "target_tlt_ratio_20",
    "kbe_ret_5",
    "kbe_shock_5",
    "kre_ret_5",
    "kre_shock_5",
    "kie_ret_5",
    "kie_shock_5",
    "iai_ret_5",
    "iai_shock_5",
)


@dataclass(frozen=True)
class TickerTrainingPreset:
    name: str
    macro_tickers: tuple[str, ...]
    supplementary_ticker_feature_suffixes: tuple[str, ...]
    market_feature_columns: tuple[str, ...]
    horizon_candidates: tuple[int, ...]
    regression_style_fixed_horizon: int
    training_embedding_pca_components: int
    top_feature_count: int
    optuna_trials: int
    random_seed: int

    def as_config_overrides(self) -> dict[str, Any]:
        return {
            "preset_name": self.name,
            "macro_tickers": self.macro_tickers,
            "supplementary_ticker_feature_suffixes": (
                self.supplementary_ticker_feature_suffixes
            ),
            "market_feature_columns": self.market_feature_columns,
            "horizon_candidates": self.horizon_candidates,
            "regression_style_fixed_horizon": self.regression_style_fixed_horizon,
            "training_embedding_pca_components": self.training_embedding_pca_components,
            "top_feature_count": self.top_feature_count,
            "optuna_trials": self.optuna_trials,
            "random_seed": self.random_seed,
        }


DEFAULT_TICKER_PRESET = TickerTrainingPreset(
    name="default",
    macro_tickers=FIXED_MACRO_TICKERS,
    supplementary_ticker_feature_suffixes=DEFAULT_SUPPLEMENTARY_TICKER_FEATURE_SUFFIXES,
    market_feature_columns=BASE_MARKET_FEATURE_COLUMNS,
    horizon_candidates=(5, 7, 10, 15),
    regression_style_fixed_horizon=5,
    training_embedding_pca_components=5,
    top_feature_count=30,
    optuna_trials=200,
    random_seed=42,
)

NAMED_TICKER_TRAINING_PRESETS: dict[str, TickerTrainingPreset] = {
    "qqq_legacy": TickerTrainingPreset(
        name="qqq_legacy",
        macro_tickers=FIXED_MACRO_TICKERS,
        supplementary_ticker_feature_suffixes=DEFAULT_SUPPLEMENTARY_TICKER_FEATURE_SUFFIXES,
        market_feature_columns=BASE_MARKET_FEATURE_COLUMNS,
        horizon_candidates=(5, 7, 10, 15),
        regression_style_fixed_horizon=5,
        training_embedding_pca_components=5,
        top_feature_count=30,
        optuna_trials=200,
        random_seed=42,
    ),
    "qqq_growth_tech": TickerTrainingPreset(
        name="qqq_growth_tech",
        macro_tickers=FIXED_MACRO_TICKERS + ("XLK", "SOXX", "IWM"),
        supplementary_ticker_feature_suffixes=DEFAULT_SUPPLEMENTARY_TICKER_FEATURE_SUFFIXES,
        market_feature_columns=QQQ_GROWTH_TECH_MARKET_FEATURE_COLUMNS,
        horizon_candidates=(5, 7, 10, 15),
        regression_style_fixed_horizon=5,
        training_embedding_pca_components=5,
        top_feature_count=30,
        optuna_trials=200,
        random_seed=42,
    ),
    "xle_energy": TickerTrainingPreset(
        name="xle_energy",
        macro_tickers=FIXED_MACRO_TICKERS + ("USO", "XOP", "OIH", "XLB"),
        supplementary_ticker_feature_suffixes=(),
        market_feature_columns=XLE_MARKET_FEATURE_COLUMNS,
        horizon_candidates=(3, 5, 10, 20),
        regression_style_fixed_horizon=5,
        training_embedding_pca_components=5,
        top_feature_count=35,
        optuna_trials=30,
        random_seed=73,
    ),
    "xlf_financials": TickerTrainingPreset(
        name="xlf_financials",
        macro_tickers=FIXED_MACRO_TICKERS + ("KBE", "KRE", "KIE", "IAI"),
        supplementary_ticker_feature_suffixes=(),
        market_feature_columns=XLF_MARKET_FEATURE_COLUMNS,
        horizon_candidates=(3, 5, 10, 20),
        regression_style_fixed_horizon=5,
        training_embedding_pca_components=5,
        top_feature_count=40,
        optuna_trials=30,
        random_seed=91,
    ),
}

TICKER_AUTO_PRESET_NAMES: dict[str, str] = {
    "QQQ": "qqq_growth_tech",
    "XLE": "xle_energy",
    "XLF": "xlf_financials",
}

TICKER_TRAINING_PRESETS: dict[str, TickerTrainingPreset] = {
    ticker: NAMED_TICKER_TRAINING_PRESETS[preset_name]
    for ticker, preset_name in TICKER_AUTO_PRESET_NAMES.items()
}


def ticker_slug(ticker: str) -> str:
    normalized = ticker.lower().replace("^", "")
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return slug or "ticker"


def available_ticker_presets() -> tuple[str, ...]:
    return tuple(sorted(("default", *NAMED_TICKER_TRAINING_PRESETS)))


def get_ticker_training_preset(
    ticker: str,
    preset: str | None = "auto",
) -> TickerTrainingPreset:
    if preset is None or preset.lower() in {"none", "default"}:
        return DEFAULT_TICKER_PRESET

    normalized_preset = preset.upper()
    normalized_ticker = ticker.upper()

    if preset.lower() == "auto":
        preset_name = TICKER_AUTO_PRESET_NAMES.get(normalized_ticker)
        if preset_name is None:
            return DEFAULT_TICKER_PRESET
        return NAMED_TICKER_TRAINING_PRESETS[preset_name]

    normalized_name = preset.lower()
    if normalized_name in NAMED_TICKER_TRAINING_PRESETS:
        return NAMED_TICKER_TRAINING_PRESETS[normalized_name]

    if normalized_preset in TICKER_TRAINING_PRESETS:
        return TICKER_TRAINING_PRESETS[normalized_preset]

    for candidate in TICKER_TRAINING_PRESETS.values():
        if preset.lower() == candidate.name.lower():
            return candidate

    available = ", ".join(available_ticker_presets())
    raise ValueError(f"Unknown ticker preset: {preset}. Available presets: {available}")
