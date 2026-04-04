from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import yfinance as yf

from shared.config.schema import MarketNewsTrainingConfig


FeatureBuilder = Callable[[pd.DataFrame], list[str]]


def _get_series(data: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    """
    yfinance 멀티인덱스 결과에서 특정 필드와 티커의 시리즈를 꺼낸다.

    이후 단계에서는 "멀티인덱스 구조"를 계속 신경 쓰지 않도록,
    가장 먼저 단일 시리즈로 풀어내는 역할을 맡는다.
    """
    series = data[field][ticker].copy()
    series.name = f"{ticker}_{field}"
    return series


def download_market_data(config: MarketNewsTrainingConfig) -> pd.DataFrame:
    """
    학습에 사용할 가격 데이터를 다운로드한다.

    기존 `training/train_regression.py`의 입력 가정을 유지하기 위해
    타깃 ETF 외에 거시 맥락용 자산도 함께 내려받는다.
    """
    tickers = [config.target_ticker, *config.macro_tickers]
    raw = yf.download(
        tickers=tickers,
        start=config.start_date,
        end=config.end_date,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError("yfinance returned an empty dataframe. Check ticker/date settings.")

    return raw


def _initialize_market_frame(raw: pd.DataFrame, target_ticker: str) -> pd.DataFrame:
    """
    시장 피처 생성의 출발점이 되는 기본 컬럼 테이블을 만든다.

    이 단계에서는 아직 파생 피처를 만들지 않고, 이후 함수들이 공통으로 참조할
    가격/거래량/거시 자산 컬럼만 정리한다.
    """
    price_field = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"

    df = pd.DataFrame(index=raw.index)
    df["target_price"] = _get_series(raw, price_field, target_ticker)
    df["SPY_price"] = _get_series(raw, price_field, "SPY")
    df["VIX_price"] = _get_series(raw, price_field, "^VIX")
    df["TLT_price"] = _get_series(raw, price_field, "TLT")
    df["HYG_price"] = _get_series(raw, price_field, "HYG")
    df["UUP_price"] = _get_series(raw, price_field, "UUP")
    df["target_open"] = _get_series(raw, "Open", target_ticker)
    df["target_high"] = _get_series(raw, "High", target_ticker)
    df["target_low"] = _get_series(raw, "Low", target_ticker)
    df["target_volume"] = _get_series(raw, "Volume", target_ticker)
    return df.reset_index().rename(columns={"index": "Date"})


def _add_return_and_volatility_features(df: pd.DataFrame) -> list[str]:
    """
    가장 기본적인 가격 변화율과 변동성 피처를 추가한다.

    이 그룹은 "최근 며칠 동안 얼마나 움직였는가"를 담기 때문에,
    거의 모든 회귀/분류 실험의 기반이 되는 신호다.
    """
    price = df["target_price"]
    volume = df["target_volume"]
    daily_ret = price.pct_change()

    df["ret_1"] = price.pct_change(1)
    df["ret_3"] = price.pct_change(3)
    df["ret_5"] = price.pct_change(5)
    df["ret_10"] = price.pct_change(10)
    df["ret_20"] = price.pct_change(20)
    df["vol_5"] = daily_ret.rolling(5).std()
    df["vol_10"] = daily_ret.rolling(10).std()
    df["vol_20"] = daily_ret.rolling(20).std()
    df["vol_shock"] = df["vol_5"] / (df["vol_20"] + 1e-9)
    df["vol_chg_1"] = volume.pct_change(1)
    df["vol_to_ma20"] = volume / volume.rolling(20).mean() - 1.0

    return [
        "ret_1",
        "ret_3",
        "ret_5",
        "ret_10",
        "ret_20",
        "vol_5",
        "vol_10",
        "vol_20",
        "vol_shock",
        "vol_chg_1",
        "vol_to_ma20",
    ]


def _add_trend_and_distance_features(df: pd.DataFrame) -> list[str]:
    """
    이동평균 대비 위치와 장기 추세를 설명하는 피처를 추가한다.

    팀원이 이 블록을 보면 "가격 수준이 아니라 추세 위치를 보고 있구나"를
    바로 이해할 수 있도록, 추세 계열 피처만 한데 모아둔다.
    """
    price = df["target_price"]

    for window in [5, 10, 20, 60, 120, 200]:
        moving_average = price.rolling(window).mean()
        df[f"price_to_ma_{window}"] = price / moving_average - 1.0

    for window in [20, 60, 120, 200]:
        df[f"slope_{window}"] = np.log(price / price.shift(window)) / window

    rolling_max = price.rolling(window=252, min_periods=1).max()
    df["drawdown"] = (price - rolling_max) / rolling_max
    df["dist_to_ma5"] = price / price.rolling(5).mean() - 1.0
    df["dist_10"] = price / price.rolling(10).mean() - 1.0

    return [
        "dist_to_ma5",
        "price_to_ma_5",
        "price_to_ma_10",
        "price_to_ma_20",
        "price_to_ma_60",
        "price_to_ma_120",
        "slope_20",
        "slope_60",
        "slope_120",
        "drawdown",
        "dist_10",
    ]


def _add_intraday_and_technical_features(df: pd.DataFrame) -> list[str]:
    """
    기술적 분석 계열 피처를 추가한다.

    일중 변동, RSI, MACD, 볼린저 밴드처럼 트레이딩 관점에서 자주 쓰는 지표를
    한 블록으로 묶어 두면 함수 이름만으로도 역할이 명확해진다.
    """
    price = df["target_price"]
    open_price = df["target_open"]
    high_price = df["target_high"]
    low_price = df["target_low"]

    df["intraday_range"] = (high_price - low_price) / price
    df["close_open"] = (price - open_price) / open_price

    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    relative_strength = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + relative_strength))

    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    bb_mid = price.rolling(20).mean()
    bb_std = price.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid
    df["bb_pos"] = (price - bb_lower) / (bb_upper - bb_lower)
    df["ret_accel"] = df["ret_1"] - df["ret_5"]
    df["vol_breakout"] = df["ret_1"] / (df["vol_5"] + 1e-9)
    df["bb_high_dist"] = (high_price - bb_upper) / (bb_upper + 1e-9)

    return [
        "intraday_range",
        "close_open",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_width",
        "bb_pos",
        "ret_accel",
        "vol_breakout",
        "bb_high_dist",
    ]


def _z_score(series: pd.Series, window: int = 20) -> pd.Series:
    """
    롤링 기준으로 상대적 고점/저점을 보기 위한 표준화 점수를 계산한다.
    """
    return (series - series.rolling(window).mean()) / series.rolling(window).std()


def _add_macro_context_features(df: pd.DataFrame) -> list[str]:
    """
    SPY, VIX, TLT, HYG, UUP 같은 거시 자산 기반 피처를 추가한다.

    타깃 ETF 단독 시계열만 보면 설명이 약해질 수 있어서,
    시장 전체 위험 선호와 금리/달러 환경을 함께 읽기 위한 블록이다.
    """
    spy = df["SPY_price"]
    vix = df["VIX_price"]
    tlt = df["TLT_price"]
    hyg = df["HYG_price"]
    uup = df["UUP_price"]

    df["spy_ret_1"] = spy.pct_change(1)
    df["spy_ret_5"] = spy.pct_change(5)
    df["spy_ret_20"] = spy.pct_change(20)
    df["spy_to_ma_60"] = spy / spy.rolling(60).mean() - 1.0

    df["vix_level"] = vix
    df["vix_ret_1"] = vix.pct_change(1)
    df["vix_ret_5"] = vix.pct_change(5)
    df["vix_to_ma_20"] = vix / vix.rolling(20).mean() - 1.0
    df["vix_z_score"] = _z_score(vix, 20)
    df["vix_z_score_5"] = (vix - vix.rolling(5).mean()) / (vix.rolling(5).std() + 1e-9)
    df["vix_speed"] = vix.pct_change(3)

    df["tlt_ret_1"] = tlt.pct_change(1)
    df["tlt_ret_5"] = tlt.pct_change(5)
    df["tlt_ret_20"] = tlt.pct_change(20)
    df["tlt_to_ma_60"] = tlt / tlt.rolling(60).mean() - 1.0
    df["tlt_shock_5"] = tlt.pct_change(5)

    df["hyg_ret"] = hyg.pct_change(10)
    df["uup_ret"] = uup.pct_change(10)
    df["uup_shock_5"] = uup.pct_change(5)
    df["hyg_z_score"] = _z_score(hyg, 20)
    df["uup_z_score"] = _z_score(uup, 20)

    return [
        "spy_ret_1",
        "spy_ret_5",
        "spy_ret_20",
        "spy_to_ma_60",
        "vix_level",
        "vix_ret_1",
        "vix_ret_5",
        "vix_to_ma_20",
        "tlt_ret_1",
        "tlt_ret_5",
        "tlt_ret_20",
        "tlt_to_ma_60",
        "tlt_shock_5",
        "hyg_ret",
        "uup_ret",
        "uup_shock_5",
        "hyg_z_score",
        "uup_z_score",
        "vix_z_score",
        "vix_z_score_5",
        "vix_speed",
    ]


def _add_relative_strength_features(df: pd.DataFrame) -> list[str]:
    """
    타깃 ETF가 다른 자산 대비 상대적으로 강한지 약한지를 표현하는 피처를 만든다.

    같은 상승장이어도 QQQ가 SPY보다 강한지, 채권 대비 강한지 같은 정보는
    방향성 예측에서 꽤 중요한 문맥이 될 수 있다.
    """
    price = df["target_price"]
    spy = df["SPY_price"]
    tlt = df["TLT_price"]

    df["target_spy_ratio"] = price / spy
    df["target_spy_ratio_20"] = df["target_spy_ratio"] / df["target_spy_ratio"].rolling(20).mean() - 1.0
    df["target_tlt_ratio"] = price / tlt
    df["target_tlt_ratio_20"] = df["target_tlt_ratio"] / df["target_tlt_ratio"].rolling(20).mean() - 1.0
    df["target_spy_rel_ret"] = (price / spy).pct_change(10)
    df["target_tlt_rel_ret"] = (price / tlt).pct_change(10)
    df["vol_ratio"] = price.pct_change().rolling(10).std() / (spy.pct_change().rolling(10).std() + 1e-9)
    df["rel_strength_5"] = price.pct_change(5) - spy.pct_change(5)

    return [
        "target_spy_ratio_20",
        "target_tlt_ratio_20",
        "target_spy_rel_ret",
        "target_tlt_rel_ret",
        "vol_ratio",
        "rel_strength_5",
    ]


def build_market_feature_frame(
    raw: pd.DataFrame,
    target_ticker: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    가격 원천 데이터를 모델 입력용 시장 피처 테이블로 변환한다.

    이 함수는 내부적으로 여러 세부 블록을 호출하지만, 외부에서는
    "시장 피처 테이블 하나를 만든다"라는 단일 책임으로 보이게 설계했다.
    """
    df = _initialize_market_frame(raw, target_ticker)

    feature_builders: list[FeatureBuilder] = [
        _add_return_and_volatility_features,
        _add_trend_and_distance_features,
        _add_intraday_and_technical_features,
        _add_macro_context_features,
        _add_relative_strength_features,
    ]

    market_feature_columns: list[str] = []
    for builder in feature_builders:
        market_feature_columns.extend(builder(df))

    return df, market_feature_columns
