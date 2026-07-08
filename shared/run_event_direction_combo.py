from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.common.utils import training_data_path, write_json
from shared.config.schema import make_training_config
from shared.config.ticker_presets import ticker_slug
from shared.run_quarterly_event_walkforward import run_quarterly_walk_forward


DEFAULT_HORIZON = 2
DEFAULT_RANDOM_SEED = 42
DEFAULT_START_QUARTER = "2022Q1"
DEFAULT_EVENT_COMMON_START_DATE = "2017-03-10"
DEFAULT_DIRECTION_COMMON_START_DATE = "2000-01-01"
DEFAULT_EVENT_THRESHOLD_PCT = 2.0
DEFAULT_EVENT_GATE = "drawdown_tertile_threshold"
DEFAULT_EVENT_SCORE_THRESHOLD = 0.60
DEFAULT_EVENT_NEWS_WEIGHT = 0.50
DEFAULT_DRAWDOWN_TERTILE_CUTPOINTS = (-0.0941599, -0.0192656)
DEFAULT_DRAWDOWN_TERTILE_THRESHOLDS = (0.50, 0.65, 0.70)
DEFAULT_PRECISION_GATE_SCORE = "average"
DEFAULT_TARGET_EVENT_PRECISION = 0.50
DEFAULT_PRECISION_GATE_MIN_RECALL = 0.20
DEFAULT_PRECISION_GATE_MIN_WARNINGS = 10
DEFAULT_PRECISION_GATE_WARMUP_QUARTERS = 4
DEFAULT_PRECISION_GATE_CALIBRATION_QUARTERS = 8


def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def _safe_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    target = np.asarray(target, dtype=int)
    if len(np.unique(target)) < 2:
        return None
    return float(roc_auc_score(target, score))


def _safe_average_precision(target: np.ndarray, score: np.ndarray) -> float | None:
    target = np.asarray(target, dtype=int)
    if len(np.unique(target)) < 2:
        return None
    return float(average_precision_score(target, score))


def _safe_brier(target: np.ndarray, score: np.ndarray) -> float | None:
    target = np.asarray(target, dtype=int)
    if len(target) == 0:
        return None
    return float(brier_score_loss(target, score))


def _horizon_root_name(prefix: str, horizon: int, suffix: str) -> str:
    return f"{prefix}_h{horizon}_{suffix}"


def default_event_root(target_ticker: str, horizon: int) -> Path:
    return training_data_path(
        ticker_slug(target_ticker),
        _horizon_root_name(
            "event_walkforward_quarterly",
            horizon,
            "quality_news_ranking",
        ),
        "report.json",
    ).parent


def default_direction_root(target_ticker: str, horizon: int) -> Path:
    return training_data_path(
        ticker_slug(target_ticker),
        _horizon_root_name(
            "event_walkforward_quarterly",
            horizon,
            "market_only_long_current_features",
        ),
        "report.json",
    ).parent


def default_combo_root(
    target_ticker: str,
    horizon: int,
    event_gate: str = DEFAULT_EVENT_GATE,
) -> Path:
    gate_slug = event_gate.lower().replace("_", "-")
    if event_gate == "news":
        suffix = "news_quality_event_market_long_raw_direction"
    else:
        suffix = f"{gate_slug}_event_market_long_raw_direction"
    return training_data_path(
        ticker_slug(target_ticker),
        _horizon_root_name(
            "event_direction_combo",
            horizon,
            suffix,
        ),
        "report.json",
    ).parent


def default_event_training_frame(target_ticker: str) -> Path:
    return training_data_path(
        ticker_slug(target_ticker),
        "market_news",
        "training_frame.csv",
    )


def default_direction_training_frame(target_ticker: str) -> Path:
    return training_data_path(
        ticker_slug(target_ticker),
        "market_only_long",
        "training_frame.csv",
    )


def _event_predictions_path(root: Path) -> Path:
    return root / "activity_sentiment_oos_predictions.csv"


def _direction_predictions_path(root: Path) -> Path:
    return root / "market_only_oos_predictions.csv"


def _binary_event_metrics(
    name: str,
    actual_event: np.ndarray,
    predicted_event: np.ndarray,
    probability: np.ndarray,
) -> dict:
    actual_event = np.asarray(actual_event, dtype=bool)
    predicted_event = np.asarray(predicted_event, dtype=bool)
    probability = np.asarray(probability, dtype=float)
    tp = int((actual_event & predicted_event).sum())
    fp = int((~actual_event & predicted_event).sum())
    fn = int((actual_event & ~predicted_event).sum())
    tn = int((~actual_event & ~predicted_event).sum())
    return {
        "Model": name,
        "Rows": int(len(actual_event)),
        "Actual_Events": int(actual_event.sum()),
        "Predicted_Events": int(predicted_event.sum()),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": float(
            precision_score(actual_event, predicted_event, zero_division=0)
        ),
        "Recall": float(recall_score(actual_event, predicted_event, zero_division=0)),
        "Accuracy": float((tp + tn) / len(actual_event)) if len(actual_event) else None,
        "Balanced_Accuracy": (
            float(balanced_accuracy_score(actual_event, predicted_event))
            if len(np.unique(actual_event)) > 1
            else None
        ),
        "AUC": _safe_auc(actual_event, probability),
        "PR_AUC": _safe_average_precision(actual_event, probability),
        "Brier": _safe_brier(actual_event, probability),
    }


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def _summarize_daily_returns(
    strategy: str,
    returns: np.ndarray,
    cost_bps: float,
    active_signal_days: int,
    average_exposure: float,
    cumulative_return_method: str,
) -> dict:
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return {
            "Strategy": strategy,
            "Cost_bps": float(cost_bps),
            "Days": 0,
            "Active_Signal_Days": int(active_signal_days),
            "Average_Abs_Exposure": float(average_exposure),
            "Total_Return": None,
            "Annualized_Return": None,
            "Annualized_Volatility": None,
            "Sharpe": None,
            "Max_Drawdown": None,
            "Ending_Equity": None,
            "Cumulative_Return_Method": cumulative_return_method,
        }

    equity = np.cumprod(1.0 + returns)
    total_return = float(equity[-1] - 1.0)
    years = len(returns) / 252.0
    annualized_return = (
        float((1.0 + total_return) ** (1.0 / years) - 1.0)
        if total_return > -1.0 and years > 0.0
        else None
    )
    volatility = (
        float(np.std(returns, ddof=1) * math.sqrt(252.0))
        if len(returns) > 1
        else None
    )
    sharpe = (
        float(np.mean(returns) / np.std(returns, ddof=1) * math.sqrt(252.0))
        if len(returns) > 1 and np.std(returns, ddof=1) > 0.0
        else None
    )
    return {
        "Strategy": strategy,
        "Cost_bps": float(cost_bps),
        "Days": int(len(returns)),
        "Active_Signal_Days": int(active_signal_days),
        "Average_Abs_Exposure": float(average_exposure),
        "Total_Return": total_return,
        "Annualized_Return": annualized_return,
        "Annualized_Volatility": volatility,
        "Sharpe": sharpe,
        "Max_Drawdown": _max_drawdown(equity),
        "Ending_Equity": float(equity[-1]),
        "Cumulative_Return_Method": cumulative_return_method,
    }


def rolling_horizon_daily_returns(
    current_price: np.ndarray,
    signal: np.ndarray,
    horizon: int,
    cost_bps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert overlapping T+N signals into a daily rolling-sleeve portfolio.

    A new 1 / horizon sleeve is opened each row and held for `horizon`
    trading days. This avoids treating overlapping T+N forward returns as
    independent full-capital daily returns.
    """

    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    prices = np.asarray(current_price, dtype=float)
    signals = np.asarray(signal, dtype=float)
    if len(prices) != len(signals):
        raise ValueError("current_price and signal must have equal length.")
    if len(prices) < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    daily_returns = prices[1:] / prices[:-1] - 1.0
    return_count = len(daily_returns)
    active_position = np.zeros(return_count, dtype=float)
    costs = np.zeros(return_count, dtype=float)
    cost_rate = float(cost_bps) / 10000.0

    for index in range(return_count):
        first_active = max(0, index - horizon + 1)
        active_position[index] = signals[first_active : index + 1].sum() / horizon

        entering_signal = signals[index]
        exiting_signal = signals[index - horizon] if index - horizon >= 0 else 0.0
        costs[index] = abs(entering_signal - exiting_signal) / horizon * cost_rate

    if cost_rate > 0.0:
        final_open_start = max(0, return_count - horizon + 1)
        final_open_signals = signals[final_open_start:return_count]
        costs[-1] += float(np.abs(final_open_signals).sum() / horizon * cost_rate)

    strategy_returns = active_position * daily_returns - costs
    return strategy_returns, active_position


def buy_and_hold_daily_returns(
    current_price: np.ndarray,
    cost_bps: float = 0.0,
) -> np.ndarray:
    prices = np.asarray(current_price, dtype=float)
    if len(prices) < 2:
        return np.asarray([], dtype=float)
    returns = prices[1:] / prices[:-1] - 1.0
    cost_rate = float(cost_bps) / 10000.0
    if cost_rate > 0.0 and len(returns):
        returns = returns.copy()
        returns[0] -= cost_rate
        returns[-1] -= cost_rate
    return returns


def _return_summary(
    combo: pd.DataFrame,
    horizon: int,
    cost_bps_values: tuple[float, ...],
) -> pd.DataFrame:
    price = combo["Current_Price"].to_numpy(dtype=float)
    signal = combo["Model_Signal"].to_numpy(dtype=float)
    rows: list[dict] = []
    for cost_bps in cost_bps_values:
        model_returns, exposure = rolling_horizon_daily_returns(
            price,
            signal,
            horizon,
            cost_bps=cost_bps,
        )
        rows.append(
            _summarize_daily_returns(
                "combo_event_cash_direction_long_short",
                model_returns,
                cost_bps,
                active_signal_days=int((signal != 0.0).sum()),
                average_exposure=float(np.mean(np.abs(exposure))) if len(exposure) else 0.0,
                cumulative_return_method="rolling_horizon_daily_sleeves",
            )
        )

        buy_hold_returns = buy_and_hold_daily_returns(price, cost_bps=cost_bps)
        rows.append(
            _summarize_daily_returns(
                "buy_and_hold",
                buy_hold_returns,
                cost_bps,
                active_signal_days=int(len(combo)),
                average_exposure=1.0,
                cumulative_return_method="daily_buy_and_hold",
            )
        )
    return pd.DataFrame(rows)


def _yearly_return_summary(
    combo: pd.DataFrame,
    horizon: int,
    cost_bps: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for year, group in combo.groupby(combo["Current_Date"].dt.year):
        price = group["Current_Price"].to_numpy(dtype=float)
        signal = group["Model_Signal"].to_numpy(dtype=float)
        model_returns, exposure = rolling_horizon_daily_returns(
            price,
            signal,
            horizon,
            cost_bps=cost_bps,
        )
        rows.append(
            {
                **_summarize_daily_returns(
                    "combo_event_cash_direction_long_short",
                    model_returns,
                    cost_bps,
                    active_signal_days=int((signal != 0.0).sum()),
                    average_exposure=(
                        float(np.mean(np.abs(exposure))) if len(exposure) else 0.0
                    ),
                    cumulative_return_method="rolling_horizon_daily_sleeves",
                ),
                "Year": int(year),
            }
        )

        buy_hold_returns = buy_and_hold_daily_returns(price, cost_bps=cost_bps)
        rows.append(
            {
                **_summarize_daily_returns(
                    "buy_and_hold",
                    buy_hold_returns,
                    cost_bps,
                    active_signal_days=int(len(group)),
                    average_exposure=1.0,
                    cumulative_return_method="daily_buy_and_hold",
                ),
                "Year": int(year),
            }
        )
    return pd.DataFrame(rows)


def _event_score(
    combo: pd.DataFrame,
    score_mode: str,
) -> np.ndarray:
    news_probability = combo["News_Event_Probability"].to_numpy(dtype=float)
    market_probability = combo["Market_Long_Event_Probability"].to_numpy(dtype=float)
    if score_mode == "news":
        return news_probability
    if score_mode == "market":
        return market_probability
    if score_mode == "average":
        return 0.5 * news_probability + 0.5 * market_probability
    if score_mode == "maximum":
        return np.maximum(news_probability, market_probability)
    raise ValueError(
        "score_mode must be one of: news, market, average, maximum"
    )


def _weighted_event_score(
    combo: pd.DataFrame,
    event_news_weight: float,
) -> np.ndarray:
    news_probability = combo["News_Event_Probability"].to_numpy(dtype=float)
    market_probability = combo["Market_Long_Event_Probability"].to_numpy(dtype=float)
    resolved_weight = float(event_news_weight)
    if not 0.0 <= resolved_weight <= 1.0:
        raise ValueError("event_news_weight must be between 0 and 1.")
    return resolved_weight * news_probability + (1.0 - resolved_weight) * market_probability


def _attach_event_gate_features(
    combo: pd.DataFrame,
    feature_frame_path: Path | None,
) -> pd.DataFrame:
    if feature_frame_path is None:
        return combo
    frame = pd.read_csv(feature_frame_path, encoding="utf-8-sig", parse_dates=["Date"])
    available = [column for column in ("Date", "drawdown") if column in frame.columns]
    if "Date" not in available:
        raise ValueError("Event gate feature frame must include a Date column.")
    if len(available) == 1:
        return combo
    feature_frame = frame[available].rename(columns={"Date": "Current_Date"})
    return combo.merge(feature_frame, on="Current_Date", how="left", validate="one_to_one")


def _threshold_event_metrics(
    target: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict:
    target = np.asarray(target, dtype=bool)
    score = np.asarray(score, dtype=float)
    prediction = score >= threshold
    tp = int((prediction & target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    tn = int((~prediction & ~target).sum())
    predicted_count = int(prediction.sum())
    actual_count = int(target.sum())
    precision = float(tp / predicted_count) if predicted_count else 0.0
    recall = float(tp / actual_count) if actual_count else 0.0
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if precision + recall > 0.0
        else 0.0
    )
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    balanced = float(0.5 * (recall + specificity))
    return {
        "threshold": float(threshold),
        "predicted_count": predicted_count,
        "actual_count": actual_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced,
    }


def _select_precision_gate_threshold(
    target: np.ndarray,
    score: np.ndarray,
    target_precision: float,
    minimum_recall: float,
    minimum_warnings: int,
) -> dict:
    target = np.asarray(target, dtype=bool)
    score = np.asarray(score, dtype=float)
    if len(target) == 0:
        return {
            "threshold": float("inf"),
            "status": "empty_calibration",
            "target_precision_met": False,
            "predicted_count": 0,
            "actual_count": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "balanced_accuracy": 0.0,
        }

    finite_score = score[np.isfinite(score)]
    if len(finite_score) == 0:
        return {
            "threshold": float("inf"),
            "status": "invalid_score",
            "target_precision_met": False,
            "predicted_count": 0,
            "actual_count": int(target.sum()),
            "tp": 0,
            "fp": 0,
            "fn": int(target.sum()),
            "tn": int((~target).sum()),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "balanced_accuracy": 0.0,
        }

    grid = np.unique(
        np.concatenate(
            [
                np.arange(0.05, 0.951, 0.01),
                np.quantile(finite_score, np.linspace(0.05, 0.95, 19)),
            ]
        )
    )
    metrics = [
        _threshold_event_metrics(target, score, float(threshold))
        for threshold in grid
    ]
    warnings_floor = max(1, int(minimum_warnings))
    usable = [
        row
        for row in metrics
        if row["predicted_count"] >= warnings_floor
    ]
    if not usable:
        usable = [row for row in metrics if row["predicted_count"] > 0]
    if not usable:
        best = _threshold_event_metrics(target, score, float("inf"))
        return {
            **best,
            "status": "no_warnings_available",
            "target_precision_met": False,
        }

    target_met = [
        row
        for row in usable
        if row["precision"] + 1e-12 >= target_precision
        and row["recall"] + 1e-12 >= minimum_recall
    ]
    if target_met:
        best = max(
            target_met,
            key=lambda row: (
                row["f1"],
                row["recall"],
                row["precision"],
                -row["threshold"],
            ),
        )
        return {
            **best,
            "status": "target_met",
            "target_precision_met": True,
        }

    precision_only = [
        row
        for row in usable
        if row["precision"] + 1e-12 >= target_precision
    ]
    if precision_only:
        best = max(
            precision_only,
            key=lambda row: (
                row["recall"],
                row["f1"],
                row["precision"],
                -row["threshold"],
            ),
        )
        return {
            **best,
            "status": "target_precision_met_recall_missed",
            "target_precision_met": True,
        }

    best = max(
        usable,
        key=lambda row: (
            row["precision"],
            row["f1"],
            row["recall"],
            -row["threshold"],
        ),
    )
    return {
        **best,
        "status": "fallback_best_precision",
        "target_precision_met": False,
    }


def _apply_precision_target_gate(
    combo: pd.DataFrame,
    score: np.ndarray,
    target_precision: float,
    minimum_recall: float,
    minimum_warnings: int,
    warmup_quarters: int,
    calibration_quarters: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    score = np.asarray(score, dtype=float)
    prediction = np.zeros(len(combo), dtype=bool)
    threshold_values = np.full(len(combo), np.nan, dtype=float)
    history: list[dict] = []
    quarters = list(dict.fromkeys(combo["Quarter"].tolist()))
    actual_event = combo["Actual_Event"].to_numpy(dtype=bool)

    for index, quarter in enumerate(quarters):
        test_mask = combo["Quarter"].eq(quarter).to_numpy()
        test_start = combo.loc[test_mask, "Current_Date"].min()
        if index < warmup_quarters:
            history.append(
                {
                    "Quarter": quarter,
                    "Status": "warmup_no_alerts",
                    "Threshold": None,
                    "Target_Precision": float(target_precision),
                    "Calibration_Quarters": [],
                    "Calibration_Rows": 0,
                    "Calibration_Actual_Events": 0,
                    "Calibration_Warnings": 0,
                    "Calibration_TP": 0,
                    "Calibration_FP": 0,
                    "Calibration_Precision": None,
                    "Calibration_Recall": None,
                    "Calibration_F1": None,
                    "Target_Precision_Met": False,
                    "Test_Warnings": 0,
                    "Test_Actual_Events": int(actual_event[test_mask].sum()),
                    "Test_TP": 0,
                    "Test_FP": 0,
                    "Test_FN": int(actual_event[test_mask].sum()),
                    "Test_Precision": None,
                    "Test_Recall": 0.0,
                }
            )
            continue

        start_index = 0 if calibration_quarters <= 0 else max(0, index - calibration_quarters)
        allowed_quarters = quarters[start_index:index]
        calibration_mask = (
            combo["Quarter"].isin(allowed_quarters).to_numpy()
            & (combo["Target_Date"] < test_start).to_numpy()
        )
        selected = _select_precision_gate_threshold(
            actual_event[calibration_mask],
            score[calibration_mask],
            target_precision=target_precision,
            minimum_recall=minimum_recall,
            minimum_warnings=minimum_warnings,
        )
        threshold = float(selected["threshold"])
        test_prediction = score[test_mask] >= threshold
        prediction[test_mask] = test_prediction
        threshold_values[test_mask] = threshold

        test_actual = actual_event[test_mask]
        test_tp = int((test_prediction & test_actual).sum())
        test_fp = int((test_prediction & ~test_actual).sum())
        test_fn = int((~test_prediction & test_actual).sum())
        test_warnings = int(test_prediction.sum())
        history.append(
            {
                "Quarter": quarter,
                "Status": selected["status"],
                "Threshold": threshold,
                "Target_Precision": float(target_precision),
                "Calibration_Quarters": "|".join(allowed_quarters),
                "Calibration_Rows": int(calibration_mask.sum()),
                "Calibration_Actual_Events": int(selected["actual_count"]),
                "Calibration_Warnings": int(selected["predicted_count"]),
                "Calibration_TP": int(selected["tp"]),
                "Calibration_FP": int(selected["fp"]),
                "Calibration_Precision": float(selected["precision"]),
                "Calibration_Recall": float(selected["recall"]),
                "Calibration_F1": float(selected["f1"]),
                "Target_Precision_Met": bool(selected["target_precision_met"]),
                "Test_Warnings": test_warnings,
                "Test_Actual_Events": int(test_actual.sum()),
                "Test_TP": test_tp,
                "Test_FP": test_fp,
                "Test_FN": test_fn,
                "Test_Precision": (
                    float(test_tp / test_warnings) if test_warnings else None
                ),
                "Test_Recall": (
                    float(test_tp / test_actual.sum()) if test_actual.sum() else None
                ),
            }
        )

    combo["Event_Gate_Threshold"] = threshold_values
    return prediction, pd.DataFrame(history)


def _apply_drawdown_tertile_gate(
    combo: pd.DataFrame,
    score: np.ndarray,
    cutpoints: tuple[float, float],
    thresholds: tuple[float, float, float],
) -> np.ndarray:
    if "drawdown" not in combo.columns:
        raise ValueError(
            "drawdown_tertile_threshold requires a feature frame with a drawdown column."
        )
    drawdown = combo["drawdown"].to_numpy(dtype=float)
    if np.isnan(drawdown).any():
        missing = int(np.isnan(drawdown).sum())
        raise ValueError(f"drawdown contains {missing} missing rows after feature merge.")
    first_cut, second_cut = (float(cutpoints[0]), float(cutpoints[1]))
    if first_cut >= second_cut:
        raise ValueError("drawdown cutpoints must be ascending.")
    low_threshold, middle_threshold, high_threshold = (
        float(thresholds[0]),
        float(thresholds[1]),
        float(thresholds[2]),
    )
    threshold_values = np.where(
        drawdown < first_cut,
        low_threshold,
        np.where(drawdown < second_cut, middle_threshold, high_threshold),
    )
    combo["Event_Gate_Threshold"] = threshold_values
    combo["Event_Gate_Regime_Feature"] = "drawdown"
    combo["Event_Gate_Regime"] = np.where(
        drawdown < first_cut,
        "deep_drawdown",
        np.where(drawdown < second_cut, "middle_drawdown", "shallow_drawdown"),
    )
    return np.asarray(score, dtype=float) >= threshold_values


def _apply_event_gate(
    combo: pd.DataFrame,
    event_gate: str,
    event_score_threshold: float,
    event_news_weight: float,
    drawdown_tertile_cutpoints: tuple[float, float],
    drawdown_tertile_thresholds: tuple[float, float, float],
    precision_gate_score: str,
    target_event_precision: float,
    precision_gate_min_recall: float,
    precision_gate_min_warnings: int,
    precision_gate_warmup_quarters: int,
    precision_gate_calibration_quarters: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if event_gate == "news":
        score = _event_score(combo, "news")
        combo["Event_Gate_Threshold"] = np.nan
        return combo["Event_Predicted_Event"].to_numpy(dtype=bool), score, pd.DataFrame()
    if event_gate == "market":
        score = _event_score(combo, "market")
        combo["Event_Gate_Threshold"] = np.nan
        return combo["Market_Predicted_Event"].to_numpy(dtype=bool), score, pd.DataFrame()
    if event_gate == "or":
        score = _event_score(combo, "maximum")
        combo["Event_Gate_Threshold"] = np.nan
        return (
            combo["Event_Predicted_Event"].to_numpy(dtype=bool)
            | combo["Market_Predicted_Event"].to_numpy(dtype=bool),
            score,
            pd.DataFrame(),
        )
    if event_gate == "and":
        score = _event_score(combo, "average")
        combo["Event_Gate_Threshold"] = np.nan
        return (
            combo["Event_Predicted_Event"].to_numpy(dtype=bool)
            & combo["Market_Predicted_Event"].to_numpy(dtype=bool),
            score,
            pd.DataFrame(),
        )
    if event_gate == "average_threshold":
        score = _event_score(combo, "average")
        combo["Event_Gate_Threshold"] = float(event_score_threshold)
        return score >= event_score_threshold, score, pd.DataFrame()
    if event_gate == "weighted_threshold":
        score = _weighted_event_score(combo, event_news_weight)
        combo["Event_Gate_Threshold"] = float(event_score_threshold)
        return score >= event_score_threshold, score, pd.DataFrame()
    if event_gate == "maximum_threshold":
        score = _event_score(combo, "maximum")
        combo["Event_Gate_Threshold"] = float(event_score_threshold)
        return score >= event_score_threshold, score, pd.DataFrame()
    if event_gate == "drawdown_tertile_threshold":
        score = _weighted_event_score(combo, event_news_weight)
        prediction = _apply_drawdown_tertile_gate(
            combo,
            score,
            drawdown_tertile_cutpoints,
            drawdown_tertile_thresholds,
        )
        return prediction, score, pd.DataFrame()
    if event_gate == "precision_target":
        score = _event_score(combo, precision_gate_score)
        prediction, history = _apply_precision_target_gate(
            combo,
            score,
            target_precision=target_event_precision,
            minimum_recall=precision_gate_min_recall,
            minimum_warnings=precision_gate_min_warnings,
            warmup_quarters=precision_gate_warmup_quarters,
            calibration_quarters=precision_gate_calibration_quarters,
        )
        return prediction, score, history
    raise ValueError(
        "event_gate must be one of: news, market, or, and, "
        "average_threshold, weighted_threshold, maximum_threshold, "
        "drawdown_tertile_threshold, precision_target"
    )


def _event_gate_score_mode_label(
    event_gate: str,
    precision_gate_score: str,
) -> str:
    if event_gate == "precision_target":
        return precision_gate_score
    if event_gate == "average_threshold":
        return "average"
    if event_gate == "weighted_threshold":
        return "weighted"
    if event_gate == "drawdown_tertile_threshold":
        return "weighted"
    if event_gate == "maximum_threshold":
        return "maximum"
    if event_gate == "or":
        return "maximum"
    if event_gate == "and":
        return "average"
    return event_gate


def combine_event_and_direction_predictions(
    event_predictions_path: Path,
    direction_predictions_path: Path,
    output_root: Path,
    target_ticker: str,
    horizon: int = DEFAULT_HORIZON,
    event_threshold_pct: float = DEFAULT_EVENT_THRESHOLD_PCT,
    cost_bps_values: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0),
    event_gate: str = DEFAULT_EVENT_GATE,
    event_gate_feature_frame_path: Path | None = None,
    event_score_threshold: float = DEFAULT_EVENT_SCORE_THRESHOLD,
    event_news_weight: float = DEFAULT_EVENT_NEWS_WEIGHT,
    drawdown_tertile_cutpoints: tuple[float, float] = DEFAULT_DRAWDOWN_TERTILE_CUTPOINTS,
    drawdown_tertile_thresholds: tuple[float, float, float] = (
        DEFAULT_DRAWDOWN_TERTILE_THRESHOLDS
    ),
    precision_gate_score: str = DEFAULT_PRECISION_GATE_SCORE,
    target_event_precision: float = DEFAULT_TARGET_EVENT_PRECISION,
    precision_gate_min_recall: float = DEFAULT_PRECISION_GATE_MIN_RECALL,
    precision_gate_min_warnings: int = DEFAULT_PRECISION_GATE_MIN_WARNINGS,
    precision_gate_warmup_quarters: int = DEFAULT_PRECISION_GATE_WARMUP_QUARTERS,
    precision_gate_calibration_quarters: int = (
        DEFAULT_PRECISION_GATE_CALIBRATION_QUARTERS
    ),
) -> dict:
    event = pd.read_csv(
        event_predictions_path,
        encoding="utf-8-sig",
        parse_dates=["Current_Date", "Target_Date"],
    )
    direction = pd.read_csv(
        direction_predictions_path,
        encoding="utf-8-sig",
        parse_dates=["Current_Date", "Target_Date"],
    )

    event["Event_Predicted_Event"] = _as_bool(event["Predicted_Event"])
    direction["Market_Predicted_Event"] = _as_bool(direction["Predicted_Event"])
    event["Actual_Event_Recomputed"] = (
        event["Actual_LogRet"].astype(float).abs() > float(event_threshold_pct)
    )
    direction["Actual_Event_Recomputed"] = (
        direction["Actual_LogRet"].astype(float).abs() > float(event_threshold_pct)
    )

    event_columns = [
        "Quarter",
        "Current_Date",
        "Target_Date",
        "Current_Price",
        "Actual_Future_Price",
        "Actual_LogRet",
        "Actual_Return",
        "Event_Probability",
        "Event_Predicted_Event",
        "Actual_Event_Recomputed",
        "Direction_Probability",
        "Direction_Class",
        "Actual_Direction_Class",
    ]
    direction_columns = [
        "Current_Date",
        "Target_Date",
        "Event_Probability",
        "Market_Predicted_Event",
        "Direction_Probability",
        "Direction_Class",
    ]
    combo = event[event_columns].merge(
        direction[direction_columns],
        on=["Current_Date", "Target_Date"],
        how="inner",
        suffixes=("_NewsEvent", "_MarketLong"),
        validate="one_to_one",
    )
    combo = combo.sort_values("Current_Date").reset_index(drop=True)
    combo = _attach_event_gate_features(combo, event_gate_feature_frame_path)
    combo["Actual_Event"] = combo["Actual_Event_Recomputed"]
    combo["News_Event_Probability"] = combo["Event_Probability_NewsEvent"].astype(float)
    combo["Market_Long_Event_Probability"] = combo[
        "Event_Probability_MarketLong"
    ].astype(float)
    combo["Market_Long_Direction_Probability"] = combo[
        "Direction_Probability_MarketLong"
    ].astype(float)
    predicted_event, event_gate_score, threshold_history = _apply_event_gate(
        combo,
        event_gate=event_gate,
        event_score_threshold=event_score_threshold,
        event_news_weight=event_news_weight,
        drawdown_tertile_cutpoints=drawdown_tertile_cutpoints,
        drawdown_tertile_thresholds=drawdown_tertile_thresholds,
        precision_gate_score=precision_gate_score,
        target_event_precision=target_event_precision,
        precision_gate_min_recall=precision_gate_min_recall,
        precision_gate_min_warnings=precision_gate_min_warnings,
        precision_gate_warmup_quarters=precision_gate_warmup_quarters,
        precision_gate_calibration_quarters=precision_gate_calibration_quarters,
    )
    combo["Event_Gate"] = event_gate
    event_gate_score_mode = _event_gate_score_mode_label(
        event_gate,
        precision_gate_score,
    )
    combo["Event_Gate_Score_Mode"] = event_gate_score_mode
    combo["Event_Gate_Score"] = event_gate_score
    combo["Predicted_Event"] = predicted_event
    combo["Predicted_Up"] = combo["Market_Long_Direction_Probability"] >= 0.5
    combo["Direction_Class_Raw"] = np.where(combo["Predicted_Up"], "up", "down")
    combo["Actual_Up"] = combo["Actual_LogRet"].astype(float) >= 0.0
    combo["Model_Signal"] = np.where(
        combo["Predicted_Event"],
        np.where(combo["Predicted_Up"], 1, -1),
        0,
    )
    combo["Model_Signal_Return"] = (
        combo["Model_Signal"].astype(float) * combo["Actual_Return"].astype(float)
    )
    combo["Correct_Event_And_Direction"] = (
        combo["Predicted_Event"]
        & combo["Actual_Event"]
        & (combo["Predicted_Up"] == combo["Actual_Up"])
    )
    combo["Wrong_Event_Direction"] = (
        combo["Predicted_Event"]
        & combo["Actual_Event"]
        & (combo["Predicted_Up"] != combo["Actual_Up"])
    )

    output_root.mkdir(parents=True, exist_ok=True)
    combo.to_csv(
        output_root / "combined_oos_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not threshold_history.empty:
        threshold_history.to_csv(
            output_root / "threshold_history.csv",
            index=False,
            encoding="utf-8-sig",
        )

    actual_event = combo["Actual_Event"].to_numpy(dtype=bool)
    predicted_event = combo["Predicted_Event"].to_numpy(dtype=bool)
    predicted_up = combo["Predicted_Up"].to_numpy(dtype=bool)
    actual_up = combo["Actual_Up"].to_numpy(dtype=bool)
    true_warning = actual_event & predicted_event
    correct_event_direction = combo["Correct_Event_And_Direction"].to_numpy(dtype=bool)

    event_tp = int(true_warning.sum())
    event_fp = int((predicted_event & ~actual_event).sum())
    event_fn = int((~predicted_event & actual_event).sum())
    event_tn = int((~predicted_event & ~actual_event).sum())
    actual_event_count = int(actual_event.sum())
    warning_count = int(predicted_event.sum())
    event_precision = float(
        precision_score(actual_event, predicted_event, zero_division=0)
    )
    event_recall = float(recall_score(actual_event, predicted_event, zero_division=0))
    actual_event_rate = (
        float(actual_event_count / len(actual_event)) if len(actual_event) else None
    )

    pooled = {
        "Target_Ticker": target_ticker.upper(),
        "Horizon_Days": int(horizon),
        "Event_Gate": event_gate,
        "Event_Gate_Score_Mode": event_gate_score_mode,
        "Event_News_Weight": float(event_news_weight),
        "Event_Market_Weight": float(1.0 - event_news_weight),
        "Event_Score_Threshold": (
            float(event_score_threshold)
            if event_gate in {"average_threshold", "weighted_threshold", "maximum_threshold"}
            else None
        ),
        "Drawdown_Tertile_Cutpoints": (
            "|".join(str(float(value)) for value in drawdown_tertile_cutpoints)
            if event_gate == "drawdown_tertile_threshold"
            else None
        ),
        "Drawdown_Tertile_Thresholds": (
            "|".join(str(float(value)) for value in drawdown_tertile_thresholds)
            if event_gate == "drawdown_tertile_threshold"
            else None
        ),
        "Target_Event_Precision": float(target_event_precision),
        "Precision_Gate_Min_Recall": float(precision_gate_min_recall),
        "Precision_Gate_Min_Warnings": int(precision_gate_min_warnings),
        "Precision_Gate_Warmup_Quarters": int(precision_gate_warmup_quarters),
        "Precision_Gate_Calibration_Quarters": int(
            precision_gate_calibration_quarters
        ),
        "Rows": int(len(combo)),
        "Start_Date": combo["Current_Date"].min().strftime("%Y-%m-%d"),
        "End_Date": combo["Current_Date"].max().strftime("%Y-%m-%d"),
        "Actual_Events": actual_event_count,
        "Warnings": warning_count,
        "Event_TP": event_tp,
        "Event_FP": event_fp,
        "Event_FN": event_fn,
        "Event_TN": event_tn,
        "Actual_Event_Rate": actual_event_rate,
        "Event_Precision": event_precision,
        "Event_Recall": event_recall,
        "Precision_Lift_vs_Random": (
            float(event_precision / actual_event_rate)
            if actual_event_rate and actual_event_rate > 0.0
            else None
        ),
        "Event_Balanced_Accuracy": (
            float(balanced_accuracy_score(actual_event, predicted_event))
            if len(np.unique(actual_event)) > 1
            else None
        ),
        "Event_AUC": _safe_auc(actual_event, combo["Event_Gate_Score"]),
        "Event_PR_AUC": _safe_average_precision(
            actual_event,
            combo["Event_Gate_Score"],
        ),
        "Event_Brier": _safe_brier(actual_event, combo["Event_Gate_Score"]),
        "Direction_Accuracy_On_All_Actual_Events": (
            float(accuracy_score(actual_up[actual_event], predicted_up[actual_event]))
            if actual_event_count
            else None
        ),
        "Direction_AUC_On_All_Actual_Events": (
            _safe_auc(
                actual_up[actual_event],
                combo.loc[actual_event, "Market_Long_Direction_Probability"],
            )
            if actual_event_count
            else None
        ),
        "Direction_Accuracy_On_True_Warnings": (
            float(accuracy_score(actual_up[true_warning], predicted_up[true_warning]))
            if event_tp
            else None
        ),
        "Correct_Event_And_Direction": int(correct_event_direction.sum()),
        "Correct_Event_And_Direction_Per_Warning": (
            float(correct_event_direction.sum() / warning_count)
            if warning_count
            else None
        ),
        "Correct_Event_And_Direction_Recall_Of_Actual_Events": (
            float(correct_event_direction.sum() / actual_event_count)
            if actual_event_count
            else None
        ),
        "Max_Possible_Correct_With_Current_Direction_If_Warn_All_Actual_Events": int(
            ((predicted_up == actual_up) & actual_event).sum()
        ),
        "Warn_All_Current_Direction_Accuracy_On_Actual_Events": (
            float(((predicted_up == actual_up) & actual_event).sum() / actual_event_count)
            if actual_event_count
            else None
        ),
    }
    pd.DataFrame([pooled]).to_csv(
        output_root / "pooled_oos_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    source_comparison = pd.DataFrame(
        [
            _binary_event_metrics(
                f"Final event gate: {event_gate}",
                actual_event,
                predicted_event,
                combo["Event_Gate_Score"],
            ),
            _binary_event_metrics(
                "News quality-only event",
                actual_event,
                combo["Event_Predicted_Event"].to_numpy(dtype=bool),
                combo["News_Event_Probability"],
            ),
            _binary_event_metrics(
                "Long market-only event",
                actual_event,
                combo["Market_Predicted_Event"].to_numpy(dtype=bool),
                combo["Market_Long_Event_Probability"],
            ),
        ]
    )
    source_comparison.to_csv(
        output_root / "source_event_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    quarter_rows: list[dict] = []
    for quarter, group in combo.groupby("Quarter", sort=False):
        quarter_actual = group["Actual_Event"].to_numpy(dtype=bool)
        quarter_predicted = group["Predicted_Event"].to_numpy(dtype=bool)
        quarter_correct = group["Correct_Event_And_Direction"].to_numpy(dtype=bool)
        quarter_true_warning = quarter_actual & quarter_predicted
        quarter_rows.append(
            {
                "Quarter": quarter,
                "Rows": int(len(group)),
                "Actual_Events": int(quarter_actual.sum()),
                "Warnings": int(quarter_predicted.sum()),
                "Event_TP": int(quarter_true_warning.sum()),
                "Event_FP": int((quarter_predicted & ~quarter_actual).sum()),
                "Event_FN": int((~quarter_predicted & quarter_actual).sum()),
                "Event_Precision": (
                    float(quarter_true_warning.sum() / quarter_predicted.sum())
                    if quarter_predicted.sum()
                    else None
                ),
                "Event_Recall": (
                    float(quarter_true_warning.sum() / quarter_actual.sum())
                    if quarter_actual.sum()
                    else None
                ),
                "Correct_Event_And_Direction": int(quarter_correct.sum()),
                "Correct_Per_Warning": (
                    float(quarter_correct.sum() / quarter_predicted.sum())
                    if quarter_predicted.sum()
                    else None
                ),
            }
        )
    pd.DataFrame(quarter_rows).to_csv(
        output_root / "quarterly_breakdown.csv",
        index=False,
        encoding="utf-8-sig",
    )

    returns = _return_summary(combo, horizon, cost_bps_values)
    returns.to_csv(
        output_root / "return_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    yearly_returns = _yearly_return_summary(combo, horizon, cost_bps=5.0)
    yearly_returns.to_csv(
        output_root / "yearly_return_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "target_ticker": target_ticker.upper(),
        "horizon": int(horizon),
        "event_threshold_pct": float(event_threshold_pct),
        "event_gate": event_gate,
        "event_gate_score_mode": event_gate_score_mode,
        "event_gate_feature_frame_path": (
            str(event_gate_feature_frame_path)
            if event_gate_feature_frame_path is not None
            else None
        ),
        "event_score_threshold": float(event_score_threshold),
        "event_news_weight": float(event_news_weight),
        "drawdown_tertile_cutpoints": [float(value) for value in drawdown_tertile_cutpoints],
        "drawdown_tertile_thresholds": [
            float(value) for value in drawdown_tertile_thresholds
        ],
        "precision_gate_score": precision_gate_score,
        "target_event_precision": float(target_event_precision),
        "precision_gate_min_recall": float(precision_gate_min_recall),
        "precision_gate_min_warnings": int(precision_gate_min_warnings),
        "precision_gate_warmup_quarters": int(precision_gate_warmup_quarters),
        "precision_gate_calibration_quarters": int(
            precision_gate_calibration_quarters
        ),
        "event_predictions_path": str(event_predictions_path),
        "direction_predictions_path": str(direction_predictions_path),
        "output_root": str(output_root),
        "pooled_oos_summary": pooled,
        "threshold_history": threshold_history.to_dict(orient="records"),
        "source_event_comparison": source_comparison.to_dict(orient="records"),
        "return_summary": returns.to_dict(orient="records"),
    }
    write_json(payload, output_root / "report.json")
    return payload


def _component_predictions_exist(event_root: Path, direction_root: Path) -> bool:
    return _event_predictions_path(event_root).exists() and _direction_predictions_path(
        direction_root
    ).exists()


def run_combo_pipeline(
    target_ticker: str,
    horizon: int,
    event_training_frame: Path,
    direction_training_frame: Path,
    event_root: Path,
    direction_root: Path,
    output_root: Path,
    start_quarter: str = DEFAULT_START_QUARTER,
    event_common_start_date: str = DEFAULT_EVENT_COMMON_START_DATE,
    direction_common_start_date: str = DEFAULT_DIRECTION_COMMON_START_DATE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    event_selection_objective: str = "ranking",
    event_min_recall: float = 0.4,
    skip_component_training: bool = False,
    force_retrain: bool = False,
    event_gate: str = DEFAULT_EVENT_GATE,
    event_score_threshold: float = DEFAULT_EVENT_SCORE_THRESHOLD,
    event_news_weight: float = DEFAULT_EVENT_NEWS_WEIGHT,
    drawdown_tertile_cutpoints: tuple[float, float] = DEFAULT_DRAWDOWN_TERTILE_CUTPOINTS,
    drawdown_tertile_thresholds: tuple[float, float, float] = (
        DEFAULT_DRAWDOWN_TERTILE_THRESHOLDS
    ),
    precision_gate_score: str = DEFAULT_PRECISION_GATE_SCORE,
    target_event_precision: float = DEFAULT_TARGET_EVENT_PRECISION,
    precision_gate_min_recall: float = DEFAULT_PRECISION_GATE_MIN_RECALL,
    precision_gate_min_warnings: int = DEFAULT_PRECISION_GATE_MIN_WARNINGS,
    precision_gate_warmup_quarters: int = DEFAULT_PRECISION_GATE_WARMUP_QUARTERS,
    precision_gate_calibration_quarters: int = (
        DEFAULT_PRECISION_GATE_CALIBRATION_QUARTERS
    ),
) -> dict:
    if force_retrain or (
        not skip_component_training
        and not _event_predictions_path(event_root).exists()
    ):
        event_config = make_training_config(
            target_ticker,
            preset="auto",
            random_seed=random_seed,
            regression_style_fixed_horizon=horizon,
            lstm_event_selection_objective=event_selection_objective,
            lstm_event_min_recall=event_min_recall,
        )
        run_quarterly_walk_forward(
            event_config,
            event_training_frame,
            event_root,
            start_quarter=start_quarter,
            common_start_date=event_common_start_date,
            news_feature_profile="legacy",
            include_market_only=False,
            training_mode="multitask",
        )

    if force_retrain or (
        not skip_component_training
        and not _direction_predictions_path(direction_root).exists()
    ):
        direction_config = make_training_config(
            target_ticker,
            preset="auto",
            random_seed=random_seed,
            regression_style_fixed_horizon=horizon,
            lstm_event_selection_objective=event_selection_objective,
            lstm_event_min_recall=event_min_recall,
        )
        run_quarterly_walk_forward(
            direction_config,
            direction_training_frame,
            direction_root,
            start_quarter=start_quarter,
            common_start_date=direction_common_start_date,
            news_feature_profile="none",
            include_market_only=True,
            training_mode="multitask",
        )

    if not _component_predictions_exist(event_root, direction_root):
        raise FileNotFoundError(
            "Component predictions are missing. Run without --skip-component-training "
            "or check the event/direction root paths."
        )

    return combine_event_and_direction_predictions(
        _event_predictions_path(event_root),
        _direction_predictions_path(direction_root),
        output_root,
        target_ticker=target_ticker,
        horizon=horizon,
        event_threshold_pct=DEFAULT_EVENT_THRESHOLD_PCT,
        event_gate=event_gate,
        event_gate_feature_frame_path=event_training_frame,
        event_score_threshold=event_score_threshold,
        event_news_weight=event_news_weight,
        drawdown_tertile_cutpoints=drawdown_tertile_cutpoints,
        drawdown_tertile_thresholds=drawdown_tertile_thresholds,
        precision_gate_score=precision_gate_score,
        target_event_precision=target_event_precision,
        precision_gate_min_recall=precision_gate_min_recall,
        precision_gate_min_warnings=precision_gate_min_warnings,
        precision_gate_warmup_quarters=precision_gate_warmup_quarters,
        precision_gate_calibration_quarters=precision_gate_calibration_quarters,
    )


def _pct(value: object) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def _parse_float_tuple(raw_value: str, expected_count: int) -> tuple[float, ...]:
    values = tuple(
        float(value.strip())
        for value in raw_value.split(",")
        if value.strip()
    )
    if len(values) != expected_count:
        raise argparse.ArgumentTypeError(
            f"Expected {expected_count} comma-separated floats, got {len(values)}."
        )
    return values


def _print_report(payload: dict, show_returns: bool = False) -> None:
    pooled = payload["pooled_oos_summary"]
    if pooled["Event_Gate"] == "drawdown_tertile_threshold":
        threshold_label = (
            f"drawdown cuts={pooled['Drawdown_Tertile_Cutpoints']} "
            f"thresholds={pooled['Drawdown_Tertile_Thresholds']}"
        )
    else:
        threshold_label = (
            pooled["Event_Score_Threshold"]
            if pooled["Event_Score_Threshold"] is not None
            else "adaptive"
        )
    print(
        f"\n{payload['target_ticker']} final combo "
        f"T+{payload['horizon']} pooled OOS"
    )
    print(
        "  게이트     : "
        f"{pooled['Event_Gate']} | "
        f"score={pooled['Event_Gate_Score_Mode']} | "
        f"{threshold_label}"
    )
    print(
        "  이벤트     : "
        f"precision {_pct(pooled['Event_Precision'])} | "
        f"recall {_pct(pooled['Event_Recall'])} | "
        f"AUC {_pct(pooled['Event_AUC'])} | "
        f"랜덤 대비 {pooled['Precision_Lift_vs_Random']:.2f}x"
    )
    print(
        "  경고 품질  : "
        f"경고 {pooled['Warnings']}개 | "
        f"맞은 경고 {pooled['Event_TP']}개 | "
        f"오탐 {pooled['Event_FP']}개 | "
        f"놓침 {pooled['Event_FN']}개"
    )
    print(
        "  동시 적중   : "
        f"{pooled['Correct_Event_And_Direction']}개 | "
        f"경고 대비 {_pct(pooled['Correct_Event_And_Direction_Per_Warning'])} | "
        f"실제 이벤트 대비 "
        f"{_pct(pooled['Correct_Event_And_Direction_Recall_Of_Actual_Events'])}"
    )
    print(
        "  방향       : "
        f"진짜 경고일 기준 "
        f"{_pct(pooled['Direction_Accuracy_On_True_Warnings'])} | "
        f"전체 실제 이벤트 기준 "
        f"{_pct(pooled['Direction_Accuracy_On_All_Actual_Events'])}"
    )

    if not show_returns:
        return

    returns = pd.DataFrame(payload["return_summary"])
    if not returns.empty:
        print("\nReturn summary")
        display = returns[
            [
                "Strategy",
                "Cost_bps",
                "Active_Signal_Days",
                "Average_Abs_Exposure",
                "Total_Return",
                "Annualized_Return",
                "Sharpe",
                "Max_Drawdown",
            ]
        ].copy()
        print(
            display.to_string(
                index=False,
                formatters={
                    "Average_Abs_Exposure": _pct,
                    "Total_Return": _pct,
                    "Annualized_Return": _pct,
                    "Max_Drawdown": _pct,
                },
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final event-direction combo. Default structure is QQQ T+2: "
            "conservative average(news, market) event gate + long market-only raw direction."
        )
    )
    parser.add_argument("--target-ticker", default="QQQ")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--start-quarter", default=DEFAULT_START_QUARTER)
    parser.add_argument("--event-common-start-date", default=DEFAULT_EVENT_COMMON_START_DATE)
    parser.add_argument(
        "--direction-common-start-date",
        default=DEFAULT_DIRECTION_COMMON_START_DATE,
    )
    parser.add_argument("--event-training-frame", default=None)
    parser.add_argument("--direction-training-frame", default=None)
    parser.add_argument("--event-root", default=None)
    parser.add_argument("--direction-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--event-selection-objective",
        choices=("ranking", "precision_at_recall"),
        default="ranking",
    )
    parser.add_argument("--event-min-recall", type=float, default=0.4)
    parser.add_argument(
        "--event-gate",
        choices=(
            "news",
            "market",
            "or",
            "and",
            "average_threshold",
            "weighted_threshold",
            "maximum_threshold",
            "drawdown_tertile_threshold",
            "precision_target",
        ),
        default=DEFAULT_EVENT_GATE,
        help=(
            "Final event gate. The default average_threshold requires the "
            "average of news and market event probabilities to clear a fixed threshold."
        ),
    )
    parser.add_argument(
        "--event-score-threshold",
        type=float,
        default=DEFAULT_EVENT_SCORE_THRESHOLD,
        help="Fixed threshold for average_threshold or maximum_threshold gates.",
    )
    parser.add_argument(
        "--event-news-weight",
        type=float,
        default=DEFAULT_EVENT_NEWS_WEIGHT,
        help="News score weight for weighted and drawdown-tertile event gates.",
    )
    parser.add_argument(
        "--drawdown-tertile-cutpoints",
        type=lambda value: _parse_float_tuple(value, 2),
        default=DEFAULT_DRAWDOWN_TERTILE_CUTPOINTS,
        help=(
            "Two drawdown cutpoints for drawdown_tertile_threshold, comma-separated. "
            "Default is the best search candidate."
        ),
    )
    parser.add_argument(
        "--drawdown-tertile-thresholds",
        type=lambda value: _parse_float_tuple(value, 3),
        default=DEFAULT_DRAWDOWN_TERTILE_THRESHOLDS,
        help=(
            "Three event score thresholds for deep/middle/shallow drawdown regimes, "
            "comma-separated."
        ),
    )
    parser.add_argument(
        "--precision-gate-score",
        choices=("news", "market", "average", "maximum"),
        default=DEFAULT_PRECISION_GATE_SCORE,
        help="Score used by precision_target.",
    )
    parser.add_argument(
        "--target-event-precision",
        type=float,
        default=DEFAULT_TARGET_EVENT_PRECISION,
        help="Minimum precision target used on past-quarter calibration.",
    )
    parser.add_argument(
        "--precision-gate-min-recall",
        type=float,
        default=DEFAULT_PRECISION_GATE_MIN_RECALL,
        help="Recall floor while selecting the past-quarter precision threshold.",
    )
    parser.add_argument(
        "--precision-gate-min-warnings",
        type=int,
        default=DEFAULT_PRECISION_GATE_MIN_WARNINGS,
        help="Minimum calibration warnings required when choosing a threshold.",
    )
    parser.add_argument(
        "--precision-gate-warmup-quarters",
        type=int,
        default=DEFAULT_PRECISION_GATE_WARMUP_QUARTERS,
        help="Initial quarters with no final alerts, used to avoid uncalibrated gates.",
    )
    parser.add_argument(
        "--precision-gate-calibration-quarters",
        type=int,
        default=DEFAULT_PRECISION_GATE_CALIBRATION_QUARTERS,
        help="Number of previous quarters used for threshold selection. Use 0 for all past.",
    )
    parser.add_argument(
        "--skip-component-training",
        action="store_true",
        help="Use existing component OOS prediction files only.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain the event and direction component walk-forwards even if outputs exist.",
    )
    parser.add_argument(
        "--show-returns",
        action="store_true",
        help="Print return summary in addition to the classification-focused report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_training_frame = (
        Path(args.event_training_frame)
        if args.event_training_frame
        else default_event_training_frame(args.target_ticker)
    )
    direction_training_frame = (
        Path(args.direction_training_frame)
        if args.direction_training_frame
        else default_direction_training_frame(args.target_ticker)
    )
    event_root = (
        Path(args.event_root)
        if args.event_root
        else default_event_root(args.target_ticker, args.horizon)
    )
    direction_root = (
        Path(args.direction_root)
        if args.direction_root
        else default_direction_root(args.target_ticker, args.horizon)
    )
    output_root = (
        Path(args.output_root)
        if args.output_root
        else default_combo_root(args.target_ticker, args.horizon, args.event_gate)
    )

    payload = run_combo_pipeline(
        target_ticker=args.target_ticker,
        horizon=args.horizon,
        event_training_frame=event_training_frame,
        direction_training_frame=direction_training_frame,
        event_root=event_root,
        direction_root=direction_root,
        output_root=output_root,
        start_quarter=args.start_quarter,
        event_common_start_date=args.event_common_start_date,
        direction_common_start_date=args.direction_common_start_date,
        random_seed=args.random_seed,
        event_selection_objective=args.event_selection_objective,
        event_min_recall=args.event_min_recall,
        skip_component_training=args.skip_component_training,
        force_retrain=args.force_retrain,
        event_gate=args.event_gate,
        event_score_threshold=args.event_score_threshold,
        event_news_weight=args.event_news_weight,
        drawdown_tertile_cutpoints=args.drawdown_tertile_cutpoints,
        drawdown_tertile_thresholds=args.drawdown_tertile_thresholds,
        precision_gate_score=args.precision_gate_score,
        target_event_precision=args.target_event_precision,
        precision_gate_min_recall=args.precision_gate_min_recall,
        precision_gate_min_warnings=args.precision_gate_min_warnings,
        precision_gate_warmup_quarters=args.precision_gate_warmup_quarters,
        precision_gate_calibration_quarters=args.precision_gate_calibration_quarters,
    )
    _print_report(payload, show_returns=args.show_returns)
    print(f"\nSaved outputs: {output_root}")


if __name__ == "__main__":
    main()
