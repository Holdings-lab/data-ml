from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.common.utils import training_data_path, write_json
from shared.config.schema import make_training_config
from shared.config.ticker_presets import ticker_slug
from shared.market.data import supplementary_ticker_feature_columns
from shared.run_event_direction_combo import (
    DEFAULT_EVENT_GATE,
    DEFAULT_EVENT_NEWS_WEIGHT,
    DEFAULT_EVENT_THRESHOLD_PCT,
    DEFAULT_RANDOM_SEED,
    DEFAULT_START_QUARTER,
    default_combo_root,
    default_direction_root,
    default_event_root,
    default_event_training_frame,
    run_combo_pipeline,
    _direction_predictions_path,
    _event_predictions_path,
)

DEFAULT_FINAL_HORIZON = 16

# QQQ h16 thresholds selected to prioritize event+direction precision.
# Search target:
# - Normal: practical coverage, event recall around 30%
# - High Confidence: balanced precision/coverage, event recall around 20%
# - Strong: highest event+direction precision with event recall around 10%
NORMAL_THRESHOLDS = (0.65, 0.69, 0.76)
HIGH_CONFIDENCE_THRESHOLDS = (0.66, 0.84, 0.99)
STRONG_THRESHOLDS = (0.71, 0.84, 0.99)
DEFAULT_DIRECTION_THRESHOLD = 0.40
DEFAULT_XGB_MIN_TRAIN_ROWS = 120


def _horizon_root_name(prefix: str, horizon: int, suffix: str) -> str:
    return f"{prefix}_h{horizon}_{suffix}"


def default_market_long_training_frame(target_ticker: str) -> Path:
    return training_data_path(
        ticker_slug(target_ticker),
        "market_only_long",
        "training_frame.csv",
    )


def default_output_root(target_ticker: str, horizon: int) -> Path:
    return training_data_path(
        ticker_slug(target_ticker),
        _horizon_root_name(
            "event_direction_combo",
            horizon,
            "lstm_event_xgb_market_long_event_only_direction",
        ),
    )


def _parse_float_tuple(raw_value: str, length: int) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw_value.split(",") if part.strip())
    if len(values) != length:
        raise argparse.ArgumentTypeError(
            f"Expected {length} comma-separated floats, got {len(values)}: {raw_value}"
        )
    return values


def _market_feature_columns(target_ticker: str) -> list[str]:
    config = make_training_config(target_ticker)
    return list(
        dict.fromkeys(
            list(config.market_feature_columns)
            + supplementary_ticker_feature_columns(
                config.macro_tickers,
                config.supplementary_ticker_feature_suffixes,
            )
        )
    )


def _build_supervised_frame(
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
    event_threshold_pct: float,
) -> pd.DataFrame:
    frame = feature_frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.sort_values("Date").reset_index(drop=True)
    frame["target_logret_horizon"] = (
        np.log(frame["target_price"].shift(-horizon) / frame["target_price"]) * 100.0
    )
    frame["target_future_price_horizon"] = frame["target_price"].shift(-horizon)
    frame["target_date_horizon"] = frame["Date"].shift(-horizon)
    frame["target_event_horizon"] = (
        frame["target_logret_horizon"].abs() >= event_threshold_pct
    )
    required_columns = [
        "Date",
        "target_price",
        "target_logret_horizon",
        "target_future_price_horizon",
        "target_date_horizon",
        "target_event_horizon",
    ] + feature_columns
    return (
        frame[required_columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )


def train_xgb_market_long_event_only_direction(
    lstm_combo: pd.DataFrame,
    market_long_training_frame_path: Path,
    target_ticker: str,
    horizon: int,
    event_threshold_pct: float,
    random_seed: int,
    min_train_rows: int = DEFAULT_XGB_MIN_TRAIN_ROWS,
) -> pd.DataFrame:
    feature_frame = pd.read_csv(market_long_training_frame_path)
    feature_columns = _market_feature_columns(target_ticker)
    missing_columns = [column for column in feature_columns if column not in feature_frame.columns]
    if missing_columns:
        raise ValueError(f"market-long training frame is missing columns: {missing_columns}")

    supervised = _build_supervised_frame(
        feature_frame,
        feature_columns,
        horizon=horizon,
        event_threshold_pct=event_threshold_pct,
    )

    combo = lstm_combo.copy()
    combo["Current_Date"] = pd.to_datetime(combo["Current_Date"], errors="coerce")
    combo["QuarterPeriod"] = combo["Current_Date"].dt.to_period("Q")
    combo = combo.sort_values("Current_Date").reset_index(drop=True)

    prediction_parts: list[pd.DataFrame] = []
    for quarter in sorted(combo["QuarterPeriod"].dropna().unique()):
        quarter_start = quarter.start_time
        quarter_end = quarter.end_time
        quarter_dates = combo.loc[combo["QuarterPeriod"] == quarter, "Current_Date"]

        # Important leakage rule:
        # A row can be used for training only if its label target date is before the
        # test quarter starts. This prevents T+h labels from spilling into the quarter
        # we are evaluating.
        train = supervised[supervised["target_date_horizon"] < quarter_start].copy()
        train = train[train["target_event_horizon"]].copy()
        test = supervised[
            (supervised["Date"].isin(quarter_dates))
            & (supervised["Date"] >= quarter_start)
            & (supervised["Date"] <= quarter_end)
        ].copy()
        if len(train) < min_train_rows or test.empty:
            continue

        x_train = train[feature_columns].astype(float)
        y_train = (train["target_logret_horizon"] >= 0).astype(int)
        x_test = test[feature_columns].astype(float)

        down_count = int((y_train == 0).sum())
        up_count = int((y_train == 1).sum())
        scale_pos_weight = float(down_count / max(up_count, 1))

        model = XGBClassifier(
            n_estimators=220,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.80,
            colsample_bytree=0.80,
            min_child_weight=3,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_seed,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(x_train, y_train)
        direction_score = model.predict_proba(x_test)[:, 1]

        prediction_parts.append(
            pd.DataFrame(
                {
                    "Current_Date": test["Date"].to_numpy(),
                    "Target_Date_XGB": test["target_date_horizon"].to_numpy(),
                    "Actual_LogRet_XGB_Horizon": test["target_logret_horizon"].to_numpy(),
                    "XGB_Direction_Score": direction_score,
                    "XGB_Predicted_Up": direction_score >= DEFAULT_DIRECTION_THRESHOLD,
                    "XGB_Source": "market_long",
                    "XGB_Model": "classifier",
                    "XGB_Train_Mode": "event_only",
                    "XGB_Train_Rows": int(len(train)),
                    "XGB_Test_Quarter": str(quarter),
                }
            )
        )

    if not prediction_parts:
        raise ValueError("XGBoost direction walk-forward produced no predictions.")
    return pd.concat(prediction_parts, ignore_index=True).sort_values("Current_Date")


def _thresholds_by_regime(frame: pd.DataFrame, thresholds: tuple[float, float, float]) -> np.ndarray:
    deep_threshold, middle_threshold, shallow_threshold = thresholds
    regimes = frame["Event_Gate_Regime"].astype(str).to_numpy()
    values = np.full(len(frame), shallow_threshold, dtype=float)
    values[regimes == "deep_drawdown"] = deep_threshold
    values[regimes == "middle_drawdown"] = middle_threshold
    values[regimes == "shallow_drawdown"] = shallow_threshold
    return values


def _apply_alert_tier(
    frame: pd.DataFrame,
    prefix: str,
    thresholds: tuple[float, float, float],
    direction_threshold: float,
    direction_confidence_margin: float = 0.0,
) -> pd.DataFrame:
    output = frame.copy()
    event_thresholds = _thresholds_by_regime(output, thresholds)
    event_gate = output["Event_Gate_Score"].astype(float).to_numpy() >= event_thresholds
    direction_score = output["XGB_Direction_Score"].astype(float).to_numpy()
    if direction_confidence_margin > 0:
        event_gate = event_gate & (np.abs(direction_score - 0.5) >= direction_confidence_margin)
    predicted_up = direction_score >= direction_threshold
    actual_event = output["Actual_Event"].astype(bool).to_numpy()
    actual_up = output["Actual_Up"].astype(bool).to_numpy()

    output[f"{prefix}_Event_Threshold"] = event_thresholds
    output[f"{prefix}_Event_Gate"] = event_gate
    output[f"{prefix}_Predicted_Up"] = predicted_up
    output[f"{prefix}_Direction_Threshold"] = float(direction_threshold)
    output[f"{prefix}_Direction_Confidence_Margin"] = float(direction_confidence_margin)
    output[f"{prefix}_Model_Signal"] = np.where(
        event_gate,
        np.where(predicted_up, 1, -1),
        0,
    )
    output[f"{prefix}_Correct_Event_And_Direction"] = (
        event_gate & actual_event & (predicted_up == actual_up)
    )
    output[f"{prefix}_Wrong_Event_Direction"] = (
        event_gate & actual_event & (predicted_up != actual_up)
    )
    return output


def _tier_metrics(frame: pd.DataFrame, prefix: str) -> dict:
    gate = frame[f"{prefix}_Event_Gate"].astype(bool).to_numpy()
    predicted_up = frame[f"{prefix}_Predicted_Up"].astype(bool).to_numpy()
    actual_event = frame["Actual_Event"].astype(bool).to_numpy()
    actual_up = frame["Actual_Up"].astype(bool).to_numpy()

    warnings = int(gate.sum())
    actual_total = int(actual_event.sum())
    tp = int((gate & actual_event).sum())
    fp = int((gate & ~actual_event).sum())
    fn = int((~gate & actual_event).sum())
    both = int((gate & actual_event & (predicted_up == actual_up)).sum())
    return {
        "Alert_Tier": prefix,
        "Rows": int(len(frame)),
        "Actual_Events": actual_total,
        "Warnings": warnings,
        "Event_TP": tp,
        "Event_FP": fp,
        "Event_FN": fn,
        "Event_Precision": tp / max(warnings, 1),
        "Event_Recall": tp / max(actual_total, 1),
        "Event_F1": 0.0 if tp == 0 else (2 * tp) / max(2 * tp + fp + fn, 1),
        "Event_And_Direction_Correct": both,
        "Event_And_Direction_Per_Warning": both / max(warnings, 1),
        "Event_And_Direction_Recall": both / max(actual_total, 1),
        "Direction_Accuracy_On_True_Warnings": both / max(tp, 1),
        "Predicted_Up_Rate": float(predicted_up.mean()) if len(predicted_up) else 0.0,
    }


def _breakdown(frame: pd.DataFrame, prefix: str, group_column: str) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(group_column):
        metrics = _tier_metrics(group, prefix)
        metrics[group_column] = key
        rows.append(metrics)
    return pd.DataFrame(rows)


def _save_threshold_reference(frame: pd.DataFrame, output_root: Path) -> None:
    rows = []
    for label, thresholds in (
        ("normal", NORMAL_THRESHOLDS),
        ("high_confidence", HIGH_CONFIDENCE_THRESHOLDS),
        ("strong", STRONG_THRESHOLDS),
    ):
        tmp = _apply_alert_tier(
            frame,
            label,
            thresholds=thresholds,
            direction_threshold=DEFAULT_DIRECTION_THRESHOLD,
        )
        row = _tier_metrics(tmp, label)
        row.update(
            {
                "Deep_Threshold": thresholds[0],
                "Middle_Threshold": thresholds[1],
                "Shallow_Threshold": thresholds[2],
                "Direction_Threshold": DEFAULT_DIRECTION_THRESHOLD,
            }
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        output_root / "threshold_reference_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def run_lstm_xgb_event_direction_flow(
    target_ticker: str,
    horizon: int,
    output_root: Path,
    lstm_combo_root: Path,
    market_long_training_frame_path: Path,
    skip_lstm_component_training: bool,
    force_retrain_lstm_components: bool,
    random_seed: int,
    start_quarter: str,
    event_threshold_pct: float,
    event_news_weight: float,
    normal_thresholds: tuple[float, float, float],
    high_confidence_thresholds: tuple[float, float, float],
    strong_thresholds: tuple[float, float, float],
    direction_threshold: float,
) -> dict:
    event_root = default_event_root(target_ticker, horizon)
    direction_root = default_direction_root(target_ticker, horizon)
    event_training_frame = default_event_training_frame(target_ticker)
    event_predictions_path = _event_predictions_path(event_root)
    direction_predictions_path = _direction_predictions_path(direction_root)

    print()
    print(f"{target_ticker.upper()} final flow T+{horizon}")
    if force_retrain_lstm_components:
        print("  LSTM components : force retrain enabled")
    elif skip_lstm_component_training:
        print("  LSTM components : reuse existing only (--skip-lstm-component-training)")
    else:
        event_status = "reuse existing" if event_predictions_path.exists() else "train now"
        direction_status = (
            "reuse existing" if direction_predictions_path.exists() else "train now"
        )
        print(f"  LSTM event      : {event_status} ({event_predictions_path})")
        print(f"  LSTM direction  : {direction_status} ({direction_predictions_path})")

    # Build/read the LSTM event base. The LSTM direction in this base is kept for
    # comparison only; final direction comes from XGBoost below.
    run_combo_pipeline(
        target_ticker=target_ticker,
        horizon=horizon,
        event_root=event_root,
        direction_root=direction_root,
        output_root=lstm_combo_root,
        start_quarter=start_quarter,
        random_seed=random_seed,
        event_training_frame=event_training_frame,
        direction_training_frame=market_long_training_frame_path,
        skip_component_training=skip_lstm_component_training,
        force_retrain=force_retrain_lstm_components,
        event_selection_objective="ranking",
        event_min_recall=0.4,
        event_gate=DEFAULT_EVENT_GATE,
        event_news_weight=event_news_weight,
    )

    combo_path = lstm_combo_root / "combined_oos_predictions.csv"
    if not combo_path.exists():
        raise FileNotFoundError(f"Missing LSTM combo predictions: {combo_path}")
    lstm_combo = pd.read_csv(combo_path, parse_dates=["Current_Date", "Target_Date"])

    print("  XGBoost direction: train walk-forward now")
    xgb_predictions = train_xgb_market_long_event_only_direction(
        lstm_combo=lstm_combo,
        market_long_training_frame_path=market_long_training_frame_path,
        target_ticker=target_ticker,
        horizon=horizon,
        event_threshold_pct=event_threshold_pct,
        random_seed=random_seed,
    )

    combined = lstm_combo.merge(xgb_predictions, on="Current_Date", how="inner")
    combined = combined.sort_values("Current_Date").reset_index(drop=True)
    combined["Quarter"] = combined["Current_Date"].dt.to_period("Q").astype(str)
    combined["Year"] = combined["Current_Date"].dt.year

    # Replace final direction with XGBoost while preserving the original LSTM
    # direction fields for audit/comparison.
    combined["Final_Direction_Model"] = "xgboost_market_long_event_only"
    combined["Final_Direction_Score"] = combined["XGB_Direction_Score"]
    combined["Final_Predicted_Up"] = combined["XGB_Direction_Score"] >= direction_threshold

    combined = _apply_alert_tier(
        combined,
        "Normal",
        thresholds=normal_thresholds,
        direction_threshold=direction_threshold,
    )
    combined = _apply_alert_tier(
        combined,
        "High_Confidence",
        thresholds=high_confidence_thresholds,
        direction_threshold=direction_threshold,
    )
    combined = _apply_alert_tier(
        combined,
        "Strong",
        thresholds=strong_thresholds,
        direction_threshold=direction_threshold,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_root / "combined_oos_predictions.csv", index=False, encoding="utf-8-sig")
    xgb_predictions.to_csv(output_root / "xgb_direction_oos_predictions.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            _tier_metrics(combined, "Normal"),
            _tier_metrics(combined, "High_Confidence"),
            _tier_metrics(combined, "Strong"),
        ]
    )
    summary.to_csv(output_root / "alert_summary.csv", index=False, encoding="utf-8-sig")

    quarterly_frames = []
    yearly_frames = []
    for prefix in ("Normal", "High_Confidence", "Strong"):
        q = _breakdown(combined, prefix, "Quarter")
        y = _breakdown(combined, prefix, "Year")
        quarterly_frames.append(q)
        yearly_frames.append(y)
    pd.concat(quarterly_frames, ignore_index=True).to_csv(
        output_root / "quarterly_breakdown.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(yearly_frames, ignore_index=True).to_csv(
        output_root / "yearly_breakdown.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _save_threshold_reference(combined, output_root)

    metadata = {
        "target_ticker": target_ticker.upper(),
        "horizon": horizon,
        "random_seed": random_seed,
        "event_threshold_pct": event_threshold_pct,
        "event_model": "LSTM quarterly event gate, quality-news + market-long event score",
        "direction_model": f"XGBoost market-long event-only T+{horizon} classifier",
        "direction_training_rule": (
            "For each test quarter, train only on prior rows whose T+h target_date is "
            "before the quarter start; then keep only historical large-move rows "
            f"where abs(T+{horizon} log return) >= {event_threshold_pct:.2f}%."
        ),
        "normal_thresholds": {
            "deep_drawdown": normal_thresholds[0],
            "middle_drawdown": normal_thresholds[1],
            "shallow_drawdown": normal_thresholds[2],
        },
        "high_confidence_thresholds": {
            "deep_drawdown": high_confidence_thresholds[0],
            "middle_drawdown": high_confidence_thresholds[1],
            "shallow_drawdown": high_confidence_thresholds[2],
        },
        "strong_thresholds": {
            "deep_drawdown": strong_thresholds[0],
            "middle_drawdown": strong_thresholds[1],
            "shallow_drawdown": strong_thresholds[2],
        },
        "direction_threshold": direction_threshold,
        "skip_lstm_component_training": bool(skip_lstm_component_training),
        "force_retrain_lstm_components": bool(force_retrain_lstm_components),
        "lstm_combo_root": str(lstm_combo_root),
        "market_long_training_frame_path": str(market_long_training_frame_path),
        "output_root": str(output_root),
    }
    write_json(metadata, output_root / "metadata.json")

    return {
        "metadata": metadata,
        "summary": summary,
        "combined": combined,
    }


def _format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def _print_summary(target_ticker: str, horizon: int, output_root: Path, summary: pd.DataFrame) -> None:
    print()
    print(f"{target_ticker.upper()} LSTM event + XGBoost market-long event-only direction T+{horizon}")
    print("  flow       : LSTM event gate -> XGBoost direction -> alert tiers")
    print("  direction  : XGBoost classifier trained only on historical large-move rows")
    print()
    for _, row in summary.iterrows():
        print(
            f"  {row['Alert_Tier']:<16}: "
            f"warnings {int(row['Warnings'])} | "
            f"precision {_format_rate(row['Event_Precision'])} | "
            f"recall {_format_rate(row['Event_Recall'])} | "
            f"event+dir {int(row['Event_And_Direction_Correct'])} | "
            f"dir@true {_format_rate(row['Direction_Accuracy_On_True_Warnings'])}"
        )
    print()
    print(f"Saved outputs: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final QQQ-style flow: LSTM event alert + XGBoost market-long "
            "event-only direction, with Normal/High Confidence/Strong alert tiers."
        )
    )
    parser.add_argument("--target-ticker", default="QQQ")
    parser.add_argument("--horizon", type=int, default=DEFAULT_FINAL_HORIZON)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--start-quarter", default=DEFAULT_START_QUARTER)
    parser.add_argument("--event-threshold-pct", type=float, default=DEFAULT_EVENT_THRESHOLD_PCT)
    parser.add_argument("--event-news-weight", type=float, default=DEFAULT_EVENT_NEWS_WEIGHT)
    parser.add_argument("--direction-threshold", type=float, default=DEFAULT_DIRECTION_THRESHOLD)
    parser.add_argument(
        "--normal-thresholds",
        type=lambda value: _parse_float_tuple(value, 3),
        default=NORMAL_THRESHOLDS,
        help="Deep,middle,shallow drawdown event thresholds for Normal alerts.",
    )
    parser.add_argument(
        "--high-confidence-thresholds",
        type=lambda value: _parse_float_tuple(value, 3),
        default=HIGH_CONFIDENCE_THRESHOLDS,
        help="Deep,middle,shallow drawdown event thresholds for High Confidence alerts.",
    )
    parser.add_argument(
        "--strong-thresholds",
        type=lambda value: _parse_float_tuple(value, 3),
        default=STRONG_THRESHOLDS,
        help="Deep,middle,shallow drawdown event thresholds for Strong alerts.",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--lstm-combo-root", default=None)
    parser.add_argument("--market-long-training-frame", default=None)
    parser.add_argument(
        "--skip-lstm-component-training",
        action="store_true",
        help="Use existing LSTM component predictions instead of retraining them.",
    )
    parser.add_argument(
        "--force-retrain-lstm-components",
        action="store_true",
        help="Retrain LSTM event/direction component predictions even if cached files exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_ticker = args.target_ticker.upper()
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else default_output_root(target_ticker, args.horizon)
    )
    lstm_combo_root = (
        Path(args.lstm_combo_root)
        if args.lstm_combo_root is not None
        else default_combo_root(target_ticker, args.horizon, DEFAULT_EVENT_GATE)
    )
    market_long_training_frame_path = (
        Path(args.market_long_training_frame)
        if args.market_long_training_frame is not None
        else default_market_long_training_frame(target_ticker)
    )

    payload = run_lstm_xgb_event_direction_flow(
        target_ticker=target_ticker,
        horizon=args.horizon,
        output_root=output_root,
        lstm_combo_root=lstm_combo_root,
        market_long_training_frame_path=market_long_training_frame_path,
        skip_lstm_component_training=args.skip_lstm_component_training,
        force_retrain_lstm_components=args.force_retrain_lstm_components,
        random_seed=args.random_seed,
        start_quarter=args.start_quarter,
        event_threshold_pct=args.event_threshold_pct,
        event_news_weight=args.event_news_weight,
        normal_thresholds=args.normal_thresholds,
        high_confidence_thresholds=args.high_confidence_thresholds,
        strong_thresholds=args.strong_thresholds,
        direction_threshold=args.direction_threshold,
    )
    _print_summary(target_ticker, args.horizon, output_root, payload["summary"])


if __name__ == "__main__":
    main()


