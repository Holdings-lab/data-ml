from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.common.utils import data_path, training_data_path, write_json
from shared.config.schema import make_training_config
from shared.config.ticker_presets import ticker_slug
from shared.run_event_direction_combo import (
    DEFAULT_DRAWDOWN_TERTILE_CUTPOINTS,
    DEFAULT_DRAWDOWN_TERTILE_THRESHOLDS,
    DEFAULT_EVENT_COMMON_START_DATE,
    DEFAULT_EVENT_NEWS_WEIGHT,
    DEFAULT_EVENT_THRESHOLD_PCT,
    DEFAULT_HORIZON,
    DEFAULT_RANDOM_SEED,
)
from shared.run_event_walkforward import _load_feature_groups
from shared.run_lstm_xgb_event_direction import (
    DEFAULT_DIRECTION_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLDS,
    STRONG_THRESHOLDS,
    _build_supervised_frame as build_xgb_supervised_frame,
    _market_feature_columns,
    default_market_long_training_frame,
    default_output_root,
)
from shared.training.lstm_pipeline import (
    _build_sequences,
    _train_event_lstm,
    _train_lstm,
)
from shared.training.metrics import (
    build_supervised_frame,
    filter_feature_frame_by_min_date,
    seed_everything,
    serialize_timestamp,
    to_serializable_config,
)


def default_bundle_root(target_ticker: str, horizon: int) -> Path:
    return data_path("bundles", f"{ticker_slug(target_ticker)}_lstm_xgb_h{horizon}")


def default_event_training_frame(target_ticker: str) -> Path:
    return training_data_path(ticker_slug(target_ticker), "market_news", "training_frame.csv")


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _prepare_lstm_training_arrays(
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
    min_date: str,
    seq_len: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    frame = feature_frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    frame = filter_feature_frame_by_min_date(frame, min_date)
    supervised = build_supervised_frame(frame, feature_columns, horizon)
    supervised = supervised.replace([np.inf, -np.inf], np.nan)
    supervised = supervised.dropna(subset=["Date", "target_date", "target_logret"] + feature_columns)
    supervised = supervised.sort_values("Date").reset_index(drop=True)

    if len(supervised) < seq_len + 20:
        raise ValueError("Not enough rows to train a production LSTM artifact.")

    sequence_rows = supervised.iloc[seq_len - 1 :].reset_index(drop=True)
    sequence_count = len(sequence_rows)
    validation_size = max(1, int(sequence_count * 0.1))
    validation_start = sequence_count - validation_size
    if validation_start <= 0:
        raise ValueError("Not enough sequences for validation split.")
    validation_origin_date = pd.to_datetime(sequence_rows["Date"].iloc[validation_start])
    representation_fit_mask = (
        pd.to_datetime(supervised["Date"], errors="coerce") < validation_origin_date
    ).to_numpy()
    if not bool(representation_fit_mask.any()):
        raise ValueError("No rows available for feature scaling fit.")

    scaler = StandardScaler()
    scaler.fit(supervised.loc[representation_fit_mask, feature_columns].to_numpy(dtype=float))
    x_scaled = scaler.transform(supervised[feature_columns].to_numpy(dtype=float))
    y = supervised["target_logret"].to_numpy(dtype=float)
    x_seq, y_seq = _build_sequences(x_scaled, y, seq_len)
    origin_dates = pd.to_datetime(sequence_rows["Date"]).to_numpy()
    target_dates = pd.to_datetime(sequence_rows["target_date"]).to_numpy()
    return supervised, x_seq, y_seq, origin_dates, target_dates, scaler


def train_lstm_artifact(
    *,
    name: str,
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
    min_date: str,
    config,
    output_path: Path,
    training_mode: str,
) -> dict[str, Any]:
    supervised, x_seq, y_seq, origin_dates, target_dates, scaler = (
        _prepare_lstm_training_arrays(
            feature_frame=feature_frame,
            feature_columns=feature_columns,
            horizon=horizon,
            min_date=min_date,
            seq_len=config.lstm_seq_len,
        )
    )

    if training_mode == "event_only":
        model, training_summary = _train_event_lstm(
            x_seq,
            y_seq,
            origin_dates,
            target_dates,
            config,
        )
        architecture = "event_only_lstm_classifier"
    elif training_mode == "multitask":
        model, training_summary = _train_lstm(
            x_seq,
            y_seq,
            origin_dates,
            target_dates,
            config,
        )
        architecture = "separate_event_and_direction_lstm_encoders"
    else:
        raise ValueError("training_mode must be event_only or multitask.")

    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": state_dict,
            "input_size": len(feature_columns),
            "hidden_size": config.lstm_hidden_size,
            "num_layers": config.lstm_num_layers,
            "dropout": config.lstm_dropout,
            "architecture": architecture,
            "feature_columns": feature_columns,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "seq_len": config.lstm_seq_len,
            "training": training_summary,
        },
        output_path,
    )
    if next(model.parameters()).device.type == "cuda":
        model.to("cpu")
        torch.cuda.empty_cache()

    return {
        "name": name,
        "path": str(output_path),
        "architecture": architecture,
        "training_mode": training_mode,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "seq_len": int(config.lstm_seq_len),
        "supervised_rows": int(len(supervised)),
        "train_start_date": serialize_timestamp(supervised["Date"].iloc[0]),
        "train_end_date": serialize_timestamp(supervised["Date"].iloc[-1]),
        "target_end_date": serialize_timestamp(supervised["target_date"].iloc[-1]),
        "training_summary": training_summary,
    }


def train_xgb_direction_artifact(
    *,
    target_ticker: str,
    market_long_training_frame_path: Path,
    horizon: int,
    event_threshold_pct: float,
    random_seed: int,
    output_path: Path,
) -> dict[str, Any]:
    feature_frame = pd.read_csv(market_long_training_frame_path)
    feature_columns = _market_feature_columns(target_ticker)
    missing_columns = [column for column in feature_columns if column not in feature_frame.columns]
    if missing_columns:
        raise ValueError(f"market-long training frame is missing columns: {missing_columns}")

    supervised = build_xgb_supervised_frame(
        feature_frame,
        feature_columns,
        horizon=horizon,
        event_threshold_pct=event_threshold_pct,
    )
    train = supervised.loc[supervised["target_event_horizon"]].copy()
    if train.empty:
        raise ValueError("No event rows available for XGBoost direction training.")
    x_train = train[feature_columns].astype(float)
    y_train = (train["target_logret_horizon"] >= 0).astype(int)
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    booster = model.get_booster()
    booster.feature_names = feature_columns
    booster.save_model(output_path)
    return {
        "path": str(output_path),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "supervised_rows": int(len(supervised)),
        "event_train_rows": int(len(train)),
        "up_count": up_count,
        "down_count": down_count,
        "scale_pos_weight": scale_pos_weight,
        "train_start_date": serialize_timestamp(supervised["Date"].iloc[0]),
        "train_end_date": serialize_timestamp(supervised["Date"].iloc[-1]),
        "target_end_date": serialize_timestamp(supervised["target_date_horizon"].iloc[-1]),
    }


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _write_bundle_readme(bundle_root: Path) -> None:
    readme = """# QQQ LSTM + XGBoost T+2 model bundle

This bundle is for inference only.

The backend should not train models from this folder. It should load:

- `lstm_news_event.pt`
- `lstm_market_event.pt`
- `xgb_direction.json`
- `feature_schema.json`
- `thresholds.json`

Important input contract:

1. Raw news text is not passed directly into the LSTM.
2. The backend must first create the same daily numeric market/news features used in training.
3. LSTM event inference requires the latest `seq_len` rows.
4. XGBoost direction inference uses the latest market-long feature row.
5. Final alert tiers are calculated from the weighted event score and drawdown regime thresholds.

Python helper:

```python
from shared.inference.lstm_xgb_bundle import load_bundle, predict_from_feature_frames

bundle = load_bundle("data/bundles/qqq_lstm_xgb_h2")
result = predict_from_feature_frames(bundle, news_event_frame, market_long_frame)
```
"""
    (bundle_root / "README.md").write_text(readme, encoding="utf-8")


def export_bundle(args: argparse.Namespace) -> Path:
    target_ticker = args.target_ticker.upper()
    horizon = int(args.horizon)
    seed_everything(args.random_seed)

    bundle_root = Path(args.output_root) if args.output_root else default_bundle_root(target_ticker, horizon)
    bundle_root.mkdir(parents=True, exist_ok=True)

    config = make_training_config(
        target_ticker,
        preset="auto",
        random_seed=args.random_seed,
        regression_style_fixed_horizon=horizon,
        verbose_output=args.verbose,
        lstm_epochs=args.lstm_epochs,
        lstm_device=args.lstm_device,
    )

    event_training_frame_path = Path(args.event_training_frame) if args.event_training_frame else default_event_training_frame(target_ticker)
    market_long_training_frame_path = Path(args.market_long_training_frame) if args.market_long_training_frame else default_market_long_training_frame(target_ticker)

    news_frame, news_market_columns, news_scalar_columns = _load_feature_groups(
        config,
        event_training_frame_path,
        news_feature_profile="legacy",
    )
    news_event_features = list(dict.fromkeys(news_market_columns + news_scalar_columns))

    market_frame = pd.read_csv(market_long_training_frame_path, encoding="utf-8-sig")
    market_frame["Date"] = pd.to_datetime(market_frame["Date"], errors="coerce")
    market_event_features = _market_feature_columns(target_ticker)

    print(f"[BUNDLE] output={bundle_root}", flush=True)
    print("[BUNDLE] training LSTM news event model", flush=True)
    news_lstm = train_lstm_artifact(
        name="lstm_news_event",
        feature_frame=news_frame,
        feature_columns=news_event_features,
        horizon=horizon,
        min_date=args.event_common_start_date,
        config=config,
        output_path=bundle_root / "lstm_news_event.pt",
        training_mode=args.lstm_training_mode,
    )
    print("[BUNDLE] training LSTM market event model", flush=True)
    market_lstm = train_lstm_artifact(
        name="lstm_market_event",
        feature_frame=market_frame,
        feature_columns=market_event_features,
        horizon=horizon,
        min_date=args.market_common_start_date,
        config=config,
        output_path=bundle_root / "lstm_market_event.pt",
        training_mode=args.lstm_training_mode,
    )
    print("[BUNDLE] training XGBoost direction model", flush=True)
    xgb_direction = train_xgb_direction_artifact(
        target_ticker=target_ticker,
        market_long_training_frame_path=market_long_training_frame_path,
        horizon=horizon,
        event_threshold_pct=args.event_threshold_pct,
        random_seed=args.random_seed,
        output_path=bundle_root / "xgb_direction.json",
    )

    thresholds = {
        "event_threshold_pct": float(args.event_threshold_pct),
        "event_news_weight": float(args.event_news_weight),
        "drawdown_tertile_cutpoints": [float(value) for value in DEFAULT_DRAWDOWN_TERTILE_CUTPOINTS],
        "normal_thresholds": {
            "deep_drawdown": float(DEFAULT_DRAWDOWN_TERTILE_THRESHOLDS[0]),
            "middle_drawdown": float(DEFAULT_DRAWDOWN_TERTILE_THRESHOLDS[1]),
            "shallow_drawdown": float(DEFAULT_DRAWDOWN_TERTILE_THRESHOLDS[2]),
        },
        "high_confidence_thresholds": {
            "deep_drawdown": float(HIGH_CONFIDENCE_THRESHOLDS[0]),
            "middle_drawdown": float(HIGH_CONFIDENCE_THRESHOLDS[1]),
            "shallow_drawdown": float(HIGH_CONFIDENCE_THRESHOLDS[2]),
        },
        "strong_thresholds": {
            "deep_drawdown": float(STRONG_THRESHOLDS[0]),
            "middle_drawdown": float(STRONG_THRESHOLDS[1]),
            "shallow_drawdown": float(STRONG_THRESHOLDS[2]),
        },
        "direction_threshold": float(args.direction_threshold),
    }
    feature_schema = {
        "target_ticker": target_ticker,
        "horizon": horizon,
        "date_column": "Date",
        "price_column": "target_price",
        "drawdown_column": "drawdown",
        "news_event_lstm": {
            "input_type": "latest_sequence",
            "seq_len": int(config.lstm_seq_len),
            "training_frame": str(event_training_frame_path),
            "news_feature_profile": "legacy",
            "feature_columns": news_event_features,
        },
        "market_event_lstm": {
            "input_type": "latest_sequence",
            "seq_len": int(config.lstm_seq_len),
            "training_frame": str(market_long_training_frame_path),
            "feature_columns": market_event_features,
        },
        "xgb_direction": {
            "input_type": "latest_row",
            "training_frame": str(market_long_training_frame_path),
            "training_rule": f"direction classifier trained only on rows with abs(T+{horizon} log return) >= {args.event_threshold_pct:.2f}%",
            "feature_columns": xgb_direction["feature_columns"],
        },
    }
    manifest = {
        "bundle_version": "qqq_lstm_xgb_h2_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_ticker": target_ticker,
        "horizon": horizon,
        "random_seed": int(args.random_seed),
        "model_family": "lstm_event_plus_xgboost_direction",
        "event_model": "weighted news-event LSTM + market-event LSTM",
        "direction_model": "XGBoost market-long event-only classifier",
        "lstm_training_mode": args.lstm_training_mode,
        "config": to_serializable_config(config),
        "artifacts": {
            "lstm_news_event": "lstm_news_event.pt",
            "lstm_market_event": "lstm_market_event.pt",
            "xgb_direction": "xgb_direction.json",
            "feature_schema": "feature_schema.json",
            "thresholds": "thresholds.json",
            "performance": "performance.json",
        },
        "training": {
            "news_lstm": news_lstm,
            "market_lstm": market_lstm,
            "xgb_direction": xgb_direction,
        },
    }

    performance_root = default_output_root(target_ticker, horizon)
    performance = {
        "source": str(performance_root),
        "alert_summary": [],
        "note": "Historical OOS performance from the evaluation pipeline; production artifacts are trained separately for inference.",
    }
    alert_summary_path = performance_root / "alert_summary.csv"
    if alert_summary_path.exists():
        performance["alert_summary"] = pd.read_csv(alert_summary_path).to_dict(orient="records")

    _json_dump(bundle_root / "manifest.json", manifest)
    _json_dump(bundle_root / "feature_schema.json", feature_schema)
    _json_dump(bundle_root / "thresholds.json", thresholds)
    _json_dump(bundle_root / "performance.json", performance)
    _write_bundle_readme(bundle_root)

    _copy_if_exists(alert_summary_path, bundle_root / "evaluation" / "alert_summary.csv")
    _copy_if_exists(performance_root / "quarterly_breakdown.csv", bundle_root / "evaluation" / "quarterly_breakdown.csv")
    _copy_if_exists(performance_root / "yearly_breakdown.csv", bundle_root / "evaluation" / "yearly_breakdown.csv")

    sample_rows = max(60, config.lstm_seq_len * 3)
    news_frame.tail(sample_rows).to_csv(
        bundle_root / "sample_news_event_recent_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    market_frame.tail(sample_rows).to_csv(
        bundle_root / "sample_market_long_recent_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[BUNDLE] saved: {bundle_root}", flush=True)
    return bundle_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export backend inference bundle.")
    parser.add_argument("--target-ticker", default="QQQ")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--event-threshold-pct", type=float, default=DEFAULT_EVENT_THRESHOLD_PCT)
    parser.add_argument("--event-news-weight", type=float, default=DEFAULT_EVENT_NEWS_WEIGHT)
    parser.add_argument("--direction-threshold", type=float, default=DEFAULT_DIRECTION_THRESHOLD)
    parser.add_argument("--event-common-start-date", default=DEFAULT_EVENT_COMMON_START_DATE)
    parser.add_argument("--market-common-start-date", default="2000-01-01")
    parser.add_argument("--event-training-frame", default=None)
    parser.add_argument("--market-long-training-frame", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--lstm-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--lstm-epochs", type=int, default=100)
    parser.add_argument(
        "--lstm-training-mode",
        choices=("multitask", "event_only"),
        default="multitask",
        help="Use multitask to match the evaluated LSTM event branch.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    export_bundle(parse_args())


if __name__ == "__main__":
    main()
