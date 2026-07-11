from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from shared.training.lstm_pipeline import _EventLSTMClassifier, _LSTMRegressor


@dataclass
class LoadedLstmModel:
    model: torch.nn.Module
    feature_columns: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    seq_len: int
    architecture: str


@dataclass
class LoadedBundle:
    bundle_dir: Path
    manifest: dict[str, Any]
    feature_schema: dict[str, Any]
    thresholds: dict[str, Any]
    news_event_lstm: LoadedLstmModel
    market_event_lstm: LoadedLstmModel
    xgb_direction: xgb.Booster


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_lstm(path: Path, device: str = "cpu") -> LoadedLstmModel:
    payload = torch.load(path, map_location=device, weights_only=False)
    architecture = str(payload["architecture"])
    feature_columns = list(payload["feature_columns"])
    if architecture == "separate_event_and_direction_lstm_encoders":
        model = _LSTMRegressor(
            input_size=int(payload["input_size"]),
            hidden_size=int(payload["hidden_size"]),
            num_layers=int(payload["num_layers"]),
            dropout=float(payload["dropout"]),
        )
    elif architecture == "event_only_lstm_classifier":
        model = _EventLSTMClassifier(
            input_size=int(payload["input_size"]),
            hidden_size=int(payload["hidden_size"]),
            num_layers=int(payload["num_layers"]),
            dropout=float(payload["dropout"]),
        )
    else:
        raise ValueError(f"Unsupported LSTM architecture: {architecture}")
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return LoadedLstmModel(
        model=model,
        feature_columns=feature_columns,
        scaler_mean=np.asarray(payload["scaler_mean"], dtype=float),
        scaler_scale=np.asarray(payload["scaler_scale"], dtype=float),
        seq_len=int(payload["seq_len"]),
        architecture=architecture,
    )


def load_bundle(bundle_dir: str | Path, device: str = "cpu") -> LoadedBundle:
    bundle_path = Path(bundle_dir)
    booster = xgb.Booster()
    booster.load_model(bundle_path / "xgb_direction.json")
    return LoadedBundle(
        bundle_dir=bundle_path,
        manifest=_read_json(bundle_path / "manifest.json"),
        feature_schema=_read_json(bundle_path / "feature_schema.json"),
        thresholds=_read_json(bundle_path / "thresholds.json"),
        news_event_lstm=_load_lstm(bundle_path / "lstm_news_event.pt", device=device),
        market_event_lstm=_load_lstm(bundle_path / "lstm_market_event.pt", device=device),
        xgb_direction=booster,
    )


def _latest_rows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    seq_len: int,
    as_of_date: str | pd.Timestamp | None,
) -> pd.DataFrame:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    data = frame.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date")
    if as_of_date is not None:
        data = data.loc[data["Date"] <= pd.to_datetime(as_of_date)]
    data = data.dropna(subset=feature_columns)
    if len(data) < seq_len:
        raise ValueError(f"Need at least {seq_len} rows, got {len(data)}.")
    return data.tail(seq_len).copy()


def _event_probability(
    loaded: LoadedLstmModel,
    frame: pd.DataFrame,
    as_of_date: str | pd.Timestamp | None,
) -> float:
    rows = _latest_rows(frame, loaded.feature_columns, loaded.seq_len, as_of_date)
    values = rows[loaded.feature_columns].to_numpy(dtype=float)
    scaled = (values - loaded.scaler_mean) / (loaded.scaler_scale + 1e-12)
    x = torch.as_tensor(scaled[None, :, :], dtype=torch.float32)
    device = next(loaded.model.parameters()).device
    x = x.to(device)
    with torch.no_grad():
        if loaded.architecture == "separate_event_and_direction_lstm_encoders":
            _, event_logit, _ = loaded.model(x)
        else:
            event_logit = loaded.model(x)
        return float(torch.sigmoid(event_logit).detach().cpu().numpy()[0])


def _latest_xgb_row(
    frame: pd.DataFrame,
    feature_columns: list[str],
    as_of_date: str | pd.Timestamp | None,
) -> pd.DataFrame:
    rows = _latest_rows(frame, feature_columns, seq_len=1, as_of_date=as_of_date)
    return rows[feature_columns].astype(float)


def _drawdown_regime(drawdown: float, cutpoints: list[float]) -> str:
    first, second = float(cutpoints[0]), float(cutpoints[1])
    if drawdown < first:
        return "deep_drawdown"
    if drawdown < second:
        return "middle_drawdown"
    return "shallow_drawdown"


def _tier_result(
    tier_name: str,
    event_score: float,
    direction_score: float,
    regime: str,
    thresholds: dict[str, float],
    direction_threshold: float,
) -> dict[str, Any]:
    threshold = float(thresholds[regime])
    alert = bool(event_score >= threshold)
    predicted_up = bool(direction_score >= direction_threshold)
    return {
        "tier": tier_name,
        "alert": alert,
        "event_threshold": threshold,
        "direction": "UP" if predicted_up else "DOWN",
        "direction_probability_up": float(direction_score),
    }


def predict_from_feature_frames(
    bundle: LoadedBundle,
    news_event_frame: pd.DataFrame,
    market_long_frame: pd.DataFrame,
    as_of_date: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Predict from already-prepared numeric feature frames.

    The backend is responsible for turning a raw news item into the same daily
    numeric features used during training. This helper only runs model inference.
    """

    news_probability = _event_probability(
        bundle.news_event_lstm,
        news_event_frame,
        as_of_date,
    )
    market_probability = _event_probability(
        bundle.market_event_lstm,
        market_long_frame,
        as_of_date,
    )

    xgb_features = bundle.feature_schema["xgb_direction"]["feature_columns"]
    xgb_row = _latest_xgb_row(market_long_frame, xgb_features, as_of_date)
    direction_matrix = xgb.DMatrix(xgb_row, feature_names=xgb_features)
    direction_score = float(bundle.xgb_direction.predict(direction_matrix)[0])

    latest_market = _latest_rows(market_long_frame, ["drawdown"], 1, as_of_date)
    drawdown = float(latest_market["drawdown"].iloc[-1])
    regime = _drawdown_regime(drawdown, bundle.thresholds["drawdown_tertile_cutpoints"])

    news_weight = float(bundle.thresholds["event_news_weight"])
    event_score = news_weight * news_probability + (1.0 - news_weight) * market_probability
    direction_threshold = float(bundle.thresholds["direction_threshold"])

    tiers = {
        "normal": _tier_result(
            "Normal",
            event_score,
            direction_score,
            regime,
            bundle.thresholds["normal_thresholds"],
            direction_threshold,
        ),
        "high_confidence": _tier_result(
            "High_Confidence",
            event_score,
            direction_score,
            regime,
            bundle.thresholds["high_confidence_thresholds"],
            direction_threshold,
        ),
        "strong": _tier_result(
            "Strong",
            event_score,
            direction_score,
            regime,
            bundle.thresholds["strong_thresholds"],
            direction_threshold,
        ),
    }
    active_tier = None
    for key in ("strong", "high_confidence", "normal"):
        if tiers[key]["alert"]:
            active_tier = tiers[key]["tier"]
            break

    return {
        "ticker": bundle.manifest["target_ticker"],
        "horizon": bundle.manifest["horizon"],
        "as_of_date": str(as_of_date) if as_of_date is not None else None,
        "event_probability_news": news_probability,
        "event_probability_market": market_probability,
        "event_score": float(event_score),
        "drawdown": drawdown,
        "event_regime": regime,
        "direction_probability_up": direction_score,
        "direction": "UP" if direction_score >= direction_threshold else "DOWN",
        "active_tier": active_tier,
        "tiers": tiers,
    }
