from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config.ticker_presets import ticker_slug
from shared.export_prediction_features import (
    DEFAULT_OUTPUT_ROWS,
    export_prediction_features,
)
from shared.inference.lstm_xgb_bundle import load_bundle, predict_from_feature_frames


DEFAULT_HORIZON = 16


def _server_root() -> Path | None:
    root = Path("/opt/riseai")
    return root if root.exists() else None


def _default_bundle_path(ticker: str) -> Path:
    slug = ticker_slug(ticker)
    server = _server_root()
    if server is not None:
        return server / "models" / "current" / slug
    return PROJECT_ROOT / "data" / "bundles" / f"{slug}_lstm_xgb_h{DEFAULT_HORIZON}"


def _default_crawler_input(ticker: str) -> Path:
    slug = ticker_slug(ticker)
    server = _server_root()
    if server is not None:
        return server / "data" / "crawler" / "policy_updates_features.csv"
    return PROJECT_ROOT / "data" / "crawler" / "features" / slug / "merged_finbert_with_embeddings.csv"


def _default_feature_root(ticker: str) -> Path:
    slug = ticker_slug(ticker)
    server = _server_root()
    if server is not None:
        return server / "data" / "features" / slug
    return PROJECT_ROOT / "data" / "features" / slug


def _default_predictions_dir(ticker: str) -> Path:
    slug = ticker_slug(ticker)
    server = _server_root()
    if server is not None:
        return server / "data" / "predictions" / slug
    return PROJECT_ROOT / "data" / "predictions" / slug


def _default_pending_path(ticker: str) -> Path:
    slug = ticker_slug(ticker)
    server = _server_root()
    if server is not None:
        return server / "data" / "training" / slug / "pending" / "pending_predictions.csv"
    return PROJECT_ROOT / "data" / "training" / slug / "pending" / "pending_predictions.csv"


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _read_feature_frame(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} feature file does not exist: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        raise ValueError(f"{label} feature file is empty: {path}")
    if "Date" not in frame.columns:
        raise ValueError(f"{label} feature file must contain a Date column: {path}")
    return frame


def _latest_common_date(
    news_frame: pd.DataFrame,
    market_frame: pd.DataFrame,
    requested_as_of_date: str | None,
) -> str:
    news_dates = pd.to_datetime(news_frame["Date"], errors="coerce").dropna().dt.normalize()
    market_dates = pd.to_datetime(market_frame["Date"], errors="coerce").dropna().dt.normalize()
    common_dates = sorted(set(news_dates).intersection(set(market_dates)))
    if not common_dates:
        raise ValueError("No common Date values between news and market feature files.")
    if requested_as_of_date:
        requested = pd.to_datetime(requested_as_of_date).normalize()
        eligible_dates = [value for value in common_dates if value <= requested]
        if not eligible_dates:
            raise ValueError(
                f"No common feature Date values on or before requested as_of_date={requested_as_of_date}."
            )
        return eligible_dates[-1].date().isoformat()
    return common_dates[-1].date().isoformat()


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, default=_json_default, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def _bool_from_tier(result: dict[str, Any], tier_key: str) -> bool:
    return bool(result.get("tiers", {}).get(tier_key, {}).get("alert", False))


def _target_date_estimated(as_of_date: str, horizon: int) -> str:
    as_of = pd.to_datetime(as_of_date)
    return (as_of + pd.offsets.BDay(horizon)).date().isoformat()


def _upsert_csv_row(path: Path, row: dict[str, Any], key_columns: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path, encoding="utf-8-sig")
        combined = pd.concat([existing, new_row], ignore_index=True, sort=False)
    else:
        combined = new_row
    combined = combined.drop_duplicates(subset=key_columns, keep="last")
    if "as_of_date" in combined.columns:
        combined = combined.sort_values(["ticker", "as_of_date"], kind="stable")
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return int(len(combined))


def _append_pending_prediction(
    *,
    result: dict[str, Any],
    pending_path: Path,
    prediction_output: Path,
    news_features_path: Path,
    market_features_path: Path,
    bundle_path: Path,
) -> tuple[dict[str, Any], int]:
    as_of_date = str(result["as_of_date"])
    horizon = int(result["horizon"])
    ticker = str(result["ticker"]).upper()
    active_tier = result.get("active_tier") or ""
    prediction_id = f"{ticker_slug(ticker)}_{as_of_date}_h{horizon}"
    normal_alert = _bool_from_tier(result, "normal")
    high_confidence_alert = _bool_from_tier(result, "high_confidence")
    strong_alert = _bool_from_tier(result, "strong")

    row = {
        "prediction_id": prediction_id,
        "ticker": ticker,
        "as_of_date": as_of_date,
        "horizon": horizon,
        "target_date_estimated": _target_date_estimated(as_of_date, horizon),
        "event_score": float(result["event_score"]),
        "event_probability_news": float(result["event_probability_news"]),
        "event_probability_market": float(result["event_probability_market"]),
        "drawdown": float(result["drawdown"]),
        "event_regime": result["event_regime"],
        "pred_direction": result["direction"],
        "direction_probability_up": float(result["direction_probability_up"]),
        "active_tier": active_tier,
        "normal_alert": normal_alert,
        "high_confidence_alert": high_confidence_alert,
        "strong_alert": strong_alert,
        "predicted_alert": bool(active_tier),
        "bundle_path": str(bundle_path),
        "news_features_path": str(news_features_path),
        "market_features_path": str(market_features_path),
        "prediction_output": str(prediction_output),
        "status": "pending",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    pending_rows = _upsert_csv_row(pending_path, row, ["prediction_id"])
    return row, pending_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the daily production prediction flow: export features, predict, "
            "save JSON, and append/upsert the prediction into pending_predictions.csv."
        )
    )
    parser.add_argument("--ticker", default="QQQ", help="Target ticker. Default: QQQ")
    parser.add_argument("--crawler-input", default=None, help="Crawler article CSV.")
    parser.add_argument("--bundle", default=None, help="Model bundle directory.")
    parser.add_argument(
        "--feature-root",
        default=None,
        help="Directory for generated feature CSVs. Default: /opt/riseai/data/features/{ticker}",
    )
    parser.add_argument(
        "--market-input",
        default=None,
        help="Optional precomputed market feature CSV. If omitted, yfinance is used.",
    )
    parser.add_argument("--predictions-dir", default=None, help="Directory for prediction JSONs.")
    parser.add_argument("--prediction-output", default=None, help="Optional exact JSON output path.")
    parser.add_argument("--pending-path", default=None, help="pending_predictions.csv path.")
    parser.add_argument("--rows", type=int, default=DEFAULT_OUTPUT_ROWS, help="Recent rows to export.")
    parser.add_argument("--start-date", default=None, help="Market download start date.")
    parser.add_argument("--end-date", default=None, help="Market download end date.")
    parser.add_argument("--as-of-date", default=None, help="Prediction date to use.")
    parser.add_argument(
        "--include-all-sectors",
        action="store_true",
        help="Do not filter crawler rows by sector.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="Inference device. Default: cpu.",
    )
    parser.add_argument(
        "--no-pending",
        action="store_true",
        help="Run prediction but do not write pending_predictions.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ticker = args.ticker.upper()
    feature_root = Path(args.feature_root) if args.feature_root else _default_feature_root(ticker)
    news_features_path = feature_root / "news_event_features.csv"
    market_features_path = feature_root / "market_long_features.csv"
    daily_news_path = feature_root / "daily_news_features.csv"
    bundle_path = Path(args.bundle) if args.bundle else _default_bundle_path(ticker)
    crawler_input = Path(args.crawler_input) if args.crawler_input else _default_crawler_input(ticker)
    predictions_dir = (
        Path(args.predictions_dir) if args.predictions_dir else _default_predictions_dir(ticker)
    )
    pending_path = Path(args.pending_path) if args.pending_path else _default_pending_path(ticker)

    export_summary = export_prediction_features(
        ticker=ticker,
        crawler_input=crawler_input,
        bundle_path=bundle_path,
        market_input=args.market_input,
        news_output=news_features_path,
        market_output=market_features_path,
        daily_news_output=daily_news_path,
        rows=args.rows,
        start_date=args.start_date,
        end_date=args.end_date,
        as_of_date=args.as_of_date,
        include_all_sectors=bool(args.include_all_sectors),
    )

    news_frame = _read_feature_frame(news_features_path, "news")
    market_frame = _read_feature_frame(market_features_path, "market")
    effective_as_of_date = _latest_common_date(news_frame, market_frame, args.as_of_date)
    bundle = load_bundle(bundle_path, device=_resolve_device(args.device))
    result = predict_from_feature_frames(
        bundle,
        news_event_frame=news_frame,
        market_long_frame=market_frame,
        as_of_date=effective_as_of_date,
    )
    result["bundle_path"] = str(bundle_path)
    result["news_features_path"] = str(news_features_path)
    result["market_features_path"] = str(market_features_path)

    prediction_output = (
        Path(args.prediction_output)
        if args.prediction_output
        else predictions_dir / f"{ticker_slug(ticker)}_signal_{effective_as_of_date}.json"
    )
    _write_json(result, prediction_output)

    pending_row = None
    pending_rows = None
    if not args.no_pending:
        pending_row, pending_rows = _append_pending_prediction(
            result=result,
            pending_path=pending_path,
            prediction_output=prediction_output,
            news_features_path=news_features_path,
            market_features_path=market_features_path,
            bundle_path=bundle_path,
        )

    summary = {
        "feature_export": export_summary,
        "prediction": result,
        "prediction_output": str(prediction_output),
        "pending_path": None if args.no_pending else str(pending_path),
        "pending_rows": pending_rows,
        "pending_row": pending_row,
    }
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, default=_json_default, indent=2))


if __name__ == "__main__":
    main()
