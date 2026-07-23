from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.inference.lstm_xgb_bundle import load_bundle, predict_from_feature_frames


def _ticker_slug(ticker: str) -> str:
    return ticker.strip().lower().replace("^", "").replace("/", "_").replace("-", "_")


def _default_bundle_path(ticker: str) -> Path:
    slug = _ticker_slug(ticker)
    server_current = Path("/opt/riseai/models/current") / slug
    if server_current.exists():
        return server_current
    return PROJECT_ROOT / "data" / "bundles" / f"{slug}_lstm_xgb_h16"


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


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(result: dict[str, Any], output_path: Path | None, pretty: bool) -> None:
    text = json.dumps(
        result,
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
        indent=2 if pretty else None,
    )
    if output_path is None:
        print(text)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the production LSTM event + XGBoost direction signal bundle. "
            "If feature CSV paths are omitted, bundle sample CSVs are used for a smoke test."
        )
    )
    parser.add_argument("--ticker", default="QQQ", help="Ticker symbol. Default: QQQ")
    parser.add_argument(
        "--bundle",
        default=None,
        help=(
            "Model bundle directory. Default: /opt/riseai/models/current/{ticker} on the server, "
            "or data/bundles/{ticker}_lstm_xgb_h16 locally."
        ),
    )
    parser.add_argument(
        "--news-features",
        default=None,
        help="Prepared news-event feature CSV. Default: sample_news_event_recent_rows.csv in the bundle.",
    )
    parser.add_argument(
        "--market-features",
        default=None,
        help="Prepared market-long feature CSV. Default: sample_market_long_recent_rows.csv in the bundle.",
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Use only feature rows on or before this date. Example: 2026-07-23",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="Inference device. Default: cpu. Use auto to use CUDA when available.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. If omitted, JSON is printed to stdout.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ticker = args.ticker.upper()
    bundle_path = Path(args.bundle) if args.bundle else _default_bundle_path(ticker)
    news_features_path = (
        Path(args.news_features)
        if args.news_features
        else bundle_path / "sample_news_event_recent_rows.csv"
    )
    market_features_path = (
        Path(args.market_features)
        if args.market_features
        else bundle_path / "sample_market_long_recent_rows.csv"
    )

    bundle = load_bundle(bundle_path, device=_resolve_device(args.device))
    news_frame = _read_feature_frame(news_features_path, "news")
    market_frame = _read_feature_frame(market_features_path, "market")
    result = predict_from_feature_frames(
        bundle,
        news_event_frame=news_frame,
        market_long_frame=market_frame,
        as_of_date=args.as_of_date,
    )
    result["bundle_path"] = str(bundle_path)
    result["news_features_path"] = str(news_features_path)
    result["market_features_path"] = str(market_features_path)
    _write_json(result, Path(args.output) if args.output else None, pretty=args.pretty)


if __name__ == "__main__":
    main()
