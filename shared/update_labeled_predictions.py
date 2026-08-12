from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config.schema import make_training_config
from shared.config.ticker_presets import ticker_slug
from shared.market.data import build_market_feature_frame, download_market_data


DEFAULT_HORIZON = 16
DEFAULT_EVENT_THRESHOLD_PCT = 2.0


def _server_root() -> Path | None:
    root = Path("/opt/riseai")
    return root if root.exists() else None


def _default_pending_path(ticker: str) -> Path:
    slug = ticker_slug(ticker)
    server = _server_root()
    if server is not None:
        return server / "data" / "training" / slug / "pending" / "pending_predictions.csv"
    return PROJECT_ROOT / "data" / "training" / slug / "pending" / "pending_predictions.csv"


def _default_labeled_path(ticker: str) -> Path:
    slug = ticker_slug(ticker)
    server = _server_root()
    if server is not None:
        return server / "data" / "training" / slug / "labeled" / "labeled_predictions.csv"
    return PROJECT_ROOT / "data" / "training" / slug / "labeled" / "labeled_predictions.csv"


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_market_price_frame(
    *,
    ticker: str,
    market_input: Path | None,
    start_date: str | None,
    end_date: str | None,
) -> tuple[pd.DataFrame, str]:
    if market_input is not None:
        if not market_input.exists():
            raise FileNotFoundError(f"market input CSV does not exist: {market_input}")
        frame = pd.read_csv(market_input, encoding="utf-8-sig")
        source = f"loaded:{market_input}"
    else:
        resolved_end_date = end_date or (date.today() + timedelta(days=1)).isoformat()
        overrides: dict[str, Any] = {"end_date": resolved_end_date}
        if start_date:
            overrides["start_date"] = start_date
        config = make_training_config(ticker, **overrides)
        raw_market = download_market_data(config)
        frame, _ = build_market_feature_frame(
            raw_market,
            config.target_ticker,
            supplementary_tickers=config.macro_tickers,
        )
        source = "downloaded:yfinance"

    missing = [column for column in ("Date", "target_price") if column not in frame.columns]
    if missing:
        raise ValueError(f"market price frame is missing required columns: {missing}")
    result = frame[["Date", "target_price"]].copy()
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce").dt.tz_localize(None)
    result["target_price"] = pd.to_numeric(result["target_price"], errors="coerce")
    result = result.dropna(subset=["Date", "target_price"]).sort_values("Date").reset_index(drop=True)
    if result.empty:
        raise ValueError("market price frame has no usable rows.")
    return result, source


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _predicted_alert(row: pd.Series) -> bool:
    if "predicted_alert" in row.index:
        return _coerce_bool(row["predicted_alert"])
    active_tier = row.get("active_tier", "")
    if pd.isna(active_tier):
        return False
    return bool(str(active_tier).strip())


def _prediction_id(row: pd.Series) -> str:
    existing = row.get("prediction_id", "")
    if not pd.isna(existing) and str(existing).strip():
        return str(existing)
    ticker = str(row.get("ticker", "")).upper()
    as_of_date = pd.to_datetime(row.get("as_of_date")).date().isoformat()
    horizon = int(row.get("horizon", DEFAULT_HORIZON))
    return f"{ticker_slug(ticker)}_{as_of_date}_h{horizon}"


def _label_pending_row(
    row: pd.Series,
    *,
    market: pd.DataFrame,
    event_threshold_pct: float,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        as_of_date = pd.to_datetime(row["as_of_date"]).normalize()
    except Exception:
        return None, "invalid_as_of_date"

    horizon = int(row.get("horizon", DEFAULT_HORIZON))
    origin_candidates = market.index[market["Date"] <= as_of_date].tolist()
    if not origin_candidates:
        return None, "as_of_date_before_market_history"

    origin_idx = int(origin_candidates[-1])
    target_idx = origin_idx + horizon
    if target_idx >= len(market):
        return None, "target_trading_day_not_available_yet"

    entry_date = market.loc[origin_idx, "Date"]
    target_date = market.loc[target_idx, "Date"]
    entry_price = float(market.loc[origin_idx, "target_price"])
    target_price = float(market.loc[target_idx, "target_price"])
    actual_logret = float(np.log(target_price / entry_price))
    actual_return_pct = float((target_price / entry_price - 1.0) * 100.0)
    actual_abs_return_pct = abs(actual_return_pct)
    actual_event = bool(abs(actual_logret) >= event_threshold_pct / 100.0)
    actual_direction = "UP" if actual_logret >= 0.0 else "DOWN"
    pred_direction = str(row.get("pred_direction", row.get("direction", ""))).upper()
    predicted_alert = _predicted_alert(row)

    labeled = row.to_dict()
    labeled.update(
        {
            "prediction_id": _prediction_id(row),
            "status": "labeled",
            "entry_date": entry_date.date().isoformat(),
            "target_date_actual": target_date.date().isoformat(),
            "entry_price": entry_price,
            "target_price": target_price,
            "actual_logret": actual_logret,
            "actual_return_pct": actual_return_pct,
            "actual_abs_return_pct": actual_abs_return_pct,
            "actual_event": actual_event,
            "actual_direction": actual_direction,
            "predicted_alert": predicted_alert,
            "event_prediction_correct": bool(predicted_alert == actual_event),
            "direction_correct": bool(pred_direction == actual_direction),
            "event_and_direction_correct": bool(
                predicted_alert and actual_event and pred_direction == actual_direction
            ),
            "event_threshold_pct": float(event_threshold_pct),
            "labeled_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
    )
    return labeled, None


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _append_labeled_rows(path: Path, rows: list[dict[str, Any]]) -> int:
    if not rows:
        existing = _read_csv_if_exists(path)
        return int(len(existing))

    new_rows = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path, encoding="utf-8-sig")
        combined = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    else:
        combined = new_rows

    combined = combined.drop_duplicates(subset=["prediction_id"], keep="last")
    if "as_of_date" in combined.columns:
        combined = combined.sort_values(["ticker", "as_of_date"], kind="stable")
    _write_csv(path, combined)
    return int(len(combined))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move mature pending predictions into labeled_predictions.csv by attaching "
            "actual T+horizon market outcomes."
        )
    )
    parser.add_argument("--ticker", default="QQQ", help="Target ticker. Default: QQQ")
    parser.add_argument("--pending-path", default=None, help="pending_predictions.csv path.")
    parser.add_argument("--labeled-path", default=None, help="labeled_predictions.csv path.")
    parser.add_argument(
        "--market-input",
        default=None,
        help="Optional market feature CSV containing Date and target_price. If omitted, yfinance is used.",
    )
    parser.add_argument("--start-date", default=None, help="Market download start date.")
    parser.add_argument("--end-date", default=None, help="Market download end date.")
    parser.add_argument(
        "--event-threshold-pct",
        type=float,
        default=DEFAULT_EVENT_THRESHOLD_PCT,
        help="Actual event threshold based on absolute log return. Default: 2.0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ticker = args.ticker.upper()
    pending_path = Path(args.pending_path) if args.pending_path else _default_pending_path(ticker)
    labeled_path = Path(args.labeled_path) if args.labeled_path else _default_labeled_path(ticker)
    pending = _read_csv_if_exists(pending_path)

    if pending.empty:
        summary = {
            "ticker": ticker,
            "pending_path": str(pending_path),
            "labeled_path": str(labeled_path),
            "pending_before": 0,
            "moved_to_labeled": 0,
            "pending_after": 0,
            "message": "No pending rows to label.",
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    market, market_source = _load_market_price_frame(
        ticker=ticker,
        market_input=Path(args.market_input) if args.market_input else None,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    labeled_rows: list[dict[str, Any]] = []
    remaining_indices: list[int] = []
    waiting_reasons: dict[str, int] = {}

    for index, row in pending.iterrows():
        labeled, reason = _label_pending_row(
            row,
            market=market,
            event_threshold_pct=float(args.event_threshold_pct),
        )
        if labeled is None:
            remaining_indices.append(index)
            waiting_reasons[str(reason)] = waiting_reasons.get(str(reason), 0) + 1
        else:
            labeled_rows.append(labeled)

    remaining = pending.loc[remaining_indices].copy()
    if not remaining.empty:
        remaining["status"] = "pending"
    _write_csv(pending_path, remaining)
    labeled_total_rows = _append_labeled_rows(labeled_path, labeled_rows)

    summary = {
        "ticker": ticker,
        "pending_path": str(pending_path),
        "labeled_path": str(labeled_path),
        "market_source": market_source,
        "market_start": market["Date"].iloc[0].date().isoformat(),
        "market_end": market["Date"].iloc[-1].date().isoformat(),
        "pending_before": int(len(pending)),
        "moved_to_labeled": int(len(labeled_rows)),
        "pending_after": int(len(remaining)),
        "labeled_total_rows": int(labeled_total_rows),
        "waiting_reasons": waiting_reasons,
    }
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, default=_json_default, indent=2))


if __name__ == "__main__":
    main()
