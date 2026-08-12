from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
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
from shared.news.features import build_daily_news_feature_table
from shared.news.merge import merge_news_features_into_market_frame


DEFAULT_HORIZON = 16
DEFAULT_OUTPUT_ROWS = 120


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


def _default_feature_output(ticker: str, filename: str) -> Path:
    slug = ticker_slug(ticker)
    server = _server_root()
    if server is not None:
        return server / "data" / "features" / slug / filename
    return PROJECT_ROOT / "data" / "features" / slug / filename


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV does not exist: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        raise ValueError(f"{label} CSV is empty: {path}")
    return frame


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _normalize_article_news_frame(
    source: pd.DataFrame,
    *,
    ticker: str,
    include_all_sectors: bool,
) -> pd.DataFrame:
    """Normalize crawler output into the article-level schema expected by news features.

    Supported inputs:
    - legacy crawler files with date/category/doc_type/title/body
    - policy monitor files with sector/release_date/body_summary/url
    """

    frame = source.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    slug = ticker_slug(ticker)

    if not include_all_sectors and "sector" in frame.columns:
        sector = frame["sector"].fillna("").astype(str).str.lower().str.strip()
        frame = frame.loc[sector.eq(slug)].copy()
        if frame.empty:
            available = sorted(source["sector"].dropna().astype(str).str.lower().unique().tolist())
            raise ValueError(
                f"No rows for ticker sector '{slug}'. "
                f"Available sectors in crawler input: {available}"
            )

    date_column = _first_existing_column(frame, ("date", "Date", "release_date", "published_at"))
    if date_column is None:
        raise ValueError(
            "Crawler input must contain one of these date columns: "
            "date, Date, release_date, published_at"
        )

    normalized = pd.DataFrame()
    normalized["date"] = pd.to_datetime(frame[date_column], errors="coerce")
    normalized["category"] = (
        frame[_first_existing_column(frame, ("category", "source", "doc_category"))]
        if _first_existing_column(frame, ("category", "source", "doc_category")) is not None
        else "Unknown"
    )
    normalized["doc_type"] = (
        frame[_first_existing_column(frame, ("doc_type", "document_type", "source"))]
        if _first_existing_column(frame, ("doc_type", "document_type", "source")) is not None
        else "unknown"
    )
    normalized["title"] = (
        frame[_first_existing_column(frame, ("title", "headline"))]
        if _first_existing_column(frame, ("title", "headline")) is not None
        else ""
    )
    normalized["body"] = (
        frame[_first_existing_column(frame, ("body", "body_summary", "body_text", "summary"))]
        if _first_existing_column(frame, ("body", "body_summary", "body_text", "summary")) is not None
        else ""
    )
    normalized["link"] = (
        frame[_first_existing_column(frame, ("link", "url"))]
        if _first_existing_column(frame, ("link", "url")) is not None
        else ""
    )

    passthrough_columns = [
        "body_original_length",
        "body_n_chunks",
        "title_positive_prob",
        "title_negative_prob",
        "title_neutral_prob",
        "title_sentiment_score",
        "body_positive_prob",
        "body_negative_prob",
        "body_neutral_prob",
        "body_sentiment_score",
        "matched_keywords",
    ]
    category_indicator_columns = [
        column for column in frame.columns if str(column).startswith("category_")
    ]
    for column in [*passthrough_columns, *category_indicator_columns]:
        if column in frame.columns:
            normalized[column] = frame[column]

    normalized = normalized.dropna(subset=["date"]).copy()
    if normalized.empty:
        raise ValueError("Crawler input has no valid dates after parsing.")

    for text_column in ("category", "doc_type", "title", "body", "link"):
        normalized[text_column] = normalized[text_column].fillna("").astype(str)

    if "body_original_length" not in normalized.columns:
        normalized["body_original_length"] = normalized["body"].str.len()
    if "body_n_chunks" not in normalized.columns:
        normalized["body_n_chunks"] = 0

    for column in (
        "title_positive_prob",
        "title_negative_prob",
        "title_neutral_prob",
        "title_sentiment_score",
        "body_positive_prob",
        "body_negative_prob",
        "body_neutral_prob",
        "body_sentiment_score",
    ):
        if column not in normalized.columns:
            normalized[column] = 1.0 if column.endswith("_neutral_prob") else 0.0
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(
            1.0 if column.endswith("_neutral_prob") else 0.0
        )

    normalized["body_original_length"] = pd.to_numeric(
        normalized["body_original_length"], errors="coerce"
    ).fillna(0).astype(int)
    normalized["body_n_chunks"] = pd.to_numeric(
        normalized["body_n_chunks"], errors="coerce"
    ).fillna(0).astype(int)
    normalized["is_negative_news"] = (normalized["body_sentiment_score"] <= -0.15).astype(int)
    normalized["is_positive_news"] = (normalized["body_sentiment_score"] >= 0.15).astype(int)
    return normalized.sort_values("date").reset_index(drop=True)


def _load_feature_schema(bundle_path: Path) -> dict[str, Any]:
    schema_path = bundle_path / "feature_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"feature_schema.json not found in bundle: {schema_path}. "
            "Pass --bundle to the deployed model bundle directory."
        )
    with schema_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _ordered_unique(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def _required_columns_from_schema(schema: dict[str, Any]) -> tuple[list[str], list[str], int]:
    news_columns = list(schema["news_event_lstm"]["feature_columns"])
    market_columns = _ordered_unique(
        list(schema["market_event_lstm"]["feature_columns"])
        + list(schema["xgb_direction"]["feature_columns"])
        + [schema.get("drawdown_column", "drawdown")]
    )
    seq_len = max(
        int(schema["news_event_lstm"].get("seq_len", 10)),
        int(schema["market_event_lstm"].get("seq_len", 10)),
    )
    return news_columns, market_columns, seq_len


def _coerce_feature_frame(frame: pd.DataFrame, *, date_column: str = "Date") -> pd.DataFrame:
    result = frame.copy()
    if date_column not in result.columns:
        raise ValueError(f"Feature frame must contain '{date_column}' column.")
    result[date_column] = pd.to_datetime(result[date_column], errors="coerce").dt.tz_localize(None)
    result = result.dropna(subset=[date_column]).sort_values(date_column).reset_index(drop=True)
    return result


def _load_or_build_market_frame(
    *,
    market_input: Path | None,
    config,
) -> tuple[pd.DataFrame, str]:
    if market_input is not None:
        market_frame = _read_csv(market_input, "market input")
        return _coerce_feature_frame(market_frame), f"loaded:{market_input}"

    raw_market = download_market_data(config)
    market_frame, _ = build_market_feature_frame(
        raw_market,
        config.target_ticker,
        supplementary_tickers=config.macro_tickers,
    )
    return _coerce_feature_frame(market_frame), "downloaded:yfinance"


def _select_feature_rows(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    label: str,
    rows: int,
    min_rows: int,
    as_of_date: str | None,
) -> pd.DataFrame:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} frame is missing required columns: {missing}")

    selected = _coerce_feature_frame(frame)
    if as_of_date:
        selected = selected.loc[selected["Date"] <= pd.to_datetime(as_of_date)].copy()
    selected = selected.replace([np.inf, -np.inf], np.nan)
    selected = selected.dropna(subset=["Date", *feature_columns]).copy()
    if len(selected) < min_rows:
        raise ValueError(
            f"{label} frame needs at least {min_rows} usable rows, got {len(selected)}."
        )
    output = selected[["Date", *feature_columns]].tail(rows).copy()
    output["Date"] = output["Date"].dt.strftime("%Y-%m-%d")
    return output


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export prediction-ready feature CSVs from crawler news rows and market features. "
            "This script does not train a model; it only prepares inputs for predict_signal.py."
        )
    )
    parser.add_argument("--ticker", default="QQQ", help="Target ticker. Default: QQQ")
    parser.add_argument(
        "--crawler-input",
        default=None,
        help=(
            "Article-level crawler CSV. Default on server: "
            "/opt/riseai/data/crawler/policy_updates_features.csv"
        ),
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help=(
            "Model bundle directory used for feature schema validation. "
            "Default on server: /opt/riseai/models/current/{ticker}"
        ),
    )
    parser.add_argument(
        "--market-input",
        default=None,
        help=(
            "Optional precomputed market feature CSV. If omitted, market data is downloaded "
            "with yfinance and converted with the existing market feature pipeline."
        ),
    )
    parser.add_argument(
        "--news-output",
        default=None,
        help="Output CSV for news-event model features.",
    )
    parser.add_argument(
        "--market-output",
        default=None,
        help="Output CSV for market-long model/XGBoost features.",
    )
    parser.add_argument(
        "--daily-news-output",
        default=None,
        help="Optional output CSV for the intermediate daily news feature table.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_OUTPUT_ROWS,
        help=f"Number of recent feature rows to write. Default: {DEFAULT_OUTPUT_ROWS}",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Market download start date. Ignored when --market-input is used.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help=(
            "Market download end date. yfinance treats this as exclusive. "
            "Default: tomorrow according to the server clock."
        ),
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Only export rows on or before this date. Example: 2026-08-07",
    )
    parser.add_argument(
        "--include-all-sectors",
        action="store_true",
        help="Do not filter crawler rows by the sector column.",
    )
    return parser.parse_args()


def export_prediction_features(
    *,
    ticker: str = "QQQ",
    crawler_input: Path | str | None = None,
    bundle_path: Path | str | None = None,
    market_input: Path | str | None = None,
    news_output: Path | str | None = None,
    market_output: Path | str | None = None,
    daily_news_output: Path | str | None = None,
    rows: int = DEFAULT_OUTPUT_ROWS,
    start_date: str | None = None,
    end_date: str | None = None,
    as_of_date: str | None = None,
    include_all_sectors: bool = False,
) -> dict[str, Any]:
    """Create prediction-ready feature CSVs and return a compact summary."""

    ticker = ticker.upper()
    crawler_input_path = (
        Path(crawler_input) if crawler_input else _default_crawler_input(ticker)
    )
    bundle = Path(bundle_path) if bundle_path else _default_bundle_path(ticker)
    news_output_path = (
        Path(news_output)
        if news_output
        else _default_feature_output(ticker, "news_event_features.csv")
    )
    market_output_path = (
        Path(market_output)
        if market_output
        else _default_feature_output(ticker, "market_long_features.csv")
    )
    daily_news_output_path = Path(daily_news_output) if daily_news_output else None

    if rows <= 0:
        raise ValueError("--rows must be a positive integer.")

    schema = _load_feature_schema(bundle)
    news_required_columns, market_required_columns, min_rows = _required_columns_from_schema(schema)

    raw_news = _read_csv(crawler_input_path, "crawler input")
    article_news = _normalize_article_news_frame(
        raw_news,
        ticker=ticker,
        include_all_sectors=include_all_sectors,
    )
    daily_news = build_daily_news_feature_table(article_news)
    if daily_news_output_path is not None:
        _write_csv(daily_news, daily_news_output_path)

    resolved_end_date = end_date or (date.today() + timedelta(days=1)).isoformat()
    overrides: dict[str, Any] = {"end_date": resolved_end_date}
    if start_date:
        overrides["start_date"] = start_date
    config = make_training_config(ticker, news_input_path=crawler_input_path, **overrides)

    market_input_path = Path(market_input) if market_input else None
    market_frame, market_source = _load_or_build_market_frame(
        market_input=market_input_path,
        config=config,
    )
    news_event_frame, _ = merge_news_features_into_market_frame(market_frame, daily_news)

    news_export = _select_feature_rows(
        news_event_frame,
        feature_columns=news_required_columns,
        label="news-event",
        rows=rows,
        min_rows=min_rows,
        as_of_date=as_of_date,
    )
    market_export = _select_feature_rows(
        market_frame,
        feature_columns=market_required_columns,
        label="market-long",
        rows=rows,
        min_rows=min_rows,
        as_of_date=as_of_date,
    )

    _write_csv(news_export, news_output_path)
    _write_csv(market_export, market_output_path)

    return {
        "ticker": ticker,
        "bundle": str(bundle),
        "crawler_input": str(crawler_input_path),
        "market_source": market_source,
        "article_news_rows": int(len(article_news)),
        "daily_news_rows": int(len(daily_news)),
        "daily_news_output": str(daily_news_output_path) if daily_news_output_path else None,
        "news_output": str(news_output_path),
        "news_output_rows": int(len(news_export)),
        "news_output_start": str(news_export["Date"].iloc[0]),
        "news_output_end": str(news_export["Date"].iloc[-1]),
        "market_output": str(market_output_path),
        "market_output_rows": int(len(market_export)),
        "market_output_start": str(market_export["Date"].iloc[0]),
        "market_output_end": str(market_export["Date"].iloc[-1]),
        "min_required_rows_for_lstm": int(min_rows),
    }


def main() -> None:
    args = parse_args()
    summary = export_prediction_features(
        ticker=args.ticker,
        crawler_input=args.crawler_input,
        bundle_path=args.bundle,
        market_input=args.market_input,
        news_output=args.news_output,
        market_output=args.market_output,
        daily_news_output=args.daily_news_output,
        rows=args.rows,
        start_date=args.start_date,
        end_date=args.end_date,
        as_of_date=args.as_of_date,
        include_all_sectors=bool(args.include_all_sectors),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
