from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.common.utils import training_data_path, write_json
from shared.config.schema import make_training_config
from shared.run_event_walkforward import _load_feature_groups
from shared.training.lstm_pipeline import run_training_experiment


SEARCH_METRICS = (
    "event_roc_auc",
    "event_pr_auc",
    "event_balanced_accuracy",
    "event_precision",
    "event_recall",
    "event_brier_score",
)


def _config_id(prefix: str, values: dict) -> str:
    encoded = "_".join(
        f"{key}-{str(value).replace('.', 'p')}" for key, value in values.items()
    )
    return f"{prefix}_{encoded}"


def _evaluate_configs(
    base_config,
    frame: pd.DataFrame,
    features: list[str],
    configs: list[tuple[str, dict]],
    years: tuple[int, ...],
    common_start_date: str,
    checkpoint_path: Path,
) -> tuple[list[dict], list[dict]]:
    fold_rows: list[dict] = []
    maximum_date = pd.to_datetime(frame["Date"].max())
    for config_name, overrides in configs:
        trial_config = replace(base_config, **overrides)
        for year in years:
            fold_start = pd.Timestamp(year=year, month=1, day=1)
            fold_end = min(pd.Timestamp(year=year, month=12, day=31), maximum_date)
            print(
                f"[HPARAM] {config_name} year={year} "
                f"seq={trial_config.lstm_seq_len} hidden={trial_config.lstm_hidden_size} "
                f"layers={trial_config.lstm_num_layers} lr={trial_config.lstm_learning_rate:g} "
                f"batch={trial_config.lstm_batch_size} wd={trial_config.lstm_weight_decay:g}",
                flush=True,
            )
            result = run_training_experiment(
                experiment_name=f"hparam_{config_name}_{year}",
                feature_df=frame,
                candidate_feature_columns=features,
                training_frame_output_path=None,
                predictions_output_path=None,
                model_output_path=None,
                metadata_output_path=None,
                config=trial_config,
                forced_horizon=trial_config.regression_style_fixed_horizon,
                forced_selected_features=features,
                min_date=common_start_date,
                test_start_date=fold_start,
                test_end_date=fold_end,
                persist_artifacts=False,
            )
            metrics = result["metrics"]
            row = {
                "config_id": config_name,
                "year": year,
                **overrides,
                "best_epoch": result["best_epoch"],
                "test_rows": result["test_rows"],
            }
            row.update({key: metrics[key] for key in SEARCH_METRICS})
            row["ranking_score"] = 0.5 * (
                metrics["event_roc_auc"] + metrics["event_pr_auc"]
            )
            fold_rows.append(row)
        pd.DataFrame(fold_rows).to_csv(
            checkpoint_path,
            index=False,
            encoding="utf-8-sig",
        )

    fold_frame = pd.DataFrame(fold_rows)
    summaries: list[dict] = []
    for config_name, group in fold_frame.groupby("config_id", sort=False):
        first = group.iloc[0]
        summary = {
            "config_id": config_name,
            "lstm_seq_len": int(first["lstm_seq_len"]),
            "lstm_hidden_size": int(first["lstm_hidden_size"]),
            "lstm_num_layers": int(first["lstm_num_layers"]),
            "lstm_dropout": float(first["lstm_dropout"]),
            "lstm_learning_rate": float(first["lstm_learning_rate"]),
            "lstm_batch_size": int(first["lstm_batch_size"]),
            "lstm_weight_decay": float(first["lstm_weight_decay"]),
            "fold_count": int(len(group)),
            "mean_ranking_score": float(group["ranking_score"].mean()),
            "std_ranking_score": float(group["ranking_score"].std(ddof=0)),
        }
        summary["robust_score"] = (
            summary["mean_ranking_score"] - 0.25 * summary["std_ranking_score"]
        )
        for key in SEARCH_METRICS:
            summary[f"mean_{key}"] = float(group[key].mean())
            summary[f"minimum_{key}"] = float(group[key].min())
        summaries.append(summary)
    summaries.sort(key=lambda row: row["robust_score"], reverse=True)
    return fold_rows, summaries


def _architecture_configs(base_config) -> list[tuple[str, dict]]:
    configs: list[tuple[str, dict]] = []
    for seq_len in (5, 10, 20):
        for hidden_size in (16, 32, 64):
            values = {
                "lstm_seq_len": seq_len,
                "lstm_hidden_size": hidden_size,
                "lstm_num_layers": 1,
                "lstm_dropout": 0.0,
                "lstm_learning_rate": base_config.lstm_learning_rate,
                "lstm_batch_size": base_config.lstm_batch_size,
                "lstm_weight_decay": base_config.lstm_weight_decay,
            }
            configs.append((_config_id("arch", values), values))
    for seq_len, hidden_size in ((5, 32), (10, 32), (20, 32), (10, 64)):
        values = {
            "lstm_seq_len": seq_len,
            "lstm_hidden_size": hidden_size,
            "lstm_num_layers": 2,
            "lstm_dropout": 0.2,
            "lstm_learning_rate": base_config.lstm_learning_rate,
            "lstm_batch_size": base_config.lstm_batch_size,
            "lstm_weight_decay": base_config.lstm_weight_decay,
        }
        configs.append((_config_id("arch", values), values))
    return configs


def _optimizer_configs(best_architecture: dict) -> list[tuple[str, dict]]:
    architecture_values = {
        "lstm_seq_len": best_architecture["lstm_seq_len"],
        "lstm_hidden_size": best_architecture["lstm_hidden_size"],
        "lstm_num_layers": best_architecture["lstm_num_layers"],
        "lstm_dropout": best_architecture["lstm_dropout"],
    }
    configs: list[tuple[str, dict]] = []
    for learning_rate in (3e-4, 1e-3, 3e-3):
        for batch_size in (32, 64):
            values = {
                **architecture_values,
                "lstm_learning_rate": learning_rate,
                "lstm_batch_size": batch_size,
                "lstm_weight_decay": 1e-4,
            }
            configs.append((_config_id("optim", values), values))
    return configs


def _weight_decay_configs(best_optimizer: dict) -> list[tuple[str, dict]]:
    configs: list[tuple[str, dict]] = []
    for weight_decay in (1e-5, 1e-4, 1e-3):
        values = {
            "lstm_seq_len": best_optimizer["lstm_seq_len"],
            "lstm_hidden_size": best_optimizer["lstm_hidden_size"],
            "lstm_num_layers": best_optimizer["lstm_num_layers"],
            "lstm_dropout": best_optimizer["lstm_dropout"],
            "lstm_learning_rate": best_optimizer["lstm_learning_rate"],
            "lstm_batch_size": best_optimizer["lstm_batch_size"],
            "lstm_weight_decay": weight_decay,
        }
        configs.append((_config_id("decay", values), values))
    return configs


def _summary_to_overrides(summary: dict) -> dict:
    return {
        key: summary[key]
        for key in (
            "lstm_seq_len",
            "lstm_hidden_size",
            "lstm_num_layers",
            "lstm_dropout",
            "lstm_learning_rate",
            "lstm_batch_size",
            "lstm_weight_decay",
        )
    }


def run_hparam_search(
    config,
    training_frame_path: Path,
    output_root: Path | None = None,
    tuning_years: tuple[int, ...] = (2022, 2023, 2024),
    holdout_years: tuple[int, ...] = (2025, 2026),
    common_start_date: str = "2017-03-10",
) -> dict:
    frame, market_columns, scalar_news_columns = _load_feature_groups(
        config,
        training_frame_path,
    )
    frame = frame.loc[frame["Date"] >= pd.Timestamp(common_start_date)].copy()
    frame = frame.sort_values("Date").reset_index(drop=True)
    features = market_columns + scalar_news_columns
    ticker_slug = config.target_ticker.lower().replace("^", "")
    root = output_root or training_data_path(
        ticker_slug,
        "event_hparam_search",
        "report.json",
    ).parent
    root.mkdir(parents=True, exist_ok=True)

    architecture_folds, architecture_summary = _evaluate_configs(
        config,
        frame,
        features,
        _architecture_configs(config),
        tuning_years,
        common_start_date,
        root / "architecture_folds.csv",
    )
    pd.DataFrame(architecture_summary).to_csv(
        root / "architecture_summary.csv", index=False, encoding="utf-8-sig"
    )
    best_architecture = architecture_summary[0]

    optimizer_folds, optimizer_summary = _evaluate_configs(
        config,
        frame,
        features,
        _optimizer_configs(best_architecture),
        tuning_years,
        common_start_date,
        root / "optimizer_folds.csv",
    )
    pd.DataFrame(optimizer_summary).to_csv(
        root / "optimizer_summary.csv", index=False, encoding="utf-8-sig"
    )
    best_optimizer = optimizer_summary[0]

    decay_folds, decay_summary = _evaluate_configs(
        config,
        frame,
        features,
        _weight_decay_configs(best_optimizer),
        tuning_years,
        common_start_date,
        root / "weight_decay_folds.csv",
    )
    pd.DataFrame(decay_summary).to_csv(
        root / "weight_decay_summary.csv", index=False, encoding="utf-8-sig"
    )
    best_final = decay_summary[0]

    baseline_values = {
        "lstm_seq_len": config.lstm_seq_len,
        "lstm_hidden_size": config.lstm_hidden_size,
        "lstm_num_layers": config.lstm_num_layers,
        "lstm_dropout": config.lstm_dropout,
        "lstm_learning_rate": config.lstm_learning_rate,
        "lstm_batch_size": config.lstm_batch_size,
        "lstm_weight_decay": config.lstm_weight_decay,
    }
    holdout_configs = [
        ("baseline", baseline_values),
        ("tuned", _summary_to_overrides(best_final)),
    ]
    holdout_folds, holdout_summary = _evaluate_configs(
        config,
        frame,
        features,
        holdout_configs,
        holdout_years,
        common_start_date,
        root / "holdout_folds.csv",
    )
    pd.DataFrame(holdout_summary).to_csv(
        root / "holdout_summary.csv", index=False, encoding="utf-8-sig"
    )

    payload = {
        "target_ticker": config.target_ticker,
        "tuning_years": list(tuning_years),
        "holdout_years": list(holdout_years),
        "selection_objective": "mean(0.5*ROC_AUC+0.5*PR_AUC)-0.25*std",
        "best_architecture": best_architecture,
        "best_optimizer": best_optimizer,
        "best_final": best_final,
        "architecture_folds": architecture_folds,
        "architecture_summary": architecture_summary,
        "optimizer_folds": optimizer_folds,
        "optimizer_summary": optimizer_summary,
        "weight_decay_folds": decay_folds,
        "weight_decay_summary": decay_summary,
        "holdout_folds": holdout_folds,
        "holdout_summary": holdout_summary,
    }
    write_json(payload, root / "report.json")
    return payload


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _print_report(payload: dict) -> None:
    best = payload["best_final"]
    print("\nBest tuning parameters")
    for key, value in _summary_to_overrides(best).items():
        print(f"  {key:24s}: {value}")
    print("\nHoldout 2025-2026")
    for row in payload["holdout_summary"]:
        print(
            f"  {row['config_id']:10s} "
            f"AUC={_pct(row['mean_event_roc_auc'])} "
            f"PR={_pct(row['mean_event_pr_auc'])} "
            f"Balanced={_pct(row['mean_event_balanced_accuracy'])} "
            f"Precision={_pct(row['mean_event_precision'])} "
            f"Recall={_pct(row['mean_event_recall'])}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time-safe Event LSTM hyperparameter search.")
    parser.add_argument(
        "--training-frame",
        default="data/training/qqq/market_news/training_frame.csv",
    )
    parser.add_argument("--target-ticker", default="QQQ")
    parser.add_argument("--ticker-preset", default="auto")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--tuning-years", default="2022,2023,2024")
    parser.add_argument("--holdout-years", default="2025,2026")
    parser.add_argument("--common-start-date", default="2017-03-10")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _parse_years(raw: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def main() -> None:
    args = parse_args()
    config = make_training_config(
        args.target_ticker,
        preset=args.ticker_preset,
        verbose_output=args.verbose,
    )
    payload = run_hparam_search(
        config,
        Path(args.training_frame),
        Path(args.output_root) if args.output_root else None,
        _parse_years(args.tuning_years),
        _parse_years(args.holdout_years),
        args.common_start_date,
    )
    _print_report(payload)


if __name__ == "__main__":
    main()
