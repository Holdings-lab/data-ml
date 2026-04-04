from __future__ import annotations

import os
import random
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from shared.config.schema import MarketNewsTrainingConfig
from shared.common.utils import write_json


warnings.filterwarnings("ignore")


def seed_everything(seed: int = 42) -> None:
    """
    재현 가능한 학습 결과를 위해 주요 랜덤 시드를 고정한다.

    실험 비교가 목적일 때는 모델 구조보다도 "같은 조건에서 다시 재현되는가"가
    더 중요할 수 있어서, 학습 시작 전에 이 함수를 먼저 호출한다.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _serialize_timestamp(value: object) -> str | None:
    """
    Timestamp 또는 날짜 비슷한 값을 JSON에 넣기 쉬운 문자열로 변환한다.
    """
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.strftime("%Y-%m-%d")


def _filter_feature_frame_by_min_date(
    feature_df: pd.DataFrame,
    min_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    학습에 사용할 피처 프레임을 날짜 기준으로 정렬하고 필요하면 시작일을 자른다.

    시장-only 실험은 긴 히스토리를 그대로 쓸 수 있지만,
    공정 비교용 실험은 뉴스 커버리지가 존재하는 시점 이후만 남겨야 한다.
    """
    filtered = feature_df.copy()

    if "Date" not in filtered.columns:
        raise ValueError("Feature dataframe must include a Date column.")

    filtered["Date"] = pd.to_datetime(filtered["Date"], errors="coerce")
    filtered = filtered.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if min_date is None:
        return filtered

    minimum_timestamp = pd.to_datetime(min_date, errors="coerce")
    if pd.isna(minimum_timestamp):
        raise ValueError(f"Invalid min_date value: {min_date}")

    filtered = filtered[filtered["Date"] >= minimum_timestamp].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No rows remain after applying the comparison start date filter.")

    return filtered


def build_supervised_frame(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
) -> pd.DataFrame:
    """
    horizon에 맞는 지도학습용 테이블을 만든다.

    입력은 오늘 기준 피처이고,
    라벨은 horizon 거래일 뒤의 로그수익률과 미래 가격이다.
    이렇게 만들어 두면 회귀 학습과 가격 복원이 모두 쉬워진다.
    """
    supervised = feature_df.copy()
    supervised["target_logret"] = (
        np.log(supervised["target_price"].shift(-horizon) / supervised["target_price"]) * 100
    )
    supervised["target_future_price"] = supervised["target_price"].shift(-horizon)
    supervised["target_date"] = supervised["Date"].shift(-horizon)

    required_columns = feature_columns + [
        "target_logret",
        "target_future_price",
        "target_date",
        "target_price",
    ]
    return supervised.dropna(subset=required_columns).copy()


def _build_feature_selector_model(config: MarketNewsTrainingConfig) -> XGBRegressor:
    """
    horizon별 대표 피처를 빠르게 고르기 위한 경량 XGBoost 모델을 만든다.

    최종 모델과 완전히 같은 설정일 필요는 없고,
    어떤 피처가 상대적으로 중요한지 안정적으로 드러내는 것이 목적이다.
    """
    return XGBRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.03,
        random_state=config.random_seed,
        tree_method="hist",
        objective="reg:squarederror",
    )


def _purge_overlapping_train_rows(train_index: np.ndarray, horizon: int) -> np.ndarray:
    """
    시계열 검증에서 미래 구간이 겹칠 수 있는 마지막 일부 train row를 제거한다.

    horizon일 뒤를 맞히는 문제에서는 경계 구간이 겹치면 검증이 과하게 낙관적으로
    보일 수 있어서, 작은 purge를 두어 데이터 누수 가능성을 줄인다.
    """
    if len(train_index) > horizon:
        return train_index[:-horizon]
    return train_index


def _compute_direction_accuracy(
    predicted_values: np.ndarray,
    actual_values: np.ndarray,
) -> float:
    """
    예측과 실제가 같은 방향인지 비율로 계산한다.
    """
    predicted_direction = (predicted_values > 0).astype(int)
    actual_direction = (actual_values > 0).astype(int)
    return float((predicted_direction == actual_direction).mean())


def select_features_for_horizon(
    feature_df: pd.DataFrame,
    candidate_feature_columns: list[str],
    horizon: int,
    config: MarketNewsTrainingConfig,
) -> tuple[list[str], float]:
    """
    특정 horizon에서 사용할 대표 피처와 CV 방향성 점수를 구한다.

    동일 horizon 비교를 하려면 "best horizon 선택"과 별개로
    해당 horizon 전용 피처 선택이 가능해야 한다.
    """
    supervised = build_supervised_frame(feature_df, candidate_feature_columns, horizon)
    if len(supervised) < 200:
        raise ValueError(f"Not enough rows to evaluate horizon={horizon}.")

    split_index = int(len(supervised) * config.train_ratio)
    X_train_full = supervised[candidate_feature_columns].iloc[:split_index]
    y_train_full = supervised["target_logret"].iloc[:split_index]

    if X_train_full.empty or y_train_full.empty:
        raise ValueError(f"Empty training split for horizon={horizon}.")

    selector = _build_feature_selector_model(config)
    selector.fit(X_train_full, y_train_full)

    importances = pd.Series(selector.feature_importances_, index=candidate_feature_columns)
    current_top_features = (
        importances.sort_values(ascending=False)
        .head(config.top_feature_count)
        .index
        .tolist()
    )

    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []

    for train_index, valid_index in tscv.split(X_train_full):
        purged_train_index = _purge_overlapping_train_rows(train_index, horizon)
        if len(purged_train_index) == 0:
            continue

        X_tr = X_train_full[current_top_features].iloc[purged_train_index]
        X_va = X_train_full[current_top_features].iloc[valid_index]
        y_tr = y_train_full.iloc[purged_train_index]
        y_va = y_train_full.iloc[valid_index]

        model = _build_feature_selector_model(config)
        model.fit(X_tr, y_tr)

        direction_accuracy = _compute_direction_accuracy(
            model.predict(X_va),
            y_va.to_numpy(),
        )
        cv_scores.append(direction_accuracy)

    if not cv_scores:
        raise ValueError(f"Could not compute CV score for horizon={horizon}.")

    return current_top_features, float(np.mean(cv_scores))


def select_best_horizon_and_features(
    feature_df: pd.DataFrame,
    candidate_feature_columns: list[str],
    config: MarketNewsTrainingConfig,
) -> tuple[int, list[str], float]:
    """
    horizon 후보를 비교하고, 각 horizon에서 상위 중요 피처를 선택한다.

    여기서는 RMSE보다 방향성 정확도를 우선으로 두는데,
    뉴스 피처가 가격 절대값보다 시장 방향성에 먼저 기여하는 경우가 많기 때문이다.
    """
    best_overall_score = float("-inf")
    best_horizon: int | None = None
    best_features: list[str] = []

    for horizon in config.horizon_candidates:
        try:
            current_top_features, average_score = select_features_for_horizon(
                feature_df,
                candidate_feature_columns,
                horizon,
                config,
            )
        except ValueError:
            continue
        if average_score > best_overall_score:
            best_overall_score = average_score
            best_horizon = horizon
            best_features = current_top_features

    if best_horizon is None or not best_features:
        raise ValueError("Could not select a valid horizon. Check data coverage and feature quality.")

    return best_horizon, best_features, best_overall_score


def optimize_model_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    horizon: int,
    config: MarketNewsTrainingConfig,
) -> dict:
    """
    Optuna로 최종 XGBoost 하이퍼파라미터를 조정한다.

    목적함수는 단일 지표 하나에만 맞추지 않고,
    방향성 정확도와 로그수익률 RMSE를 함께 반영한다.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 4, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "random_state": config.random_seed,
            "tree_method": "hist",
            "objective": "reg:squarederror",
        }

        tscv = TimeSeriesSplit(n_splits=4)
        combined_scores = []

        for train_index, valid_index in tscv.split(X_train):
            purged_train_index = _purge_overlapping_train_rows(train_index, horizon)
            if len(purged_train_index) == 0:
                continue

            X_tr = X_train.iloc[purged_train_index]
            X_va = X_train.iloc[valid_index]
            y_tr = y_train.iloc[purged_train_index]
            y_va = y_train.iloc[valid_index]

            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr)

            predicted_logret = model.predict(X_va)
            direction_accuracy = _compute_direction_accuracy(predicted_logret, y_va.to_numpy())
            rmse = float(np.sqrt(mean_squared_error(y_va, predicted_logret)))
            combined_scores.append(direction_accuracy - (rmse * 0.1))

        return float(np.mean(combined_scores))

    sampler = optuna.samplers.TPESampler(seed=config.random_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=config.optuna_trials)
    return study.best_params


def evaluate_model(
    model: XGBRegressor,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict, pd.DataFrame]:
    """
    최종 모델을 테스트 구간에서 평가하고 예측 테이블을 만든다.

    학습 라벨은 로그수익률이지만,
    팀원이 해석하기 쉽도록 최종 평가는 미래 가격 기준 메트릭으로 정리한다.
    """
    X_test = test_frame[feature_columns]
    y_test = test_frame["target_logret"]

    predicted_logret = model.predict(X_test)
    current_price = test_frame["target_price"].to_numpy()
    future_price = test_frame["target_future_price"].to_numpy()
    predicted_future_price = current_price * np.exp(predicted_logret / 100.0)

    mae = float(mean_absolute_error(future_price, predicted_future_price))
    rmse = float(np.sqrt(mean_squared_error(future_price, predicted_future_price)))
    r2 = float(r2_score(future_price, predicted_future_price))
    direction_accuracy = _compute_direction_accuracy(
        predicted_future_price - current_price,
        future_price - current_price,
    )
    mape = float(np.mean(np.abs((future_price - predicted_future_price) / future_price)) * 100)

    baseline_future_price = current_price.copy()
    baseline_rmse = float(np.sqrt(mean_squared_error(future_price, baseline_future_price)))
    baseline_mae = float(mean_absolute_error(future_price, baseline_future_price))
    baseline_mape = float(
        np.mean(np.abs((future_price - baseline_future_price) / future_price)) * 100
    )

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2_score": r2,
        "direction_accuracy": direction_accuracy,
        "mape": mape,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "baseline_mape": baseline_mape,
    }

    predictions = pd.DataFrame(
        {
            "Current_Date": pd.to_datetime(test_frame["Date"]),
            "Target_Date": pd.to_datetime(test_frame["target_date"]),
            "Current_Price": current_price,
            "Actual_Future_Price": future_price,
            "Pred_Future_Price": predicted_future_price,
            "Pred_LogRet": predicted_logret,
            "Actual_LogRet": y_test.to_numpy(),
        }
    )
    return metrics, predictions


def _align_supervised_frames_on_date(
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    두 지도학습 프레임을 동일한 Date 축으로 정렬한다.

    동일 horizon, 동일 기간 비교를 주장하려면
    train/test 분할 직전의 입력 날짜 집합이 실제로 같아야 한다.
    """
    left_aligned = left_frame.copy()
    right_aligned = right_frame.copy()

    left_aligned["Date"] = pd.to_datetime(left_aligned["Date"], errors="coerce")
    right_aligned["Date"] = pd.to_datetime(right_aligned["Date"], errors="coerce")

    common_dates = pd.Index(left_aligned["Date"]).intersection(pd.Index(right_aligned["Date"]))
    if common_dates.empty:
        raise ValueError("No common dates remain after aligning supervised frames.")

    left_aligned = (
        left_aligned[left_aligned["Date"].isin(common_dates)]
        .sort_values("Date")
        .reset_index(drop=True)
    )
    right_aligned = (
        right_aligned[right_aligned["Date"].isin(common_dates)]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if len(left_aligned) != len(right_aligned):
        raise ValueError("Aligned supervised frames have different lengths.")

    if not left_aligned["Date"].equals(right_aligned["Date"]):
        raise ValueError("Aligned supervised frames do not share the same Date index.")

    return left_aligned, right_aligned


def _to_serializable_config(config: MarketNewsTrainingConfig) -> dict:
    """
    dataclass 설정 객체를 JSON 저장 가능한 형태로 바꾼다.
    """
    raw_config = asdict(config)
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in raw_config.items()
    }


def _fit_and_evaluate_supervised_experiment(
    experiment_name: str,
    supervised_frame: pd.DataFrame,
    selected_features: list[str],
    horizon: int,
    horizon_score: float | None,
    config: MarketNewsTrainingConfig,
) -> tuple[dict, pd.DataFrame, XGBRegressor]:
    """
    이미 준비된 supervised frame을 받아 최종 튜닝, 학습, 평가를 수행한다.

    일반 실험과 aligned 비교 실험이 같은 학습 코드를 공유하도록 분리했다.
    """
    split_index = int(len(supervised_frame) * config.train_ratio)
    train_frame = supervised_frame.iloc[:split_index].copy()
    test_frame = supervised_frame.iloc[split_index:].copy()

    if train_frame.empty or test_frame.empty:
        raise ValueError(
            f"{experiment_name} experiment produced an empty train/test split. "
            "Check input data coverage."
        )

    best_params = optimize_model_hyperparameters(
        train_frame[selected_features],
        train_frame["target_logret"],
        horizon,
        config,
    )

    final_model = XGBRegressor(
        **best_params,
        random_state=config.random_seed,
        tree_method="hist",
        objective="reg:squarederror",
    )
    final_model.fit(train_frame[selected_features], train_frame["target_logret"])

    metrics, predictions = evaluate_model(final_model, test_frame, selected_features)
    metadata = {
        "experiment_name": experiment_name,
        "best_horizon": int(horizon),
        "best_horizon_direction_score": (
            None if horizon_score is None else float(horizon_score)
        ),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "feature_frame_start_date": _serialize_timestamp(supervised_frame["Date"].iloc[0]),
        "feature_frame_end_date": _serialize_timestamp(supervised_frame["Date"].iloc[-1]),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "train_start_date": _serialize_timestamp(train_frame["Date"].iloc[0]),
        "train_end_date": _serialize_timestamp(train_frame["Date"].iloc[-1]),
        "test_start_date": _serialize_timestamp(test_frame["Date"].iloc[0]),
        "test_end_date": _serialize_timestamp(test_frame["Date"].iloc[-1]),
        "metrics": metrics,
    }
    return metadata, predictions, final_model


def run_training_experiment(
    experiment_name: str,
    feature_df: pd.DataFrame,
    candidate_feature_columns: list[str],
    training_frame_output_path: Path | None,
    predictions_output_path: Path | None,
    model_output_path: Path | None,
    metadata_output_path: Path | None,
    config: MarketNewsTrainingConfig,
    forced_horizon: int | None = None,
    forced_selected_features: list[str] | None = None,
    min_date: str | pd.Timestamp | None = None,
    persist_artifacts: bool = True,
) -> dict:
    """
    하나의 XGBoost 실험 단위를 처음부터 끝까지 수행한다.

    `market_only`와 `market_news`가 같은 절차를 공유해야
    결과 차이를 피처 집합의 차이로 해석할 수 있기 때문에,
    학습/평가/저장을 공통 함수로 통일했다.
    """
    filtered_feature_df = _filter_feature_frame_by_min_date(feature_df, min_date)

    if forced_selected_features is not None and forced_horizon is None:
        raise ValueError("forced_selected_features can only be used together with forced_horizon.")

    if forced_selected_features is not None:
        best_horizon = int(forced_horizon)
        best_features = list(forced_selected_features)
        missing_columns = [
            column for column in best_features if column not in filtered_feature_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Forced feature columns are missing from {experiment_name}: {missing_columns}"
            )
        horizon_score = None
        horizon_selection_mode = "fixed"
        feature_selection_mode = "fixed_regression_style"
    elif forced_horizon is None:
        best_horizon, best_features, horizon_score = select_best_horizon_and_features(
            filtered_feature_df,
            candidate_feature_columns,
            config,
        )
        horizon_selection_mode = "best_of_candidates"
        feature_selection_mode = "model_importance_top_k"
    else:
        best_horizon = int(forced_horizon)
        best_features, horizon_score = select_features_for_horizon(
            filtered_feature_df,
            candidate_feature_columns,
            best_horizon,
            config,
        )
        horizon_selection_mode = "fixed"
        feature_selection_mode = "model_importance_top_k"

    supervised_frame = build_supervised_frame(
        filtered_feature_df,
        candidate_feature_columns,
        best_horizon,
    )
    metadata, predictions, final_model = _fit_and_evaluate_supervised_experiment(
        experiment_name=experiment_name,
        supervised_frame=supervised_frame,
        selected_features=best_features,
        horizon=best_horizon,
        horizon_score=horizon_score,
        config=config,
    )
    metadata["config"] = _to_serializable_config(config)
    metadata["comparison_min_date"] = _serialize_timestamp(min_date)
    metadata["horizon_selection_mode"] = horizon_selection_mode
    metadata["feature_selection_mode"] = feature_selection_mode

    if persist_artifacts:
        if training_frame_output_path is not None:
            supervised_frame.to_csv(
                training_frame_output_path,
                index=False,
                encoding="utf-8-sig",
            )
        if predictions_output_path is not None:
            predictions.to_csv(predictions_output_path, index=False, encoding="utf-8-sig")
        if model_output_path is not None:
            final_model.save_model(str(model_output_path))
        if metadata_output_path is not None:
            write_json(metadata, metadata_output_path)

    return metadata


def _build_delta_metrics(
    market_only_metrics: dict,
    market_news_metrics: dict,
) -> dict:
    """
    market+news 실험이 market-only 대비 얼마나 변했는지 계산한다.
    """
    return {
        "direction_accuracy": (
            market_news_metrics["direction_accuracy"]
            - market_only_metrics["direction_accuracy"]
        ),
        "rmse": market_news_metrics["rmse"] - market_only_metrics["rmse"],
        "mae": market_news_metrics["mae"] - market_only_metrics["mae"],
        "r2_score": market_news_metrics["r2_score"] - market_only_metrics["r2_score"],
        "mape": market_news_metrics["mape"] - market_only_metrics["mape"],
    }


def _build_comparison_rows(
    market_only_result: dict,
    market_news_result: dict,
    delta_metrics: dict,
    extra_columns: dict | None = None,
) -> list[dict]:
    """
    비교 CSV에 들어갈 row 3개를 생성한다.
    """
    extra_columns = extra_columns or {}
    market_only_metrics = market_only_result["metrics"]
    market_news_metrics = market_news_result["metrics"]

    return [
        {
            **extra_columns,
            "experiment_name": market_only_result["experiment_name"],
            "best_horizon": market_only_result["best_horizon"],
            "selected_feature_count": market_only_result["selected_feature_count"],
            "direction_accuracy": market_only_metrics["direction_accuracy"],
            "rmse": market_only_metrics["rmse"],
            "mae": market_only_metrics["mae"],
            "r2_score": market_only_metrics["r2_score"],
            "mape": market_only_metrics["mape"],
        },
        {
            **extra_columns,
            "experiment_name": market_news_result["experiment_name"],
            "best_horizon": market_news_result["best_horizon"],
            "selected_feature_count": market_news_result["selected_feature_count"],
            "direction_accuracy": market_news_metrics["direction_accuracy"],
            "rmse": market_news_metrics["rmse"],
            "mae": market_news_metrics["mae"],
            "r2_score": market_news_metrics["r2_score"],
            "mape": market_news_metrics["mape"],
        },
        {
            **extra_columns,
            "experiment_name": "market_news_minus_market_only",
            "best_horizon": market_news_result["best_horizon"] - market_only_result["best_horizon"],
            "selected_feature_count": (
                market_news_result["selected_feature_count"]
                - market_only_result["selected_feature_count"]
            ),
            "direction_accuracy": delta_metrics["direction_accuracy"],
            "rmse": delta_metrics["rmse"],
            "mae": delta_metrics["mae"],
            "r2_score": delta_metrics["r2_score"],
            "mape": delta_metrics["mape"],
        },
    ]


def build_comparison_artifacts(
    market_only_result: dict,
    market_news_result: dict,
) -> tuple[pd.DataFrame, dict]:
    """
    두 실험 결과를 CSV/JSON 저장에 적합한 비교 결과로 변환한다.

    CSV는 사람이 빠르게 읽는 용도이고,
    JSON은 후속 자동화나 대시보드 연결을 염두에 둔 구조다.
    """
    delta_metrics = _build_delta_metrics(
        market_only_result["metrics"],
        market_news_result["metrics"],
    )
    comparison_rows = _build_comparison_rows(
        market_only_result,
        market_news_result,
        delta_metrics,
    )
    comparison_df = pd.DataFrame(comparison_rows)

    comparison_payload = {
        "market_only": market_only_result,
        "market_news": market_news_result,
        "delta_market_news_minus_market_only": delta_metrics,
    }
    return comparison_df, comparison_payload


def run_aligned_horizon_comparison_suite(
    market_only_feature_df: pd.DataFrame,
    market_news_feature_df: pd.DataFrame,
    market_only_feature_columns: list[str],
    market_news_feature_columns: list[str],
    aligned_start_date: str | pd.Timestamp,
    config: MarketNewsTrainingConfig,
    forced_market_only_features: list[str] | None = None,
    forced_market_news_features: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    뉴스 커버리지가 실제로 존재하는 기간만 남기고,
    동일 horizon으로 두 실험을 나란히 비교한다.

    기존 best-horizon 결과는 유지하되, 공정 비교가 필요할 때는
    이 aligned comparison 결과를 함께 보는 것이 목적이다.
    """
    aligned_start_date_str = _serialize_timestamp(aligned_start_date)
    if aligned_start_date_str is None:
        raise ValueError("Aligned comparison start date is invalid.")

    comparison_frames: list[pd.DataFrame] = []
    horizon_payloads: list[dict] = []
    skipped_horizons: list[dict] = []

    filtered_market_only_df = _filter_feature_frame_by_min_date(
        market_only_feature_df,
        aligned_start_date_str,
    )
    filtered_market_news_df = _filter_feature_frame_by_min_date(
        market_news_feature_df,
        aligned_start_date_str,
    )

    for horizon in config.horizon_candidates:
        try:
            if forced_market_only_features is None:
                market_only_features, market_only_score = select_features_for_horizon(
                    filtered_market_only_df,
                    market_only_feature_columns,
                    horizon,
                    config,
                )
            else:
                market_only_features = list(forced_market_only_features)
                market_only_score = None

            if forced_market_news_features is None:
                market_news_features, market_news_score = select_features_for_horizon(
                    filtered_market_news_df,
                    market_news_feature_columns,
                    horizon,
                    config,
                )
            else:
                market_news_features = list(forced_market_news_features)
                market_news_score = None

            market_only_supervised = build_supervised_frame(
                filtered_market_only_df,
                market_only_feature_columns,
                horizon,
            )
            market_news_supervised = build_supervised_frame(
                filtered_market_news_df,
                market_news_feature_columns,
                horizon,
            )
            market_only_supervised, market_news_supervised = _align_supervised_frames_on_date(
                market_only_supervised,
                market_news_supervised,
            )

            market_only_result, _, _ = _fit_and_evaluate_supervised_experiment(
                experiment_name="market_only_aligned",
                supervised_frame=market_only_supervised,
                selected_features=market_only_features,
                horizon=horizon,
                horizon_score=market_only_score,
                config=config,
            )
            market_news_result, _, _ = _fit_and_evaluate_supervised_experiment(
                experiment_name="market_news_aligned",
                supervised_frame=market_news_supervised,
                selected_features=market_news_features,
                horizon=horizon,
                horizon_score=market_news_score,
                config=config,
            )
        except ValueError as exc:
            skipped_horizons.append(
                {
                    "shared_horizon": int(horizon),
                    "reason": str(exc),
                }
            )
            continue

        market_only_result["config"] = _to_serializable_config(config)
        market_news_result["config"] = _to_serializable_config(config)
        market_only_result["comparison_min_date"] = aligned_start_date_str
        market_news_result["comparison_min_date"] = aligned_start_date_str
        market_only_result["horizon_selection_mode"] = "shared_fixed"
        market_news_result["horizon_selection_mode"] = "shared_fixed"
        if forced_market_only_features is not None:
            market_only_result["feature_selection_mode"] = "fixed_regression_style"
        if forced_market_news_features is not None:
            market_news_result["feature_selection_mode"] = "fixed_regression_style"

        pair_df, pair_payload = build_comparison_artifacts(
            market_only_result,
            market_news_result,
        )
        pair_df["comparison_scope"] = "aligned_shared_period"
        pair_df["shared_horizon"] = int(horizon)
        pair_df["aligned_start_date"] = aligned_start_date_str
        pair_df["aligned_row_count"] = int(len(market_only_supervised))
        comparison_frames.append(pair_df)

        pair_payload["comparison_scope"] = "aligned_shared_period"
        pair_payload["shared_horizon"] = int(horizon)
        pair_payload["aligned_start_date"] = aligned_start_date_str
        pair_payload["aligned_row_count"] = int(len(market_only_supervised))
        horizon_payloads.append(pair_payload)

    if not comparison_frames:
        raise ValueError("Could not build any aligned comparison results.")

    comparison_df = pd.concat(comparison_frames, ignore_index=True)
    best_summary_item = max(
        horizon_payloads,
        key=lambda item: (
            item["delta_market_news_minus_market_only"]["direction_accuracy"],
            -item["delta_market_news_minus_market_only"]["rmse"],
        ),
    )
    summary = {
        "best_shared_horizon_by_direction_accuracy_delta": best_summary_item["shared_horizon"],
        "direction_accuracy_delta": (
            best_summary_item["delta_market_news_minus_market_only"]["direction_accuracy"]
        ),
        "rmse_delta": best_summary_item["delta_market_news_minus_market_only"]["rmse"],
        "mae_delta": best_summary_item["delta_market_news_minus_market_only"]["mae"],
        "r2_score_delta": best_summary_item["delta_market_news_minus_market_only"]["r2_score"],
        "mape_delta": best_summary_item["delta_market_news_minus_market_only"]["mape"],
    }
    payload = {
        "comparison_mode": "shared_horizon_shared_period",
        "aligned_start_date": aligned_start_date_str,
        "shared_horizon_results": horizon_payloads,
        "skipped_horizons": skipped_horizons,
        "summary": summary,
    }
    return comparison_df, payload
