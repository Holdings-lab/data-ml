from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from shared.config.schema import make_training_config
from shared.news.merge import (
    T1_EVENT_NEWS_FEATURE_COLUMNS,
    NEWS_QUANTITY_FEATURE_COLUMNS,
    _merge_daily_news_table,
    build_t1_event_news_features,
)
from shared.run_quarterly_event_walkforward import _completed_quarters
from shared.run_quarterly_event_calibration import (
    _apply_platt_scaler,
    _fit_platt_scaler,
    _select_threshold,
)
from shared.training.lstm_pipeline import (
    _DirectionLSTMClassifier,
    _EventLSTMClassifier,
    _calibrate_event_probability_threshold,
)
from shared.training.metrics import (
    build_supervised_frame,
    compute_signal_return_metrics,
    compute_thresholded_binary_direction_metrics,
    compute_two_stage_signal_metrics,
    split_supervised_frame,
    split_supervised_frame_at_date,
)


class LSTMSafetyTests(unittest.TestCase):
    def test_default_qqq_preset_matches_lstm_config(self) -> None:
        config = make_training_config("QQQ")

        self.assertEqual(config.preset_name, "qqq_growth_tech")
        self.assertEqual(config.regression_style_fixed_horizon, 5)
        self.assertFalse(config.use_news_embeddings)
        self.assertEqual(config.lstm_device, "auto")
        self.assertEqual(config.lstm_huber_delta, 1.0)
        self.assertEqual(config.lstm_return_loss_weight, 0.2)
        self.assertEqual(config.lstm_event_loss_weight, 1.0)
        self.assertEqual(config.lstm_direction_loss_weight, 1.0)
        self.assertEqual(config.lstm_direction_return_threshold, 2.0)
        self.assertEqual(config.lstm_event_min_recall, 0.4)
        self.assertEqual(config.lstm_event_selection_objective, "ranking")
        self.assertEqual(config.random_seed, 42)
        self.assertIsNone(config.lstm_event_probability_threshold)
        self.assertEqual(config.lstm_direction_probability_threshold, 0.5)
        self.assertFalse(config.verbose_output)

    def test_event_threshold_prioritizes_precision_with_recall_floor(self) -> None:
        threshold, metrics = _calibrate_event_probability_threshold(
            torch.tensor([0.90, 0.80, 0.40, 0.30, 0.70, 0.20, 0.10, 0.05]),
            torch.tensor([True, True, True, True, False, False, False, False]),
            minimum_recall=0.5,
        )

        self.assertGreater(threshold, 0.7)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertGreaterEqual(metrics["recall"], 0.5)

    def test_event_only_lstm_has_a_single_event_output(self) -> None:
        model = _EventLSTMClassifier(
            input_size=3,
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
        )

        output = model(torch.zeros(5, 10, 3))

        self.assertEqual(tuple(output.shape), (5,))
        self.assertFalse(hasattr(model, "direction_head"))

    def test_direction_only_lstm_has_a_single_direction_output(self) -> None:
        model = _DirectionLSTMClassifier(
            input_size=3,
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
        )

        output = model(torch.zeros(5, 10, 3))

        self.assertEqual(tuple(output.shape), (5,))
        self.assertFalse(hasattr(model, "event_head"))

    def test_quarterly_folds_exclude_incomplete_final_quarter(self) -> None:
        quarters = _completed_quarters("2025Q4", pd.Timestamp("2026-04-24"))

        self.assertEqual([str(quarter) for quarter in quarters], ["2025Q4", "2026Q1"])

    def test_platt_calibration_returns_valid_monotonic_probabilities(self) -> None:
        raw_probability = np.array([0.20, 0.30, 0.60, 0.80])
        target = np.array([0, 0, 1, 1])

        scaler = _fit_platt_scaler(raw_probability, target, random_seed=42)
        calibrated = _apply_platt_scaler(scaler, raw_probability)

        self.assertTrue(np.all((calibrated > 0.0) & (calibrated < 1.0)))
        self.assertTrue(np.all(np.diff(calibrated) > 0.0))

    def test_past_threshold_keeps_recall_floor(self) -> None:
        threshold, metrics = _select_threshold(
            np.array([1, 1, 1, 0, 0, 0]),
            np.array([0.80, 0.70, 0.20, 0.60, 0.10, 0.05]),
            minimum_recall=0.5,
        )

        self.assertGreaterEqual(metrics["recall"], 0.5)
        self.assertGreater(threshold, 0.6)

    def test_split_purges_labels_that_reach_test_period(self) -> None:
        row_count = 40
        frame = pd.DataFrame(
            {
                "Date": pd.bdate_range("2025-01-01", periods=row_count),
                "target_price": np.linspace(100.0, 120.0, row_count),
                "feature": np.arange(row_count, dtype=float),
            }
        )
        supervised = build_supervised_frame(frame, ["feature"], horizon=5)

        train, test, _pre_test, purged_rows = split_supervised_frame(supervised, 0.8)

        self.assertEqual(purged_rows, 5)
        self.assertLess(
            pd.to_datetime(train["target_date"]).max(),
            pd.to_datetime(test["Date"]).min(),
        )

        date_train, date_test, _date_pre_test, date_purged = (
            split_supervised_frame_at_date(supervised, test["Date"].iloc[0])
        )
        self.assertEqual(date_purged, purged_rows)
        self.assertEqual(date_train["Date"].tolist(), train["Date"].tolist())
        self.assertEqual(date_test["Date"].tolist(), test["Date"].tolist())

    def test_overlapping_returns_use_staggered_sleeves(self) -> None:
        log_return_10_pct = np.log(1.1) * 100.0

        metrics = compute_signal_return_metrics(
            np.ones(4),
            np.full(4, log_return_10_pct),
            horizon=2,
        )

        self.assertAlmostEqual(metrics["buy_and_hold_cumulative_return_pct"], 21.0)
        self.assertEqual(
            metrics["cumulative_return_method"],
            "staggered_non_overlapping_sleeves",
        )

    def test_thresholded_binary_metrics_ignore_small_moves(self) -> None:
        metrics = compute_thresholded_binary_direction_metrics(
            np.array([False, True, True]),
            np.array([-1.0, 0.1, 1.0]),
            direction_threshold=0.5,
        )

        self.assertEqual(metrics["direction_accuracy"], 1.0)
        self.assertEqual(metrics["macro_balanced_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["actionable_label_coverage"], 2.0 / 3.0)
        self.assertEqual(metrics["direction_sample_count"], 2)

    def test_thresholded_binary_metrics_allow_no_events(self) -> None:
        metrics = compute_thresholded_binary_direction_metrics(
            np.array([False, True]),
            np.array([-0.1, 0.1]),
            direction_threshold=0.5,
        )

        self.assertIsNone(metrics["direction_accuracy"])
        self.assertEqual(metrics["direction_sample_count"], 0)
        self.assertEqual(metrics["ignored_small_move_count"], 2)

    def test_two_stage_accuracy_includes_event_false_positives(self) -> None:
        metrics = compute_two_stage_signal_metrics(
            np.array([True, True, False, True]),
            np.array([True, False, True, True]),
            np.array([3.0, -3.0, 3.0, 0.1]),
            direction_threshold=2.0,
        )

        self.assertAlmostEqual(metrics["event_precision"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["event_recall"], 2.0 / 3.0)
        self.assertEqual(metrics["conditional_direction_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["strong_signal_accuracy"], 2.0 / 3.0)
        self.assertEqual(metrics["predicted_event_up_count"], 2)
        self.assertEqual(metrics["predicted_event_down_count"], 1)

    def test_news_values_are_delayed_one_market_row(self) -> None:
        market = pd.DataFrame(
            {
                "Date": pd.bdate_range("2025-01-06", periods=3),
                "market_feature": [np.nan, 1.0, 2.0],
            }
        )
        news = pd.DataFrame(
            {
                "date": [pd.Timestamp("2025-01-06")],
                "news_count": [2.0],
                "body_sentiment_score": [0.4],
                "body_emb_0": [1.0],
            }
        )

        merged = _merge_daily_news_table(market, news)

        self.assertEqual(merged["news_count"].tolist(), [0.0, 2.0, 0.0])
        self.assertEqual(merged["news_count_lag1"].tolist(), [0.0, 2.0, 0.0])
        self.assertTrue(np.isnan(merged["market_feature"].iloc[0]))

    def test_t1_news_features_exclude_quantity_inputs(self) -> None:
        frame = pd.DataFrame(
            {
                "news_count": [0.0, 2.0, 0.0],
                "title_sentiment_score": [0.0, -0.4, 0.0],
                "body_sentiment_score": [0.0, -0.6, 0.0],
                "negative_news_ratio": [0.0, 1.0, 0.0],
                "positive_news_ratio": [0.0, 0.0, 0.0],
                "category_FOMC": [0.0, 1.0, 0.0],
            }
        )

        featured, columns = build_t1_event_news_features(frame)

        self.assertEqual(columns, T1_EVENT_NEWS_FEATURE_COLUMNS)
        self.assertTrue(set(columns).isdisjoint(NEWS_QUANTITY_FEATURE_COLUMNS))
        self.assertIn("body_sentiment_score", columns)
        self.assertIn("negative_news_ratio", columns)
        self.assertEqual(featured["body_sentiment_intensity_1d"].iloc[1], 0.6)
        self.assertEqual(featured["fomc_sentiment_intensity_1d"].iloc[1], 0.6)


if __name__ == "__main__":
    unittest.main()
