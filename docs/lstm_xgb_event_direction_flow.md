# LSTM Event + XGBoost Direction Flow

This branch's final QQQ research flow is:

```text
LSTM event gate
→ XGBoost market-long event-only direction
→ Normal / High Confidence / Strong alert tiers
```

## Run

```powershell
python shared/run_lstm_xgb_event_direction.py --target-ticker QQQ --horizon 2 --skip-lstm-component-training
```

Remove `--skip-lstm-component-training` when the LSTM component predictions need to be regenerated.

## Model roles

| Stage | Model | Input | Target |
|---|---|---|---|
| Event alert | LSTM | market + news-quality event features | whether QQQ has a large T+2 move |
| Direction | XGBoost classifier | market-only long feature frame | up/down direction on historical large-move rows |

The XGBoost direction model is trained quarter-by-quarter. For each test quarter, it uses only past rows whose T+2 target date is before that quarter starts. It then keeps only historical large-move rows:

```text
abs(T+2 log return) >= 2.0%
```

This is the event-only direction training rule.

## Default alert tiers

| Tier | Deep drawdown | Middle drawdown | Shallow drawdown | Purpose |
|---|---:|---:|---:|---|
| Normal | 0.50 | 0.65 | 0.70 | catch more events / higher recall |
| High Confidence | 0.60 | 0.80 | 0.80 | fewer but cleaner alerts |
| Strong | 0.75 | 0.80 | 0.80 | strongest warning label |

Direction threshold is `XGB_Direction_Score >= 0.50`.

## Current QQQ T+2 OOS result

| Tier | Warnings | Event precision | Event recall | Event + direction correct | Direction accuracy on true warnings |
|---|---:|---:|---:|---:|---:|
| Normal | 398 | 52.0% | 67.9% | 118 | 57.0% |
| High Confidence | 276 | 57.2% | 51.8% | 94 | 59.5% |
| Strong | 160 | 61.9% | 32.5% | 65 | 65.7% |

## Output files

Default output folder:

```text
data/training/qqq/event_direction_combo_h2_lstm_event_xgb_market_long_event_only_direction/
```

Main files:

| File | Meaning |
|---|---|
| `combined_oos_predictions.csv` | final OOS predictions with all alert tiers |
| `alert_summary.csv` | pooled summary for Normal / High Confidence / Strong |
| `quarterly_breakdown.csv` | quarter-level stability |
| `yearly_breakdown.csv` | year-level stability |
| `xgb_direction_oos_predictions.csv` | quarter walk-forward XGBoost direction predictions |
| `metadata.json` | model-flow settings and leakage rule |

## Interpretation

Raising thresholds improves precision but lowers recall.

```text
Lower threshold  → more alerts, higher recall, more false positives
Higher threshold → fewer alerts, higher precision, more missed events
```

So the recommended product behavior is not one universal alert, but tiered alerts:

```text
Normal Alert          = broad risk warning
High Confidence Alert = cleaner warning
Strong Alert          = strongest but sparse warning
```
