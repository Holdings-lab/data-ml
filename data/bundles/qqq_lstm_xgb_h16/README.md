# QQQ LSTM + XGBoost T+16 model bundle

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

bundle = load_bundle("data/bundles/qqq_lstm_xgb_h16")
result = predict_from_feature_frames(bundle, news_event_frame, market_long_frame)
```
