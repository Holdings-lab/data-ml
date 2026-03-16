import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import TimeSeriesSplit

# =========================================================
# 1. 데이터 다운로드
# =========================================================
tickers = ["QQQ", "SPY", "^VIX", "TLT"]

raw = yf.download(
    tickers=tickers,
    start="2010-01-01",
    end="2026-01-01",
    auto_adjust=False,
    progress=False
)

def get_series(data: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    s = data[field][ticker].copy()
    s.name = f"{ticker}_{field}"
    return s

price_field = "Adj Close" if ("Adj Close" in raw.columns.get_level_values(0)) else "Close"

qqq_price = get_series(raw, price_field, "QQQ")
spy_price = get_series(raw, price_field, "SPY")
vix_price = get_series(raw, price_field, "^VIX")
tlt_price = get_series(raw, price_field, "TLT")

qqq_open = get_series(raw, "Open", "QQQ")
qqq_high = get_series(raw, "High", "QQQ")
qqq_low = get_series(raw, "Low", "QQQ")
qqq_volume = get_series(raw, "Volume", "QQQ")

# =========================================================
# 2. 단일 DataFrame으로 정리
# =========================================================
df = pd.DataFrame(index=raw.index)
df["QQQ_price"] = qqq_price
df["SPY_price"] = spy_price
df["VIX_price"] = vix_price
df["TLT_price"] = tlt_price
df["QQQ_open"] = qqq_open
df["QQQ_high"] = qqq_high
df["QQQ_low"] = qqq_low
df["QQQ_volume"] = qqq_volume

df = df.reset_index()

price = df["QQQ_price"]
spy = df["SPY_price"]
vix = df["VIX_price"]
tlt = df["TLT_price"]
open_price = df["QQQ_open"]
high_price = df["QQQ_high"]
low_price = df["QQQ_low"]
volume = df["QQQ_volume"]

# =========================================================
# 3. 기본 수익률 / 변동성
# =========================================================
df["ret_1"] = price.pct_change(1)
df["ret_3"] = price.pct_change(3)
df["ret_5"] = price.pct_change(5)
df["ret_10"] = price.pct_change(10)
df["ret_20"] = price.pct_change(20)

daily_ret = price.pct_change()
df["vol_5"] = daily_ret.rolling(5).std()
df["vol_10"] = daily_ret.rolling(10).std()
df["vol_20"] = daily_ret.rolling(20).std()

# =========================================================
# 4. 장기 추세 feature
# =========================================================
for w in [5, 10, 20, 60, 120, 200]:
    ma = price.rolling(w).mean()
    df[f"price_to_ma_{w}"] = price / ma - 1.0

for w in [20, 60, 120, 200]:
    df[f"slope_{w}"] = np.log(price / price.shift(w)) / w

# =========================================================
# 5. 거래량 / 캔들 feature
# =========================================================
df["vol_chg_1"] = volume.pct_change(1)
df["vol_to_ma20"] = volume / volume.rolling(20).mean() - 1.0

df["intraday_range"] = (high_price - low_price) / price
df["close_open"] = (price - open_price) / open_price

# =========================================================
# 6. RSI / MACD / Bollinger
# =========================================================
delta = price.diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df["rsi_14"] = 100 - (100 / (1 + rs))

ema12 = price.ewm(span=12, adjust=False).mean()
ema26 = price.ewm(span=26, adjust=False).mean()
df["macd"] = ema12 - ema26
df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
df["macd_hist"] = df["macd"] - df["macd_signal"]

bb_mid = price.rolling(20).mean()
bb_std = price.rolling(20).std()
bb_upper = bb_mid + 2 * bb_std
bb_lower = bb_mid - 2 * bb_std

df["bb_width"] = (bb_upper - bb_lower) / bb_mid
df["bb_pos"] = (price - bb_lower) / (bb_upper - bb_lower)

# =========================================================
# 7. SPY / VIX / TLT feature
# =========================================================
df["spy_ret_1"] = spy.pct_change(1)
df["spy_ret_5"] = spy.pct_change(5)
df["spy_ret_20"] = spy.pct_change(20)
df["spy_to_ma_60"] = spy / spy.rolling(60).mean() - 1.0

df["vix_level"] = vix
df["vix_ret_1"] = vix.pct_change(1)
df["vix_ret_5"] = vix.pct_change(5)
df["vix_to_ma_20"] = vix / vix.rolling(20).mean() - 1.0

df["tlt_ret_1"] = tlt.pct_change(1)
df["tlt_ret_5"] = tlt.pct_change(5)
df["tlt_ret_20"] = tlt.pct_change(20)
df["tlt_to_ma_60"] = tlt / tlt.rolling(60).mean() - 1.0

df["qqq_spy_ratio"] = price / spy
df["qqq_spy_ratio_20"] = df["qqq_spy_ratio"] / df["qqq_spy_ratio"].rolling(20).mean() - 1.0

df["qqq_tlt_ratio"] = price / tlt
df["qqq_tlt_ratio_20"] = df["qqq_tlt_ratio"] / df["qqq_tlt_ratio"].rolling(20).mean() - 1.0

# =========================================================
# 8. 분류 타깃: 5일 뒤 상승/하락
# =========================================================
horizon = 5
df["target_future_price"] = price.shift(-horizon)
df["target_date"] = df["Date"].shift(-horizon)
df["target_up_5d"] = (df["target_future_price"] > price).astype(int)

# =========================================================
# 9. feature 선택
# =========================================================
feature_cols = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "vol_5", "vol_10", "vol_20",

    "price_to_ma_5", "price_to_ma_10", "price_to_ma_20",
    "price_to_ma_60", "price_to_ma_120", "price_to_ma_200",

    "slope_20", "slope_60", "slope_120", "slope_200",

    "vol_chg_1", "vol_to_ma20",
    "intraday_range", "close_open",

    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_pos",

    "spy_ret_1", "spy_ret_5", "spy_ret_20", "spy_to_ma_60",
    "vix_level", "vix_ret_1", "vix_ret_5", "vix_to_ma_20",
    "tlt_ret_1", "tlt_ret_5", "tlt_ret_20", "tlt_to_ma_60",

    "qqq_spy_ratio_20",
    "qqq_tlt_ratio_20",
]

df = df.dropna().copy()

X = df[feature_cols]
y = df["target_up_5d"]

print("Target distribution:")
print(y.value_counts(normalize=True))

# =========================================================
# 10. train / test 분리
# =========================================================
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

test_current_price = df.iloc[split:]["QQQ_price"].values
test_future_price = df.iloc[split:]["target_future_price"].values
test_target_date = pd.to_datetime(df.iloc[split:]["target_date"].values)
test_current_date = pd.to_datetime(df.iloc[split:]["Date"].values)

# =========================================================
# 11. walk-forward CV로 설정 선택
# =========================================================
param_grid = [
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03},
    {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.03},
    {"n_estimators": 700, "max_depth": 4, "learning_rate": 0.02},
    {"n_estimators": 900, "max_depth": 4, "learning_rate": 0.02},
]

tscv = TimeSeriesSplit(n_splits=4)

best_cfg = None
best_score = -np.inf

for cfg in param_grid:
    fold_scores = []

    for tr_idx, va_idx in tscv.split(X_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        scale_pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=5,
            gamma=0.2,
            reg_alpha=0.1,
            reg_lambda=1.5,
            scale_pos_weight=scale_pos_weight,
            **cfg
        )

        model.fit(X_tr, y_tr)
        pred_va = model.predict(X_va)
        acc = accuracy_score(y_va, pred_va)
        fold_scores.append(acc)

    avg_acc = np.mean(fold_scores)

    if avg_acc > best_score:
        best_score = avg_acc
        best_cfg = cfg

print("Best CV config:", best_cfg)
print("Best CV Accuracy:", round(best_score, 6))

# =========================================================
# 12. 최종 학습
# =========================================================
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    gamma=0.2,
    reg_alpha=0.1,
    reg_lambda=1.5,
    scale_pos_weight=scale_pos_weight,
    **best_cfg
)

model.fit(X_train, y_train)

# =========================================================
# 13. 예측
# =========================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 0.5 기준 상승/하락
pred_dir = y_pred
actual_dir = y_test.values

# 참고용: 확률을 가격 방향으로 해석
pred_future_price_direction = np.where(y_prob >= 0.5, "UP", "DOWN")
actual_future_price_direction = np.where(actual_dir == 1, "UP", "DOWN")

# =========================================================
# 14. 평가
# =========================================================
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n===== Classification Metrics =====")
print("Accuracy:", round(acc, 4))
print("ROC-AUC:", round(auc, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# baseline: 전부 상승 예측
baseline_pred = np.ones_like(y_test.values)
baseline_acc = accuracy_score(y_test, baseline_pred)

print("\n===== Baseline (always UP) =====")
print("Baseline Accuracy:", round(baseline_acc, 4))

# =========================================================
# 15. 중요도
# =========================================================
score = model.get_booster().get_score(importance_type="gain")
if len(score) > 0:
    plt.figure(figsize=(10, 8))
    plot_importance(model, max_num_features=25, importance_type="gain")
    plt.title("Feature Importance (QQQ 5-day up/down classification)")
    plt.tight_layout()
    plt.show()
else:
    print("\nFeature importance를 그릴 split이 없습니다.")

# =========================================================
# 16. 결과 테이블
# =========================================================
result_df = pd.DataFrame({
    "Current_Date": test_current_date,
    "Target_Date": test_target_date,
    "Current_Price": test_current_price,
    "Actual_Future_Price": test_future_price,
    "Actual_UpDown": actual_future_price_direction,
    "Pred_UpDown": pred_future_price_direction,
    "Prob_Up": y_prob
})

print("\nPrediction Sample:")
print(result_df.head(10))

# =========================================================
# 17. 그래프 1: 상승 확률
# =========================================================
plt.figure(figsize=(14, 7))
plt.plot(result_df["Current_Date"], result_df["Prob_Up"], label="Predicted Probability of Up")
plt.axhline(0.5, linestyle="--", linewidth=1, label="Threshold 0.5")
plt.title("QQQ: Predicted Probability of Going Up in 5 Days")
plt.xlabel("Prediction Date")
plt.ylabel("Probability")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# 18. 그래프 2: 실제 가격 + 상승확률(보조축)
# =========================================================
fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.plot(result_df["Target_Date"], result_df["Actual_Future_Price"], label="Actual Future Price")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")

ax2 = ax1.twinx()
ax2.plot(result_df["Current_Date"], result_df["Prob_Up"], linestyle="--", label="Prob Up")
ax2.set_ylabel("Probability of Up")

ax1.set_title("QQQ Future Price and Predicted Up Probability")
ax1.grid(True, alpha=0.3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.tight_layout()
plt.show()