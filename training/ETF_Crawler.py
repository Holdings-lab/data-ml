import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================================================
# 1. 데이터 다운로드
#    yfinance는 여러 티커를 한 번에 download 가능
# =========================================================
tickers = ["QQQ", "SPY", "^VIX"]

raw = yf.download(
    tickers=tickers,
    start="2010-01-01",
    end="2026-01-01",
    auto_adjust=False,
    progress=False
)

# raw.columns 예시:
# MultiIndex([('Adj Close','QQQ'), ('Adj Close','SPY'), ('Adj Close','^VIX'), ...])

# ---------------------------------------------------------
# 안전하게 특정 티커 / 특정 컬럼을 Series로 꺼내는 함수
# ---------------------------------------------------------
def get_series(data: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    s = data[field][ticker].copy()
    s.name = f"{ticker}_{field}"
    return s

# 가격/거래량 Series
qqq_price = get_series(raw, "Adj Close", "QQQ") if ("Adj Close" in raw.columns.get_level_values(0)) else get_series(raw, "Close", "QQQ")
spy_price = get_series(raw, "Adj Close", "SPY") if ("Adj Close" in raw.columns.get_level_values(0)) else get_series(raw, "Close", "SPY")
vix_price = get_series(raw, "Adj Close", "^VIX") if ("Adj Close" in raw.columns.get_level_values(0)) else get_series(raw, "Close", "^VIX")

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
df["QQQ_open"] = qqq_open
df["QQQ_high"] = qqq_high
df["QQQ_low"] = qqq_low
df["QQQ_volume"] = qqq_volume

df = df.reset_index()  # Date 컬럼 생성

price = df["QQQ_price"]
spy = df["SPY_price"]
vix = df["VIX_price"]
open_price = df["QQQ_open"]
high_price = df["QQQ_high"]
low_price = df["QQQ_low"]
volume = df["QQQ_volume"]

# =========================================================
# 3. QQQ 기존 feature
# =========================================================
# 수익률 (%)
df["return_1d"] = price.pct_change(1) * 100
df["return_3d"] = price.pct_change(3) * 100
df["return_5d"] = price.pct_change(5) * 100
df["return_10d"] = price.pct_change(10) * 100
df["return_20d"] = price.pct_change(20) * 100

# 변동성
daily_ret = price.pct_change() * 100
df["volatility_5d"] = daily_ret.rolling(5).std()
df["volatility_10d"] = daily_ret.rolling(10).std()
df["volatility_20d"] = daily_ret.rolling(20).std()

# 이동평균
df["ma_5"] = price.rolling(5).mean()
df["ma_10"] = price.rolling(10).mean()
df["ma_20"] = price.rolling(20).mean()

# 이동평균 비율 (%)
df["ma_ratio_5"] = (price / df["ma_5"] - 1) * 100
df["ma_ratio_10"] = (price / df["ma_10"] - 1) * 100
df["ma_ratio_20"] = (price / df["ma_20"] - 1) * 100

# 거래량
df["vol_change"] = volume.pct_change() * 100
df["vol_ma_5"] = volume.rolling(5).mean()
df["vol_ma_ratio_5"] = (volume / df["vol_ma_5"] - 1) * 100

# RSI
delta = price.diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df["rsi_14"] = 100 - (100 / (1 + rs))

# MACD
ema_12 = price.ewm(span=12, adjust=False).mean()
ema_26 = price.ewm(span=26, adjust=False).mean()
df["macd"] = ema_12 - ema_26
df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
df["macd_hist"] = df["macd"] - df["macd_signal"]

# 추가 feature
df["price_std_20"] = price.rolling(20).std()

bb_mid = price.rolling(20).mean()
bb_std = price.rolling(20).std()
bb_upper = bb_mid + 2 * bb_std
bb_lower = bb_mid - 2 * bb_std

df["bb_upper_dist"] = (price / bb_upper - 1) * 100
df["bb_lower_dist"] = (price / bb_lower - 1) * 100
df["bb_width"] = ((bb_upper - bb_lower) / bb_mid) * 100
df["ema_ratio_12_26"] = (ema_12 / ema_26 - 1) * 100
df["high_low_range"] = ((high_price - low_price) / price) * 100
df["close_open_change"] = ((price - open_price) / open_price) * 100

# =========================================================
# 4. SPY feature 추가
# =========================================================
spy_ret = spy.pct_change() * 100
df["spy_return_1d"] = spy.pct_change(1) * 100
df["spy_return_5d"] = spy.pct_change(5) * 100
df["spy_return_20d"] = spy.pct_change(20) * 100
df["spy_volatility_5d"] = spy_ret.rolling(5).std()
df["spy_ma_20"] = spy.rolling(20).mean()
df["spy_ma_ratio_20"] = (spy / df["spy_ma_20"] - 1) * 100

# =========================================================
# 5. VIX feature 추가
# =========================================================
df["vix_level"] = vix
df["vix_change_1d"] = vix.pct_change(1) * 100
df["vix_change_5d"] = vix.pct_change(5) * 100
df["vix_ma_10"] = vix.rolling(10).mean()
df["vix_ma_ratio_10"] = (vix / df["vix_ma_10"] - 1) * 100

# =========================================================
# 6. 타겟: QQQ 5일 뒤 상승/하락
# =========================================================
df["target_5d_updown"] = (price.shift(-5) > price).astype(int)

# =========================================================
# 7. 학습용 feature 목록
# =========================================================
feature_cols = [
    # QQQ feature
    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "ma_ratio_5", "ma_ratio_10", "ma_ratio_20",
    "vol_change", "vol_ma_ratio_5",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "price_std_20", "bb_upper_dist", "bb_lower_dist", "bb_width",
    "ema_ratio_12_26", "high_low_range", "close_open_change",

    # SPY feature
    "spy_return_1d", "spy_return_5d", "spy_return_20d",
    "spy_volatility_5d", "spy_ma_ratio_20",

    # VIX feature
    "vix_level", "vix_change_1d", "vix_change_5d", "vix_ma_ratio_10"
]

df = df.dropna().copy()

X = df[feature_cols]
y = df["target_5d_updown"]

print("Target distribution:")
print(y.value_counts(normalize=True))

# =========================================================
# 8. 시계열 분리
# =========================================================
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

# =========================================================
# 9. XGBoost 분류
# =========================================================
model = XGBClassifier(
    n_estimators=700,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="logloss",
    tree_method="hist",   # CPU
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# 10. 평가
# =========================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:", acc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================================================
# 11. 중요도
# =========================================================
plt.figure(figsize=(10, 8))
plot_importance(model, max_num_features=20, importance_type="gain")
plt.title("Feature Importance (QQQ + SPY + VIX)")
plt.tight_layout()
plt.show()

# =========================================================
# 12. 예측 결과 일부
# =========================================================
result_df = pd.DataFrame({
    "Date": df.iloc[split:]["Date"].values,
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Prob_Up": y_prob
})

print("\nPrediction Sample:")
print(result_df.head(50))