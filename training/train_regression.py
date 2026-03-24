import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# =========================================================
# 1. 데이터 다운로드
# =========================================================
tickers = ["QQQ", "SPY", "^VIX", "TLT"] 
#QQQ: 나스닥 100 지수, SPY: S&P 500 지수, ^VIX: 변동성 지수, TLT: 장기 국채 ETF

raw = yf.download(
    tickers=tickers,
    start="2010-01-01",
    end="2026-01-01",
    auto_adjust=True, #주가 조정 (배당금, 액면분할 등) 반영된 가격 사용/ if, False면 조정 안된 가격 사용
    progress=False
)
#여러 자산의 가격을 하나의 테이블로 다운로드. 멀티인덱스 형태로 (필드, 티커) 구조로 저장됨.

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
#ret = (오늘 가격 - 과거 가격) / 과거 가격. ret_5는 5일 전과 비교한 수익률. 미래 수익률 예측이 목표이므로 과거 수익률을 feature로 사용.

daily_ret = price.pct_change()
#daily_ret는 일간 수익률. 변동성 계산에 사용됨.

df["vol_5"] = daily_ret.rolling(5).std()
df["vol_10"] = daily_ret.rolling(10).std()
df["vol_20"] = daily_ret.rolling(20).std()
#vol = 수익률의 표준편차로 계산된 변동성. vol_5는 최근 5일간의 일간 수익률의 표준편차. 미래 수익률 예측에 변동성이 중요한 역할을 할 수 있기 때문에 feature로 사용.

#추가 feature 생성: 이동평균과의 비율, 장기 추세, 거래량 변화, 캔들 특징, RSI/MACD/Bollinger, SPY/VIX/TLT 관련 feature 등

# =========================================================
# 4. 장기 추세 feature (평평해지는 문제 해결 핵심)
# =========================================================
for w in [5, 10, 20, 60, 120, 200]:
    ma = price.rolling(w).mean()
    df[f"price_to_ma_{w}"] = price / ma - 1.0
#price_to_ma_20는 현재 가격이 20일 이동평균보다 몇 % 높은지 나타냄. 이동평균과의 비율은 장기 추세를 반영하는 중요한 feature가 될 수 있음.

for w in [20, 60, 120, 200]:
    df[f"slope_{w}"] = np.log(price / price.shift(w)) / w
#slope_20은 20일 전 가격과 비교한 로그 수익률을 20으로 나눈 값. 장기 추세의 기울기를 나타냄. 평평해지는 문제를 해결하는 데 중요한 역할을 할 수 있음.

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
#rsi_14는 14일 RSI. RSI는 최근 상승폭과 하락폭을 비교하여 과매수/과매도 상태를 나타내는 지표로, 미래 수익률 예측에 유용한 feature가 될 수 있음.
#과열상태 감지

ema12 = price.ewm(span=12, adjust=False).mean()
ema26 = price.ewm(span=26, adjust=False).mean()
df["macd"] = ema12 - ema26
df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
df["macd_hist"] = df["macd"] - df["macd_signal"]
#macd는 12일 EMA와 26일 EMA의 차이. macd_signal은 macd의 9일 EMA. macd_hist는 macd와 macd_signal의 차이. MACD는 모멘텀 지표로, 추세의 강도와 방향을 나타내며 미래 수익률 예측에 도움이 될 수 있음.
# 단기추세 vs 장기추세 비교

bb_mid = price.rolling(20).mean()
bb_std = price.rolling(20).std()
bb_upper = bb_mid + 2 * bb_std
bb_lower = bb_mid - 2 * bb_std

df["bb_width"] = (bb_upper - bb_lower) / bb_mid
df["bb_pos"] = (price - bb_lower) / (bb_upper - bb_lower)
# 가격의 “정상 범위”

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

# 상대강도
df["qqq_spy_ratio"] = price / spy
df["qqq_spy_ratio_20"] = df["qqq_spy_ratio"] / df["qqq_spy_ratio"].rolling(20).mean() - 1.0

df["qqq_tlt_ratio"] = price / tlt
df["qqq_tlt_ratio_20"] = df["qqq_tlt_ratio"] / df["qqq_tlt_ratio"].rolling(20).mean() - 1.0

# =========================================================
# 8. 타깃: 2일 뒤 로그수익률
# =========================================================
horizon = 25 #타겟 날짜
df["target_logret_2d"] = np.log(price.shift(-horizon) / price)
df["target_future_price"] = price.shift(-horizon)
df["target_date"] = df["Date"].shift(-horizon)

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
y = df["target_logret_2d"]

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
# 11. walk-forward CV로 n_estimators 선택
# =========================================================
param_grid = [
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.01},
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.01},
    
]
#n_estimators는 트리의 개수, max_depth는 각 트리의 최대 깊이, learning_rate는 각 트리가 예측에 기여하는 정도를 조절하는 하이퍼파라미터. 이들을 조합하여 모델의 복잡도와 학습 속도를 조절할 수 있음. walk-forward CV로 최적의 조합을 찾음.

tscv = TimeSeriesSplit(n_splits=4)
#TimeSeriesSplit는 시계열 데이터를 순차적으로 분할하여 훈련과 검증을 수행하는 방법. 과거 데이터로 모델을 훈련시키고, 이후의 데이터로 검증하는 방식으로 시계열 데이터의 시간적 특성을 고려한 교차 검증을 가능하게 함.

best_cfg = None
best_score = np.inf

for cfg in param_grid:
    fold_scores = []

    for tr_idx, va_idx in tscv.split(X_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=5,
            gamma=0.2,
            reg_alpha=0.1,
            reg_lambda=1.5,
            **cfg
        )

        model.fit(X_tr, y_tr) #훈련 세트로 모델 학습
        pred_va = model.predict(X_va) #검증 세트에 대한 예측값 생성
        rmse = np.sqrt(mean_squared_error(y_va, pred_va)) #검증 세트의 실제값과 예측값 간의 RMSE 계산
        fold_scores.append(rmse) 

    avg_rmse = np.mean(fold_scores)

    if avg_rmse < best_score:
        best_score = avg_rmse
        best_cfg = cfg

print("Best CV config:", best_cfg)
print("Best CV RMSE(logret):", round(best_score, 6))

# =========================================================
# 12. 최종 학습
# =========================================================
model = XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    gamma=0.2,
    reg_alpha=0.1,
    reg_lambda=1.5,
    **best_cfg
)

model.fit(X_train, y_train)

# =========================================================
# 13. 예측
# =========================================================
pred_logret = model.predict(X_test)

# 로그수익률 -> 미래 가격 복원
pred_future_price = test_current_price * np.exp(pred_logret)

# =========================================================
# 14. 평가
# =========================================================
mae = mean_absolute_error(test_future_price, pred_future_price) #실제 미래 가격과 예측된 미래 가격 간의 MAE 계산. MAE는 예측값과 실제값 간의 절대 오차의 평균으로, 예측의 정확도를 평가하는 지표로 사용됨.
rmse = np.sqrt(mean_squared_error(test_future_price, pred_future_price)) #실제 미래 가격과 예측된 미래 가격 간의 RMSE 계산. RMSE는 예측값과 실제값 간의 제곱 오차의 평균의 제곱근으로, MAE보다 큰 오차에 더 큰 패널티를 주는 지표로 사용됨.
r2 = r2_score(test_future_price, pred_future_price) #실제 미래 가격과 예측된 미래 가격 간의 R² 계산. R²는 모델이 실제값의 분산을 얼마나 설명하는지를 나타내는 지표로, 1에 가까울수록 모델의 설명력이 높음을 의미함.

actual_dir = test_future_price > test_current_price
pred_dir = pred_future_price > test_current_price
direction_acc = (actual_dir == pred_dir).mean() #예측된 미래 가격이 현재 가격보다 높은지 여부와 실제 미래 가격이 현재 가격보다 높은지 여부가 일치하는 비율을 계산하여 방향성 예측의 정확도를 평가하는 지표로 사용됨.

mape = np.mean(np.abs((test_future_price - pred_future_price) / test_future_price)) * 100

print("\n===== Regression Metrics =====")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2:", round(r2, 4))
print("MAPE(%):", round(mape, 4))
print("Direction Accuracy:", round(direction_acc, 4))

# baseline: 그냥 현재가격 = 5일 뒤 가격이라고 가정
baseline_pred_price = test_current_price.copy()
baseline_rmse = np.sqrt(mean_squared_error(test_future_price, baseline_pred_price))
baseline_mae = mean_absolute_error(test_future_price, baseline_pred_price)
baseline_mape = np.mean(np.abs((test_future_price - baseline_pred_price) / test_future_price)) * 100

print("\n===== Baseline (future price = current price) =====")
print("Baseline MAE:", round(baseline_mae, 4))
print("Baseline RMSE:", round(baseline_rmse, 4))
print("Baseline MAPE(%):", round(baseline_mape, 4))

# =========================================================
# 15. 중요도
# =========================================================
#plt.figure(figsize=(10, 8))
#plot_importance(model, max_num_features=25, importance_type="gain")
#plt.title("Feature Importance (QQQ 5-day log-return regression)")
#plt.tight_layout()
#plt.show()

# =========================================================
# 16. 결과 테이블
# =========================================================
result_df = pd.DataFrame({
    "Current_Date": test_current_date,
    "Target_Date": test_target_date,
    "Current_Price": test_current_price,
    "Actual_Future_Price": test_future_price,
    "Pred_Future_Price": pred_future_price,
    "Pred_LogRet": pred_logret,
})

print("\nPrediction Sample:")
print(result_df.head(10))

"""
# =========================================================
# 17. 그래프 1: 미래 실제가격 vs 미래 예측가격
# =========================================================
plt.figure(figsize=(14, 7))
plt.plot(result_df["Target_Date"], result_df["Actual_Future_Price"], label="Actual Future Price")
plt.plot(result_df["Target_Date"], result_df["Pred_Future_Price"], label="Predicted Future Price")
plt.title("QQQ: Actual vs Predicted Future Price (5-day ahead)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# 18. 그래프 2: 마지막 150개만 확대
# =========================================================
zoom_df = result_df.tail(150).copy()

plt.figure(figsize=(14, 7))
plt.plot(zoom_df["Target_Date"], zoom_df["Actual_Future_Price"], label="Actual Future Price")
plt.plot(zoom_df["Target_Date"], zoom_df["Pred_Future_Price"], label="Predicted Future Price")
plt.title("QQQ: Actual vs Predicted Future Price (Zoom)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""