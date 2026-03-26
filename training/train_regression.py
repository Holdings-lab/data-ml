"""
1. threshold 추가
2. 거시경제 HYG, UUP 추가
3. 

전체적인 흐름을 보고 급변하는 시기를 맞춰서 기간을 정해야함.
2000 ~ 2020 주가가 많이 오르지 않았기 때문에 2020~ 2025 퀀텀점프 예측은 힘들다
target ==> 2015 시작이 적정
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import seaborn as sns

from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from optuna.samplers import TPESampler

import random
import os

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # 만약 다른 라이브러리(torch 등)를 쓴다면 거기 시드도 고정
    
seed_everything(42)

# =========================================================
# 1. 데이터 다운로드
# =========================================================
target = "QQQ"
tickers = [target, "SPY", "^VIX", "TLT", "HYG", "UUP"] 
#target: 나스닥 100 지수, SPY: S&P 500 지수, ^VIX: 변동성 지수, TLT: 장기 국채 ETF

raw = yf.download(
    tickers=tickers,
    start="2015-01-01",
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

target_price = get_series(raw, price_field, target)
spy_price = get_series(raw, price_field, "SPY")
vix_price = get_series(raw, price_field, "^VIX")
tlt_price = get_series(raw, price_field, "TLT")
hyg_price = get_series(raw, price_field, "HYG")
uup_price = get_series(raw, price_field, "UUP")

target_open = get_series(raw, "Open", target)
target_high = get_series(raw, "High", target)
target_low = get_series(raw, "Low", target)
target_volume = get_series(raw, "Volume", target)

# =========================================================
# 2. 단일 DataFrame으로 정리
# =========================================================
df = pd.DataFrame(index=raw.index)
df["target_price"] = target_price
df["SPY_price"] = spy_price
df["VIX_price"] = vix_price
df["TLT_price"] = tlt_price
df["target_open"] = target_open
df["target_high"] = target_high
df["target_low"] = target_low
df["target_volume"] = target_volume
df["HYG_price"] = hyg_price
df["UUP_price"] = uup_price

df = df.reset_index()

price = df["target_price"]
spy = df["SPY_price"]
vix = df["VIX_price"]
tlt = df["TLT_price"]
hyg = df["HYG_price"]
uup = df["UUP_price"]

open_price = df["target_open"]
high_price = df["target_high"]
low_price = df["target_low"]
volume = df["target_volume"]

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
df["target_spy_ratio"] = price / spy
df["target_spy_ratio_20"] = df["target_spy_ratio"] / df["target_spy_ratio"].rolling(20).mean() - 1.0

df["target_tlt_ratio"] = price / tlt
df["target_tlt_ratio_20"] = df["target_tlt_ratio"] / df["target_tlt_ratio"].rolling(20).mean() - 1.0

df["hyg_ret"] = hyg.pct_change(10)
df["uup_ret"] = uup.pct_change(10)

rolling_max = price.rolling(window=252, min_periods=1).max()
df["drawdown"] = (price - rolling_max) / rolling_max

df["target_spy_rel_ret"] = (df["target_price"] / df["SPY_price"]).pct_change(10)
df["target_tlt_rel_ret"] = (df["target_price"] / df["TLT_price"]).pct_change(10)

def get_z_score(series, window=20):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

df["hyg_z_score"] = get_z_score(df["HYG_price"], 20)
df["uup_z_score"] = get_z_score(df["UUP_price"], 20)
df["vix_z_score"] = get_z_score(df["VIX_price"], 20)

# 3. 변동성의 역전 현상 (target 변동성 / SPY 변동성)
# 나스닥이 지수보다 유난히 요동치면 위험 신호입니다.
df["vol_ratio"] = df["target_price"].pct_change().rolling(10).std() / \
                  (df["SPY_price"].pct_change().rolling(10).std() + 1e-9)

df["ret_accel"] = df["ret_5"].diff()
df["vix_speed"] = df["VIX_price"].pct_change(3)
df["dist_10"] = price / price.rolling(10).mean() - 1.0
# 최근 5일간의 변동성 대비 오늘 움직임의 강도
df["vol_breakout"] = df["ret_1"] / (df["vol_5"] + 1e-9)

# 볼린저 밴드 상단 돌파 여부
df["bb_high_dist"] = (df["target_high"] - bb_upper) / (bb_upper + 1e-9)

# =========================================================
# 9. feature 선택
# =========================================================
feature_cols = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "vol_5", "vol_10", "vol_20",

    "price_to_ma_5", "price_to_ma_10", "price_to_ma_20",
    "price_to_ma_60", "price_to_ma_120", 

    "slope_20", "slope_60", "slope_120", 

    "vol_chg_1", "vol_to_ma20",
    "intraday_range", "close_open",

    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_pos",

    "spy_ret_1", "spy_ret_5", "spy_ret_20", "spy_to_ma_60",
    "vix_level", "vix_ret_1", "vix_ret_5", "vix_to_ma_20",
    "tlt_ret_1", "tlt_ret_5", "tlt_ret_20", "tlt_to_ma_60",

    "target_spy_ratio_20",
    "target_tlt_ratio_20",

    "hyg_ret", "uup_ret", "drawdown", "target_spy_rel_ret", "target_tlt_rel_ret", "hyg_z_score", "uup_z_score", 
    "vix_z_score", "vol_ratio","ret_accel", "vix_speed", "dist_10", "vol_breakout", "bb_high_dist"
]

"""
# [디버깅] dropna 하기 직전에 실행하세요
print("\n" + "="*50)
print("🔎 결측치(NaN) 점검 리포트")
print(f"전체 행 개수: {len(df)}")
print("="*50)

# 1. 컬럼별 NaN 개수 확인 (NaN이 하나라도 있는 것들만)
nan_counts = df.isnull().sum()
nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)

if not nan_cols.empty:
    print("\n[NaN 발생 컬럼 목록]")
    for col, count in nan_cols.items():
        print(f"{col}: {count}개 (비율: {count/len(df)*100:.1f}%)")
else:
    print("\n✅ 모든 컬럼이 깨끗합니다 (NaN 없음).")

# 2. 모든 행을 지워버리는 주범(100% NaN) 찾기
deadly_cols = nan_counts[nan_counts >= len(df)].index.tolist()
if deadly_cols:
    print(f"\n❌ 위험: 아래 컬럼들이 모든 데이터를 삭제하고 있습니다:")
    print(deadly_cols)
    print("\n💡 해결책: 위 컬럼의 계산식을 확인하거나, 해당 티커 데이터가 잘 다운로드됐는지 확인하세요.")
else:
    print("\n✅ 데이터 전체를 삭제하는 '독성 컬럼'은 없습니다.")

print("="*50 + "\n")
"""


best_overall_acc = 0
best_horizon = None
best_features = []

print("=== 🔍 [통합 최적화] 최적의 조합 탐색 시작 ===")

for h in range(15,16):
    # 1. 해당 Horizon용 데이터 세팅 (타겟 생성 등)
    df_h = df.copy()
    df_h["target_logret"] = np.log(df_h["target_price"].shift(-h) / df_h["target_price"])
    df_h = df_h.dropna().copy()
    
    # 2. 피처/타겟 분리
    X_full = df_h[feature_cols]
    y_full = df_h["target_logret"]
    split = int(len(df_h) * 0.8)
    X_train_full = X_full.iloc[:split]
    y_train_full = y_full.iloc[:split]

    # -----------------------------------------------------
    # Step A: 이 Horizon에서 가장 중요한 피처 25개 뽑기
    # -----------------------------------------------------
    selector = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    selector.fit(X_train_full, y_train_full)
    
    # 중요도 순으로 상위 25개 피처 추출
    importances = pd.Series(selector.feature_importances_, index=feature_cols)
    current_top_25 = importances.sort_values(ascending=False).head(25).index.tolist()

    # -----------------------------------------------------
    # Step B: 뽑힌 25개 피처로만 교차 검증 (실력 테스트)
    # -----------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=3)
    cv_accs = []
    
    for tr_idx, va_idx in tscv.split(X_train_full):
        tr_idx_purged = tr_idx[:-h] if len(tr_idx) > h else tr_idx
        
        X_tr, X_va = X_train_full[current_top_25].iloc[tr_idx_purged], X_train_full[current_top_25].iloc[va_idx]
        y_tr, y_va = y_train_full.iloc[tr_idx_purged], y_train_full.iloc[va_idx]
        
        m = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.005, random_state=42)
        m.fit(X_tr, y_tr)
        
        # 방향성 정확도 측정
        acc = accuracy_score((y_va > 0).astype(int), (m.predict(X_va) > 0).astype(int))
        cv_accs.append(acc)
    
    avg_acc = np.mean(cv_accs)

    # -----------------------------------------------------
    # Step C: 전체 Horizon 중 최고 성적 업데이트
    # -----------------------------------------------------
    if avg_acc > best_overall_acc:
        best_overall_acc = avg_acc
        best_horizon = h
        best_features = current_top_25 # 최고 성적일 때의 피처 20개를 저장

# =========================================================
# 🏆 최종 결과 발표
# =========================================================
print("\n" + "="*50)
print(f"🎉 최적의 Horizon 발견: {best_horizon}일")
print(f"📈 최고 예측 정확도: {best_overall_acc*100:.2f}%")
print(f"🛠 선택된 정예 피처 30개: \n{best_features}")
print("="*50)

horizon = int(best_horizon) #타겟 날짜
df["target_logret_2d"] = np.log(price.shift(-horizon) / price)
df["target_future_price"] = price.shift(-horizon)
df["target_date"] = df["Date"].shift(-horizon)

df = df.dropna().copy()

X = df[best_features]
y = df["target_logret_2d"]

# =========================================================
# 10. train / test 분리
# =========================================================
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

test_current_price = df.iloc[split:]["target_price"].values
test_future_price = df.iloc[split:]["target_future_price"].values
test_target_date = pd.to_datetime(df.iloc[split:]["target_date"].values)
test_current_date = pd.to_datetime(df.iloc[split:]["Date"].values)

# =========================================================
# 11. walk-forward CV로 n_estimators 선택
# =========================================================
print(f"\n[Optuna] 최적화 시작... (데이터 크기: {len(X_train)})")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8), # 조금 더 깊게 탐색 허용 10일 예측시 (3,8) 20일 예측시 (5,8)
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "random_state": 42,
        "tree_method": "hist",
        "objective": "reg:squarederror"
    }
    
    tscv = TimeSeriesSplit(n_splits=4)
    combined_scores = []
    
    for tr_idx, va_idx in tscv.split(X_train):
        # Purging (미래 데이터 누수 방지)
        tr_idx_purged = tr_idx[:-horizon] if len(tr_idx) > horizon else tr_idx
        
        X_tr, X_va = X_train.iloc[tr_idx_purged], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx_purged], y_train.iloc[va_idx]
        
        va_current_prices = df.iloc[va_idx]["target_price"].values
        
        model = XGBRegressor(**param)
        model.fit(X_tr, y_tr)
        
        # 1. 예측 및 복원
        preds_logret = model.predict(X_va)
        
        # 2. [토끼 1] 방향성 정확도 (Accuracy) 계산
        # 실제 수익률(y_va)과 예측 수익률(preds_logret)의 부호가 같은지 확인
        actual_dir = (y_va.values > 0).astype(int)
        pred_dir = (preds_logret > 0).astype(int)
        acc = (actual_dir == pred_dir).mean()
        
        # 3. [토끼 2] 수치적 오차 (RMSE) 계산 
        # 로그수익률 단위에서의 오차를 측정 (값이 작을수록 그래프가 잘 달라붙음)
        rmse = np.sqrt(mean_squared_error(y_va, preds_logret))
        
        # 4. [핵심] 두 마리 토끼 합치기
        # 정확도(0.7~1.0)는 높을수록 좋고, RMSE(0.01~0.10)는 낮을수록 좋음.
        # 가중치 5는 RMSE의 영향을 정확도와 비슷한 체급으로 맞추기 위한 수치입니다.
        # 이 값을 높이면 그래프가 더 잘 달라붙고, 낮추면 방향성에 더 집중합니다.
        """
        1. 숫자를 낮추면 (예: rmse * 2) → "도박꾼 모델"
            목표: "수치는 좀 틀려도 돼! 오를지 내릴지만 확실히 맞춰!"
            정확도(Acc): 올라갑니다. 모델이 방향을 맞추는 데 모든 수단과 방법을 가동하기 때문입니다.
            그래프: 잘 안 달라붙습니다. 실제 주가는 5% 올랐는데 모델은 "응, 어쨌든 올라(0.1% 상승)"라고 대답해도 점수를 잘 받기 때문에, 그래프가 실제 가격의 굴곡을 무시하고 평평해지거나 자기 마음대로 움직입니다.
        
        2. 숫자를 높이면 (예: rmse * 50) → "겁쟁이/완벽주의자 모델"
            목표: "틀려서 감점당하는 게 제일 무서워! 실제 가격이랑 최대한 비슷하게 대답할래."
            정확도(Acc): 낮아질 가능성이 큽니다. 방향을 맞추는 것보다 실제 가격과의 '거리'를 좁히는 데 에너지를 다 쓰기 때문입니다.
            그래프: 실제 가격에 찰떡처럼 달라붙습니다. 오차를 줄여야 점수를 받으므로, 모델이 실제 주가의 출렁임을 필사적으로 흉내 냅니다."""
        score = acc - (rmse * 3) 
        
        combined_scores.append(score)
    
    return np.mean(combined_scores) # 이 결합 점수를 최대화(maximize)


sampler = optuna.samplers.TPESampler(seed=42)

study = optuna.create_study(
    direction="maximize",
    sampler=sampler
)
study.optimize(objective, n_trials=200)

print(f"Best Params: {study.best_params}")

# =========================================================
# 12. 최종 학습
# =========================================================
best_model = XGBRegressor(**study.best_params)
best_model.fit(X_train, y_train)

# =========================================================
# 13. 예측
# =========================================================
pred_logret = best_model.predict(X_test)

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

print("\n===== 🎯 최적화 후 회귀 모델 성과 =====")
print(f"MAE              : {mae:.4f}")
print(f"RMSE             : {rmse:.4f}")
print(f"R2 Score         : {r2:.4f}")
print(f"Direction Accuracy: {direction_acc * 100:.2f}%")

# baseline: 그냥 현재가격 = 5일 뒤 가격이라고 가정
baseline_pred_price = test_current_price.copy()
baseline_rmse = np.sqrt(mean_squared_error(test_future_price, baseline_pred_price))
baseline_mae = mean_absolute_error(test_future_price, baseline_pred_price)
baseline_mape = np.mean(np.abs((test_future_price - baseline_pred_price) / test_future_price)) * 100

print("\n===== Baseline (future price = current price) =====")
print("Baseline MAE:", round(baseline_mae, 4))
print("Baseline RMSE:", round(baseline_rmse, 4))
print("Baseline MAPE(%):", round(baseline_mape, 4))

# 중요도 확인
from xgboost import plot_importance
plot_importance(best_model, max_num_features=15, importance_type="gain")
plt.show()

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

# 한글 깨짐 방지 (필요시)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 실제 미래 수익률 vs 예측 미래 수익률 (산점도)
# 이 그래프가 대각선 우상향으로 모여있을수록 모델이 천재인 겁니다.
plt.figure(figsize=(8, 8))
sns.regplot(x=y_test, y=pred_logret, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title(f"수익률 예측 정확도 (Actual vs Predicted Returns)\nHorizon: {horizon}일")
plt.xlabel("실제 미래 수익률 (Log Return)")
plt.ylabel("예측 미래 수익률 (Log Return)")
plt.grid(True)
plt.show()

# 2. 누적 수익률 비교 (백테스트 가상 체험)
# 모델이 "상승"한다고 했을 때만 샀다면 어떻게 됐을까?
result_df['Signal'] = (result_df['Pred_LogRet'] > 0).astype(int)
result_df['Daily_Ret'] = result_df['Actual_Future_Price'].pct_change() # 단순 비교용

# 모델 전략 수익률 vs 시장 홀딩 수익률
# 주의: 간단한 시뮬레이션입니다.
strategy_ret = (result_df['Signal'].shift(1) * result_df['Daily_Ret']).fillna(0)
cum_strategy = (1 + strategy_ret).cumprod()
cum_market = (1 + result_df['Daily_Ret'].fillna(0)).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(result_df['Target_Date'], cum_market, label='Market (Hold target)', alpha=0.7)
plt.plot(result_df['Target_Date'], cum_strategy, label='Model Strategy', linewidth=2)
plt.title("모델 신호에 따른 누적 수익률 시뮬레이션")
plt.legend()
plt.grid(True)
plt.show()

# 3. 예측 오차 분포 (Error Distribution)
errors = result_df['Actual_Future_Price'] - result_df['Pred_Future_Price']
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, bins=50)
plt.axvline(0, color='red', linestyle='--')
plt.title("예측 오차 분포 (0에 가까울수록 정확)")
plt.xlabel("Price Error (Actual - Predicted)")
plt.show()

# 1. 테스트할 문턱(Threshold) 후보들 
# (예: 0%, 1%, 2%, 3% 이상 상승 예측 시에만 매수)
thresholds = [0, 0.002, 0.004, 0.006] 

plt.figure(figsize=(14, 8))

# 시장 수익률 (Benchmark: 그냥 사서 들고 있기)
market_ret = result_df['Actual_Future_Price'].pct_change().fillna(0)
cum_market = (1 + market_ret).cumprod()
plt.plot(result_df['Target_Date'], cum_market, label='Market (Hold target)', color='black', linestyle='--', alpha=0.6)

# 각 문턱별 전략 수익률 계산
for th in thresholds:
    # 모델이 th(문턱)보다 큰 수익률을 예측했을 때만 1 (매수), 아니면 0 (현금)
    # e.g. 모델이 "10일 뒤 0.2% 이상 오를 거야"라고 확신할 때만 투자한 것이 훨씬 더 높은 수익
    result_df[f'Signal_{th}'] = (result_df['Pred_LogRet'] > th).astype(int)
    
    # 전략 수익률 = 어제 발생한 신호 * 오늘 시장 수익률
    # (주의: 미래 데이터를 보고 오늘 결정할 수 없으므로 shift(1) 적용)
    strat_ret = result_df[f'Signal_{th}'].shift(1).fillna(0) * market_ret
    cum_strat = (1 + strat_ret).cumprod()
    
    plt.plot(result_df['Target_Date'], cum_strat, label=f'Strategy (Th > {th*100:.1f}%)', linewidth=2)

plt.title(f"예측 확신도(Threshold)에 따른 수익률 비교 (Horizon: {horizon}일)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 최종 성과 요약 출력
print("=== 📊 문턱별 최종 누적 수익률 ===")
print(f"Market (Hold) : {cum_market.iloc[-1]:.2f}")
for th in thresholds:
    final_ret = (1 + (result_df[f'Signal_{th}'].shift(1).fillna(0) * market_ret)).cumprod().iloc[-1]
    print(f"Threshold {th*100:.1f}% : {final_ret:.2f}")

# =========================================================
# 12-1. 모델 및 메타데이터 저장
# =========================================================
# 1) XGBoost 모델을 JSON으로 저장
best_model._estimator_type = "regressor" 
best_model.save_model("qqq_xgboost_model.json")

# 2) 모델 사용에 필요한 부가 정보(피처 리스트, Horizon) 저장
import json
metadata = {
    "target_ticker": target,
    "best_horizon": int(best_horizon),
    "best_features": best_features
}

with open("qqq_model_metadata.json", "w") as f:
    json.dump(metadata, f)

print("\n✅ 모델과 메타데이터가 저장되었습니다. (qqq_xgboost_model.json, qqq_model_metadata.json)")

"""
0번 그래프: 각각의 feature 기여도

1번 그래프: 산점도 (Scatter Plot) - "모델의 정직함"
보는 법: 점들이 빨간 선 주위에 몰려 있는지 보세요.
평가: 그냥 동그랗게 뭉쳐 있다면 모델은 사실상 '아무 말'이나 하고 있는 겁니다. 대각선 방향으로 점들이 길쭉하게 늘어서야 모델이 미래의 변동을 읽고 있는 것입니다.

2번 그래프: 누적 수익률 - "모델의 쓸모"
보는 법: 파란 선(시장)보다 주황 선(모델)이 위에 있는지 보세요.
평가: 모델이 상승한다고 예측했을 때만 샀는데 시장 수익률보다 낮다면, 그 모델은 정확도가 아무리 높게 나와도 실전에서는 돈을 잃게 만드는 모델입니다. 하락장에서 주황색 선이 평평하게(안 사고 관망) 유지되는지 보세요.

3번 그래프: 오차 분포 - "모델의 안정성"
보는 법: 산의 정점이 0에 있는지, 꼬리가 왼쪽(과대평가)이나 오른쪽(과소평가)으로 길지 않은지 보세요.
평가: 만약 산이 왼쪽으로 치우쳐 있다면, 모델이 항상 실제보다 가격을 높게 부르는 '낙관주의자'라는 뜻입니다.

4번 그래프:
    1) 하락장에서 평평해지는가?:
        시장이 급락할 때(파란색 선이 꺾일 때), 전략 선(주황색, 초록색 등)이 수평으로 유지된다면 모델이 하락을 예측하고 현금화에 성공했다는 뜻입니다.
    2) 수익률이 시장보다 높은가?:
        최종 지점이 검은 점선(Market)보다 높은 선이 있다면, 그 문턱값이 이 모델의 **'진짜 실력'**이 발휘되는 지점입니다.
    3) 문턱이 너무 높으면?:
        만약 문턱을 3%(0.03)로 잡았는데 수익률이 1.0 근처에서 거의 안 움직인다면, 모델이 그만큼 확신하는 경우가 거의 없어서 아무것도 안 했다는 뜻입니다.
"""