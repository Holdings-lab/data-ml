import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, plot_importance


# 1. QQQ 데이터 다운로드
# auto_adjust=False로 두면 Adj Close를 그대로 사용할 수 있음

df = yf.download("QQQ", start="2024-01-01", end="2026-01-01", auto_adjust=False)

df = df.reset_index()

# yfinance 버전에 따라 MultiIndex 컬럼이 나올 수 있으므로 방어 코드 추가
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

# 2. Feature 생성
# 현재 시점까지의 정보만 사용하도록 구성

df["return_1d"] = df[price_col].pct_change(1)
df["return_5d"] = df[price_col].pct_change(5)
df["ma_5"] = df[price_col].rolling(5).mean()
df["ma_10"] = df[price_col].rolling(10).mean()
df["vol_change"] = df["Volume"].pct_change()

# 3. Label 생성: 다음날 상승(1) / 하락 또는 동일(0)
df["target"] = (df[price_col].shift(-1) > df[price_col]).astype(int)

# 4. 결측 제거
df = df.dropna().copy()

features = [
    "return_1d",
    "return_5d",
    "ma_5",
    "ma_10",
    "vol_change",
]

X = df[features]
y = df["target"]

# 5. 시계열 데이터이므로 shuffle=False 유지
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False,
)


# 6. GPU 사용 모델
# NVIDIA CUDA 환경이 없으면 예외가 날 수 있으므로 CPU fallback 추가

def build_gpu_model():
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cuda",
        eval_metric="logloss",
        random_state=42,
    )


def build_cpu_model():
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cpu",
        eval_metric="logloss",
        random_state=42,
    )


try:
    model = build_gpu_model()
    model.fit(X_train, y_train)
    used_device = "cuda"
except Exception as e:
    print("GPU 학습 실패. CPU로 전환합니다.")
    print("원인:", e)
    model = build_cpu_model()
    model.fit(X_train, y_train)
    used_device = "cpu"

# 7. 예측 및 평가
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Used device: {used_device}")
print("Accuracy:", acc)

# 8. Feature importance 시각화
plot_importance(model)
plt.tight_layout()
plt.show()
