import pandas as pd

df = pd.read_csv("data/crawler/features/policy_updates_features.csv")
sampled = df[["body_summary"]]

# 컬럼 내부 문자열 생략(Truncation) 해제
pd.set_option("display.max_colwidth", None)

print(sampled)