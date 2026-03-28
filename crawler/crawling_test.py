import pandas as pd
from data_paths import csv_path

merged = pd.read_csv(csv_path("merged_finbert.csv"))
print(merged.columns)
print(merged[merged["body_sentiment_score"]>0.5])

# dataset2 = pd.read_csv(csv_path("whitehouse_qqq_policy.csv"))
# text_lengths2 = dataset2["body"].fillna("").astype(str).str.len()
# print(text_lengths2.to_string())

# dataset3 = pd.read_csv(csv_path("bis_press_releases.csv"))
# text_lengths3 = dataset3["body"].fillna("").astype(str).str.len()
# print(text_lengths3.to_string())
