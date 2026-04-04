import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.support_legacy.data_paths import feature_csv_path

merged = pd.read_csv(feature_csv_path("merged_finbert.csv"))
print(merged.columns)
print(merged[merged["body_sentiment_score"]>0.5])

# dataset2 = pd.read_csv(collected_csv_path("whitehouse_qqq_policy.csv"))
# text_lengths2 = dataset2["body"].fillna("").astype(str).str.len()
# print(text_lengths2.to_string())

# dataset3 = pd.read_csv(collected_csv_path("bis_press_releases.csv"))
# text_lengths3 = dataset3["body"].fillna("").astype(str).str.len()
# print(text_lengths3.to_string())
