import pandas as pd

dataset = pd.read_csv("fed_fomc_links_summarized.csv")
text_lengths = dataset["body"].fillna("").astype(str).str.len()
print(text_lengths.to_string())
print(dataset["doc_type"].head(30).to_string(index=False))

# dataset2 = pd.read_csv("whitehouse_qqq_policy.csv")
# text_lengths2 = dataset2["body"].fillna("").astype(str).str.len()
# print(text_lengths2.to_string())

# dataset3 = pd.read_csv("bis_press_releases.csv")
# text_lengths3 = dataset3["body"].fillna("").astype(str).str.len()
# print(text_lengths3.to_string())
