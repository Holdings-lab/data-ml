import pandas as pd

dataset = pd.read_csv("fed_fomc_links.csv")
example_text = dataset.iloc[2]["body_text"]



dataset2 = pd.read_csv("whitehouse_qqq_policy.csv")
example_text = dataset2["category"]
print(example_text)
print(dataset2.columns)