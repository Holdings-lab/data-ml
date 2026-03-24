import pandas as pd

dataset = pd.read_csv("fed_fomc_links.csv")
example_text = dataset.iloc[2]["body_text"]



dataset2 = pd.read_csv("whitehouse_qqq_policy.csv")
print(dataset2.columns)


dataset3 = pd.read_csv("bis_press_releases.csv")
print(dataset3.columns)