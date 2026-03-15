import pandas as pd

dataset = pd.read_csv("fed_fomc_links.csv")
example_text = dataset.iloc[2]["body_text"]
print(example_text)
print(dataset.columns)