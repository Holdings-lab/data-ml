from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "fed_fomc_links.csv"

dataset = pd.read_csv(DATA_PATH)
example_text = dataset.iloc[2]["body_text"]
print(example_text)
print(dataset.columns)
