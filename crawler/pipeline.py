from pathlib import Path

import pandas as pd


def deduplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
