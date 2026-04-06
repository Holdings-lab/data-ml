from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = str(PROJECT_ROOT)

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from crawler.support_legacy.data_paths import collected_csv_path


def main() -> None:
    csv_path = collected_csv_path("ucsb_presidential_documents.csv")
    df = pd.read_csv(csv_path)

    keyword_columns = [
        "published_date",
        "matched_keyword_groups",
        "matched_keywords",
    ]

    available_columns = [column for column in keyword_columns if column in df.columns]
    print(df[available_columns].to_string(index=False))


if __name__ == "__main__":
    main()
