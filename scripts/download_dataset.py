"""Download the full Zomato Kaggle dataset (~9,551 restaurants)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

DATASET_URLS = [
    "https://raw.githubusercontent.com/AnaghaBandaru/ZomatoDatasetAnalysis/main/data/zomato.csv",
    "https://raw.githubusercontent.com/akarshxydv/Zomato-Dataset-Exploratory-Data-Analysis/master/zomato.csv",
    "https://raw.githubusercontent.com/You-sha/Restaurant-Ratings-Prediction/main/zomato.csv",
]


def download_dataset(output_path: Path | None = None) -> Path:
    output = output_path or Path(__file__).resolve().parent.parent / "data" / "zomato.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for url in DATASET_URLS:
        try:
            print(f"Downloading from {url} ...")
            urllib.request.urlretrieve(url, output)
            import pandas as pd

            df = pd.read_csv(output, encoding="ISO-8859-1")
            if len(df) < 9000:
                raise ValueError(f"Unexpected row count: {len(df)}")
            print(f"Saved {len(df)} restaurants to {output}")
            return output
        except Exception as exc:
            last_error = exc
            print(f"  Failed: {exc}")

    raise RuntimeError(f"Could not download dataset. Last error: {last_error}")


if __name__ == "__main__":
    download_dataset()
