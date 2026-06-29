"""Shared data path resolution and metadata helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .preprocess import load_dataset


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_data_path() -> Path:
    root = project_root()
    for candidate in (root / "data" / "zomato.csv", root / "data" / "sample_zomato.csv"):
        if candidate.exists():
            return candidate
    return root / "data" / "zomato.csv"


def load_default_dataset() -> pd.DataFrame:
    path = default_data_path()
    if not path.exists():
        from scripts.download_dataset import download_dataset

        path = download_dataset(path)
    return load_dataset(path)


def get_cities(df: pd.DataFrame) -> list[str]:
    return sorted(df["City"].dropna().unique().tolist())


def get_cuisines(df: pd.DataFrame) -> list[str]:
    all_cuisines: set[str] = set()
    for val in df["Cuisines"].dropna():
        for c in str(val).split(","):
            c = c.strip()
            if c:
                all_cuisines.add(c)
    return sorted(all_cuisines)


def get_restaurant_names(df: pd.DataFrame) -> list[str]:
    return sorted(df["Restaurant Name"].dropna().unique().tolist())
