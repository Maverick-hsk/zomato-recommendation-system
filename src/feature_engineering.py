"""Build composite restaurant profile text from metadata and reviews."""

from __future__ import annotations

import pandas as pd

from .preprocess import tokenize_text

PRICE_LABELS = {
    1: "budget affordable cheap",
    2: "moderate midrange",
    3: "expensive premium",
    4: "luxury fine dining",
}


def _price_text(price_range: int) -> str:
    return PRICE_LABELS.get(int(price_range), "moderate midrange")


def _rating_text(rating: float, rating_label: str) -> str:
    if rating >= 4.5:
        bucket = "excellent top rated highly recommended"
    elif rating >= 4.0:
        bucket = "very good popular well reviewed"
    elif rating >= 3.5:
        bucket = "good decent satisfactory"
    elif rating >= 3.0:
        bucket = "average mixed reviews"
    else:
        bucket = "below average low rated"

    label = tokenize_text(str(rating_label))
    return f"{bucket} {label}".strip()


def build_profile_text(row: pd.Series) -> str:
    """Combine cuisine, location, price, and review signals into one document."""
    cuisines = tokenize_text(str(row["Cuisines"]).replace(",", " "))
    location = tokenize_text(f"{row['City']} {row['Locality']}")
    name = tokenize_text(str(row["Restaurant Name"]))
    price = tokenize_text(_price_text(row["Price range"]))
    reviews = _rating_text(float(row["Aggregate rating"]), str(row["Rating text"]))

    # Repeat cuisines to weight them higher in TF-IDF
    return " ".join(
        [
            cuisines,
            cuisines,
            location,
            price,
            reviews,
            name,
        ]
    ).strip()


def add_profile_column(df: pd.DataFrame, column_name: str = "profile_text") -> pd.DataFrame:
    """Add tokenized composite profile column used for vectorization."""
    enriched = df.copy()
    enriched[column_name] = enriched.apply(build_profile_text, axis=1)
    return enriched
