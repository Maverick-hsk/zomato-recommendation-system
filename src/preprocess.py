"""Data loading, cleaning, and text tokenization for restaurant profiles."""

from __future__ import annotations

import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources quietly on first import
for resource in ("stopwords", "wordnet", "punkt", "punkt_tab"):
    try:
        nltk.data.find(f"corpora/{resource}" if resource != "punkt_tab" else "tokenizers/punkt_tab")
    except LookupError:
        nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Standard Zomato Kaggle dataset columns
REQUIRED_COLUMNS = [
    "Restaurant ID",
    "Restaurant Name",
    "City",
    "Locality",
    "Cuisines",
    "Price range",
    "Aggregate rating",
    "Rating text",
    "Votes",
]

COLUMN_ALIASES = {
    "restaurant_id": "Restaurant ID",
    "restaurant_name": "Restaurant Name",
    "name": "Restaurant Name",
    "city": "City",
    "locality": "Locality",
    "location": "Locality",
    "cuisines": "Cuisines",
    "price_range": "Price range",
    "aggregate_rating": "Aggregate rating",
    "rating": "Aggregate rating",
    "rating_text": "Rating text",
    "votes": "Votes",
}


def load_dataset(path: str | Path, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    """Load Zomato CSV and normalize column names."""
    df = pd.read_csv(path, encoding=encoding)
    df.columns = [c.strip() for c in df.columns]

    rename_map = {col: COLUMN_ALIASES[col.lower()] for col in df.columns if col.lower() in COLUMN_ALIASES}
    if rename_map:
        df = df.rename(columns=rename_map)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean restaurant metadata and fill missing textual fields."""
    cleaned = df.copy()

    text_cols = ["Restaurant Name", "City", "Locality", "Cuisines", "Rating text"]
    for col in text_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        cleaned[col] = cleaned[col].replace({"nan": "", "None": ""})

    cleaned["Cuisines"] = cleaned["Cuisines"].str.replace(r"\s*,\s*", ", ", regex=True)
    cleaned["Price range"] = pd.to_numeric(cleaned["Price range"], errors="coerce").fillna(2).astype(int)
    cleaned["Aggregate rating"] = pd.to_numeric(cleaned["Aggregate rating"], errors="coerce").fillna(0.0)
    cleaned["Votes"] = pd.to_numeric(cleaned["Votes"], errors="coerce").fillna(0).astype(int)

    cleaned = cleaned.drop_duplicates(subset=["Restaurant ID"]).reset_index(drop=True)
    cleaned = cleaned[cleaned["Cuisines"].str.len() > 0]

    return cleaned


def tokenize_text(text: str) -> str:
    """Lowercase, remove punctuation, lemmatize, and drop stopwords."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [LEMMATIZER.lemmatize(tok) for tok in text.split() if len(tok) > 1]
    tokens = [tok for tok in tokens if tok not in STOP_WORDS]
    return " ".join(tokens)


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline returning a cleaned dataframe."""
    cleaned = clean_dataframe(df)
    return cleaned
