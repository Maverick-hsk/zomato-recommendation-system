"""Content-based recommender using TF-IDF and cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .feature_engineering import add_profile_column
from .preprocess import preprocess_dataset


@dataclass
class RecommendationResult:
    restaurant_id: int
    restaurant_name: str
    city: str
    locality: str
    cuisines: str
    rating: float
    price_range: int
    similarity_score: float


class ZomatoRecommender:
    """TF-IDF + cosine similarity content-based restaurant recommender."""

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words="english",
        )
        self.df: pd.DataFrame | None = None
        self.tfidf_matrix = None
        self.similarity_matrix = None

    def fit(self, df: pd.DataFrame) -> "ZomatoRecommender":
        """Fit vectorizer on restaurant profiles."""
        processed = preprocess_dataset(df)
        self.df = add_profile_column(processed)

        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["profile_text"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        return self

    def _find_index(self, restaurant_name: str) -> int:
        if self.df is None:
            raise RuntimeError("Recommender is not fitted. Call fit() first.")

        matches = self.df[self.df["Restaurant Name"].str.lower() == restaurant_name.lower()]
        if matches.empty:
            matches = self.df[self.df["Restaurant Name"].str.contains(restaurant_name, case=False, na=False)]
        if matches.empty:
            raise ValueError(f"Restaurant not found: {restaurant_name}")
        return int(matches.index[0])

    def recommend_by_restaurant(
        self,
        restaurant_name: str,
        top_n: int = 10,
        min_rating: float = 0.0,
        same_city: bool = False,
    ) -> list[RecommendationResult]:
        """Recommend restaurants similar to a given restaurant."""
        if self.df is None or self.similarity_matrix is None:
            raise RuntimeError("Recommender is not fitted. Call fit() first.")

        idx = self._find_index(restaurant_name)
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores.sort(key=lambda x: x[1], reverse=True)

        source_city = self.df.loc[idx, "City"]
        results: list[RecommendationResult] = []

        for i, score in scores[1:]:  # skip self
            row = self.df.loc[i]
            if float(row["Aggregate rating"]) < min_rating:
                continue
            if same_city and row["City"] != source_city:
                continue

            results.append(
                RecommendationResult(
                    restaurant_id=int(row["Restaurant ID"]),
                    restaurant_name=str(row["Restaurant Name"]),
                    city=str(row["City"]),
                    locality=str(row["Locality"]),
                    cuisines=str(row["Cuisines"]),
                    rating=float(row["Aggregate rating"]),
                    price_range=int(row["Price range"]),
                    similarity_score=float(score),
                )
            )
            if len(results) >= top_n:
                break

        return results

    def recommend_by_preferences(
        self,
        cuisines: str,
        city: str | None = None,
        price_range: int | None = None,
        min_rating: float = 3.5,
        top_n: int = 10,
    ) -> list[RecommendationResult]:
        """Recommend restaurants matching a synthetic user preference profile."""
        if self.df is None or self.tfidf_matrix is None:
            raise RuntimeError("Recommender is not fitted. Call fit() first.")

        from .feature_engineering import build_profile_text

        pref_row = pd.Series(
            {
                "Restaurant Name": "user preference",
                "City": city or "",
                "Locality": city or "",
                "Cuisines": cuisines,
                "Price range": price_range or 2,
                "Aggregate rating": 4.5,
                "Rating text": "excellent",
            }
        )
        pref_text = build_profile_text(pref_row)
        pref_vector = self.vectorizer.transform([pref_text])
        scores = cosine_similarity(pref_vector, self.tfidf_matrix).flatten()

        ranked = np.argsort(scores)[::-1]
        results: list[RecommendationResult] = []

        for i in ranked:
            row = self.df.loc[i]
            if float(row["Aggregate rating"]) < min_rating:
                continue
            if city and str(row["City"]).lower() != city.lower():
                continue
            if price_range and int(row["Price range"]) != price_range:
                continue

            results.append(
                RecommendationResult(
                    restaurant_id=int(row["Restaurant ID"]),
                    restaurant_name=str(row["Restaurant Name"]),
                    city=str(row["City"]),
                    locality=str(row["Locality"]),
                    cuisines=str(row["Cuisines"]),
                    rating=float(row["Aggregate rating"]),
                    price_range=int(row["Price range"]),
                    similarity_score=float(scores[i]),
                )
            )
            if len(results) >= top_n:
                break

        return results

    def get_params(self) -> dict[str, Any]:
        return {
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "min_df": self.min_df,
        }
