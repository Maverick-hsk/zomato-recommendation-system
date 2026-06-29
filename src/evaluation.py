"""Precision@K and related metrics for recommendation evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .recommender import ZomatoRecommender


@dataclass
class EvaluationMetrics:
    precision_at_k: float
    recall_at_k: float
    hit_rate: float
    mean_average_precision: float
    evaluated_queries: int


def _cuisine_overlap(cuisines_a: str, cuisines_b: str) -> bool:
    set_a = {c.strip().lower() for c in cuisines_a.split(",") if c.strip()}
    set_b = {c.strip().lower() for c in cuisines_b.split(",") if c.strip()}
    return len(set_a & set_b) > 0


def build_relevance_labels(
    df: pd.DataFrame,
    query_idx: int,
    same_city: bool = True,
    min_rating: float = 3.5,
) -> set[int]:
    """
    Derive pseudo ground-truth: restaurants sharing cuisine and city
    with rating >= min_rating are considered relevant.
    """
    query = df.loc[query_idx]
    relevant: set[int] = set()

    for idx, row in df.iterrows():
        if idx == query_idx:
            continue
        if float(row["Aggregate rating"]) < min_rating:
            continue
        if same_city and row["City"] != query["City"]:
            continue
        if _cuisine_overlap(str(query["Cuisines"]), str(row["Cuisines"])):
            relevant.add(int(idx))

    return relevant


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if r in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(relevant)


def average_precision(recommended: list[int], relevant: set[int]) -> float:
    if not relevant:
        return 0.0

    score = 0.0
    hits = 0
    for i, item in enumerate(recommended, start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant)


def evaluate_recommender(
    recommender: ZomatoRecommender,
    k: int = 10,
    sample_size: int | None = 200,
    random_state: int = 42,
) -> EvaluationMetrics:
    """Evaluate recommender using cuisine-overlap pseudo labels."""
    if recommender.df is None or recommender.similarity_matrix is None:
        raise RuntimeError("Recommender must be fitted before evaluation.")

    df = recommender.df
    indices = df.index.tolist()

    if sample_size and sample_size < len(indices):
        rng = np.random.default_rng(random_state)
        indices = rng.choice(indices, size=sample_size, replace=False).tolist()

    precisions: list[float] = []
    recalls: list[float] = []
    hits: list[float] = []
    maps: list[float] = []

    for query_idx in indices:
        relevant = build_relevance_labels(df, query_idx)
        if not relevant:
            continue

        scores = list(enumerate(recommender.similarity_matrix[query_idx]))
        scores.sort(key=lambda x: x[1], reverse=True)
        recommended = [i for i, _ in scores[1 : k + 1]]

        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))
        hits.append(1.0 if any(r in relevant for r in recommended) else 0.0)
        maps.append(average_precision(recommended, relevant))

    if not precisions:
        return EvaluationMetrics(0.0, 0.0, 0.0, 0.0, 0)

    return EvaluationMetrics(
        precision_at_k=float(np.mean(precisions)),
        recall_at_k=float(np.mean(recalls)),
        hit_rate=float(np.mean(hits)),
        mean_average_precision=float(np.mean(maps)),
        evaluated_queries=len(precisions),
    )
