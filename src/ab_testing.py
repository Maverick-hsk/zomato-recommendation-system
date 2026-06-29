"""A/B testing harness to compare TF-IDF hyperparameter configurations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .evaluation import EvaluationMetrics, evaluate_recommender
from .recommender import ZomatoRecommender


@dataclass
class ABTestVariant:
    name: str
    params: dict
    metrics: EvaluationMetrics
    user_satisfaction_score: float


@dataclass
class ABTestResult:
    winner: str
    variants: list[ABTestVariant]
    improvement_pct: float


def _satisfaction_score(metrics: EvaluationMetrics) -> float:
    """
    Composite satisfaction proxy combining precision, hit rate, and MAP.
    Weighted toward precision@K for business relevance.
    """
    return (
        0.5 * metrics.precision_at_k
        + 0.3 * metrics.hit_rate
        + 0.2 * metrics.mean_average_precision
    )


def run_ab_test(
    df: pd.DataFrame,
    variants: list[dict] | None = None,
    k: int = 10,
    sample_size: int = 150,
) -> ABTestResult:
    """
    Compare multiple TF-IDF configurations and pick the best variant.

    Each variant dict may include: name, max_features, ngram_range, min_df.
    """
    if variants is None:
        variants = [
            {"name": "A_baseline", "max_features": 3000, "ngram_range": (1, 1), "min_df": 1},
            {"name": "B_bigrams", "max_features": 5000, "ngram_range": (1, 2), "min_df": 1},
            {"name": "C_high_dim", "max_features": 8000, "ngram_range": (1, 2), "min_df": 2},
        ]

    results: list[ABTestVariant] = []

    for variant in variants:
        name = variant.get("name", "variant")
        params = {
            "max_features": variant.get("max_features", 5000),
            "ngram_range": variant.get("ngram_range", (1, 2)),
            "min_df": variant.get("min_df", 1),
        }

        model = ZomatoRecommender(**params)
        model.fit(df)
        metrics = evaluate_recommender(model, k=k, sample_size=sample_size)
        satisfaction = _satisfaction_score(metrics)

        results.append(
            ABTestVariant(
                name=name,
                params=params,
                metrics=metrics,
                user_satisfaction_score=satisfaction,
            )
        )

    results.sort(key=lambda v: v.user_satisfaction_score, reverse=True)
    winner = results[0]
    runner_up = results[1] if len(results) > 1 else results[0]

    if runner_up.user_satisfaction_score > 0:
        improvement = (
            (winner.user_satisfaction_score - runner_up.user_satisfaction_score)
            / runner_up.user_satisfaction_score
            * 100
        )
    else:
        improvement = 0.0

    return ABTestResult(
        winner=winner.name,
        variants=results,
        improvement_pct=round(improvement, 2),
    )
