"""CLI entry point for the Zomato recommendation system."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ab_testing import run_ab_test
from src.data_utils import default_data_path, load_default_dataset
from src.evaluation import evaluate_recommender
from src.recommender import ZomatoRecommender


def _print_recommendations(results) -> None:
    print(f"\n{'Rank':<5} {'Restaurant':<35} {'City':<12} {'Rating':<8} {'Score':<8} Cuisines")
    print("-" * 100)
    for rank, rec in enumerate(results, start=1):
        print(
            f"{rank:<5} {rec.restaurant_name[:34]:<35} {rec.city[:11]:<12} "
            f"{rec.rating:<8.1f} {rec.similarity_score:<8.3f} {rec.cuisines[:40]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Zomato content-based restaurant recommender")
    parser.add_argument("--data", type=Path, default=None, help="Path to Zomato CSV")
    parser.add_argument("--restaurant", type=str, help="Restaurant name for similarity recommendations")
    parser.add_argument("--cuisines", type=str, help="Preferred cuisines (e.g. 'North Indian, Chinese')")
    parser.add_argument("--city", type=str, help="Filter recommendations by city")
    parser.add_argument("--price-range", type=int, choices=[1, 2, 3, 4], help="Price range 1-4")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--evaluate", action="store_true", help="Run precision@K evaluation")
    parser.add_argument("--ab-test", action="store_true", help="Run A/B test across TF-IDF configs")
    args = parser.parse_args()

    if args.data:
        from src.preprocess import load_dataset

        data_path = args.data
        if not data_path.exists():
            print(f"Dataset not found at {data_path}")
            return
        print(f"Loading data from {data_path}")
        df = load_dataset(data_path)
    else:
        data_path = default_data_path()
        if not data_path.exists():
            print("Full dataset not found. Downloading...")
            from scripts.download_dataset import download_dataset

            download_dataset(data_path)
        print(f"Loading data from {data_path}")
        df = load_default_dataset()

    if args.ab_test:
        print("\n=== A/B Testing TF-IDF Configurations ===")
        result = run_ab_test(df)
        for variant in result.variants:
            m = variant.metrics
            print(
                f"\n{variant.name} | satisfaction={variant.user_satisfaction_score:.3f} | "
                f"precision@{args.top_n}={m.precision_at_k:.3f} | "
                f"recall@{args.top_n}={m.recall_at_k:.3f} | hit_rate={m.hit_rate:.3f} | MAP={m.mean_average_precision:.3f}"
            )
            print(f"  params: {variant.params}")
        print(f"\nWinner: {result.winner} (+{result.improvement_pct}% vs runner-up)")
        return

    model = ZomatoRecommender(max_features=5000, ngram_range=(1, 2))
    model.fit(df)
    print(f"Fitted on {len(model.df)} restaurants")

    if args.evaluate:
        metrics = evaluate_recommender(model, k=args.top_n)
        print("\n=== Evaluation Metrics ===")
        print(f"Precision@{args.top_n}: {metrics.precision_at_k:.3f}")
        print(f"Recall@{args.top_n}:    {metrics.recall_at_k:.3f}")
        print(f"Hit Rate:              {metrics.hit_rate:.3f}")
        print(f"MAP:                   {metrics.mean_average_precision:.3f}")
        print(f"Queries evaluated:     {metrics.evaluated_queries}")

    if args.cuisines:
        print(f"\n=== Recommendations for preferences: {args.cuisines} ===")
        results = model.recommend_by_preferences(
            cuisines=args.cuisines,
            city=args.city,
            price_range=args.price_range,
            top_n=args.top_n,
        )
        _print_recommendations(results)
    elif args.restaurant:
        print(f"\n=== Restaurants similar to '{args.restaurant}' ===")
        results = model.recommend_by_restaurant(args.restaurant, top_n=args.top_n)
        _print_recommendations(results)
    elif not args.evaluate:
        # Demo: pick first restaurant in dataset
        sample_name = df.iloc[0]["Restaurant Name"]
        print(f"\n=== Demo: Restaurants similar to '{sample_name}' ===")
        results = model.recommend_by_restaurant(sample_name, top_n=args.top_n)
        _print_recommendations(results)
        print("\nTip: use --restaurant, --cuisines, --evaluate, or --ab-test")


if __name__ == "__main__":
    main()
