"""Streamlit UI for Zomato content-based restaurant recommendations."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.ab_testing import run_ab_test
from src.data_utils import default_data_path, get_cities, get_cuisines, get_restaurant_names, load_default_dataset
from src.evaluation import evaluate_recommender
from src.recommender import ZomatoRecommender
from src.visualizations import (
    plot_ab_test_results,
    plot_city_distribution,
    plot_cuisine_distribution,
    plot_evaluation_summary,
    plot_price_range_distribution,
    plot_rating_distribution,
    plot_rating_vs_price,
    plot_recommendation_scores,
    plot_similarity_heatmap,
    plot_tfidf_top_terms,
)

PRICE_LABELS = {1: "Budget (₹)", 2: "Moderate (₹₹)", 3: "Expensive (₹₹₹)", 4: "Luxury (₹₹₹₹)"}

st.set_page_config(
    page_title="Zomato Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading Zomato dataset...")
def load_data() -> pd.DataFrame:
    path = default_data_path()
    if not path.exists():
        from scripts.download_dataset import download_dataset

        try:
            download_dataset(path)
        except RuntimeError as exc:
            sample = ROOT / "data" / "sample_zomato.csv"
            if sample.exists():
                st.warning(
                    "Full dataset download failed; using bundled sample data (~500 restaurants). "
                    "Refresh later for the full catalog."
                )
                return load_default_dataset()
            st.error(f"Could not load dataset: {exc}")
            st.stop()
    return load_default_dataset()


@st.cache_resource(show_spinner="Training TF-IDF recommender on full dataset...")
def load_model(_df: pd.DataFrame) -> ZomatoRecommender:
    model = ZomatoRecommender(max_features=5000, ngram_range=(1, 2))
    model.fit(_df)
    return model


def _results_to_df(results) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Restaurant": r.restaurant_name,
                "City": r.city,
                "Locality": r.locality,
                "Cuisines": r.cuisines,
                "Rating": r.rating,
                "Price": PRICE_LABELS.get(r.price_range, str(r.price_range)),
                "Similarity": round(r.similarity_score, 3),
            }
            for r in results
        ]
    )


def render_sidebar(df: pd.DataFrame) -> dict:
    st.sidebar.title("🍽️ Zomato Recommender")
    st.sidebar.caption("Content-based · TF-IDF · Cosine Similarity")

    mode = st.sidebar.radio(
        "Recommendation mode",
        ["Similar Restaurant", "User Preferences"],
    )

    top_n = st.sidebar.slider("Number of recommendations", 5, 20, 10)
    min_rating = st.sidebar.slider("Minimum rating", 0.0, 5.0, 3.5, 0.5)

    params: dict = {"mode": mode, "top_n": top_n, "min_rating": min_rating}

    if mode == "Similar Restaurant":
        restaurants = get_restaurant_names(df)
        params["restaurant"] = st.sidebar.selectbox(
            "Select a restaurant",
            restaurants,
            index=0,
        )
        params["same_city"] = st.sidebar.checkbox("Same city only", value=False)
    else:
        cities = ["Any city"] + get_cities(df)
        cuisines = get_cuisines(df)
        params["city"] = st.sidebar.selectbox("City", cities)
        params["cuisines"] = st.sidebar.multiselect(
            "Preferred cuisines",
            cuisines,
            default=cuisines[:2] if len(cuisines) >= 2 else cuisines,
        )
        params["price_range"] = st.sidebar.selectbox(
            "Price range",
            [None, 1, 2, 3, 4],
            format_func=lambda x: "Any" if x is None else PRICE_LABELS[x],
        )

    return params


def main() -> None:
    df = load_data()
    model = load_model(df)

    st.title("Zomato Restaurant Recommendation System")
    st.markdown(
        f"**{len(df):,} restaurants** loaded from `{default_data_path().name}` · "
        "Powered by TF-IDF vectorization and cosine similarity"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Restaurants", f"{len(df):,}")
    col2.metric("Cities", df["City"].nunique())
    col3.metric("Avg Rating", f"{df['Aggregate rating'].mean():.2f}")
    col4.metric("Cuisines", len(get_cuisines(df)))

    params = render_sidebar(df)

    tab_rec, tab_viz, tab_eval = st.tabs(["Recommendations", "Visualizations", "Evaluation & A/B Test"])

    with tab_rec:
        if st.button("Get Recommendations", type="primary", use_container_width=True):
            with st.spinner("Computing recommendations..."):
                if params["mode"] == "Similar Restaurant":
                    results = model.recommend_by_restaurant(
                        params["restaurant"],
                        top_n=params["top_n"],
                        min_rating=params["min_rating"],
                        same_city=params["same_city"],
                    )
                    st.subheader(f"Restaurants similar to **{params['restaurant']}**")
                else:
                    if not params["cuisines"]:
                        st.warning("Select at least one cuisine.")
                        st.stop()
                    city = None if params["city"] == "Any city" else params["city"]
                    results = model.recommend_by_preferences(
                        cuisines=", ".join(params["cuisines"]),
                        city=city,
                        price_range=params["price_range"],
                        min_rating=params["min_rating"],
                        top_n=params["top_n"],
                    )
                    st.subheader("Restaurants matching your preferences")

                if not results:
                    st.info("No restaurants matched your filters. Try relaxing min rating or price range.")
                else:
                    st.dataframe(_results_to_df(results), use_container_width=True, hide_index=True)
                    st.plotly_chart(plot_recommendation_scores(results), use_container_width=True)

                    if params["mode"] == "Similar Restaurant":
                        st.plotly_chart(
                            plot_similarity_heatmap(model, params["restaurant"], top_n=min(8, params["top_n"])),
                            use_container_width=True,
                        )
        else:
            st.info("Configure options in the sidebar, then click **Get Recommendations**.")

    with tab_viz:
        st.subheader("Dataset Explorer")
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.plotly_chart(plot_city_distribution(df), use_container_width=True)
            st.plotly_chart(plot_rating_distribution(df), use_container_width=True)
            st.plotly_chart(plot_rating_vs_price(df), use_container_width=True)
        with vcol2:
            st.plotly_chart(plot_cuisine_distribution(df), use_container_width=True)
            st.plotly_chart(plot_price_range_distribution(df), use_container_width=True)
            st.plotly_chart(plot_tfidf_top_terms(model), use_container_width=True)

    with tab_eval:
        st.subheader("Model Evaluation")
        eval_k = st.slider("K for Precision@K", 5, 20, 10, key="eval_k")

        if st.button("Run Evaluation", use_container_width=True):
            with st.spinner("Evaluating on sample queries..."):
                metrics = evaluate_recommender(model, k=eval_k, sample_size=300)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Precision@K", f"{metrics.precision_at_k:.3f}")
                c2.metric("Recall@K", f"{metrics.recall_at_k:.3f}")
                c3.metric("Hit Rate", f"{metrics.hit_rate:.3f}")
                c4.metric("MAP", f"{metrics.mean_average_precision:.3f}")
                st.caption(f"Evaluated {metrics.evaluated_queries} queries")
                st.plotly_chart(plot_evaluation_summary(metrics), use_container_width=True)

        st.divider()
        st.subheader("A/B Testing — TF-IDF Configurations")
        if st.button("Run A/B Test", use_container_width=True):
            with st.spinner("Running A/B test across 3 variants (may take a minute)..."):
                ab = run_ab_test(df, sample_size=150, k=eval_k)
                st.success(f"Winner: **{ab.winner}** (+{ab.improvement_pct}% satisfaction vs runner-up)")
                st.plotly_chart(plot_ab_test_results(ab), use_container_width=True)

                ab_df = pd.DataFrame(
                    [
                        {
                            "Variant": v.name,
                            "Satisfaction": round(v.user_satisfaction_score, 3),
                            "Precision@K": round(v.metrics.precision_at_k, 3),
                            "Recall@K": round(v.metrics.recall_at_k, 3),
                            "Hit Rate": round(v.metrics.hit_rate, 3),
                            "MAP": round(v.metrics.mean_average_precision, 3),
                            "Params": str(v.params),
                        }
                        for v in ab.variants
                    ]
                )
                st.dataframe(ab_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
