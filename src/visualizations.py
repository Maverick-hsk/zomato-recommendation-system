"""Charts and plots for dataset exploration and recommendation insights."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .recommender import RecommendationResult, ZomatoRecommender

PRICE_LABELS = {1: "Budget", 2: "Moderate", 3: "Expensive", 4: "Luxury"}


def _explode_cuisines(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for cuisine in str(row["Cuisines"]).split(","):
            cuisine = cuisine.strip()
            if cuisine:
                rows.append({"Cuisine": cuisine, "City": row["City"]})
    return pd.DataFrame(rows)


def plot_city_distribution(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    counts = df["City"].value_counts().head(top_n).reset_index()
    counts.columns = ["City", "Restaurants"]
    fig = px.bar(
        counts,
        x="Restaurants",
        y="City",
        orientation="h",
        title=f"Top {top_n} Cities by Restaurant Count",
        color="Restaurants",
        color_continuous_scale="Reds",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
    return fig


def plot_cuisine_distribution(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    cuisine_df = _explode_cuisines(df)
    counts = cuisine_df["Cuisine"].value_counts().head(top_n).reset_index()
    counts.columns = ["Cuisine", "Count"]
    fig = px.bar(
        counts,
        x="Count",
        y="Cuisine",
        orientation="h",
        title=f"Top {top_n} Cuisines",
        color="Count",
        color_continuous_scale="Oranges",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
    return fig


def plot_rating_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="Aggregate rating",
        nbins=30,
        title="Restaurant Rating Distribution",
        color_discrete_sequence=["#e23744"],
    )
    fig.update_layout(bargap=0.05, xaxis_title="Aggregate Rating", yaxis_title="Count")
    return fig


def plot_price_range_distribution(df: pd.DataFrame) -> go.Figure:
    price_df = df.copy()
    price_df["Price Label"] = price_df["Price range"].map(PRICE_LABELS).fillna("Unknown")
    counts = price_df["Price Label"].value_counts().reset_index()
    counts.columns = ["Price Range", "Count"]
    fig = px.pie(
        counts,
        names="Price Range",
        values="Count",
        title="Restaurants by Price Range",
        color_discrete_sequence=px.colors.sequential.RdBu,
    )
    return fig


def plot_rating_vs_price(df: pd.DataFrame) -> go.Figure:
    plot_df = df.copy()
    plot_df["Price Label"] = plot_df["Price range"].map(PRICE_LABELS)
    fig = px.box(
        plot_df,
        x="Price Label",
        y="Aggregate rating",
        title="Rating vs Price Range",
        color="Price Label",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_recommendation_scores(results: list[RecommendationResult]) -> go.Figure:
    if not results:
        return go.Figure().update_layout(title="No recommendations to display")

    names = [r.restaurant_name[:30] for r in results]
    scores = [r.similarity_score for r in results]
    ratings = [r.rating for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Similarity", x=names, y=scores, marker_color="#e23744"))
    fig.add_trace(
        go.Scatter(
            name="Rating (scaled)",
            x=names,
            y=[r / 5.0 for r in ratings],
            mode="lines+markers",
            line={"color": "#2d2d2d", "width": 2},
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Recommendation Similarity Scores",
        xaxis_tickangle=-35,
        yaxis_title="Cosine Similarity",
        yaxis2={"title": "Rating (÷5)", "overlaying": "y", "side": "right", "range": [0, 1]},
        height=420,
    )
    return fig


def plot_similarity_heatmap(
    recommender: ZomatoRecommender,
    restaurant_name: str,
    top_n: int = 8,
) -> go.Figure:
    if recommender.df is None or recommender.similarity_matrix is None:
        return go.Figure().update_layout(title="Model not fitted")

    idx = recommender._find_index(restaurant_name)
    recs = recommender.recommend_by_restaurant(restaurant_name, top_n=top_n)
    indices = [idx] + [
        recommender.df[recommender.df["Restaurant Name"] == r.restaurant_name].index[0]
        for r in recs
    ]
    labels = [recommender.df.loc[i, "Restaurant Name"][:25] for i in indices]
    sub_matrix = recommender.similarity_matrix[np.ix_(indices, indices)]

    fig = go.Figure(
        data=go.Heatmap(
            z=sub_matrix,
            x=labels,
            y=labels,
            colorscale="Reds",
            zmin=0,
            zmax=1,
        )
    )
    fig.update_layout(
        title=f"Similarity Heatmap: {restaurant_name[:40]}",
        height=500,
        xaxis_tickangle=-30,
    )
    return fig


def plot_tfidf_top_terms(recommender: ZomatoRecommender, top_n: int = 20) -> go.Figure:
    if recommender.vectorizer is None:
        return go.Figure().update_layout(title="Vectorizer not fitted")

    feature_names = recommender.vectorizer.get_feature_names_out()
    idf_scores = recommender.vectorizer.idf_
    top_indices = np.argsort(idf_scores)[:top_n]

    terms = [feature_names[i] for i in top_indices]
    scores = [idf_scores[i] for i in top_indices]

    fig = px.bar(
        x=scores,
        y=terms,
        orientation="h",
        title=f"Top {top_n} Distinctive TF-IDF Terms (lowest IDF = most common)",
        labels={"x": "IDF Score", "y": "Term"},
        color=scores,
        color_continuous_scale="Greys",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    return fig


def plot_evaluation_summary(metrics) -> go.Figure:
    labels = ["Precision@K", "Recall@K", "Hit Rate", "MAP"]
    values = [
        metrics.precision_at_k,
        metrics.recall_at_k,
        metrics.hit_rate,
        metrics.mean_average_precision,
    ]
    fig = go.Figure(
        data=go.Bar(x=labels, y=values, marker_color=["#e23744", "#ff7e8a", "#2d2d2d", "#f4c430"])
    )
    fig.update_layout(
        title="Recommendation Evaluation Metrics",
        yaxis={"range": [0, 1]},
        yaxis_title="Score",
        height=380,
    )
    return fig


def plot_ab_test_results(ab_result) -> go.Figure:
    names = [v.name for v in ab_result.variants]
    satisfaction = [v.user_satisfaction_score for v in ab_result.variants]
    precision = [v.metrics.precision_at_k for v in ab_result.variants]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Satisfaction", x=names, y=satisfaction, marker_color="#e23744"))
    fig.add_trace(go.Bar(name="Precision@K", x=names, y=precision, marker_color="#2d2d2d"))
    fig.update_layout(
        title=f"A/B Test Results (Winner: {ab_result.winner})",
        barmode="group",
        yaxis={"range": [0, 1]},
        height=400,
    )
    return fig
