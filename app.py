"""Interactive Streamlit dashboard for the SVD and PMF recommenders."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import load_movielens_data
from utils.recommendation import generate_recommendations

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("SVD vs PMF Movie Recommender")


@st.cache_data
def load_all():
    """Load the ratings, movies, prediction matrices and movie details."""
    ratings, movies, _ = load_movielens_data("data")
    matrix = pd.read_csv("processed/user_item_matrix.csv", index_col=0)
    matrix.columns = matrix.columns.astype(int)
    svd_predictions = np.load("reports/svd_predictions.npy")
    pmf_predictions = np.load("reports/pmf_predictions.npy")
    return ratings, movies, matrix, svd_predictions, pmf_predictions


def top_genres(user_id, ratings, movies, n=3):
    """Return the user's most rated genres."""
    history = ratings[ratings.UserID == user_id].merge(movies, on="MovieID")
    genres = history.Genres.str.split("|").explode()
    return set(genres.value_counts().head(n).index)


def add_reason(recommendations, genres):
    """Add one short sentence explaining each recommendation."""

    def explain(row):
        overlap = set(str(row.Genres).split("|")) & genres
        if overlap:
            return "Genre match: " + ", ".join(sorted(overlap))
        return "High latent-factor score"

    return recommendations.assign(Reason=recommendations.apply(explain, axis=1))


try:
    ratings, movies, matrix, svd_predictions, pmf_predictions = load_all()

    selected = st.sidebar.selectbox("User ID:", matrix.index.tolist())
    typed = st.sidebar.text_input("Or type a User ID:", "")

    if typed.strip():
        try:
            user_id = int(typed)
        except ValueError:
            st.error("User ID must be a whole number.")
            st.stop()
    else:
        user_id = selected

    if user_id not in matrix.index:
        st.error(f"Invalid user ID: {user_id}. Please choose a valid user.")
        st.stop()

    svd_model = (svd_predictions, matrix, movies, ratings)
    pmf_model = (pmf_predictions, matrix, movies, ratings)
    svd_recs = generate_recommendations(user_id, svd_model)
    pmf_recs = generate_recommendations(user_id, pmf_model)

    if svd_recs.empty or pmf_recs.empty:
        st.error(f"No recommendations available for user {user_id}.")
        st.stop()

    genres = top_genres(user_id, ratings, movies)
    svd_recs = add_reason(svd_recs, genres)
    pmf_recs = add_reason(pmf_recs, genres)

    st.subheader(f"Top rated movies by user {user_id}")
    history = ratings[ratings.UserID == user_id].merge(movies, on="MovieID")
    st.dataframe(
        history.sort_values("Rating", ascending=False)[["Title", "Genres", "Rating"]].head(5),
        width="stretch",
    )
    st.caption("Favorite genres: " + (", ".join(sorted(genres)) or "unknown"))

    left, right = st.columns(2)
    with left:
        st.subheader("SVD recommendations")
        st.dataframe(
            svd_recs[["Title", "Genres", "PredictedRating", "Reason"]],
            width="stretch",
        )
    with right:
        st.subheader("PMF recommendations")
        st.dataframe(
            pmf_recs[["Title", "Genres", "PredictedRating", "Reason"]],
            width="stretch",
        )

    st.subheader("Why these PMF recommendations?")
    matches = pmf_recs.Genres.apply(
        lambda value: bool(set(str(value).split("|")) & genres)
    )
    colors = matches.map({True: "#2ecc71", False: "#95a5a6"})
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.barh(pmf_recs.Title, pmf_recs.PredictedRating, color=colors)
    axis.invert_yaxis()
    axis.set(xlabel="Predicted rating", ylabel="Movie", xlim=(0, 5.5))
    axis.set_title("Green = favorite genre, gray = latent-factor match")
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    st.subheader(f"SVD vs PMF predictions for user {user_id}")
    user_index = matrix.index.get_loc(user_id)
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(svd_predictions[user_index, :100], label="SVD")
    axis.plot(pmf_predictions[user_index, :100], label="PMF")
    axis.set(xlabel="Movie index (first 100)", ylabel="Predicted rating")
    axis.legend()
    st.pyplot(figure, width="stretch")
    plt.close(figure)

except Exception as error:
    st.error(f"Could not load the models. Run the notebook first. Details: {error}")
