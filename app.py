import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import load_movielens_data
from utils.recommendation import generate_recommendations

st.set_page_config(page_title="Movie Recommender System", layout="wide")
st.title("🎬 SVD vs PMF Movie Recommender")


@st.cache_data
def load_all():
    ratings, movies, _ = load_movielens_data("data")
    matrix = pd.read_csv("processed/user_item_matrix.csv", index_col=0)
    matrix.columns = matrix.columns.astype(int)
    svd_preds = np.load("reports/svd_predictions.npy")
    pmf_preds = np.load("reports/pmf_predictions.npy")
    return ratings, movies, matrix, svd_preds, pmf_preds


def top_genres(user_id, ratings, movies, n_genres=3):
    history = ratings[ratings.UserID == user_id].merge(movies, on="MovieID")
    return set(history.Genres.str.split("|").explode().value_counts().head(n_genres).index)


def add_reasons(recs, genres):
    def explain(row):
        overlap = set(str(row.Genres).split("|")) & genres
        if overlap:
            return "Genre match: " + ", ".join(sorted(overlap))
        return "High predicted rating from latent factors"

    return recs.assign(Reason=recs.apply(explain, axis=1))


try:
    ratings, movies, matrix, svd_preds, pmf_preds = load_all()
    selected_user = st.sidebar.selectbox("Select User ID:", matrix.index.tolist())
    manual_user = st.sidebar.text_input("Or enter a User ID manually:", "")
    if manual_user.strip():
        try:
            user_id = int(manual_user)
        except ValueError:
            st.error("User ID must be an integer.")
            st.stop()
    else:
        user_id = selected_user
    if user_id not in matrix.index:
        st.error(f"Invalid user ID: {user_id}. Please choose a valid user.")
        st.stop()

    svd_model = (svd_preds, matrix, movies, ratings)
    pmf_model = (pmf_preds, matrix, movies, ratings)
    svd_recs = generate_recommendations(user_id, svd_model)
    pmf_recs = generate_recommendations(user_id, pmf_model)
    if svd_recs.empty or pmf_recs.empty:
        st.error(f"No recommendations could be generated for user {user_id}.")
        st.stop()

    genres = top_genres(user_id, ratings, movies)
    svd_recs = add_reasons(svd_recs, genres)
    pmf_recs = add_reasons(pmf_recs, genres)

    st.subheader(f"⭐ Top Rated Movies by User {user_id}")
    history = ratings[ratings.UserID == user_id].merge(movies, on="MovieID")
    top_history = history.sort_values("Rating", ascending=False).head(5)
    st.dataframe(top_history[["Title", "Genres", "Rating"]], width="stretch")
    st.caption("User's top genres: " + (", ".join(sorted(genres)) or "unknown"))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔵 SVD Top 10")
        st.dataframe(svd_recs[["Title", "Genres", "PredictedRating", "Reason"]], width="stretch")
    with col2:
        st.subheader("🟢 PMF Top 10")
        st.dataframe(pmf_recs[["Title", "Genres", "PredictedRating", "Reason"]], width="stretch")

    st.subheader("🧠 Why these PMF recommendations?")
    reasons = pmf_recs.assign(
        GenreMatch=pmf_recs.Genres.apply(lambda value: bool(set(str(value).split("|")) & genres))
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = reasons.GenreMatch.map({True: "#2ecc71", False: "#95a5a6"})
    ax.barh(reasons.Title, reasons.PredictedRating, color=colors)
    ax.invert_yaxis()
    ax.set(xlabel="Predicted rating", ylabel="Movie", xlim=(0, 5.5))
    ax.set_title("Green = matches a top genre, gray = driven by latent factors")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.subheader(f"📊 SVD vs PMF Predictions — User {user_id}")
    user_idx = matrix.index.get_loc(user_id)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(svd_preds[user_idx, :100], label="SVD")
    ax.plot(pmf_preds[user_idx, :100], label="PMF")
    ax.set(xlabel="Movie index (first 100)", ylabel="Predicted rating")
    ax.legend()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
except Exception as error:
    st.error(f"Run the notebook first to train the models. Details: {error}")
