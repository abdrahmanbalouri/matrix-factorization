"""Build and save movie recommendations."""

import os

import pandas as pd


def generate_recommendations(user_id, model, top_n=10):
    """Recommend unseen movies using (predictions, matrix, movies, ratings)."""
    predictions, matrix, movies, ratings = model
    if user_id not in matrix.index:
        return pd.DataFrame()

    user_index = matrix.index.get_loc(user_id)
    seen = ratings.loc[ratings.UserID == user_id, "MovieID"]
    recommendations = pd.DataFrame({
        "MovieID": matrix.columns,
        "PredictedRating": predictions[user_index],
    })
    recommendations = recommendations[~recommendations.MovieID.isin(seen)]
    return recommendations.merge(movies, on="MovieID").sort_values(
        "PredictedRating", ascending=False
    ).head(top_n)


def favorite_genres(user_id, ratings, movies, n=3):
    """Return the genres the user rated the most."""
    history = ratings[ratings.UserID == user_id].merge(movies, on="MovieID")
    return set(history.Genres.str.split("|").explode().value_counts().head(n).index)


def add_reasons(recommendations, genres):
    """Add a short reason to every recommended movie."""

    def explain(row):
        overlap = set(str(row.Genres).split("|")) & genres
        if overlap:
            return "Genre match: " + ", ".join(sorted(overlap))
        return "High latent-factor score"

    return recommendations.assign(Reason=recommendations.apply(explain, axis=1))


def save_user_recommendations(user_id, rec_df, reports_dir="reports"):
    """Save a user's recommendations as a CSV file."""
    os.makedirs(reports_dir, exist_ok=True)
    rec_df.to_csv(
        os.path.join(reports_dir, f"user_{user_id}_recommendations.csv"), index=False
    )
