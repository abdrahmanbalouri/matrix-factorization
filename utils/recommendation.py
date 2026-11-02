import os

import pandas as pd


def generate_recommendations(user_id, model, top_n=10):
    """Recommend unseen movies using (predictions, matrix, movies, ratings)."""
    predictions, matrix, movies, ratings = model
    if user_id not in matrix.index:
        return pd.DataFrame()
    user_idx = matrix.index.get_loc(user_id)
    seen = ratings.loc[ratings.UserID == user_id, "MovieID"]
    recommendations = pd.DataFrame({
        "MovieID": matrix.columns,
        "PredictedRating": predictions[user_idx],
    })
    recommendations = recommendations[~recommendations.MovieID.isin(seen)]
    return recommendations.merge(movies, on="MovieID").sort_values(
        "PredictedRating", ascending=False
    ).head(top_n)


def save_user_recommendations(user_id, rec_df, reports_dir="reports"):
    os.makedirs(reports_dir, exist_ok=True)
    rec_df.to_csv(
        os.path.join(reports_dir, f"user_{user_id}_recommendations.csv"), index=False
    )
