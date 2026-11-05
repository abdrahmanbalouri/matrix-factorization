"""Split the ratings and build the normalized user-item matrix."""

import os

import pandas as pd
from sklearn.model_selection import train_test_split


def create_and_save_interaction_matrix(ratings, processed_dir="processed", test_size=0.1):
    """Return train/test split, the rating matrix, its baseline and residuals.

    The baseline uses the global mean plus a user bias plus a movie bias.
    Subtracting it centers the matrix around zero, which helps SVD and PMF.
    """

    train, test = train_test_split(ratings, test_size=test_size, random_state=42)

    users = sorted(ratings.UserID.unique())
    movies = sorted(ratings.MovieID.unique())
    matrix = train.pivot(
        index="UserID", columns="MovieID", values="Rating"
    ).reindex(index=users, columns=movies)

    global_mean = train.Rating.mean()
    user_bias = train.groupby("UserID").Rating.mean() - global_mean
    item_bias = train.groupby("MovieID").Rating.mean() - global_mean
    baseline = (
        global_mean
        + user_bias.reindex(users).fillna(0).to_numpy()[:, None]
        + item_bias.reindex(movies).fillna(0).to_numpy()[None, :]
    )

    normalized = (matrix - baseline).fillna(0)

    os.makedirs(processed_dir, exist_ok=True)
    normalized.round(2).to_csv(
        os.path.join(processed_dir, "user_item_matrix.csv")
    )

    return train, test, matrix, normalized, baseline
