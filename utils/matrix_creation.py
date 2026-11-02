import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_and_save_interaction_matrix(ratings, processed_dir="processed", test_size=0.2):
    """Split data, learn regularized user/item biases, and save the matrix."""
    os.makedirs(processed_dir, exist_ok=True)
    train_data, test_data = train_test_split(ratings, test_size=test_size, random_state=42)

    users = sorted(ratings.UserID.unique())
    movies = sorted(ratings.MovieID.unique())
    interaction_matrix = train_data.pivot(index="UserID", columns="MovieID", values="Rating").reindex(
        index=users, columns=movies
    )
    global_mean = train_data.Rating.mean()
    user_index = pd.Index(users).get_indexer(train_data.UserID)
    item_index = pd.Index(movies).get_indexer(train_data.MovieID)
    values = train_data.Rating.to_numpy(dtype=float)

    user_bias = np.zeros(len(users))
    item_bias = np.zeros(len(movies))
    for _ in range(5):
        user_bias = np.bincount(
            user_index, weights=values - global_mean - item_bias[item_index], minlength=len(users)
        ) / (np.bincount(user_index, minlength=len(users)) + 10)
        item_bias = np.bincount(
            item_index, weights=values - global_mean - user_bias[user_index], minlength=len(movies)
        ) / (np.bincount(item_index, minlength=len(movies)) + 10)

    baseline = global_mean + user_bias[:, None] + item_bias[None, :]
    normalized_matrix = interaction_matrix - baseline
    normalized_matrix = normalized_matrix.fillna(0)
    normalized_matrix.round(2).to_csv(os.path.join(processed_dir, "user_item_matrix.csv"))
    return train_data, test_data, interaction_matrix, normalized_matrix, baseline
