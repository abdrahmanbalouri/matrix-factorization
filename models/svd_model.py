"""Truncated SVD recommender."""

import os

import numpy as np
import scipy.sparse.linalg as sps
from sklearn.metrics import root_mean_squared_error


def train_svd_model(normalized_matrix, baseline, test_data, k=30, reports_dir="reports"):
    """Factorize the normalized matrix with svds and evaluate on the test set."""

    U, sigma, Vt = sps.svds(normalized_matrix.values, k=k, random_state=42)

    # Rebuild ratings: low-rank part + the user/movie baseline.
    predictions = np.clip((U * sigma) @ Vt + baseline, 1, 5).astype(np.float32)

    os.makedirs(reports_dir, exist_ok=True)
    np.save(os.path.join(reports_dir, "svd_predictions.npy"), predictions)

    users = {user: i for i, user in enumerate(normalized_matrix.index)}
    items = {movie: i for i, movie in enumerate(normalized_matrix.columns)}

    actuals, preds = [], []
    for row in test_data.itertuples():
        if row.UserID in users and row.MovieID in items:
            actuals.append(row.Rating)
            preds.append(predictions[users[row.UserID], items[row.MovieID]])

    return predictions, root_mean_squared_error(actuals, preds), actuals, preds
