import os

import numpy as np
import scipy.sparse.linalg as sps
from sklearn.metrics import root_mean_squared_error


def train_svd_model(normalized_matrix, baseline, test_data, k=30, reports_dir="reports",
                    validation_data=None):
    """Train bias-centered SVD, optionally calibrate on validation data, evaluate on test."""
    os.makedirs(reports_dir, exist_ok=True)
    U, sigma, Vt = sps.svds(normalized_matrix.values, k=k, random_state=42)
    raw_predictions = (U * sigma) @ Vt + baseline
    raw_predictions = np.clip(raw_predictions, 1, 5)

    users = {uid: i for i, uid in enumerate(normalized_matrix.index)}
    items = {mid: i for i, mid in enumerate(normalized_matrix.columns)}

    # Optional validation calibration: corrects the SVD scale/offset with a
    # linear fit learned on held-out validation pairs (no test leakage).
    slope, intercept = 1.0, 0.0
    if validation_data is not None:
        val_actuals, val_preds = [], []
        for row in validation_data.itertuples():
            if row.UserID in users and row.MovieID in items:
                val_actuals.append(row.Rating)
                val_preds.append(raw_predictions[users[row.UserID], items[row.MovieID]])
        if len(val_actuals) > 1:
            slope, intercept = np.polyfit(val_preds, val_actuals, 1)

    predictions = np.clip(slope * raw_predictions + intercept, 1, 5).astype(np.float32)
    np.save(os.path.join(reports_dir, "svd_predictions.npy"), predictions)

    actuals, preds = [], []
    for row in test_data.itertuples():
        if row.UserID in users and row.MovieID in items:
            actuals.append(row.Rating)
            preds.append(predictions[users[row.UserID], items[row.MovieID]])
    return predictions, root_mean_squared_error(actuals, preds), actuals, preds
