"""A small PMF model: learn user and movie factors with SGD."""

import json
import os

import numpy as np
from numba import njit
from sklearn.metrics import root_mean_squared_error


@njit
def _sgd(U, V, users, items, ratings, lr, reg):
    """One SGD pass over all training ratings. Returns the MSE."""
    total = 0.0
    for n in range(len(users)):
        user, item = users[n], items[n]

        prediction = 0.0
        for f in range(U.shape[1]):
            prediction += U[user, f] * V[item, f]

        error = ratings[n] - prediction
        total += error * error

        for f in range(U.shape[1]):
            U[user, f] += lr * (error * V[item, f] - reg * U[user, f])
            V[item, f] += lr * (error * U[user, f] - reg * V[item, f])

    return total / len(users)


def _index(matrix, data):
    """Turn UserID/MovieID values into row and column positions."""
    users = matrix.index.get_indexer(data.UserID)
    items = matrix.columns.get_indexer(data.MovieID)
    return users, items, data.Rating.to_numpy()


def train_pmf(train, test, matrix, validation=None, factors=20, epochs=100,
              lr=0.005, reg=0.05, members=5, reports="reports"):
    """Train a small PMF ensemble, save its files and return the test RMSE."""

    users, items, ratings = _index(matrix, train)
    if validation is not None:
        valid_users, valid_items, valid_ratings = _index(matrix, validation)

    predictions = np.zeros((len(matrix.index), len(matrix.columns)), dtype=np.float32)
    train_history, validation_history = [], []
    first_U = first_V = None

    for member in range(members):
        rng = np.random.default_rng(42 + member)
        U = rng.normal(0, 0.1, (len(matrix.index), factors))
        V = rng.normal(0, 0.1, (len(matrix.columns), factors))

        for _ in range(epochs):
            mse = _sgd(U, V, users, items, ratings, lr, reg)

            # Log curves from the first member only (they are all similar).
            if member == 0:
                train_history.append(float(mse))
                if validation is not None:
                    valid_pred = np.sum(U[valid_users] * V[valid_items], axis=1)
                    validation_history.append(
                        float(np.mean((valid_ratings - np.clip(valid_pred, 1, 5)) ** 2))
                    )

        predictions += np.clip(U @ V.T, 1, 5) / members
        if member == 0:
            first_U, first_V = U, V

    predictions = np.clip(predictions, 1, 5).astype(np.float32)

    os.makedirs(os.path.join(reports, "pmf_factors"), exist_ok=True)
    np.save(os.path.join(reports, "pmf_factors", "U.npy"), first_U.astype(np.float32))
    np.save(os.path.join(reports, "pmf_factors", "V.npy"), first_V.astype(np.float32))
    np.save(os.path.join(reports, "pmf_predictions.npy"), predictions)

    metadata = {
        "factors": factors,
        "epochs": epochs,
        "members": members,
        "mse_history": train_history,
        "validation_mse_history": validation_history,
    }
    with open(os.path.join(reports, "pmf_factors", "training_metadata.json"), "w") as file:
        json.dump(metadata, file, indent=2)
    test_users = matrix.index.get_indexer(test.UserID)
    test_items = matrix.columns.get_indexer(test.MovieID)
    valid = (test_users >= 0) & (test_items >= 0)

    rmse = root_mean_squared_error(
    test.Rating.to_numpy()[valid],
    predictions[test_users[valid], test_items[valid]]
)

    return predictions, rmse, train_history, validation_history
