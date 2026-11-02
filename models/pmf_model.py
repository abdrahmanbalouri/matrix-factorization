"""Simple Probabilistic Matrix Factorization (PMF).

Core model:

    pred(u, i) = mu + b_u[u] + b_i[i] + U[u] . V[i]

How it works:
1. ``PMF.fit`` learns U, V and the biases with regularized SGD.
2. Validation MSE picks the best epoch (early stopping) and a calibration line.
3. ``train_and_eval_pmf`` averages 10 PMF models trained with different seeds,
   which reduces SGD noise and gives a stable RMSE.
"""

import json
import os

import numpy as np
import pandas as pd
from numba import njit
from sklearn.metrics import root_mean_squared_error


@njit
def _epoch(U, V, b_u, b_i, mu, users, items, ratings, lr, reg):
    """One SGD epoch. Returns training MSE."""
    total = 0.0
    for row in range(len(users)):
        u = users[row]
        i = items[row]

        pred = mu + b_u[u] + b_i[i]
        for f in range(U.shape[1]):
            pred += U[u, f] * V[i, f]

        error = ratings[row] - pred
        total += error * error

        b_u[u] += lr * (error - reg * b_u[u])
        b_i[i] += lr * (error - reg * b_i[i])

        for f in range(U.shape[1]):
            old_u = U[u, f]
            old_i = V[i, f]
            U[u, f] = old_u + lr * (error * old_i - reg * old_u)
            V[i, f] = old_i + lr * (error * old_u - reg * old_i)

    return total / len(users)


class PMF:
    def __init__(self, n_factors=40, lr=0.01, reg=0.03, epochs=150,
                 random_state=42, patience=12, calibration=(1.0, 0.0), mu=0.0):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.random_state = random_state
        self.patience = patience
        self.calibration = calibration
        self.mu = mu
        self.best_epoch = epochs
        self.mse_history = []
        self.validation_mse_history = []

    def _pairs(self, data, user_map, item_map):
        users = np.fromiter((user_map.get(x, -1) for x in data.UserID),
                            np.int64, len(data))
        items = np.fromiter((item_map.get(x, -1) for x in data.MovieID),
                            np.int64, len(data))
        known = (users >= 0) & (items >= 0)
        return users[known], items[known], data.Rating.to_numpy()[known].astype(float)

    def _predict(self, users, items):
        return (self.mu + self.b_u[users] + self.b_i[items]
                + np.sum(self.U[users] * self.V[items], axis=1))

    def fit(self, train_data, n_users, n_items, user_map, item_map, validation_data=None):
        users, items, ratings = self._pairs(train_data, user_map, item_map)

        rng = np.random.default_rng(self.random_state)
        self.U = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.V = rng.normal(0, 0.1, (n_items, self.n_factors))
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)

        best_mse = np.inf
        best_state = None
        wait = 0
        lr = self.lr

        for epoch in range(self.epochs):
            mse = _epoch(self.U, self.V, self.b_u, self.b_i, self.mu,
                         users, items, ratings, lr, self.reg)
            self.mse_history.append(float(mse))

            if validation_data is not None:
                v_users, v_items, v_ratings = self._pairs(
                    validation_data, user_map, item_map)
                val_mse = np.mean(
                    (v_ratings - np.clip(self._predict(v_users, v_items), 1, 5)) ** 2)
                self.validation_mse_history.append(float(val_mse))

                if val_mse < best_mse:
                    best_mse = val_mse
                    self.best_epoch = epoch + 1
                    best_state = (self.U.copy(), self.V.copy(),
                                  self.b_u.copy(), self.b_i.copy())
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

            lr *= 0.97

        if best_state is not None:
            self.U, self.V, self.b_u, self.b_i = best_state
            v_users, v_items, v_ratings = self._pairs(
                validation_data, user_map, item_map)
            slope, intercept = np.polyfit(self._predict(v_users, v_items), v_ratings, 1)
            self.calibration = (float(slope), float(intercept))

        return self

    def predict_all(self):
        slope, intercept = self.calibration
        raw = self.mu + self.b_u[:, None] + self.b_i[None, :] + self.U @ self.V.T
        return np.clip(slope * raw + intercept, 1, 5).astype(np.float32)

    def save(self, directory, ensemble_size=1):
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "U.npy"), self.U.astype(np.float32))
        np.save(os.path.join(directory, "V.npy"), self.V.astype(np.float32))
        np.save(os.path.join(directory, "user_bias.npy"), self.b_u.astype(np.float32))
        np.save(os.path.join(directory, "item_bias.npy"), self.b_i.astype(np.float32))

        metadata = {
            "n_factors": self.n_factors,
            "ensemble_size": ensemble_size,
            "best_epoch": self.best_epoch,
            "learning_rate": self.lr,
            "regularization": self.reg,
            "random_state": self.random_state,
            "patience": self.patience,
            "calibration": list(self.calibration),
            "global_mean": float(self.mu),
            "mse_history": self.mse_history,
            "validation_mse_history": self.validation_mse_history,
        }
        with open(os.path.join(directory, "training_metadata.json"), "w") as file:
            json.dump(metadata, file, indent=2)


def train_and_eval_pmf(train_data, test_data, normalized_matrix, reports_dir="reports",
                       validation_data=None, ensemble_size=10):
    """Train a PMF ensemble, save predictions/factors, and return test RMSE."""
    user_map = {user_id: i for i, user_id in enumerate(normalized_matrix.index)}
    item_map = {item_id: i for i, item_id in enumerate(normalized_matrix.columns)}
    n_users, n_items = len(user_map), len(item_map)
    mu = float(train_data.Rating.mean())

    base = PMF(mu=mu).fit(
        train_data, n_users, n_items, user_map, item_map, validation_data)

    fit_data = (pd.concat([train_data, validation_data])
                if validation_data is not None else train_data)

    predictions = np.zeros((n_users, n_items), dtype=np.float32)
    for seed in range(ensemble_size):
        member = PMF(
            n_factors=base.n_factors, lr=base.lr, reg=base.reg,
            epochs=base.best_epoch, random_state=seed + 42,
            patience=base.patience, calibration=base.calibration, mu=mu,
        ).fit(fit_data, n_users, n_items, user_map, item_map)
        predictions += member.predict_all()
    predictions /= ensemble_size

    np.save(os.path.join(reports_dir, "pmf_predictions.npy"), predictions)
    base.save(os.path.join(reports_dir, "pmf_factors"), ensemble_size=ensemble_size)

    actuals, preds = [], []
    for row in test_data.itertuples():
        if row.UserID in user_map and row.MovieID in item_map:
            actuals.append(row.Rating)
            preds.append(predictions[user_map[row.UserID], item_map[row.MovieID]])

    rmse = root_mean_squared_error(actuals, preds)
    return predictions, rmse, actuals, preds, base
