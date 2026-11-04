import json
import os

import numpy as np
import pandas as pd
from numba import njit
from sklearn.metrics import root_mean_squared_error


@njit
def _epoch(U, V, u, i, y, lr, reg):
    total = 0.0
    for k in range(len(u)):
        error = y[k]
        for f in range(U.shape[1]):
            error -= U[u[k], f] * V[i[k], f]
        total += error * error
        for f in range(U.shape[1]):
            ou, ov = U[u[k], f], V[i[k], f]
            U[u[k], f] = ou + lr * (error * ov - reg * ou)
            V[i[k], f] = ov + lr * (error * ou - reg * ov)
    return total / len(u)


class PMF:
    def __init__(self, n_factors=20, lr=0.005, reg=0.05, epochs=80, random_state=42):
        self.n_factors, self.lr, self.reg, self.epochs = n_factors, lr, reg, epochs
        self.random_state = random_state
        self.calibration = (1.0, 0.0)
        self.best_epoch = epochs

    def _ratings(self, data, user_map, item_map):
        users = np.fromiter((user_map.get(v, -1) for v in data.UserID), dtype=np.int64, count=len(data))
        items = np.fromiter((item_map.get(v, -1) for v in data.MovieID), dtype=np.int64, count=len(data))
        valid = (users >= 0) & (items >= 0)
        return users[valid], items[valid], data.Rating.to_numpy()[valid].astype(float)

    def fit(self, train_data, n_users, n_items, user_map, item_map, validation_data=None):
        u, i, y = self._ratings(train_data, user_map, item_map)
        rng = np.random.default_rng(self.random_state)
        self.U = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.V = rng.normal(0, 0.1, (n_items, self.n_factors))
        best, wait = np.inf, 0
        best_u, best_v = self.U.copy(), self.V.copy()
        self.mse_history, self.validation_mse_history = [], []
        for epoch in range(self.epochs):
            self.mse_history.append(float(_epoch(self.U, self.V, u, i, y, self.lr, self.reg)))
            if validation_data is None:
                self.best_epoch = epoch + 1
                continue
            vu, vi, vy = self._ratings(validation_data, user_map, item_map)
            raw = np.sum(self.U[vu] * self.V[vi], axis=1)
            val_mse = float(np.mean((vy - np.clip(raw, 1, 5)) ** 2))
            self.validation_mse_history.append(val_mse)
            if val_mse < best:
                best, wait = val_mse, 0
                self.best_epoch = epoch + 1
                best_u, best_v = self.U.copy(), self.V.copy()
            else:
                wait += 1
            if wait >= 8:
                break
        if validation_data is not None:
            self.U, self.V = best_u, best_v
            vu, vi, vy = self._ratings(validation_data, user_map, item_map)
            raw = np.sum(self.U[vu] * self.V[vi], axis=1)
            slope, intercept = np.polyfit(raw, vy, 1)
            self.calibration = (float(slope), float(intercept))
        return self

    def predict_all(self):
        slope, intercept = self.calibration
        return np.clip(slope * (self.U @ self.V.T) + intercept, 1, 5).astype(np.float32)

    def save_factors(self, directory):
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "U.npy"), self.U.astype(np.float32))
        np.save(os.path.join(directory, "V.npy"), self.V.astype(np.float32))
        meta = {
            "n_factors": self.n_factors, "best_epoch": self.best_epoch,
            "learning_rate": self.lr, "regularization": self.reg,
            "random_state": self.random_state, "calibration": list(self.calibration),
            "mse_history": self.mse_history,
            "validation_mse_history": self.validation_mse_history,
        }
        with open(os.path.join(directory, "training_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)


def train_and_eval_pmf(train_data, test_data, normalized_matrix, reports_dir="reports",
                       validation_data=None, ensemble_size=10):
    user_map = {uid: i for i, uid in enumerate(normalized_matrix.index)}
    item_map = {mid: i for i, mid in enumerate(normalized_matrix.columns)}
    pmf = PMF().fit(train_data, len(user_map), len(item_map), user_map, item_map, validation_data)
    best_epoch, calibration = pmf.best_epoch, pmf.calibration
    fit_data = pd.concat([train_data, validation_data]) if validation_data is not None else train_data
    Us, Vs = [], []
    for seed in range(ensemble_size):
        member = PMF(random_state=seed + 42, epochs=best_epoch).fit(
            fit_data, len(user_map), len(item_map), user_map, item_map)
        Us.append(member.U / (ensemble_size ** 0.5))
        Vs.append(member.V / (ensemble_size ** 0.5))
    pmf.U = np.concatenate(Us, axis=1)
    pmf.V = np.concatenate(Vs, axis=1)
    pmf.calibration = calibration
    pmf.n_factors = pmf.U.shape[1]
    pmf.save_factors(os.path.join(reports_dir, "pmf_factors"))
    predictions = pmf.predict_all()
    actuals, preds = [], []
    for row in test_data.itertuples():
        if row.UserID in user_map and row.MovieID in item_map:
            actuals.append(row.Rating)
            preds.append(predictions[user_map[row.UserID], item_map[row.MovieID]])
    rmse = root_mean_squared_error(actuals, preds)
    return predictions, rmse, actuals, preds, pmf
