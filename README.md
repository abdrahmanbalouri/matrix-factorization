# Matrix Factorization Movie Recommender

Movie recommendations with **SVD** and **PMF** on the MovieLens 1M dataset, plus a Streamlit dashboard.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The files `ratings.dat`, `movies.dat`, and `users.dat` are in `data/`.

## Train and create reports

Open and run all cells in `Movie_Recommender_System.ipynb`. The notebook:

- explores the MovieLens data (rating distribution, activity, genre popularity),
- builds a reproducible 80/10/10 train/validation/test split with `random_state=42`,
- trains SVD and a regularized PMF ensemble,
- evaluates both with MSE/RMSE and compares against item-based collaborative filtering,
- analyzes global and local interpretability,
- selects two training-set users and one test-set user for recommendation examples,
- writes all required files to `processed/` and `reports/`.

Key outputs:

```text
processed/user_item_matrix.csv
reports/model_metrics.json
reports/svd_predictions.npy
reports/pmf_predictions.npy
reports/pmf_factors/U.npy
reports/pmf_factors/V.npy
reports/pmf_convergence.png
reports/rmse_comparison.png
reports/predicted_vs_actual.png
reports/user_comparison.png
reports/eda_overview.png
reports/user_accuracy_comparison.png
reports/recommendations_three_users.png
reports/user_<id>_recommendations.csv
reports/user_<id>_recommendations.png
```

## Run the dashboard

```bash
streamlit run app.py
```

Select a user ID to see their top-rated movies, SVD and PMF recommendations with reasons, a genre-match visualization, and a prediction comparison. Invalid IDs are handled with an error message.

## Results

On the held-out test set:

| Model | MSE | RMSE |
|---|---:|---:|
| SVD | 0.7897 | 0.8886 |
| PMF | 0.7119 | 0.8437 |

PMF improved RMSE by **5.06%** over SVD. An item-based collaborative filtering benchmark scored **0.9104 RMSE**.

## Method

- Ratings are split with `random_state=42` (80/10/10 train/validation/test).
- SVD is bias-centered with global, user, and item biases, uses `scipy.sparse.linalg.svds`, and is calibrated on validation data.
- PMF learns user and movie factors with regularized SGD and explicit user/item biases. The core model is simple:

  ```text
  pred(u, i) = global_mean + user_bias[u] + item_bias[i] + U[u] . V[i]
  ```

- Validation-based early stopping selects the best epoch, and a linear calibration is learned on validation pairs.
- A 10-member PMF ensemble is trained with different random seeds and its predictions are averaged to reduce SGD noise. The saved `U.npy` and `V.npy` are the 40 factors of the main model for interpretability; the averaged predictions are saved in `reports/pmf_predictions.npy`.
- Recommendations exclude movies already rated by the user.
- RMSE and MSE are calculated on known user/movie pairs in the test set.
- Overfitting is controlled with L2 regularization, validation-based early stopping, and calibration. The notebook compares training and validation curves and explains the stopping rule.

Dataset: [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/).
