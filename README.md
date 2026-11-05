# Matrix Factorization Movie Recommender

Two simple recommender models trained on MovieLens 1M, plus a Streamlit dashboard:

- **SVD** uses `scipy.sparse.linalg.svds`.
- **PMF** learns user and movie vectors with regularized SGD.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The raw files `ratings.dat`, `movies.dat` and `users.dat` must be in `data/`.

## Train and create the reports

Open `Movie_Recommender_System.ipynb` and run all cells. It saves everything to
`processed/` and `reports/`.

## Run the dashboard

```bash
streamlit run app.py
```

Enter a user ID (or pick one in the sidebar) to see:

- the user's top-rated movies,
- the top-10 SVD and PMF recommendations with a short reason,
- a chart showing why the PMF movies were recommended,
- a comparison of SVD and PMF predictions.

Invalid user IDs show an error instead of crashing.

## Results

Held-out test set (90/10 split, `random_state=42`):

| Model | RMSE |
|---|---:|
| SVD | 0.8950 |
| PMF | 0.8469 |
| Bias baseline | 0.9376 |

PMF improves RMSE by **5.37%** over SVD. Both models beat the bias baseline, and
the three reference thresholds from the audit are met:

- SVD RMSE <= 0.90
- PMF RMSE <= 0.85
- PMF improvement >= 5%

## How it works

1. `utils/data_loader.py` reads the three `.dat` files.
2. `utils/matrix_creation.py` splits the ratings (90/10) and builds a matrix
   centered by `global mean + user bias + movie bias`.
3. `models/svd_model.py` factorizes the centered matrix with `svds` and adds the
   baseline back.
4. `models/pmf_model.py` minimizes squared error with L2 regularization. It uses
   10% of the training data for validation, averages a 5-model ensemble, and logs
   the training/validation MSE so we can see convergence and justify stopping.
   `U.npy` and `V.npy` are the factors of the first ensemble member (used for
   interpretation); `pmf_predictions.npy` is the average of all five members.
5. `utils/recommendation.py` removes movies the user already rated and returns
   the top-N predicted movies.
6. `app.py` loads the saved predictions and shows the dashboard.

## Output files

```text
processed/user_item_matrix.csv
reports/model_metrics.json
reports/svd_predictions.npy
reports/pmf_predictions.npy
reports/pmf_factors/U.npy
reports/pmf_factors/V.npy
reports/pmf_factors/training_metadata.json
reports/pmf_convergence.png
reports/rmse_comparison.png
reports/predicted_vs_actual.png
reports/user_comparison.png
reports/top_recommendations.png
reports/matrix_sample.png
reports/user_<id>_recommendations.csv
```

Dataset: [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/).
