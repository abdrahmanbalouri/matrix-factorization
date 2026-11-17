# Matrix Factorization Movie Recommender

An end-to-end collaborative-filtering project built on the **MovieLens 1M** dataset. It trains two matrix-factorization models — **SVD** and **Probabilistic Matrix Factorization (PMF)** — evaluates them on a held-out test set, explains the recommendations, and exposes everything through an interactive **Streamlit** dashboard.

---

## Table of contents

1. [What this project does](#what-this-project-does)
2. [Repository structure](#repository-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [How to run everything](#how-to-run-everything)
6. [Methodology, step by step](#methodology-step-by-step)
7. [Configuration / hyperparameters](#configuration--hyperparameters)
8. [Results](#results)
9. [Generated files](#generated-files)
10. [The Streamlit dashboard](#the-streamlit-dashboard)
11. [Troubleshooting](#troubleshooting)
12. [Dataset license and citation](#dataset-license-and-citation)

---

## What this project does

MovieLens 1M contains about **1,000,000 ratings** from **6,040 users** on roughly **3,900 movies**, which makes the user–item matrix about **95% empty**. Storing or comparing every rating directly is useless because almost all user/movie pairs are unknown. Matrix factorization solves this by learning a small number of hidden ("latent") factors that explain the observed ratings, then predicting the missing ones.

This project:

- loads and cleans the three raw `.dat` files;
- builds a rating matrix and centers it with a **bias baseline** (global mean + user bias + movie bias);
- trains **SVD** (`scipy.sparse.linalg.svds`, `k = 30`) on the centered matrix;
- trains **PMF** with regularized SGD (`numba`-accelerated), a 10%-validation curve, and a 5-model ensemble;
- computes held-out **RMSE** for both models and a bias-only benchmark;
- generates top-10 recommendations for sample users and explains them with genre overlap / latent factors;
- saves all metrics, plots, factors, and prediction matrices to `reports/` and `processed/`;
- serves the results through `app.py` (Streamlit), where you can type any user ID and see recommendations live.

---

## Repository structure

```text
matrix-factorization/
├── Movie_Recommender_System.ipynb   # Main pipeline: EDA -> train -> evaluate -> plots -> reports
├── app.py                            # Streamlit dashboard (reads the saved artifacts)
├── requirements.txt                  # Pinned Python dependencies
├── data/                             # Raw MovieLens 1M files (place them here)
│   ├── README                        # Official dataset description / license
│   ├── ratings.dat                   # UserID::MovieID::Rating::Timestamp
│   ├── movies.dat                    # MovieID::Title::Genres
│   └── users.dat                     # UserID::Gender::Age::Occupation::Zip-code
├── utils/
│   ├── data_loader.py                # load_movielens_data(): read + clean the .dat files
│   ├── matrix_creation.py            # train/test split, pivot, bias baseline, normalized matrix
│   └── recommendation.py             # top-N unseen movies, favorite genres, save CSVs
├── models/
│   ├── svd_model.py                  # train_svd_model(): truncated SVD + baseline
│   └── pmf_model.py                  # train_pmf(): regularized SGD, ensemble, factors
├── processed/
│   └── user_item_matrix.csv          # Saved normalized residual matrix
├── reports/                          # All generated artifacts (see section 9)
│   ├── model_metrics.json
│   ├── svd_predictions.npy
│   ├── pmf_predictions.npy
│   ├── pmf_factors/
│   │   ├── U.npy
│   │   ├── V.npy
│   │   └── training_metadata.json
│   ├── *.png                         # Convergence, RMSE, predictions, recommendations plots
│   └── user_<id>_recommendations.csv
└── venv/                             # Local virtual environment (git-ignored)
```

---

## Requirements

- **Python 3.12** (the code was developed and tested with 3.12.3).
- The raw MovieLens 1M files in `data/` (see below).
- A Jupyter environment to run the notebook (Jupyter / `nbconvert` / `ipykernel`).
- The pinned libraries from `requirements.txt`:

| Package | Version | Used for |
|---|---|---|
| numpy | 2.5.4 | arrays, factors, predictions |
| pandas | 2.3.3 | data loading, pivoting, results |
| scipy | 1.18.1 | `svds` (truncated SVD) |
| scikit-learn | 1.9.1 | `train_test_split`, RMSE |
| numba | 0.68.1 | JIT-compiled SGD loop |
| matplotlib | 3.11.2 | all report figures |
| streamlit | 1.66.1 | interactive dashboard |

> Note: Jupyter is **not** listed in `requirements.txt`; if you do not already have it, install it as described below.

---

## Installation

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install the pinned dependencies
pip install -r requirements.txt

# 3. (Only if you don't already have Jupyter)
pip install jupyter nbconvert ipykernel
```

The dataset files must be placed in `data/`:

```text
data/ratings.dat
data/movies.dat
data/users.dat
```

They use `::` as separator and `latin-1` encoding, and are read with column names
`UserID/MovieID/Rating/Timestamp`, `MovieID/Title/Genres`, and
`UserID/Gender/Age/Occupation/Zip-code` respectively. The loader drops duplicate
ratings (keeping the last), duplicate movies/users, and any missing rows.

---

## How to run everything

The project has **two stages**: first generate the artifacts with the notebook, then launch the dashboard. The dashboard cannot work until stage 1 has been run at least once.

### Stage 1 — Train the models and generate reports

**Option A (recommended): open the notebook in Jupyter and run all cells**

```bash
jupyter notebook            # or: jupyter lab
# Open Movie_Recommender_System.ipynb -> Kernel > Restart & Run All
```

**Option B: execute it headlessly with nbconvert**

```bash
jupyter nbconvert --to notebook --execute Movie_Recommender_System.ipynb \
  --output Movie_Recommender_System.executed.ipynb
```

Both options run the same 8 sections in order:

1. load and inspect the data,
2. build and save the normalized matrix,
3. train SVD,
4. train PMF,
5. compare the models,
6. interpret latent factors,
7. build recommendations for 3 sample users,
8. compare SVD vs PMF for a single user and summarize.

When it finishes, check that `processed/user_item_matrix.csv`, the files in
`reports/`, and `model_metrics.json` exist.

### Stage 2 — Launch the dashboard

```bash
streamlit run app.py
```

Then open the local URL printed in the terminal (usually `http://localhost:8501`).
Type a valid User ID in the sidebar (e.g. `1680`, `1941`, `4169` — or any ID between
1 and 6040 that appears in the data) to see personalized recommendations.

---

## Methodology, step by step

### 1. Data loading and cleaning — `utils/data_loader.py`

`load_movielens_data("data")` reads the three `.dat` files with `sep="::"`,
`engine="python"` and `encoding="latin-1"`, drops NaNs, keeps only the last
rating per `(UserID, MovieID)`, and de-duplicates movies and users.

The notebook then reports the size, sparsity (~95%), and the rating distribution
(4 and 5 stars dominate — the data is positively skewed).

### 2. Normalized user–item matrix — `utils/matrix_creation.py`

`create_and_save_interaction_matrix(ratings)`:

- splits ratings **90/10** with `random_state=42` into train and test;
- pivots the training ratings into a `users × movies` matrix;
- computes a **baseline**:

  ```text
  baseline = global_mean + user_bias + movie_bias
  user_bias  = mean rating of user  - global_mean
  movie_bias = mean rating of movie - global_mean
  ```

- subtracts the baseline and fills missing entries with `0`, producing the
  centered **residual** matrix. Modeling residuals instead of raw ratings is
  easier for both SVD and PMF because the average taste is removed first;
- saves the rounded matrix to `processed/user_item_matrix.csv`.

### 3. SVD — `models/svd_model.py`

`train_svd_model(normalized_matrix, baseline, test, k=30)`:

- factorizes the centered matrix with `scipy.sparse.linalg.svds(k=30,
  random_state=42)`;
- reconstructs ratings as `(U * sigma) @ Vt + baseline` and clips them to the
  valid rating range `[1, 5]`;
- saves `reports/svd_predictions.npy`;
- computes RMSE against the held-out test rows (only users/movies present in the
  matrix are scored).

### 4. PMF — `models/pmf_model.py`

`train_pmf(train, test, matrix, validation=None, ...)`:

- maps user and movie IDs to row/column positions;
- initializes user factors `U` (`n_users × 20`) and movie factors `V`
  (`n_movies × 20`) from a normal distribution `N(0, 0.1)`;
- runs **100 epochs** of SGD over all training ratings, minimizing squared error
  with **L2 regularization** (`reg = 0.05`, `lr = 0.005`). The hot loop is
  compiled with `numba.njit`;
- uses **10% of the training set as validation** and records training/validation
  MSE per epoch. Training MSE keeps falling while validation MSE flattens, which
  justifies stopping at 100 epochs instead of overfitting;
- averages an **ensemble of 5 members** (random seeds `42..46`) for stability;
- saves the factors of the **first** ensemble member (`U.npy`, `V.npy`) for
  interpretation, the averaged `pmf_predictions.npy`, and
  `training_metadata.json` with the hyperparameters and MSE curves;
- computes RMSE on the held-out test set.

### 5. Benchmark and metrics — notebook section 5

A simple bias-only predictor (global mean + user bias + movie bias) is evaluated
as a collaborative-filtering baseline. All scores are written to
`reports/model_metrics.json`.

### 6. Interpretation and recommendations — `utils/recommendation.py`

- `generate_recommendations(user_id, model, top_n=10)` takes the tuple
  `(predictions, matrix, movies, ratings)`, removes movies the user already
  rated, joins titles/genres, sorts by predicted rating, and returns the top 10.
- `favorite_genres(user_id, ratings, movies, n=3)` finds the user's three most
  frequent genres from their history.
- `save_user_recommendations()` writes each sample user's combined SVD+PMF
  recommendations to `reports/user_<id>_recommendations.csv`.
- `V.npy` columns are treated as learned movie communities: the notebook prints
  the top-5 movies per factor for the first three factors (a **global**
  explanation of what PMF learned).
- The dashboard colors a recommended movie **green** when it matches one of the
  user's favorite genres, otherwise **gray** (attributed to latent factors).

### 7. Plots — notebook sections 2, 4, 5, 7, 8

`reports/` receives figures for the normalized matrix sample, PMF convergence,
held-out RMSE comparison, predicted-vs-actual scatter, per-user top
recommendations, and the per-user SVD-vs-PMF prediction rows.

---

## Configuration / hyperparameters

| Component | Parameter | Value | Where |
|---|---|---|---|
| Split | `test_size` / `random_state` | `0.1` / `42` | `matrix_creation.py` |
| SVD | `k` / `random_state` | `30` / `42` | `svd_model.py` |
| PMF | `factors` | `20` | `pmf_model.py` |
| PMF | `epochs` | `100` | `pmf_model.py` |
| PMF | learning rate `lr` | `0.005` | `pmf_model.py` |
| PMF | regularization `reg` | `0.05` | `pmf_model.py` |
| PMF | `members` (ensemble) | `5` | `pmf_model.py` |
| PMF | validation split | `0.1` of train | notebook section 4 |
| Recommendations | `top_n` | `10` | `recommendation.py` |
| Genres | `n` | `3` | `recommendation.py` |

They are function defaults, so you can override them from the notebook without
editing the source files.

---

## Results

Held-out test set (90/10 split, `random_state=42`); values come from
`reports/model_metrics.json`:

| Model | RMSE | MSE |
|---|---:|---:|
| SVD | 0.8950 | 0.8010 |
| **PMF** | **0.8469** | **0.7173** |
| Bias baseline | 0.9376 | — |

PMF improves RMSE by **5.37%** over SVD. Both models beat the bias baseline, and
the three reference thresholds from the audit are met:

- SVD RMSE ≤ 0.90,
- PMF RMSE ≤ 0.85,
- PMF improvement ≥ 5%.

Notes on the comparison: SVD keeps only the strongest directions of the centered
matrix, so its top scores are often pushed to `5.0`; PMF spreads predictions more
smoothly, which is why the two models can share very few top-10 movies for the
same user. Per-user accuracy depends on history size and consistency — noisier
histories are harder for the latent factors to fit.

---

## Generated files

After running the notebook:

```text
processed/
├── user_item_matrix.csv              # normalized residuals (saved for the dashboard)

reports/
├── model_metrics.json                # RMSE/MSE for SVD, PMF, baseline + improvement
├── svd_predictions.npy               # full SVD prediction matrix
├── pmf_predictions.npy               # ensemble-averaged PMF prediction matrix
├── pmf_factors/
│   ├── U.npy                         # user factors, first ensemble member
│   ├── V.npy                         # movie factors, first ensemble member
│   └── training_metadata.json        # hyperparameters + train/validation MSE curves
├── matrix_sample.png                 # heatmap of normalized residuals
├── pmf_convergence.png               # training vs validation MSE per epoch
├── rmse_comparison.png               # bar chart of held-out RMSE
├── predicted_vs_actual.png           # scatter plots for both models
├── top_recommendations.png           # per-user PMF top-10 (green = favorite genre)
├── user_comparison.png               # SVD vs PMF prediction rows for one user
└── user_<id>_recommendations.csv      # 10 SVD + 10 PMF rows for a sample user
```

---

## The Streamlit dashboard

`app.py` is a thin read-only viewer over the generated artifacts:

- loads the raw data, the saved matrix, and both prediction matrices (cached with
  `@st.cache_data`);
- asks for a User ID in the sidebar and validates it — non-integer input and
  unknown IDs show a friendly error instead of crashing;
- shows the user's **top-5 rated movies** and favorite genres;
- shows side-by-side **top-10 SVD and PMF recommendations** (title, genres,
  predicted rating);
- plots **why the PMF movies were recommended** (green = favorite-genre match,
  gray = latent-factor match);
- plots **SVD vs PMF predictions** for the first 100 movies.

If the artifacts are missing, the app displays: *"Could not load the models. Run
the notebook first."* — follow [Stage 1](#stage-1--train-the-models-and-generate-reports).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError` for `ratings.dat` etc. | Put the three MovieLens 1M files in `data/` (see [Installation](#installation)). |
| `Could not load the models. Run the notebook first.` | Run all notebook cells (or nbconvert) before starting Streamlit. |
| `Invalid user ID` / `User ID must be a whole number.` | Enter an integer that exists in the dataset (1–6040, e.g. 1680). |
| `python: command not found` | Use `python3`, and activate the virtual environment first. |
| Jupyter not found | `pip install jupyter nbconvert ipykernel`. |
| NumPy 2.x / Numba compatibility errors | Install exactly the versions in `requirements.txt` inside a fresh venv. |
| Slow PMF training | It is a ~100-epoch SGD loop over ~900k ratings × 5 members; let it finish once, the artifacts are then reused by the dashboard. |

---

## Dataset license and citation

Dataset: [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) (GroupLens
Research, University of Minnesota). It may be used for research purposes under
the conditions in `data/README`; see that file for the full license.

If you publish results using this dataset, cite:

> F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History
> and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4,
> Article 19, 19 pages. DOI: 10.1145/2827872.
