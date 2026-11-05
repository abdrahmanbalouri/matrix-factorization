import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, BaselineOnly


def create_and_save_interaction_matrix(ratings, processed_dir="processed", test_size=0.2):

    os.makedirs(processed_dir, exist_ok=True)

    # 1. Split data
    train_data, test_data = train_test_split(
        ratings,
        test_size=test_size,
        random_state=42
    )

    # 2. Get users and movies
    users = sorted(ratings.UserID.unique())
    movies = sorted(ratings.MovieID.unique())

    # 3. Create interaction matrix
    interaction_matrix = train_data.pivot(
        index="UserID",
        columns="MovieID",
        values="Rating"
    ).reindex(index=users, columns=movies)

    # 4. Prepare data for Surprise
    reader = Reader(
        rating_scale=(
            float(ratings.Rating.min()),
            float(ratings.Rating.max())
        )
    )

    data = Dataset.load_from_df(
        train_data[["UserID", "MovieID", "Rating"]],
        reader
    )

    trainset = data.build_full_trainset()

    # 5. Calculate biases using Surprise
    model = BaselineOnly(
        bsl_options={
            "method": "als",
            "n_epochs": 5,
            "reg_u": 10,
            "reg_i": 10
        }
    )

    model.fit(trainset)

    # 6. Get global mean
    global_mean = trainset.global_mean
    print(global_mean, "global_mean")

    # 7. Get user and movie biases
    user_bias = np.zeros(len(users))
    item_bias = np.zeros(len(movies))

    for i, user in enumerate(users):
        try:
            user_bias[i] = model.bu[trainset.to_inner_uid(user)]
        except ValueError:
            pass

    for i, movie in enumerate(movies):
        try:
            item_bias[i] = model.bi[trainset.to_inner_iid(movie)]
        except ValueError:
            pass

    # 8. Build baseline matrix
    baseline = (
        global_mean
        + user_bias[:, None]
        + item_bias[None, :]
    )

    # 9. Normalize matrix
    normalized_matrix = (
        interaction_matrix - baseline
    ).fillna(0)

    # 10. Save matrix
    normalized_matrix.round(2).to_csv(
        os.path.join(processed_dir, "user_item_matrix.csv")
    )

    return (
        train_data,
        test_data,
        interaction_matrix,
        normalized_matrix,
        baseline
    )