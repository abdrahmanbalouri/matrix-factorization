"""Load the MovieLens 1M files into pandas DataFrames."""

import os

import pandas as pd


def load_movielens_data(data_dir="data"):
    """Read ratings.dat, movies.dat and users.dat and remove bad rows."""

    def read(filename, columns):
        path = os.path.join(data_dir, filename)
        return pd.read_csv(
            path,
            sep="::",
            engine="python",
            names=columns,
            encoding="latin-1",
        ).dropna()

    ratings = read(
        "ratings.dat", ["UserID", "MovieID", "Rating", "Timestamp"]
    ).drop_duplicates(["UserID", "MovieID"], keep="last")

    movies = read("movies.dat", ["MovieID", "Title", "Genres"]).drop_duplicates(
        "MovieID"
    )

    users = read(
        "users.dat", ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
    ).drop_duplicates("UserID")

    return ratings, movies, users
