import os
import pandas as pd


def load_movielens_data(data_dir="data"):
    ratings = pd.read_csv(
        os.path.join(data_dir, "ratings.dat"), sep="::", engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"], encoding="latin-1",
    )
    movies = pd.read_csv(
        os.path.join(data_dir, "movies.dat"), sep="::", engine="python",
        names=["MovieID", "Title", "Genres"], encoding="latin-1",
    )
    users = pd.read_csv(
        os.path.join(data_dir, "users.dat"), sep="::", engine="python",
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"], encoding="latin-1",
    )
    ratings = ratings.dropna().drop_duplicates(["UserID", "MovieID"], keep="last")
    movies = movies.dropna().drop_duplicates("MovieID")
    users = users.dropna().drop_duplicates("UserID")
    return ratings, movies, users
