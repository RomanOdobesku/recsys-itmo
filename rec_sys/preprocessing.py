import numpy as np
import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset


def load_dataset(path: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    interactions = pd.read_csv(f"{path}/interactions.csv")
    users = pd.read_csv(f"{path}/users.csv")
    items = pd.read_csv(f"{path}/items.csv")

    return interactions, users, items


class Preprocessing:
    def __init__(self, users: pd.DataFrame, items: pd.DataFrame, interactions: pd.DataFrame):
        train, test = self.__preprocess(users, items, interactions)
        test = self.__filter_cold_users(train, test)
        users, items, user_features = self.__prepare_features(users, items, train)
        item_features = self.__explode_genres(items)
        self.dataset = Dataset.construct(
            interactions_df=train,
            user_features_df=user_features,
            cat_user_features=["sex", "age", "income"],
            item_features_df=item_features,
            cat_item_features=["genre", "content_type"],
        )

    @staticmethod
    def __preprocess(users, items, interactions):
        Columns.Datetime = "last_watch_dt"
        interactions.drop(interactions[interactions[Columns.Datetime].str.len() != 10].index, inplace=True)
        interactions[Columns.Datetime] = pd.to_datetime(interactions[Columns.Datetime], format="%Y-%m-%d")
        max_date = interactions[Columns.Datetime].max()
        interactions[Columns.Weight] = np.where(interactions["watched_pct"] > 10, 3, 1)
        train = interactions[interactions[Columns.Datetime] < max_date - pd.Timedelta(days=7)].copy()
        test = interactions[interactions[Columns.Datetime] >= max_date - pd.Timedelta(days=7)].copy()
        train.drop(train.query("total_dur < 300").index, inplace=True)
        return train, test

    @staticmethod
    def __filter_cold_users(train, test):
        # фильтруем холодных пользователей из теста
        cold_users = set(test[Columns.User]) - set(train[Columns.User])
        test.drop(test[test[Columns.User].isin(cold_users)].index, inplace=True)
        return test

    @staticmethod
    def __prepare_features(users, items, train):
        users.fillna("Unknown", inplace=True)
        users = users.loc[users[Columns.User].isin(train[Columns.User])].copy()
        user_features_frames = []
        for feature in ["sex", "age", "income"]:
            feature_frame = users.reindex(columns=[Columns.User, feature])
            feature_frame.columns = ["id", "value"]
            feature_frame["feature"] = feature
            user_features_frames.append(feature_frame)
        user_features = pd.concat(user_features_frames)
        # Item features
        items = items.loc[items[Columns.Item].isin(train[Columns.Item])].copy()
        return users, items, user_features

    @staticmethod
    def __explode_genres(items):
        # Genre
        # Explode genres to flatten table
        items["genre"] = items["genres"].str.lower().str.replace(", ", ",", regex=False).str.split(",")
        genre_feature = items[["item_id", "genre"]].explode("genre")
        genre_feature.columns = ["id", "value"]
        genre_feature["feature"] = "genre"
        genre_feature["value"].value_counts()
        content_feature = items.reindex(columns=[Columns.Item, "content_type"])
        content_feature.columns = ["id", "value"]
        content_feature["feature"] = "content_type"
        item_features = pd.concat((genre_feature, content_feature))
        return item_features

    def get_dataset(self):
        return self.dataset
