import typing

import numpy as np
import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset


class VisualAnalysis:
    def __init__(
        self,
        trained_model,
        interactions_df: pd.DataFrame,
        user_ids: typing.Union[typing.List[int], np.ndarray, Columns.User],
        item_data: pd.DataFrame,
    ) -> None:
        self.model = trained_model
        self.interactions_df = interactions_df
        self.user_ids = user_ids
        self.item_data = item_data

    def user_data_enrichment(self, data: pd.DataFrame, user: str, columns: list) -> pd.DataFrame:
        """
        Enriches user data with additional information from interactions_df

        Parameters:
            data (pandas.DataFrame): The input data to be enriched
            user (str): The user ID to filter the interactions
            columns (list): The list of columns to include in the enriched data

        Returns:
            pandas.DataFrame: The enriched user data
        """
        enriched_data = (
            data.merge(
                self.interactions_df[self.interactions_df.user_id == user]
                .groupby("item_id")
                .size()
                .reset_index(name="user_views"),
                on="item_id",
                how="left",
            )
            .fillna({"user_views": 0})
            .astype({"user_views": int})
            .merge(
                self.interactions_df.groupby("item_id").size().reset_index(name="total_views"), on="item_id", how="left"
            )
            .fillna({"total_views": 0})
            .astype({"total_views": int})
            .merge(self.item_data, on="item_id")
            .drop_duplicates()[columns]
        )
        return enriched_data

    def visualize(self):
        reco = self.model.recommend(
            users=self.user_ids,
            dataset=Dataset.construct(self.interactions_df),
            k=10,
            filter_viewed=True,
        )

        user_history_columns = ["title", "genres", "datetime", "user_views", "total_views"]
        user_recommendations_columns = ["title", "genres", "user_views", "total_views", "score", "rank"]

        result = {}

        for user in self.user_ids:
            user_recommendations = reco[reco.user_id == user]
            user_interactions = self.interactions_df[self.interactions_df.user_id == user]

            user_recommendations = self.user_data_enrichment(user_recommendations, user, user_recommendations_columns)
            user_history = self.user_data_enrichment(user_interactions, user, user_history_columns)

            result[user] = {
                "recommendations": user_recommendations,
                "user_history": user_history,
            }

        return result
