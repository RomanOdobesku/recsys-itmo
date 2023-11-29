import numpy as np
import pandas as pd
from rectools.dataset import Dataset
from rectools.models.popular import PopularModel


class Popular:
    def __init__(self, interactions: pd.DataFrame = None):
        self.dataset = Dataset.construct(
            interactions_df=interactions,
            user_features_df=None,
            item_features_df=None,
        )
        popular = interactions.merge(
            interactions.groupby("item_id").size().reset_index(name="user_item_repeats"), on="item_id"
        )
        self.popular_items = (
            popular.sort_values("user_item_repeats", ascending=False)[["item_id"]].drop_duplicates().to_numpy()[:, 0]
        )
        self.popular_model = PopularModel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise Exception("Model is not fitted")
        recommends = []
        try:
            recommends = self.popular_model.recommend(X[0], dataset=self.dataset, k=10, filter_viewed=False)[
                ["item_id"]
            ].to_numpy()[:, 0]
        except KeyError:
            recommends = self.popular_items
        return recommends

    def fit(self, dataset: Dataset):
        self.popular_model.fit(dataset)
        self.is_fitted = True
