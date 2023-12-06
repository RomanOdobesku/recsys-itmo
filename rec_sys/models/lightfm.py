from copy import deepcopy

import dill
import numpy as np


class LightFM:
    def __init__(self, dataset, model=None):
        self.model = model
        self.dataset = deepcopy(dataset)

    def get_vectors(self):
        user_embeddings, item_embeddings = None, None
        if self.model and self.dataset:
            user_embeddings, item_embeddings = self.model.get_vectors(self.dataset)
        _, augmented_item_embeddings = self.augment_inner_product(item_embeddings)
        extra_zero = np.zeros((user_embeddings.shape[0], 1))
        augmented_user_embeddings = np.append(user_embeddings, extra_zero, axis=1)
        return augmented_item_embeddings, augmented_user_embeddings

    @staticmethod
    def augment_inner_product(factors):
        normed_factors = np.linalg.norm(factors, axis=1)
        max_norm = normed_factors.max()

        extra_dim = np.sqrt(max_norm**2 - normed_factors**2).reshape(-1, 1)
        augmented_factors = np.append(factors, extra_dim, axis=1)
        return max_norm, augmented_factors

    def load(self, path="../models/lightfm.dill"):
        with open(path, "rb") as f:
            self.model = dill.load(f)

    def predict(self, X, K_RECOS=10):
        if self.dataset is None:
            raise ValueError("Dataset not found")
        if self.model is None:
            raise ValueError("Model not found")
        try:
            item_ids = self.model.recommend(
                users=[X],
                dataset=self.dataset,
                k=K_RECOS,
                filter_viewed=False,
            )["item_id"].to_numpy()
            return item_ids
        except ValueError:
            return []
