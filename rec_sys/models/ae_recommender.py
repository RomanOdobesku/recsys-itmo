import numpy as np

class AERecommender:

    MODEL_NAME = 'Autoencoder'

    def __init__(self, X_preds, X_train_and_val, X_test, users_key_dict, items_keys):
        super(AERecommender, self).__init__()
        self.X_preds = X_preds.cpu().detach().numpy()
        self.X_total = X_train_and_val + X_test
        self.users_key_dict = users_key_dict
        self.items_keys = items_keys

    def get_model_name(self):
        return self.MODEL_NAME

    def get_items_to_select_idx(self, user_id):
        all_nonzero = np.argwhere(self.X_total[user_id] > 0).ravel()
        select_from = np.setdiff1d(np.arange(self.X_total.shape[1]), all_nonzero)
        return select_from

    def recommend(self, user_id, topn=10, verbose=False):
        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        try:
            inner_user_id = self.users_key_dict[user_id]
            items_to_select_idx = self.get_items_to_select_idx(inner_user_id)
            user_preds = self.X_preds[inner_user_id][items_to_select_idx]
            items_idx = items_to_select_idx[np.argsort(-user_preds)[:topn]]

            items = [self.items_keys[item_idx] for item_idx in items_idx]
            return items
        except KeyError:
            return []