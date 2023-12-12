import dill
import pickle
import numpy as np

from rec_sys.models.faiss import FAISS
from rec_sys.models.lightfm import LightFM
from rec_sys.models.random_model import RandomModel
from rec_sys.models.ae_recommender import AERecommender
from rec_sys.preprocessing import Preprocessing, load_dataset

interactions_df, users_df, items_df = load_dataset(path="data/")

# Init models

model_popular = None
with open("models/popular.dill", "rb") as f:
    model_popular = dill.load(f)

random_model = RandomModel()

preprocessing_dataset = Preprocessing(users=users_df.copy(), items=items_df.copy(), interactions=interactions_df.copy())
dataset = preprocessing_dataset.get_dataset()


with open("models/tuned_lightfm.dill", "rb") as f:
    lightfm_model = dill.load(f)

lightfm = LightFM(dataset=dataset, model=lightfm_model)
aug_item_emb, aug_user_emb = lightfm.get_vectors()
faiss = FAISS(aug_user_emb, aug_item_emb)

ae_path = 'models/autoencoder.dill'
with open(ae_path, 'rb') as f:
    ae_recommender = dill.load(f)


def extend_to_k_recs(reco, user_id, k_recs):
    reco = np.squeeze(reco)
    reco_popular = model_popular.predict([[user_id]], k=20)
    reco = np.concatenate([reco, reco_popular])
    reco_set = []
    for r in reco:
        if r in reco_set:
            continue
        if len(reco_set) == k_recs:
            break
        reco_set.append(r)
        if len(reco_set) == 10:
            break
    reco = np.array(reco_set)
    return reco
