import dill

from rec_sys.data.utils import load_dataset
from rec_sys.models.random_model import RandomModel

interactions_df, users_df, items_df = load_dataset(path="data/")

# Init models

model_knn = None
with open("models/userknn_tfidf_50.dill", "rb") as f:
    model_knn = dill.load(f)

model_popular = None
with open("models/popular.dill", "rb") as f:
    model_popular = dill.load(f)

random_model = RandomModel()
