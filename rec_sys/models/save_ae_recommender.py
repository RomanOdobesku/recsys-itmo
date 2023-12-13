import math

import dill
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from rec_sys.models.ae_recommender import AERecommender
from rec_sys.models.torch_ae import AEModel

interactions_df = pd.read_csv("data/interactions.csv")
users_df = pd.read_csv("data/users.csv")
items_df = pd.read_csv("data/items.csv")

interactions_df = interactions_df[interactions_df["last_watch_dt"] < "2021-04-01"]
users_interactions_count_df = interactions_df.groupby(["user_id", "item_id"]).size().groupby("user_id").size()
print("# users: %d" % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[
    ["user_id"]
]
print("# users with at least 5 interactions: %d" % len(users_with_enough_interactions_df))

print("# of interactions: %d" % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(
    users_with_enough_interactions_df, how="right", left_on="user_id", right_on="user_id"
)
print("# of interactions from users with at least 5 interactions: %d" % len(interactions_from_selected_users_df))


def smooth_user_preference(x):
    return math.log(1 + x, 2)


interactions_full_df = (
    interactions_from_selected_users_df.groupby(["user_id", "item_id"])["watched_pct"]
    .sum()
    .apply(smooth_user_preference)
    .reset_index()
)
print("# of unique user/item interactions: %d" % len(interactions_full_df))

interactions_train_df, interactions_test_df = train_test_split(
    interactions_full_df, stratify=interactions_full_df["user_id"], test_size=0.20, random_state=42
)

print("# interactions on Train set: %d" % len(interactions_train_df))
print("# interactions on Test set: %d" % len(interactions_test_df))


# Indexing by personId to speed up the searches during evaluation
interactions_full_indexed_df = interactions_full_df.set_index("user_id")
interactions_train_indexed_df = interactions_train_df.set_index("user_id")
interactions_test_indexed_df = interactions_test_df.set_index("user_id")


def get_items_interacted(person_id, interactions_df):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc[person_id]["item_id"]
    return set(interacted_items if type(interacted_items) is pd.Series else [interacted_items])


# Constants
SEED = 42  # random seed for reproducibility
LR = 1e-3  # learning rate, controls the speed of the training
WEIGHT_DECAY = 0.01  # lambda for L2 reg. ()
NUM_EPOCHS = 3  # num training epochs (how many times each instance will be processed)
GAMMA = 0.9995  # learning rate scheduler parameter
BATCH_SIZE = 3000  # training batch size
EVAL_BATCH_SIZE = 3000  # evaluation batch size.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device to make the calculations on
ALPHA = 0.000002  # kl_divergence coefficient

interactions_train_df.append(interactions_test_indexed_df.reset_index())

total_df = interactions_train_df.append(interactions_test_indexed_df.reset_index())
total_df["user_id"], users_keys = total_df.user_id.factorize()
total_df["item_id"], items_keys = total_df.item_id.factorize()

train_encoded = total_df.iloc[: len(interactions_train_df)].values
test_encoded = total_df.iloc[len(interactions_train_df):].values

users_key_dict = {}
for i in range(len(users_keys)):
    users_key_dict[users_keys[i]] = i

items_key_dict = {}
for i in range(len(items_keys)):
    items_key_dict[items_keys[i]] = i

shape = [int(total_df["user_id"].max() + 1), int(total_df["item_id"].max() + 1)]
X_train = csr_matrix((train_encoded[:, 2], (train_encoded[:, 0], train_encoded[:, 1])), shape=shape).toarray()
X_test = csr_matrix((test_encoded[:, 2], (test_encoded[:, 0], test_encoded[:, 1])), shape=shape).toarray()


model = AEModel(device=DEVICE)
model.load_state_dict(torch.load("models/torch_ae.ckpt"))

with torch.no_grad():
    X_pred_total, mu, sigma = model(torch.Tensor(X_train + X_test).to(DEVICE))

ae_recommender_model = AERecommender(X_pred_total, X_train, X_test, users_key_dict, items_keys)


with open("models/autoencoder.dill", "wb") as model_file:
    dill.dump(ae_recommender_model, model_file)

print("done")
