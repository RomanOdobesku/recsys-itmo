import pandas as pd
from rectools import Columns


def load_dataset(path: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    interactions = pd.read_csv(f"{path}/interactions.csv")
    users = pd.read_csv(f"{path}/users.csv")
    items = pd.read_csv(f"{path}/items.csv")
    interactions.rename(
        columns={"last_watch_dt": Columns.Datetime, "total_dur": Columns.Weight},
        inplace=True,
    )

    interactions["datetime"] = pd.to_datetime(interactions["datetime"])
    return interactions, users, items
