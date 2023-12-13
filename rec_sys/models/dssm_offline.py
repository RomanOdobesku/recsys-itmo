import pandas as pd


class DSSM_Offline:
    def __init__(self, path_data) -> None:
        self.data = pd.read_csv(path_data)

    def predict(self, user_id):
        if user_id not in self.data["user_id"].unique():
            return []
        return self.data[self.data["user_id"] == user_id].item_id.to_numpy().tolist()
