import pandas as pd
import typing
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from rectools import Columns
from rectools.metrics import calc_metrics
from rectools.dataset import Interactions, Dataset
from rectools.model_selection import TimeRangeSplitter

class CVMetrics:
    def __init__(
            self,
            models: typing.Dict,
            metrics: typing.Dict,
            splitter: TimeRangeSplitter,
            k: int = 10
        ) -> None:
        self.models = deepcopy(models)
        self.metrics = metrics
        self.splitter = splitter
        self.k = k
    
    def calculate_metrics(self, dataset_df: pd.DataFrame) -> typing.Dict:
        self.interactions = Interactions(dataset_df)
        self.splitter.get_test_fold_borders(self.interactions)
        self.dataset = Dataset.construct(dataset_df)
        self.results = []
        fold_splitter = self.splitter.split(self.interactions, collect_fold_stats=True)

        for train_ids, test_ids, fold_info in tqdm(fold_splitter, total=self.splitter.n_splits):

            df_train = self.interactions.df.iloc[train_ids]
            dataset = Dataset.construct(df_train)

            df_test = self.interactions.df.iloc[test_ids][Columns.UserItem]
            test_users = np.unique(df_test[Columns.User])

            catalog = df_train[Columns.Item].unique()

            for model_name, model in self.models.items():
                model.fit(dataset)
                recos = model.recommend(
                    users=test_users,
                    dataset=dataset,
                    k=self.k,
                    filter_viewed=True,
                )
                metric_values = calc_metrics(
                    self.metrics,
                    reco=recos,
                    interactions=df_test,
                    prev_interactions=df_train,
                    catalog=catalog,
                )
                fold_result = {"fold": fold_info["i_split"], "model": model_name}
                fold_result.update(metric_values)
                self.results.append(fold_result)
    
    def print_metrics(self):
        pivot_results = pd.DataFrame(self.results).drop(columns="fold").groupby(["model"], sort=False).agg(["mean", "std"])
        mean_metric_subset = [(metric, agg) for metric, agg in pivot_results.columns if agg == 'mean']
        beautiful_pivot = (
            pivot_results.style
            .highlight_min(subset=mean_metric_subset, color='lightcoral', axis=0)
            .highlight_max(subset=mean_metric_subset, color='lightgreen', axis=0)
        )
        return beautiful_pivot
        