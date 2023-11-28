import random
import typing

from rec_sys.models.base_model import Model


class RandomModel(Model):
    def fit(self, X: typing.Optional[typing.Any] = None, y: typing.Optional[typing.Any] = None) -> None:
        pass

    def predict(self, X: typing.Optional[typing.Any] = None, k_recs: int = 10) -> typing.List[int]:
        return random.sample(range(100), k_recs)
