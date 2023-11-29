import typing


class Model:
    def fit(self, X, y) -> None:
        raise NotImplementedError()

    def predict(self, X) -> typing.List[int]:
        raise NotImplementedError()
