import pickle


class Ranker:
    def __init__(self, path_data="../data/ranker_recos.pickle"):
        self.ranker_preds = pickle.load(open(path_data, "rb"))  # это датафрейм, через pickle быстрее грузится
        self.cols = [
            "lfm_score",
            "lfm_rank",
            "popular_score",
            "popular_rank",
            "age",
            "income",
            "sex",
            "kids_flg",
            "user_hist",
            "user_avg_pop",
            "user_last_pop",
            "content_type",
            "release_year",
            "for_kids",
            "age_rating",
            "studios",
            "item_pop",
            "item_avg_hist",
        ]

    def recommend(self, user_id):
        reco = []
        try:
            recos = self.ranker_preds[self.ranker_preds.user_id == user_id].item_id.tolist()[0]
            recos = list(set(list(map(int, reco))))
        except IndexError:
            # здесь можно делать предсказания на 
            # холодных пользователях, если заполнить значения по умолчанию.
            # т.к. офлайн предсказания, то на холодных просто будет популярное
            recos = []
        return recos
