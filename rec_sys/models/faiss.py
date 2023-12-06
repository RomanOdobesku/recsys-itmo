import faiss
import numpy as np


class FAISS:
    def __init__(self, user_embeddings, item_embeddings, efC=200, efS=200, M=48):
        self.index = faiss.index_factory(user_embeddings.shape[1], f"HNSW{M}", faiss.METRIC_L2)
        self.index.hnsw.efConstruction = efC
        self.index.hnsw.efSearch = efS
        self.user_embeddings = user_embeddings
        self.index.add(item_embeddings)

    def search(self, user_id, K=10):
        try:
            dist, indexes = self.index.search(np.array([self.user_embeddings[user_id, :]]), K)
        except IndexError:
            return [], []
        except ValueError:
            return [], []
        return dist, indexes
