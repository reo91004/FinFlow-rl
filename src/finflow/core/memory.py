import numpy as np

class EpisodicMemory:
    def __init__(self, dim: int, max_items: int = 10000):
        self.Z = np.zeros((0, dim), dtype=np.float32)
        self.meta = []

    def add(self, z_vec, info):
        self.Z = np.vstack([self.Z, z_vec.reshape(1,-1)])
        self.meta.append(info)

    def recall(self, z_vec, k=5):
        if len(self.Z)==0: return []
        sims = self.Z @ z_vec / (np.linalg.norm(self.Z, axis=1)*np.linalg.norm(z_vec)+1e-12)
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i]), self.meta[i]) for i in idx]
