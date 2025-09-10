import numpy as np
from sklearn.ensemble import IsolationForest
from ..utils.explain import shap_feature_importance

class TCell:
    def __init__(self, contamination=0.1, shap_topk=5, ema_alpha=0.1, random_state=42):
        self.m = IsolationForest(contamination=contamination, random_state=random_state)
        self.shap_topk = shap_topk
        self.ema_alpha = ema_alpha
        self.z_ema = 0.0
        self.fitted = False
        self._Xref = None

    def fit(self, X: np.ndarray):
        self.m.fit(X)
        self._Xref = X[:1024]
        self.fitted = True

    def score(self, X: np.ndarray) -> float:
        if not self.fitted: raise RuntimeError("T-Cell not fitted")
        s = - self.m.decision_function(X)  # higher -> more anomalous
        z = float(np.clip((s - s.min())/(s.max()-s.min()+1e-12), 0.0, 1.0))
        self.z_ema = (1 - self.ema_alpha)*self.z_ema + self.ema_alpha*z
        return self.z_ema

    def explain(self):
        if self._Xref is None: return []
        return shap_feature_importance(self.m, self._Xref, self.shap_topk)
