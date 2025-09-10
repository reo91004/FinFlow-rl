import numpy as np
from typing import List, Tuple
import shap

def shap_feature_importance(tree_model, X: np.ndarray, topk: int = 5) -> List[Tuple[int,float]]:
    # For IsolationForest, TreeExplainer can still attribute trees' splits wrt pseudo-loss.
    try:
        explainer = shap.TreeExplainer(tree_model)
        sv = explainer.shap_values(X[:512])
        vals = np.abs(sv).mean(axis=0)
        idx = np.argsort(vals)[::-1][:topk]
        return [(int(i), float(vals[i])) for i in idx]
    except Exception:
        # fallback generic
        explainer = shap.Explainer(tree_model, X[:512])
        sv = explainer(X[:512])
        vals = np.abs(sv.values).mean(axis=0)
        idx = np.argsort(vals)[::-1][:topk]
        return [(int(i), float(vals[i])) for i in idx]
