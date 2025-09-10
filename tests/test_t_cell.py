import numpy as np
from finflow.agents.t_cell import TCell

def test_tcell():
    X = np.random.randn(1000, 10)
    t = TCell()
    t.fit(X)
    z = t.score(X[:1])
    assert 0.0 <= z <= 1.0
