from finflow.core.distributional import Actor, QuantileMLP
def test_imports():
    a = Actor(10,3); q = QuantileMLP(10,3,25)
