import matplotlib.pyplot as plt
import numpy as np

def plot_equity(equity: np.ndarray, path: str):
    plt.figure()
    plt.plot(equity)
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
