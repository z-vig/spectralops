# utils/polyfit.py

# External Imports
import numpy as np


def vandermonde(N: int, X: np.ndarray):
    """
    Returns vandermonde matrix of order `N` using `X` data.
    """
    return np.vstack([X ** i for i in range(N)])


if __name__ == "__main__":
    rng = np.random.default_rng()
    x = rng.uniform(0, 10, 100)
    v = vandermonde(4, x)
    print(v.shape)
