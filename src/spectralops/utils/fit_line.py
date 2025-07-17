# fit_line.py

import numpy as np
from numba import njit


@njit
def fit_line(x: np.ndarray, y: np.ndarray, xfit: np.ndarray):
    """
    Fits a single line.

    Parameters
    ----------
    x: np.ndarray
        X Data
    y: np.ndarray
        Y Data
    xfit: np.ndarray
        Returns fitted data for these x-values.

    Returns
    -------
    yfit
    """
    # Building G matrix
    G = np.empty((x.size, 2), dtype=np.float64)
    G[:, 0] = 1
    G[:, 1] = x

    # Building data matrix
    d = np.empty((y.size, 1), dtype=np.float64)
    d[:, 0] = y

    # Solving L2-Norm
    m = G.T @ G
    m = np.linalg.inv(m)
    m = m @ G.T
    m = m @ d

    # Setting up fit line
    intercept = m[0, 0]
    slope = m[1, 0]
    yfit = slope*xfit + intercept

    return yfit


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rng = np.random.default_rng()
    xtest = rng.uniform(0, 100, 500)
    ytest = 2*xtest + rng.normal(0, 20, 500)

    xfit = np.linspace(0, 100, 100)
    yfit = fit_line(xtest, ytest, xfit)

    plt.scatter(xtest, ytest)
    plt.plot(xfit, yfit, color='red')
    plt.show()
