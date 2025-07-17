# linear_interpolation.py

import numpy as np
from numba import njit


@njit
def linear_interpolation(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    interp_x: np.ndarray
):
    interp = np.zeros(interp_x.size, dtype=np.float64)
    slopes = np.empty(x_pts.size-1)
    intercepts = np.empty(x_pts.size-1)

    for n in range(x_pts.size-1):
        x1 = x_pts[n]
        x2 = x_pts[n+1]
        y1 = y_pts[n]
        y2 = y_pts[n+1]
        m = (y2-y1)/(x2-x1)
        b = y2 - m*x2
        slopes[n] = m
        intercepts[n] = b

        idx = np.argwhere((interp_x <= x2) & (interp_x > x1)).flatten()
        interp[idx] = m * interp_x[idx] + b

    for n in np.argwhere(interp_x <= x_pts[0]).flatten():
        interp[n] = interp_x[n]*slopes[0] + intercepts[0]

    for n in np.argwhere(interp_x > x_pts[-1]).flatten():
        interp[n] = interp_x[n]*slopes[-1] + intercepts[-1]

    return interp
