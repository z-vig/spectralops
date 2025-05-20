# spectral_cube/moving_average

import numpy as np
from numba import njit

from spectralops.utils import fit_line


@njit
def moving_average(
    original_spectrum: np.ndarray,
    window_size: int = 5,
):
    """
    Numba-optimized version of `moving_average`.

    Parameters
    ----------
    original_spectrum: np.ndarray
        Non-smooth spectrum.
    window_size: optional, int
        Window size to use for the moving average. Default is 5.
    """
    window = np.ones(window_size)
    endcap_size = len(window) // 2

    # We are going to fix the number of points used for the linear
    # extrapolation based on the length of the spectrum (10% of the
    # spectrum length). Changing the box size will simply effect how many
    # points are extrapolated, not the number of points used for the
    # extrapolation.
    edge_length = np.maximum(round(len(original_spectrum) * 0.1, 0), 1)

    left_idx = np.arange(
        0,
        1+edge_length,
        dtype=np.int32
    )

    right_idx = np.arange(
        len(original_spectrum) - edge_length,
        len(original_spectrum),
        dtype=np.int32
    )

    left_fit = fit_line(
        left_idx,
        original_spectrum[left_idx],
        np.arange(-window_size, 0)
    )

    right_fit = fit_line(
        right_idx,
        original_spectrum[right_idx],
        np.arange(len(original_spectrum), len(original_spectrum) + window_size)
    )

    spectrum = np.empty(
        original_spectrum.size + left_fit.size + right_fit.size
    )

    spectrum[:left_fit.size] = left_fit
    spectrum[left_fit.size:-right_fit.size] = original_spectrum
    spectrum[-right_fit.size:] = right_fit

    mu = np.convolve(spectrum, window)[endcap_size:-endcap_size] / window.size
    musq = np.convolve(spectrum**2, window)[endcap_size:-endcap_size]\
        / window.size
    sigma = np.sqrt(musq - mu**2)
    # spectrum_idx = np.arange(0, len(spectrum))

    mu = mu[window_size:-window_size]
    sigma = sigma[window_size:-window_size]
    # spectrum_idx = spectrum_idx[window_size:-window_size]

    return mu, sigma
