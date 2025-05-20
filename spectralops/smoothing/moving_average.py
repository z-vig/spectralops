# spectral_cube/moving_average

import numpy as np
from numba import njit

from spectralops.utils import fit_line, get_options_errors


@njit
def moving_average_nb(
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


def moving_average(
    original_spectrum: np.ndarray,
    window_size: int = 5,
    edge_handling: str = "extrapolate",
    fit_order: int = 1
):
    """
    Smoothes a spectrum using a moving average.

    Parameters
    ----------
    original_spectrum: np.ndarray
        Non-smooth spectrum.
    window_size: optional, int
        Window size to use for the moving average. Default is 5.
    edge_handling: optional, str
        How to handle the edge cases of spectrum after convolution. Default is
        `"extrapolate"`
    remove_outliers: optional, bool
        If true, outliers are removed from the spectrum. A threshold value
        can be passed. Default is False.
    outlier_threshold: optional, float
        Sigma threshold to be used in outlier removal. See
        `outlier_removal.py`. Default is 2.
    """
    spectrum = np.copy(original_spectrum)

    window = np.ones(window_size)
    endcap_size = len(window) // 2

    edge_handling_cases = ["mirror", "extrapolate", "fill_ends", "cut_ends"]
    if edge_handling not in edge_handling_cases:
        raise ValueError(
            get_options_errors(
                edge_handling, edge_handling_cases, option_name="edge handler"
            )
        )

    if edge_handling == "mirror":
        spectrum = np.concatenate([
            np.flip(spectrum[:window_size+1]),
            spectrum,
            np.flip(spectrum[-1 * window_size:])
        ])

    elif edge_handling == "extrapolate":
        # We are going to fix the number of points used for the linear
        # extrapolation based on the length of the spectrum (10% of the
        # spectrum length). Changing the box size will simply effect how many
        # points are extrapolated, not the number of points used for the
        # extrapolation.
        edge_length = np.maximum(round(len(spectrum) * 0.1, 0), 1)
        left_idx = np.arange(0, 1+edge_length, dtype=np.int32)
        right_idx = np.arange(
            len(spectrum) - edge_length, len(spectrum), dtype=np.int32
        )

        fit_left = np.poly1d(
            np.polyfit(left_idx, spectrum[left_idx], fit_order)
        )
        fit_right = np.poly1d(
            np.polyfit(right_idx, spectrum[right_idx], fit_order)
        )

        spectrum = np.concatenate([
            fit_left(np.arange(-window_size, 0)),
            spectrum,
            fit_right(np.arange(len(spectrum), len(spectrum) + window_size))
        ])

    mu = np.convolve(spectrum, window)[endcap_size:-endcap_size] / window.size
    musq = np.convolve(spectrum**2, window)[endcap_size:-endcap_size]\
        / window.size
    sigma = np.sqrt(musq - mu**2)
    spectrum_idx = np.arange(0, len(spectrum))

    if edge_handling == "mirror" or edge_handling == "extrapolate":
        mu = mu[window_size:-window_size]
        sigma = sigma[window_size:-window_size]
        spectrum_idx = spectrum_idx[window_size:-window_size]
    elif edge_handling == "fill_ends":
        mu[:endcap_size] = spectrum[:endcap_size]
        mu[-endcap_size:] = spectrum[-endcap_size:]

        sigma[:endcap_size] = 0
        sigma[-endcap_size:] = 0
    elif edge_handling == "cut_ends":
        mu = mu[endcap_size:-endcap_size]
        sigma = sigma[endcap_size:-endcap_size]
        spectrum_idx = spectrum_idx[endcap_size:-endcap_size]

    return mu, sigma
