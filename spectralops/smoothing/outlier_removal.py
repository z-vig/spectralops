# spectral_cube/smoothing/outlier_removal.py

import numpy as np
from numba import njit

from .moving_average import moving_average


@njit
def outlier_removal(
    original_spectrum: np.ndarray,
    threshold: float = 2
) -> np.ndarray:
    """
    Numba-optimized version of `outlier_removal`.

    Parameters
    ----------
    original_spectrum : np.ndarray
        The 1D input spectrum array to process. Must be a numeric NumPy array.
    threshold : float, optional
        The Z-score threshold to use for detecting outliers. Any value with a
        Z-score greater than `threshold` (in absolute value) is considered
        an outlier. Default is 2.

    Returns
    -------
    np.ndarray
        A new spectrum array with outliers replaced by the mean of their
        immediate neighbors. The original input is not modified.

    Notes
    -----
    - The local mean and standard deviation are computed using a moving
      window that spans 10% of the spectrum length (minimum size of 3
      and always rounded to an odd number).
    - To avoid circular wraparound, the first and last elements only
      use their single available neighbor for replacement.
    - This method is robust to isolated spikes, but may not perform well
      for broad or clustered outliers.

    Examples
    --------
    >>> import numpy as np
    >>> spectrum = np.array([1.0, 1.1, 1.2, 10.0, 1.3, 1.2, 1.1])
    >>> outlier_removal(spectrum)
    array([1. , 1.1, 1.2, 1.25, 1.3, 1.2, 1.1])
    """
    spectrum = np.copy(original_spectrum)

    # Re-implementation of utils.round_to_odd()
    r = round(spectrum.size * 0.1, 0)
    if r % 2 == 0:
        if (spectrum.size - r) != 0:
            r = int(r + ((spectrum.size - r)/abs(spectrum.size - r)))
        else:
            r = int(r - 1)
    else:
        r = int(r)

    window_size = np.maximum(r, 3)

    mu, sig = moving_average(spectrum, window_size=window_size)

    zscore = (spectrum - mu) / sig
    outlier_idx = np.abs(zscore) > threshold

    neighbors = np.empty((spectrum.size, 2))
    neighbors[:, 0] = np.roll(spectrum, -1)
    neighbors[:, 1] = np.roll(spectrum, 1)

    # Avoiding edge effects
    neighbors[0, 1] = np.nan
    neighbors[-1, 0] = np.nan

    replacement = np.empty(neighbors.shape[0])
    for n in np.arange(replacement.size):
        replacement[n] = np.nanmean(neighbors[n, :])

    spectrum[outlier_idx] = replacement[outlier_idx]

    return spectrum, np.nan
