# single_line.py

import numpy as np
from scipy.interpolate import interp1d

from spectralops.utils import find_wvl, linear_interpolation


def single_line_nb(
    spectrum: np.ndarray,
    wvls: np.ndarray,
    tie_points: tuple
):
    """
    Numba-optimized version of `single_line`.

    Parameters
    ----------
    wvls: np.ndarray
        Wavelength values of the spectrum (in nm).
    spectrum: np.ndarray
        Spectrum values.
    tie_points: tuple of floats
        Tie points used to interpolate continuum.

    Returns
    -------
    continuum_removed: np.ndarray
        Spectrum with the continuum removed.
    continuum: np.ndarray
        The continuum values.
    """
    # Getting initial continuum line parameters
    anchor_pts = np.array([700, 1550, 2600])

    cont1_band_idx = np.empty(anchor_pts.size, dtype=np.int16)
    cont1_band_wvls = np.empty(anchor_pts.size, dtype=np.float64)
    for n in range(anchor_pts.size):
        idx = np.argmin(np.abs(wvls - anchor_pts[n]))
        cont1_band_idx[n] = idx
        cont1_band_wvls[n] = wvls[idx]

    cont1_spectrum_values = spectrum[cont1_band_idx]

    continuum1 = linear_interpolation(
        cont1_band_wvls, cont1_spectrum_values, wvls
    )

    continuum1_removed = spectrum / continuum1

    return continuum1_removed, continuum1


def single_line(
    spectrum: np.ndarray,
    wvls: np.ndarray,
    tie_points: tuple
):
    """
    Removes a single line continuum based on two tie point wavelengths.

    Parameters
    ----------
    wvls: np.ndarray
        Wavelength values of the spectrum (in nm).
    spectrum: np.ndarray
        Spectrum values.
    tie_points: tuple of floats
        Tie points used to interpolate continuum.

    Returns
    -------
    continuum_removed: np.ndarray
        Spectrum with the continuum removed.
    continuum: np.ndarray
        The continuum values.
    """

    cont_idx = [find_wvl(wvls, i)[0] for i in tie_points]
    cont_wvl = [find_wvl(wvls, i)[1] for i in tie_points]
    cont_spectrum_vals = spectrum[cont_idx]

    f = interp1d(
        cont_wvl,
        cont_spectrum_vals,
        kind='linear',
        fill_value='extrapolate'
    )

    continuum = f(wvls)

    continuum_removed = spectrum/continuum

    return continuum_removed, continuum
