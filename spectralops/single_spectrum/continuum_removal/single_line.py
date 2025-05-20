# single_line.py

import numpy as np
from scipy.interpolate import interp1d

from spectralops.utils import find_wvl


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
