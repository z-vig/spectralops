# band_parameters/fit_absorption.py

# External Imports
import numpy as np

# Local Imports
from spectralops.utils import find_wvl
from spectralops.polyfit import polyfit


def fit_absorption(
    contrem_spectrum: np.ndarray,
    wvl: np.ndarray,
    wvl_search_range: tuple,
    fit_order: int
):
    """
    Fit a portion of a spectrum that is defined as an absorption band.

    Parameters
    ----------
    contrem_spectrum: np.ndarray
        Continuum-Removed spectrum.
    wvl: np.ndarray
        Wavelength values.
    wvl_search_range: tuple
        Tuple definining the wavelength range to search for an absorption
        feature.
    fit_order: int
        Order of the polynomial fit.

    Returns
    -------
    fitted_absorption: np.ndarray
        Fitted polynomial line to absorption feature.
    absorption_wvl: np.ndarray
        Wavelength values definined the absorption feature.
    """
    absorption_min_idx, absoprtion_min = find_wvl(wvl, wvl_search_range[0])
    absorption_max_idx, absoprtion_max = find_wvl(wvl, wvl_search_range[1])
    absorption_indices = np.arange(
        absorption_min_idx, absorption_max_idx, 1, dtype=int
    )

    if contrem_spectrum.ndim == 3:
        absorption_spec = contrem_spectrum[:, :, absorption_indices]
    else:
        absorption_spec = contrem_spectrum[absorption_indices]

    absorption_wvl = wvl[absorption_indices]

    fitted_absorption = polyfit(absorption_wvl, absorption_spec, fit_order)

    return fitted_absorption, absorption_spec, absorption_wvl
