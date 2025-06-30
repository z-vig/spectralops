# band_parameters/calculate_center.py

# External Imports
import numpy as np
from typing import Tuple, Union, Optional

# Local Imports
from .fit_absorption import fit_absorption


def calculate_center_from_spectrum(
    wvl: np.ndarray,
    contrem_spectrum: np.ndarray,
    wvl_search_range: Tuple[float, float],
    fit_order: int = 4
) -> Union[float, np.ndarray]:
    """
    Calculate the absorption center from a continuum-removed spectrum.

    Parameters
    ----------
    wvl : np.ndarray
        Wavelength values.
    contrem_spectrum : np.ndarray
        Continuum-removed spectrum.
    wvl_search_range : tuple
        Tuple defining the wavelength range to search for an absorption
        feature.
    fit_order : int, optional
        Order of the polynomial fit (default is 4).

    Returns
    -------
    center : float or np.ndarray
        The wavelength(s) of the absorption center(s).
    """
    fitted_absorption, absorption_spec, absorption_wvl = fit_absorption(
        contrem_spectrum, wvl, wvl_search_range, fit_order
    )
    return calculate_center_from_fit(
        wvl,
        fitted_absorption,
        absorption_spec,
        absorption_wvl
    )


def calculate_center_from_fit(
    wvl: np.ndarray,
    fitted_absorption: np.ndarray,
    absorption_spec: np.ndarray,
    absorption_wvl: np.ndarray
) -> Union[float, np.ndarray]:
    """
    Calculate the absorption center from a fitted absorption feature.

    Parameters
    ----------
    wvl : np.ndarray
        Wavelength values.
    fitted_absorption : np.ndarray
        Fitted absorption feature.
    absorption_spec : np.ndarray
        Absorption spectrum.
    absorption_wvl : np.ndarray
        Wavelengths corresponding to the absorption feature.

    Returns
    -------
    center : float or np.ndarray
        The wavelength(s) of the absorption center(s).
    """
    if fitted_absorption.ndim == 3:
        absorption_center = np.argmin(fitted_absorption, axis=2)
        absorption_center[absorption_center == absorption_spec.size] = 0
        absorption_center_wvl = absorption_wvl[absorption_center]
        absorption_center_wvl[absorption_center_wvl == wvl[0]] = np.nan
        return absorption_center_wvl
    else:
        absorption_center = np.argmin(fitted_absorption)
        if (absorption_center == 0) or \
           (absorption_center == absorption_spec.size):
            return np.nan
        else:
            return wvl[absorption_center]


def calculate_center(
    wvl: np.ndarray,
    contrem_spectrum: Optional[np.ndarray] = None,
    wvl_search_range: Optional[Tuple[float, float]] = None,
    fitted_absorption: Optional[np.ndarray] = None,
    absorption_spec: Optional[np.ndarray] = None,
    absorption_wvl: Optional[np.ndarray] = None,
    fit_order: int = 4
) -> Union[float, np.ndarray]:
    """
    Main API to calculate the absorption center.

    Usage:
        - To calculate from spectrum: provide wvl, contrem_spectrum,
          wvl_search_range
        - To calculate from fit: provide wvl, fitted_absorption,
          absorption_spec, absorption_wvl

    Returns
    -------
    center : float or np.ndarray
        The wavelength(s) of the absorption center(s).
    """

    use_spectrum_mode = (
        contrem_spectrum is not None and wvl_search_range is not None
    )

    use_fit_mode = (
        fitted_absorption is not None and
        absorption_spec is not None and
        absorption_wvl is not None
    )

    if use_spectrum_mode:
        contrem_spectrum_checked = contrem_spectrum
        wvl_search_range_checked = wvl_search_range

        assert contrem_spectrum_checked is not None
        assert wvl_search_range_checked is not None

        return calculate_center_from_spectrum(
            wvl,
            contrem_spectrum_checked,
            wvl_search_range_checked,
            fit_order
        )
    elif use_fit_mode:
        fitted_absorption_checked = fitted_absorption
        absorption_spec_checked = absorption_spec
        absorption_wvl_checked = absorption_wvl

        assert fitted_absorption_checked is not None
        assert absorption_spec_checked is not None
        assert absorption_wvl_checked is not None

        return calculate_center_from_fit(
            wvl,
            fitted_absorption_checked,
            absorption_spec_checked,
            absorption_wvl_checked
        )
    else:
        raise ValueError("Invalid arguments. See docstring for usage.")
