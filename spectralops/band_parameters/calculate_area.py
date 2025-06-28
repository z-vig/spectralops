# band_parameters/calculate_area.py

# Standard Libraries
from typing import Union

# External Imports
import numpy as np

# Local Imports
from spectralops.utils import find_wvl


def calculate_area(
    contrem_spectrum: np.ndarray,
    wvl: np.ndarray,
    wvl_search_range: tuple[float, float],
    spectral_resolution: Union[float, np.ndarray]
):
    """
    Calculates the area of the absorption band below the spectral continuum.
    This operation is performed by taking the difference between the value at
    every spectral band and the continuum value at that band, multiply this
    value by the spectral resolution of the band, repeating this for all bands
    in the wavelength search range and summing the results.

    Parameters
    ----------
    contrem_spectrum: np.ndarray
        Continuum-removed spectral data for a single spectrum.
    wvl: np.ndarray
        Corresponding wavelength values.
    wvl_search_range: tuple[float, float]
        Range over which to perform band math.
    spectral_resolution: Union[float, np.ndarray]
        Spectral resolution data. If float, this is the constants resolution.
        Otherwise, specify an array of values corresponding to each band.
    """
    wvl_min_idx, wvl_min = find_wvl(wvl, wvl_search_range[0])
    wvl_max_idx, wvl_max = find_wvl(wvl, wvl_search_range[1])
    wvl_indices = np.arange(wvl_min_idx, wvl_max_idx, 1, dtype=int)

    area_components = (1 - contrem_spectrum[wvl_indices]) * spectral_resolution

    area = np.sum(area_components)

    return area
