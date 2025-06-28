# band_parameters/absorption_band_stats.py

# External Imports
# import numpy as np

# Local Imports
from spectralops import Spectrum


class AbsorptionBandStats():
    """
    Stores statistics and parameters of an absorption band in a spectrum.

    Parameters
    ----------
    spectrum: Spectrum
        Whole spectrum of data.
    wvl_search_range: tuple[float, float]
        Range of wavelengths to search for absorption feature.

    Attributes
    ----------
    """
    def __init__(
        self,
        spectrum: Spectrum,
        wvl_search_range: tuple[float, float]
    ):
        self._spec = spectrum
        self._wvl_search_range = wvl_search_range
