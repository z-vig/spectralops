# band_parameters/absorption_band_stats.py

# Standard Libraries
from typing import Tuple

# External Imports
# import numpy as np

# Local Imports
from spectralops import Spectrum
from spectralops import SpectralCube

from .fit_absorption import fit_absorption
from . calculate_area import calculate_area
from .calculate_center import calculate_center

from spectralops.utils import apply_over_cube


class AbsorptionFeature():
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
    polyfit: np.ndarray
        Polynomial fit of the feature.
    spectrum: np.ndarray
        Truncated spectrum of the feature.
    wvl: np.ndarray
        Truncated wavelength values corresponding to `spectrum`.
    area: float
        Area of absorption feature.
    center: float
        Center wavelength of absorption feature.
    """
    def __init__(
        self,
        spectrum: Spectrum,
        wvl_search_range: Tuple[float, float]
    ) -> None:
        self._original_spec = spectrum
        self._wvl_search_range = wvl_search_range

        self.polyfit, self.spectrum, self.wvl = \
            fit_absorption(
                spectrum.contrem,
                spectrum.wvl,
                self._wvl_search_range,
                4
            )

        self.area, self._area_components = calculate_area(
            spectrum.contrem,
            spectrum.wvl,
            *self._wvl_search_range,
            spectrum.spec_res
        )

        self.center = calculate_center(
            wvl=spectrum.wvl,
            fitted_absorption=self.polyfit,
            absorption_spec=self.spectrum,
            absorption_wvl=self.wvl
        )


class AbsorptionFeatureCube():
    def __init__(
        self,
        spectral_cube: SpectralCube,
        wvl_search_range: Tuple
    ) -> None:
        self._original_spec = spectral_cube
        self._wvl_search_range = wvl_search_range

        self.polyfit, self.cube, self.wvl = \
            fit_absorption(
                spectral_cube.contrem,
                spectral_cube.wvl,
                self._wvl_search_range,
                4
            )

        self.area = apply_over_cube(
            spectral_cube.contrem, calculate_area, 1,
            spectral_cube.wvl, self._wvl_search_range, spectral_cube.spec_res
        )

        self.center = calculate_center(
            wvl=spectral_cube.wvl,
            fitted_absorption=self.polyfit,
            absorption_spec=self.cube,
            absorption_wvl=self.wvl
        )
