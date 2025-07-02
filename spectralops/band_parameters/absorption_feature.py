# band_parameters/absorption_band_stats.py

# Standard Libraries
from typing import Tuple, Optional

# External Imports
import numpy as np
import matplotlib.pyplot as plt

# Local Imports
from spectralops.spectral_classes import Spectrum
from spectralops.spectral_classes import SpectralCube
from spectralops.cube_ops import apply_calculate_area_over_cube

from .fit_absorption import fit_absorption
from .calculate_area import calculate_area
from .calculate_center import calculate_center
from .calculate_depth import calculate_depth

from spectralops.utils import rgb_composite


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

        self.area = calculate_area(
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

        self.depth = calculate_depth(
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

        fit_order = 4

        self.polyfit, self.cube, self.wvl = \
            fit_absorption(
                spectral_cube.contrem,
                spectral_cube.wvl,
                self._wvl_search_range,
                fit_order
            )
        print(f"Polynomial of order {fit_order} was fit to feature.")

        area = apply_calculate_area_over_cube(
            spectral_cube.contrem,
            spectral_cube.wvl,
            spectral_cube.spec_res,
            *self._wvl_search_range
        )
        print("Feature area was calculated.")

        center = calculate_center(
            wvl=spectral_cube.wvl,
            fitted_absorption=self.polyfit,
            absorption_spec=self.cube,
            absorption_wvl=self.wvl
        )
        print("Feature center was calculated.")

        depth = calculate_depth(
            wvl=spectral_cube.wvl,
            fitted_absorption=self.polyfit,
            absorption_spec=self.cube,
            absorption_wvl=self.wvl
        )
        print("Feature depth was calculated.")

        # Ensuring type stability.
        if isinstance(area, np.ndarray):
            self.area = area
        if isinstance(center, np.ndarray):
            self.center = center
        if isinstance(depth, np.ndarray):
            self.depth = depth

    def plot_test_spectrum(
        self,
        xtest: Optional[int] = None,
        ytest: Optional[int] = None,
        ax: Optional[np.ndarray] = None,
        feature_only: bool = False,
        feature_name: Optional[str] = None
    ):
        """
        Plots a test spectrum, with all band parameter information shown.

        Parameters
        ----------
        xtest: int, optional
            Test X coordinate. If None (default), it is randomly chosen.
        ytest: int, optional
            Test X coordinate. If None (default), it is randomly chosen.
        ax: array of Axes, optional
            List of 2 axes to plot the spectrum and continuumed-removed
            spectrm into, respectively. If None (default) a new figure is
            generated.
        feature_only: bool, optional
            If False (default), both the original spectrum and the absorption
            feature are plotted. If True, only the feature is plotted.
        feature_name: str, optional
            Name of the feature. If None (default), no name will be listed.
        """
        rng = np.random.default_rng()
        if (xtest is None) or (ytest is None):
            xtest = rng.integers(0, self.polyfit.shape[0])
            ytest = rng.integers(0, self.polyfit.shape[1])

        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            assert isinstance(ax, np.ndarray)

        if not feature_only:
            # Original Whole Spectrum
            ax[0].plot(
                self._original_spec.wvl,
                self._original_spec.cube[xtest, ytest, :],
                color="k"
            )

            # Smoothed Whole Spectrum
            ax[0].plot(
                self._original_spec.wvl,
                self._original_spec.smoothed[xtest, ytest, :],
                color="red"
            )

            # Continuum Removed Whole Spectrum
            ax[1].plot(
                self._original_spec.wvl,
                self._original_spec.contrem[xtest, ytest, :],
                color="k"
            )

        # Continuum Removed Absorption Feature Fit
        ax[1].plot(
            self.wvl,
            self.polyfit[xtest, ytest, :],
            color="red"
        )

        # Marker for absorption depth
        ax[1].vlines(
            self.center[xtest, ytest],
            1-self.depth[xtest, ytest], 1,
            color="blue", linestyle='--'
        )

        # Marker for continuum
        ax[1].hlines(
            1,
            self._original_spec.wvl.min(), self._original_spec.wvl.max(),
            color="blue", linestyle="--"
        )

        ax[0].set_title(f"Plotted Point: ({xtest}, {ytest})")

        if feature_only:
            ax[1].set_title(ax[1].get_title() +
                            f"   {feature_name} Depth: "
                            f"{self.depth[xtest, ytest]:.4f}")
        else:
            ax[1].set_title(f"{feature_name} Depth: "
                            f"{self.depth[xtest, ytest]:.4f}")

    def false_color_composite(
        self,
        red_band_attribute: str,
        green_band_attribute: str,
        blue_band_attrbiute: str
    ) -> np.ndarray:
        """
        Creates an RGB false color composite image from three image-like
        attributes.
        """
        r = getattr(self, red_band_attribute)
        g = getattr(self, green_band_attribute)
        b = getattr(self, blue_band_attrbiute)

        return rgb_composite(r, g, b)
