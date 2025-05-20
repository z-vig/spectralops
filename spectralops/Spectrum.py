# Spectrum.py

import numpy as np
import matplotlib.pyplot as plt

from .smoothing import outlier_removal, moving_average


class Spectrum():
    def __init__(
        self,
        wvls: np.ndarray,
        spectrum: np.ndarray,
        spectral_units: str = "Reflectance"
    ):
        self.wvls = wvls
        self._wavelength_units = "nm"
        self._spectrum_units = spectral_units
        self.spectrum = spectrum
        self.no_outliers = self._remove_outliers()
        self.smoothed = self._smooth()

    def _remove_outliers(self):
        return outlier_removal(self.spectrum)

    def _smooth(self):
        mu, sigma, idx = moving_average(self.spectrum)
        return mu

    def to_microns(self):
        if self._wavelength_units == "nm":
            self.wvls /= 1000
            self._wavelength_units = "\u03BCm"
        else:
            print("Wavelengths already in microns.")

    def to_nm(self):
        if self._wavelength_units == "\u03BCm":
            self.wvls *= 1000
            self._wavelength_units = "nm"
        else:
            print("Wavelengths already in nm.")

    def plot(
        self,
        fig=None,
        ax=None,
        to_plot: dict = {
            "original": True,
            "outliers_removed": False,
            "smooth": True
        }
    ):
        if (fig is None) or (ax is None):
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel(f"Wavelength ({self._wavelength_units})")
            ax.set_ylabel(self._spectrum_units)

        if to_plot.get("original"):
            ax.plot(self.wvls, self.spectrum)

        if to_plot.get("outliers_removed"):
            ax.plot(self.wvls, self.no_outliers)

        if to_plot.get("smooth"):
            ax.plot(self.wvls, self.smoothed)
