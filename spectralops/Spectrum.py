# Spectrum.py

import numpy as np
import matplotlib.pyplot as plt

from .smoothing import outlier_removal, moving_average


class Spectrum():
    """
    Stores information about a single spectrum and allows for single spectrum
    processing steps.

    Parameters
    ----------
    spectrum: np.ndarray
        Single spectrum data.
    wvls: np.ndarray
        Wavelength (in nm) information corresponding to the spectrum.
    spectral_units: optional, str
        Units of the spectral data. Default is `"Reflectance"`.

    Attributes
    ----------
    wvls: Wavelengths
    spectrum: Spectrum
    no_outliers: Spectrum with outliers removed
    smoothed: Smoothed spectrum with no outliers

    Methods
    -------
    to_microns()
        Converts wavelength units from nm to microns.
    to_nm()
        Converts wavelength units from microns to nm.
    plot(fig, ax, to_plot)
        Plots all initialized spectral data.
    """
    def __init__(
        self,
        spectrum: np.ndarray,
        wvls: np.ndarray,
        spectral_units: str = "Reflectance"
    ):
        self.wvls = wvls
        self._wavelength_units = "nm"
        self._spectrum_units = spectral_units
        self.spectrum = spectrum
        self.no_outliers = self._remove_outliers()
        self.smoothed = self._smooth(starting_data=self.no_outliers)

    def _remove_outliers(self, starting_data: np.ndarray = None):
        if starting_data is None:
            no_outliers, _ = outlier_removal(self.spectrum)
        else:
            no_outliers, _ = outlier_removal(starting_data)
        return no_outliers

    def _smooth(self, starting_data: np.ndarray = None):
        if starting_data is None:
            mu, sigma = moving_average(self.spectrum)
        else:
            mu, sigma = moving_average(starting_data)
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
            "outliers_removed": True,
            "smooth": True
        }
    ):
        if (fig is None) or (ax is None):
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel(f"Wavelength ({self._wavelength_units})")
            ax.set_ylabel(self._spectrum_units)

        if to_plot.get("original"):
            ax.plot(self.wvls, self.spectrum, label="Original", alpha=0.6)

        if to_plot.get("outliers_removed"):
            ax.plot(
                self.wvls, self.no_outliers, label="No Outliers", alpha=0.6
            )

        if to_plot.get("smooth"):
            ax.plot(self.wvls, self.smoothed, label="Smoothed", alpha=0.6)
        ax.legend()
