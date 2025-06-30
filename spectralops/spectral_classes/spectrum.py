# Spectrum.py

# Standard Libraries
from typing import Union

# External Imports
import numpy as np
import matplotlib.pyplot as plt

# Local Imports
from spectralops.smoothing import outlier_removal, moving_average
from spectralops.continuum_removal import double_line_nb


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
    spectral_resolution: optional, Union[None, float, np.ndarray]

    Attributes
    ----------
    wvls: 1-D Array
        Wavelengths
    spec_res: Union[float, 1-D Array]
        Spectral resolution. Either constant or by band. If None, a constant
        resolution will be calculated from wavelength data.
    spectrum: 1-D Array
        Spectrum
    no_outliers: 1-D Array
        Spectrum with outliers removed
    smoothed: 1-D Array
        Smoothed spectrum with no outliers

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
        spectral_units: str = "Reflectance",
        spectral_resolution: Union[None, np.ndarray, float] = None
    ):
        self.wvl = wvls
        self._wavelength_units = "nm"
        self._spectrum_units = spectral_units
        self.spectrum = spectrum
        self.no_outliers = self._remove_outliers()
        self.smoothed = self._smooth(starting_data=self.no_outliers)
        self.contrem = self._remove_continuum(starting_data=self.smoothed)
        self.nbands = spectrum.size

        if spectral_resolution is None:
            self.spec_res = (wvls.max() - wvls.min()) / wvls.size
        else:
            self.spec_res = spectral_resolution

    def _remove_outliers(self, starting_data: Union[np.ndarray, None] = None):
        if starting_data is None:
            no_outliers, _ = outlier_removal(self.spectrum)
        else:
            no_outliers, _ = outlier_removal(starting_data)
        return no_outliers

    def _smooth(self, starting_data: Union[np.ndarray, None] = None):
        if starting_data is None:
            mu, sigma = moving_average(self.spectrum)
        else:
            mu, sigma = moving_average(starting_data)
        return mu

    def _remove_continuum(self, starting_data: Union[np.ndarray, None] = None):
        if starting_data is None:
            contrem, continuum = double_line_nb(self.spectrum, self.wvl)
        else:
            contrem, continuum = double_line_nb(starting_data, self.wvl)
        return contrem

    def to_microns(self):
        if self._wavelength_units == "nm":
            self.wvl /= 1000
            self._wavelength_units = "\u03BCm"
        else:
            print("Wavelengths already in microns.")

    def to_nm(self):
        if self._wavelength_units == "\u03BCm":
            self.wvl *= 1000
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
        """
        Plots the original and processed spectra.

        Parameters
        ----------
        fig: Figure
            Matplotlib figure object to add the plot to.
        axis: Axis
            Axis within `fig` to plot into.
        to_plot: dict
            Options dictionary for which spectra to plot. If the value of the
            key is True, it will be added to the plot:

            - original: Original specturm
            - outliers_removed: No outliers
            - smooth: Moving average applied
        """
        if (fig is None) or (ax is None):
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel(f"Wavelength ({self._wavelength_units})")
            ax.set_ylabel(self._spectrum_units)

        if to_plot.get("original"):
            ax.plot(self.wvl, self.spectrum, label="Original", alpha=0.6)

        if to_plot.get("outliers_removed"):
            ax.plot(
                self.wvl, self.no_outliers, label="No Outliers", alpha=0.6
            )

        if to_plot.get("smooth"):
            ax.plot(self.wvl, self.smoothed, label="Smoothed", alpha=0.6)
        ax.legend()
