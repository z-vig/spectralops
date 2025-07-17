# SpectralCube.py

# Standard Libraries
from typing import Union, Optional

# External Imports
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Local Imports
from spectralops.utils import pretty_print_runtime
from spectralops.cube_ops import apply_remove_outliers_over_cube
from spectralops.cube_ops import apply_smoothing_over_cube
from spectralops.cube_ops import apply_continuum_removal_over_cube


class SpectralCube():
    """
    Stores data from a spectral cubes and allows for pipeline-based spectral
    processing.

    Parameters
    ----------
    cube: np.ndarray
        Spectral cube data where axis=2 is the spectral dimension.
    wvl: np.ndarray
        Wavelength values corresponding to axis=2.
    pixel_mask: np.ndarray, optional
        Pixels to be masked are =1 and valid pixels are =0.
    spectral_resolution: Union[None, np.ndarray, float], optional.
        Spectral resolution of dataset. Can either be a single value or an
        array of values corresponding to spectral resolution of each band.
        If None (default), a constant resolution will be calculated.
    init_pipeline: bool, optional
        Switch to enable running the pipeline at initialization.

    Attributes
    ----------
    cube: original data
    wvl: wavelengths
    no_outliers: Outliers removed. Init with `remove_outliers` or when
                 `init_pipeline` is True.
    smoothed: Smoothed spectra. Init with `smooth_spectra` or when
                `init_pipeline` is True.

    Methods
    -------
    remove_outliers(starting_data=None)
        Remove spectral outliers from starting_data (or `cube` attribute if
        `starting_data` is None).
    smooth_spectra(starting_data=None)
        Smooths spectra in the starting_data (or `cube` attribute if
        `starting_data` is None).
    plot_test_spectrum()
        Plots a random test spectrum from within the cube.
    """
    def __init__(
        self,
        cube: np.ndarray,
        wvl: np.ndarray,
        pixel_mask: Optional[np.ndarray] = None,
        spectral_resolution: Union[None, np.ndarray, float] = None,
        init_pipeline: bool = False
    ):
        self.cube = cube
        self.wvl = wvl
        self.mask = pixel_mask
        if spectral_resolution is None:
            self.spec_res = (wvl.max() - wvl.min()) / wvl.size
        else:
            self.spec_res = spectral_resolution

        if init_pipeline:
            print("Running spectral processing pipeline...")
            pipeline_start = time()

            self.no_outliers = self.remove_outliers()
            self.smoothed, self.err = self.smooth_spectra(self.no_outliers)
            self.contrem, self.continuum = self.remove_continuum(self.smoothed)

            pipeline_runtime = time() - pipeline_start
            pretty_print_runtime(pipeline_runtime, "Pipeline")

    def remove_outliers(self, starting_data=None):
        step_start = time()

        if starting_data is None:
            step = apply_remove_outliers_over_cube(self.cube)
        else:
            step = apply_remove_outliers_over_cube(starting_data)

        step_runtime = time() - step_start
        pretty_print_runtime(step_runtime, "Outlier removal")
        return step

    def smooth_spectra(self, starting_data=None):
        step_start = time()

        if starting_data is None:
            step = apply_smoothing_over_cube(self.cube)
        else:
            step = apply_smoothing_over_cube(starting_data)

        step_runtime = time() - step_start
        pretty_print_runtime(step_runtime, "Spectral smoothing")
        return step[:, :, :, 0], step[:, :, :, 1]

    def remove_continuum(self, starting_data=None):
        step_start = time()

        if starting_data is None:
            step = apply_continuum_removal_over_cube(self.cube, self.wvl)
        else:
            step = apply_continuum_removal_over_cube(starting_data, self.wvl)

        step_runtime = time() - step_start
        pretty_print_runtime(step_runtime, "Continuum removal")
        return step[:, :, :, 0], step[:, :, :, 1]

    def with_mask(self, attr: str):
        data_nomask = getattr(self, attr)
        data_withmask = data_nomask.copy()
        data_withmask[self.mask == 1] = np.nan
        return data_withmask

    def plot_test_spectrum(self):
        attr_list = ["cube", "no_outliers", "smoothed", "contrem"]

        rng = np.random.default_rng()
        x = rng.integers(0, self.cube.shape[0])
        y = rng.integers(0, self.cube.shape[1])

        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"X: {x}, Y: {y}")
        ax[0].set_ylabel("Reflectance")
        ax[0].set_xlabel("Wavelength")
        ax[1].set_ylabel("Continuum-Removed Reflectance")
        ax[1].set_xlabel("Wavelength")

        for i in attr_list:
            try:
                dataset = getattr(self, i)
            except AttributeError:
                continue

            yvals = dataset[x, y, :]

            if i != "contrem":
                ax[0].plot(self.wvl, yvals, label=i, alpha=0.6)
            else:
                ax[1].plot(self.wvl, yvals, label=i, alpha=1)

        ax[0].legend()
        plt.show()
