# SpectralCube.py

import numpy as np
import matplotlib.pyplot as plt
from time import time
from numba import njit, prange

from spectralops.smoothing import outlier_removal, moving_average
from spectralops.utils import pretty_print_runtime


@njit(parallel=True)
def apply_over_cube(cube, func):
    """
    Applies a spectral function over an entire cube.

    Parameters
    ----------
    cube: np.ndarray
        Spectral image cube to modify via `func`. Spectral dimension must be
        in the third axis.
    func: Callable
        Function that will be applied to the cube. The first argument must be
        a single spectrum and the first return must be the modified spectrum.
        Other required arguments can be passed via `*args` and other returns
        will be ignored.
    *args
        Remaining arguments to be passed to `func`.
    """

    xsize, ysize, nbands = cube.shape

    new_cube = np.empty_like(cube)

    for i in prange(xsize):
        for j in prange(ysize):
            if np.isnan(cube[i, j, 0]):
                for k in range(nbands):
                    new_cube[i, j, k] = np.nan
            else:
                new_cube[i, j, :], _ = func(cube[i, j, :])
    return new_cube


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
    init_pipeline: bool
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
    """
    def __init__(
        self,
        cube: np.ndarray,
        wvl: np.ndarray,
        init_pipeline: bool = False
    ):
        self.cube = cube
        self.wvl = wvl

        if init_pipeline:
            print("Running spectral processing pipeline...")
            pipeline_start = time()

            self.no_outliers = self.remove_outliers()
            self.smoothed = self.smooth_spectra(self.no_outliers)

            pipeline_runtime = time() - pipeline_start
            pretty_print_runtime(pipeline_runtime, "Pipeline")

    def remove_outliers(self, starting_data=None):
        step_start = time()

        if starting_data is None:
            step = apply_over_cube(self.cube, outlier_removal)
        else:
            step = apply_over_cube(starting_data, outlier_removal)

        step_runtime = time() - step_start
        pretty_print_runtime(step_runtime, "Outlier removal")
        return step

    def smooth_spectra(self, starting_data=None):
        step_start = time()

        if starting_data is None:
            step = apply_over_cube(self.cube, moving_average)
        else:
            step = apply_over_cube(starting_data, moving_average)

        step_runtime = time() - step_start
        pretty_print_runtime(step_runtime, "Spectral smoothing")
        return step

    def plot_test_spectrum(self):
        attr_list = ["cube", "no_outliers", "smoothed"]

        rng = np.random.default_rng()
        x = rng.integers(0, self.cube.shape[0])
        y = rng.integers(0, self.cube.shape[1])

        fig, ax = plt.subplots()
        ax.set_title(f"X: {x}, Y: {y}")

        for i in attr_list:
            try:
                dataset = getattr(self, i)
            except AttributeError:
                pass

            ax.plot(self.wvl, dataset[x, y, :], label=i, alpha=0.6)

        ax.legend()
        plt.show()
