# utils/cube_ops.py

# External Imports
import numpy as np
from numba import njit, prange

from spectralops.smoothing import outlier_removal_nb
from spectralops.smoothing import moving_average_nb
from spectralops.continuum_removal import double_line_nb
from spectralops.band_parameters.calculate_area import calculate_area


@njit(parallel=True)
def apply_over_cube(cube, func, output_size, *args) -> np.ndarray:
    """
    Applies a spectral processing function over an entire cube.

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
    output_size: None or int
        Size of the output result for the `func` that is applied to a single
        spectrum. If None (default), it will be assumed that the function
        is a spectral processing step and output_size will be set to the size
        of the input spectra.
    *args
        Remaining arguments to be passed to `func`.

    Returns
    -------
    analysis_result: np.ndarray
        Spectral cube with processing applied.
    """

    xsize, ysize, nbands = cube.shape

    analysis_result = np.empty(
        (xsize, ysize, output_size, 2), dtype=cube.dtype
    )

    for i in prange(xsize):
        for j in prange(ysize):
            if np.isnan(cube[i, j, 0]):
                for k in range(nbands):
                    analysis_result[i, j, k] = np.nan
            else:
                result = func(cube[i, j, :], *args)
                analysis_result[i, j, :, 0] = result[0]
                analysis_result[i, j, :, 1] = result[1]

    return analysis_result


@njit(parallel=True)
def apply_remove_outliers_over_cube(cube):
    """Applies remove_outliers function"""
    xsize, ysize, nbands = cube.shape

    analysis_result = np.empty(
        (xsize, ysize, nbands), dtype=cube.dtype
    )

    for i in prange(xsize):
        for j in prange(ysize):
            if np.isnan(cube[i, j, 0]):
                for k in range(nbands):
                    analysis_result[i, j, k] = np.nan
            else:
                result = outlier_removal_nb(cube[i, j, :])
                analysis_result[i, j, :] = result[0]

    return analysis_result


@njit(parallel=True)
def apply_smoothing_over_cube(cube):
    """Applies moving_average_nb function"""
    xsize, ysize, nbands = cube.shape

    analysis_result = np.empty(
        (xsize, ysize, nbands, 2), dtype=cube.dtype
    )

    for i in prange(xsize):
        for j in prange(ysize):
            if np.isnan(cube[i, j, 0]):
                for k in range(nbands):
                    analysis_result[i, j, k] = np.nan
            else:
                result = moving_average_nb(cube[i, j, :])
                analysis_result[i, j, :, 0] = result[0]
                analysis_result[i, j, :, 1] = result[1]

    return analysis_result


@njit(parallel=True)
def apply_continuum_removal_over_cube(cube, wvls):
    """Applies double_line_nb function"""
    xsize, ysize, nbands = cube.shape

    analysis_result = np.empty(
        (xsize, ysize, nbands, 2), dtype=cube.dtype
    )

    for i in prange(xsize):
        for j in prange(ysize):
            if np.isnan(cube[i, j, 0]):
                for k in range(nbands):
                    analysis_result[i, j, k] = np.nan
            else:
                result = double_line_nb(cube[i, j, :], wvls)
                analysis_result[i, j, :, 0] = result[0]
                analysis_result[i, j, :, 1] = result[1]

    return analysis_result


@njit(parallel=True)
def apply_polyfit_over_cube(cube, X, Xt, XtX):
    """Applies polynomial absorption fitting function"""
    xsize, ysize, nbands = cube.shape

    analysis_result = np.empty(
        (xsize, ysize, X.shape[0]), dtype=cube.dtype
    )

    for i in prange(xsize):
        for j in prange(ysize):
            if np.isnan(cube[i, j, 0]):
                for k in range(nbands):
                    analysis_result[i, j, k] = np.nan
            else:
                beta = np.linalg.inv(XtX) @\
                       (Xt @ np.ascontiguousarray(cube[i, j, :]))
                fit_line = X @ beta
                analysis_result[i, j, :] = fit_line

    return analysis_result


@njit(parallel=True)
def apply_calculate_area_over_cube(
    cube,
    wvls,
    spec_res,
    low_search,
    high_search
):
    """Applies calculate_area fitting function"""
    xsize, ysize, nbands = cube.shape

    analysis_result = np.empty(
        (xsize, ysize), dtype=cube.dtype
    )

    for i in prange(xsize):
        for j in prange(ysize):
            if np.isnan(cube[i, j, 0]):
                analysis_result[i, j] = np.nan
            else:
                analysis_result[i, j] = calculate_area(
                    cube[i, j], wvls, low_search, high_search, spec_res
                )

    return analysis_result
