# utils/cube_ops.py

# External Imports
import numpy as np
from numba import njit, prange


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
