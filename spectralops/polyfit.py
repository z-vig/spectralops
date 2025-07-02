# utils/polyfit.py

# Standard Libraries
from typing import Optional, Union

# External Imports
import numpy as np
from numba import njit

# Local imports
from .cube_ops import apply_polyfit_over_cube


def polyfit_single(
    spectrum: np.ndarray,
    wvl: Union[None, np.ndarray],
    order: Union[None, int],
    design_matrices: Optional[tuple[np.ndarray, ...]] = None,
    return_coefficients: bool = True
) -> np.ndarray:
    """
    Fits a polynomial of order `N` to `x` and `y` data. Optionally can be used
    in a loop by specifying the `design_matrix` elements.

    Parameters
    ----------
    spectrum: np.ndarray
        Y Data. Spectral Data.
    wvl: np.ndarray
        X Data. Wavelengths. Can be None if `design_matrices` are supplied.
    order: int
        Order of polynomial fit. Can be None if `design_matrices` are supplied.
    design_matrices: Nothing or tuple[np.ndarray]
        Three design matrix components. X, Xt and XtX. If None (default),
        these are calculated from `wvl` data.
    return_coefficients: bool, optional
        If True (defualt), returns fit coefficients rather than a fit line.

    Returns
    -------
    beta: np.ndarray
        Coefficients of the fit.
    """

    if design_matrices is None:
        if (wvl is None) or (order is None):
            raise ValueError(
                "If design matrices are not specified, both x data and order"
                "must be specified"
            )
        else:
            X = np.vander(wvl, order+1, increasing=True)
            Xt = X.T
            XtX = Xt @ X
    else:
        X, Xt, XtX = design_matrices

    beta = np.linalg.inv(XtX) @ (Xt @ spectrum)

    if return_coefficients:
        return beta
    else:
        return X @ beta


@njit
def polyfit_single_nb(
    spectrum: np.ndarray,
    X: np.ndarray,
    Xt: np.ndarray,
    XtX: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fits a polynomial of order `N` to `x` and `y` data. Optionally can be used
    in a loop by specifying the `design_matrix` elements.

    Parameters
    ----------
    spectrum: np.ndarray
        Y Data. Spectral Data.
    design_matrices: tuple[np.ndarray]
        Three design matrix components. X, Xt and XtX.

    Returns
    -------
    beta: np.ndarray
        Coefficients of the fit.
    """
    beta = np.linalg.inv(XtX) @ (Xt @ spectrum)

    fit_line = X @ beta

    return fit_line, np.full(fit_line.shape, np.nan)


def polyfit_spectral_cube(
    spectral_cube: np.ndarray,
    wvl: np.ndarray,
    order: int
) -> np.ndarray:
    """
    Performs polynomial fits for an entire spectral cube of data.

    Parameters
    ----------
    spectral_cube: np.ndarray
        Spectral data cube.
    wvl: np.ndarray
        Wavelength values.
    order: int
        Order of polyfit.
    return_coefficients: bool, optional
        If True (defualt), returns a cube of coefficient rather than fit lines.

    Returns
    -------
    fit_cube: np.ndarray
        Either a cube of fitted coefficients or fitted lines.
    """
    X = np.vander(wvl, order + 1)
    Xt = np.ascontiguousarray(X.T)
    XtX = Xt @ X
    design_matrices = (X, Xt, XtX)

    fit_cube = apply_polyfit_over_cube(spectral_cube, *design_matrices)
    return fit_cube


def polyfit(
    xdata: np.ndarray,
    ydata: np.ndarray,
    order: int
):
    if ydata.ndim == 1:
        return polyfit_single(ydata, xdata, order)
    elif ydata.ndim == 3:
        return polyfit_spectral_cube(ydata, xdata, order)
    else:
        raise ValueError(f"Y Data of {ydata.ndim} dimensions is unsupported.")
