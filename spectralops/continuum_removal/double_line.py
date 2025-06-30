# double_line.py

import numpy as np
# from scipy.interpolate import interp1d
from numba import njit

from spectralops.utils import linear_interpolation


@njit
def double_line_nb(
    spectrum: np.ndarray,
    wvls: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Numba-optimized double-line continuum removal.
    """
    # Getting initial continuum line parameters
    anchor_pts = np.array([700, 1550, 2600])

    cont1_band_idx = np.empty(anchor_pts.size, dtype=np.int16)
    cont1_band_wvls = np.empty(anchor_pts.size, dtype=np.float64)
    for n in range(anchor_pts.size):
        idx = np.argmin(np.abs(wvls - anchor_pts[n]))
        cont1_band_idx[n] = idx
        cont1_band_wvls[n] = wvls[idx]

    cont1_spectrum_values = spectrum[cont1_band_idx]

    continuum1 = linear_interpolation(
        cont1_band_wvls, cont1_spectrum_values, wvls
    )

    continuum1_removed = spectrum / continuum1

    cont2_ranges = {
        "range1": (650, 1000),
        "range2": (1350, 1600),
        "range3": (2000, 2600)
    }

    cont2_band_idx = np.zeros(len(cont2_ranges), dtype=np.int16)

    # Locating dynamic maxima for second continuum parameters
    for n, val in enumerate(cont2_ranges.values()):
        lo_idx = np.argmin(np.abs(wvls - val[0]))
        hi_idx = np.argmin(np.abs(wvls - val[1]))

        max_spectrum_idx = np.argmax(
            continuum1_removed[np.arange(lo_idx, hi_idx, dtype=np.int16)]
        ) + lo_idx
        cont2_band_idx[n] = max_spectrum_idx

    cont2_band_wvl = wvls[cont2_band_idx]
    cont2_spectrum_values = spectrum[cont2_band_idx]

    continuum2 = linear_interpolation(
        cont2_band_wvl, cont2_spectrum_values, wvls
    )
    continuum2_removed = spectrum / continuum2

    return continuum2_removed, continuum2


# def double_line(
#     spectrum: np.ndarray,
#     wvls: np.ndarray
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Removes the continuum of a spectrum using the double line method of
#     Henderson et al., 2023. First, a rough continuum is removed using fixed
#     points at 700, 1550 and 2600 nm. Next, three points are chosen from the
#     maxima of this spectrum at:
#         - 650 to 1000 nm
#         - 1350 to 1600 nm
#         - 2000 to 2600 nm
#     Finally, with these endpoints, the final continuum is calculated from the
#     rfl values at these points on the original spectrum.

#     Parameters
#     ----------
#     wvls: ndarray
#         Wavelength array (in nm).
#     spectrum: ndarray
#         Spectrum array.

#     Returns
#     -------
#     continuum2_removed: ndarray
#         Values of input spectrum with the continuum removed.
#     continuum2: ndarray
#         Values of continuum.
#     """

#     # Getting initial continuum line parameters
#     cont1_band_idx = [find_wvl(wvls, i)[0] for i in [700, 1550, 2600]]
#     cont1_band_wvl = [find_wvl(wvls, i)[1] for i in [700, 1550, 2600]]
#     cont1_spectrum_values = spectrum[cont1_band_idx]

#     # Running a linear interpolation over continuum line
#     f1 = interp1d(
#         cont1_band_wvl,
#         cont1_spectrum_values,
#         kind='linear',
#         fill_value='extrapolate'
#     )

#     continuum1 = f1(wvls)

#     continuum1_removed = spectrum / continuum1

#     cont2_ranges = {
#         "range1": (650, 1000),
#         "range2": (1350, 1600),
#         "range3": (2000, 2600)
#     }

#     cont2_band_idx = []

#     # Locating dynamic maxima for second continuum parameters
#     for val in cont2_ranges.values():
#         lo_idx = find_wvl(wvls, val[0])[0]
#         hi_idx = find_wvl(wvls, val[1])[0]
#         max_spectrum_idx = np.argmax(
#             continuum1_removed[np.arange(lo_idx, hi_idx, dtype=int)]
#         ) + lo_idx
#         cont2_band_idx.append(max_spectrum_idx)

#     cont2_band_wvl = wvls[cont2_band_idx]
#     cont2_spectrum_values = spectrum[cont2_band_idx]

#     f2 = interp1d(
#         cont2_band_wvl,
#         cont2_spectrum_values,
#         kind='linear',
#         fill_value='extrapolate'
#     )

#     continuum2 = f2(wvls)

#     continuum2_removed = spectrum / continuum2

#     return continuum2_removed, continuum2
