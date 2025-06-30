# band_parameters/calculate_center.py

# External Imports
import numpy as np

# Local Imports
from spectralops.utils import find_wvl


def calculate_center(
    contrem_spectrum: np.ndarray,
    wvl: np.ndarray,
    wvl_search_range: np.ndarray
):
    absorption_min_idx, absoprtion_min = find_wvl(wvl, wvl_search_range[0])
    absorption_max_idx, absoprtion_max = find_wvl(wvl, wvl_search_range[1])
    absorption_indices = np.arange(
        absorption_min_idx, absorption_max_idx, 1, dtype=int
    )

    N = 4  # Fit Order

    absorption_spec = contrem_spectrum[absorption_indices]
    absortion_wvl = wvl[absorption_indices]
    
