# utils/create_synthetic_spectra.py

# Standard Libraries
# from typing import Union

# External Imports
import numpy as np
# import matplotlib.pyplot as plt
# import h5py as h5

# Local Imports
import spectralops as spop


def gaussian(x, mu, sigma):
    return (np.exp((-1 * (x - mu)**2) / (2 * sigma**2))) /\
           (sigma * np.sqrt(2 * np.pi))


def make_absorption_band(
    wvl: np.ndarray,
    band_location: float,
    band_width: float
) -> np.ndarray:
    """
    Makes a synthetic absorption band using a gaussian.
    """
    return gaussian(wvl, band_location, band_width)


def create_synthetic_lunar_spectrum(
    wvl_arr: np.ndarray,
    short_wvl_rfl: float,
    long_wvl_rfl: float,
    noise_level: float,
    depth_1um: float = 0,
    depth_2um: float = 0
) -> tuple[np.ndarray, ...]:
    """
    Creates a single synthetic spectrum with given characteristics.

    Parameters
    ----------
    nbands: int
        Number of spectral bands in the spectrum.
    wvl_range: tuple[int, int]
        Range of wavelengths for the spectrum.
    short_wvl_rfl: float
        Reflectance value at the shortest wavelength of the spectrum.
    long_wvl_rfl: float
        Reflectance value at the longest wavelength of the spectrum.
    noise_level: float
        Noise level in units of # of standard deviations.
    depth_1um: Union[None, float], optional.
        Adds a 1 micron absorption. Specify the band depth. Default is 0.
    depth_2um: Union[None, float], optional.
        Adds a 2 micron absorption. Specify the band depth. Default is 0.
    """
    rfl_arr = spop.utils.linear_interpolation(
        np.array([wvl_arr.min(), wvl_arr.max()]),
        np.array([short_wvl_rfl, long_wvl_rfl]),
        wvl_arr
    )

    abs1 = gaussian(wvl_arr, 1000, 200) * rfl_arr * depth_1um
    abs2 = gaussian(wvl_arr, 2000, 200) * rfl_arr * depth_2um

    rfl_arr -= abs1
    rfl_arr -= abs2

    rng = np.random.default_rng()
    noise = rng.normal(0, noise_level, rfl_arr.size) * rfl_arr

    rfl_arr += noise

    return wvl_arr, rfl_arr


def create_synthetic_spectral_cube(
    wvl_arr: np.ndarray,
    cube_shape: tuple[int, int]
):
    """
    Creates an entire spectral cube with various characteristics.
    """
    N = cube_shape[0] * cube_shape[1]
    spec_arr = np.zeros((N, len(wvl_arr)))
    rng = np.random.default_rng()

    for i in range(N):
        print(f'\r{i+1} of {N} ({i/N:.2%})', end="")
        shortwvl = rng.uniform(0.06, 0.14)
        spec_config = {
            "wvl_arr": wvl_arr,
            "short_wvl_rfl": shortwvl,
            "long_wvl_rfl": shortwvl + rng.uniform(0.1, 0.2),
            "noise_level": rng.uniform(0.01, 0.04),
            "depth_1um": rng.uniform(50, 100),
            "depth_2um": rng.uniform(0, 50)
        }
        _, spec_arr[i, :] = create_synthetic_lunar_spectrum(**spec_config)

    spec_arr = np.reshape(spec_arr, (*cube_shape, len(wvl_arr)))

    return spec_arr
