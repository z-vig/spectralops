"""
# spectralops

Processing operations for individual spectra and spectral image cubes in
Python.

### Available Modules:
- smoothing
- continuum_removal

### Base Classes:
- Spectrum
- SpectralCube
"""

from . import smoothing
from . import continuum_removal
from . import band_parameters
from .spectrum import Spectrum
from .spectral_cube import SpectralCube, apply_over_cube
from . import utils

__all__ = [
    "smoothing",
    "continuum_removal",
    "band_parameters",
    "Spectrum",
    "SpectralCube",
    "apply_over_cube",
    "utils"
]
