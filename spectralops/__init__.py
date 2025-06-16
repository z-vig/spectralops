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
from .spectrum import Spectrum
from .spectral_cube import SpectralCube

__all__ = [
    smoothing,
    continuum_removal,
    Spectrum,
    SpectralCube
]
