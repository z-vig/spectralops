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
from .band_parameters import AbsorptionFeature, AbsorptionFeatureCube
from .spectral_classes import Spectrum
from .spectral_classes import SpectralCube
from . import utils
from . import cube_ops
from .polyfit import polyfit


__all__ = [
    "Spectrum",
    "SpectralCube",
    "AbsorptionFeature",
    "AbsorptionFeatureCube",
    "smoothing",
    "continuum_removal",
    "band_parameters",
    "utils",
    "cube_ops",
    "polyfit"
]
