# spectralops/__init__.py

"""
spectralops
---

Processing operations for individual spectra and spectral image cubes in
Python.
"""

from .Spectrum import Spectrum
from .SpectralCube import SpectralCube

__all__ = [
    "Spectrum",
    "SpectralCube"
]
