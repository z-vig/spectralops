# unmixing/endmember.py

# Standard Libraries
from dataclasses import dataclass

# External Imports
import numpy as np


@dataclass
class EndMember:
    """
    Defines one spectral endmember for a spectral unmixing model.

    Attributes
    ----------
    name: str
        Name of the endmember.
    spec: 1-D Array
        Spectrum of the endmember.
    nbands: int
        Number of bands in the endmember.
    """
    name: str
    spec: np.ndarray
    nbands: int
