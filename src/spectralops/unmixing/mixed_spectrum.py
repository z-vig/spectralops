# unmixing/mixed_spectrum.py

# External Imports
import numpy as np

# Local Imports
from spectralops.spectral_classes.spectrum import Spectrum
from .endmember import EndMember


class MixedSpectrum:
    def __init__(
        self,
        mixed_spectrum: Spectrum,
        endmembers: list[EndMember],
    ):
        self.mixed_spectrum = mixed_spectrum
        self.endmembers = endmembers

        self._mixed_spectrum = mixed_spectrum
        self._G = np.empty(
            [
                self._mixed_spectrum.nbands,
                len(endmembers)
            ],
            dtype=float
        )
        self._d = self._mixed_spectrum.spectrum

        self._unmix_spectrum()

    def _unmix_spectrum(self, sum2one: bool = True):
        GtG = self._G.T @ self._G
        Gd = self._G @ self._d

        if sum2one:
            # Adding a column of ones
            GtG = np.hstack(
                (GtG, np.ones((GtG.shape[0], 1)))
            )

            # Adding a row of all ones except for last column, which is 0
            row = np.ones((1, GtG.shape[1]))
            row = row[0, -1] = 0
            GtG = np.vstack((GtG, row))

            # Adding a 1 to the end of Gd
            Gd = np.vstack((Gd, 1))

        m = np.linalg.solve(GtG, Gd)

        self.prediction = self._G @ m

        self.residual_spectrum = self._d - self.prediction

        self.residual = np.sqrt(np.sum(
            self.residual_spectrum**2 / self.residual_spectrum.size
        ))

        self.fractions = {
            self.endmembers[i].name: m[i, 0]
            for i in range(m.shape[0])
        }
