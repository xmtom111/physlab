"""Virtual plate reader and pH probe.

Both instruments add Gaussian noise from a seeded numpy Generator, so the
same seed repeats the same readings.
"""

from __future__ import annotations

import numpy as np


class PlateReader:
    """Absorbance reader with a fixed optical path supplied by the caller."""

    def __init__(self, seed: int, noise_std: float = 0.001) -> None:
        """Create the reader.

        Args:
            seed: Generator seed.
            noise_std: Standard deviation of the absorbance noise.
        """
        self._generator = np.random.default_rng(seed)
        self._noise_std = noise_std

    def read(self, absorbance: float) -> float:
        """Return a noisy absorbance.

        Args:
            absorbance: Ideal Beer-Lambert value.

        Returns:
            Absorbance plus one normal draw.
        """
        return float(absorbance + self._generator.normal(0.0, self._noise_std))


class PhProbe:
    """pH probe with seeded noise."""

    def __init__(self, seed: int, noise_std: float = 0.005) -> None:
        """Create the probe.

        Args:
            seed: Generator seed.
            noise_std: Standard deviation of the pH noise.
        """
        self._generator = np.random.default_rng(seed)
        self._noise_std = noise_std

    def read(self, ph: float) -> float:
        """Return a noisy pH.

        Args:
            ph: Ideal equilibrium pH.

        Returns:
            pH plus one normal draw.
        """
        return float(ph + self._generator.normal(0.0, self._noise_std))
