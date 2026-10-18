from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


RECON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RECON_ROOT))

from utils.noise_prewhitening import (  # noqa: E402
    estimate_noise_covariance,
    make_noise_whitener,
    validate_noise_whitener,
)


class NoisePrewhiteningTests(unittest.TestCase):
    """Verify covariance convention and whitening numerical behavior."""

    def test_correlated_samples_are_whitened_to_identity(self) -> None:
        """Estimated covariance and row-coil whitener should be consistent."""

        generator = np.random.default_rng(11)
        mixing = np.array(
            [[1.0, 0.2 + 0.1j], [0.0, 0.7]], dtype=np.complex128
        )
        samples = (
            generator.standard_normal((20000, 2))
            + 1j * generator.standard_normal((20000, 2))
        ) @ mixing
        covariance = estimate_noise_covariance(samples)
        whitener, _ = make_noise_whitener(covariance)
        self.assertLess(validate_noise_whitener(covariance, whitener), 2e-5)


if __name__ == "__main__":
    unittest.main()
