from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from recon.utils.bart_io import export_wave_inputs, write_cfl


def _read_cfl(base: Path) -> np.ndarray:
    shape = tuple(
        int(value)
        for value in base.with_suffix(".hdr").read_text(encoding="utf-8").splitlines()[1].split()
    )
    return np.fromfile(base.with_suffix(".cfl"), np.complex64).reshape(shape, order="F")


class BartIoTests(unittest.TestCase):
    def test_write_cfl_round_trip(self) -> None:
        expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4).astype(np.complex64)
        with tempfile.TemporaryDirectory() as folder:
            base = write_cfl(Path(folder) / "array", expected)
            np.testing.assert_array_equal(_read_cfl(base), expected)

    def test_required_wave_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            export_wave_inputs(
                folder,
                wave_kspace=np.ones((8, 3, 2, 1, 2), np.complex64),
                calibrated_psf=np.ones((1, 8, 3, 2), np.complex64),
                coil_sens=np.ones((2, 4, 3, 2), np.complex64),
                kspace_calib=np.ones((4, 3, 2, 2), np.complex64),
            )
            self.assertEqual(_read_cfl(Path(folder) / "wave_kspace").shape, (8, 3, 2, 2, 1))
            self.assertEqual(_read_cfl(Path(folder) / "psf").shape, (8, 3, 2, 1, 1))
            self.assertEqual(_read_cfl(Path(folder) / "coil_sens").shape, (4, 3, 2, 2, 1))
            self.assertEqual(_read_cfl(Path(folder) / "kspace_calib").shape, (4, 3, 2, 2))


if __name__ == "__main__":
    unittest.main()
