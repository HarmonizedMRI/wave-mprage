from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from recon.bart.bart_utils.bart_io import export_wave_inputs, read_cfl, write_cfl


def _read_cfl(base: Path) -> np.ndarray:
    return read_cfl(base, trim_trailing_singletons=False)


class BartIoTests(unittest.TestCase):
    def test_write_cfl_round_trip(self) -> None:
        expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4).astype(np.complex64)
        with tempfile.TemporaryDirectory() as folder:
            base = write_cfl(Path(folder) / "array", expected)
            np.testing.assert_array_equal(read_cfl(base), expected)

    def test_read_cfl_rejects_truncated_data(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            base = write_cfl(Path(folder) / "array", np.ones((2, 3), np.complex64))
            base.with_suffix(".cfl").write_bytes(b"\x00" * 8)
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                read_cfl(base)

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
