from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from recon.bart.bart_utils.bart_io import write_cfl
from recon.bart.wave_to_nifti import discover_bart_echoes, restore_bart_intensity


class BartWaveToNiftiTests(unittest.TestCase):
    def test_discovers_single_mprage_output(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root / "inputs"
            outputs = root / "outputs"
            inputs.mkdir()
            outputs.mkdir()
            write_cfl(inputs / "wave_kspace", np.ones((4, 2, 2, 1, 1), np.complex64))
            write_cfl(outputs / "image_wave", np.ones((2, 2, 2, 1, 1), np.complex64))
            (inputs / "manifest.json").write_text(
                json.dumps({"echoes": [{"echo": 1, "wave_kspace": "wave_kspace"}]}),
                encoding="utf-8",
            )
            resolved = discover_bart_echoes(inputs, outputs)
            self.assertEqual(resolved[0]["image"].name, "image_wave")

    def test_restores_bart_kspace_norm(self) -> None:
        image = np.ones((2, 2, 2), np.complex64)
        kspace = np.ones((3, 3), np.complex64)
        restored, scale = restore_bart_intensity(image, kspace)
        self.assertEqual(scale, 3.0)
        np.testing.assert_array_equal(restored, image * 3.0)


if __name__ == "__main__":
    unittest.main()
