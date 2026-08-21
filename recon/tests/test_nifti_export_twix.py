from __future__ import annotations

import unittest

import nibabel as nib
import numpy as np

from recon.utils.nifti_export_twix import canonicalize_arrays_to_ras


class CanonicalRasTests(unittest.TestCase):
    def test_reorients_matched_arrays_without_resampling(self) -> None:
        magnitude = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        phase = -magnitude
        affine = np.array(
            [
                [0.0, 0.0, -1.0, 3.0],
                [0.0, 1.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        arrays, canonical_affine, transform = canonicalize_arrays_to_ras(
            (magnitude, phase), affine
        )
        expected = nib.as_closest_canonical(nib.Nifti1Image(magnitude, affine))

        self.assertEqual(nib.aff2axcodes(canonical_affine), ("R", "A", "S"))
        np.testing.assert_array_equal(arrays[0], np.asanyarray(expected.dataobj))
        np.testing.assert_array_equal(arrays[1], -arrays[0])
        np.testing.assert_allclose(canonical_affine, expected.affine)
        self.assertEqual(np.asarray(transform).shape, (3, 2))

    def test_rejects_mismatched_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "matched 3D image shapes"):
            canonicalize_arrays_to_ras(
                (np.zeros((2, 3, 4)), np.zeros((2, 3, 5))), np.eye(4)
            )


if __name__ == "__main__":
    unittest.main()
