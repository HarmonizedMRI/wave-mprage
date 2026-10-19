from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np

from recon.bart.bart_utils.bart_io import write_cfl
from recon.bart.wave_to_nifti import (
    BART_AFFINE_AXIS_FLIPS,
    _build_parser,
    discover_bart_echoes,
    restore_bart_intensity,
)
from recon import recon_wave_mprage_from_twix_integrated_nifti as reconstruction


def _sagittal_twix_header():
    """Return a fake mapVBVD scan, header and MeasYaps for sagittal MPRAGE.

    Returns:
        Tuple ``(scan, hdr, yaps)`` whose MeasYaps uses the tuple keys read by
        ``utils.nifti_export_twix``: slice normal dSag = +1 at the isocentre
        and no in-plane rotation, as in the head-first supine Wave-MPRAGE scans.
    """

    yaps = {("sSliceArray", "asSlice", "0", "sNormal", "dSag"): 1.0}
    hdr = {"MeasYaps": yaps}
    return {"hdr": hdr}, hdr, yaps


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

    def test_discovers_cartesian_nowave_output(self) -> None:
        """Cartesian manifests should resolve PICS inputs without a PSF."""

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root / "inputs"
            outputs = root / "outputs"
            inputs.mkdir()
            outputs.mkdir()
            write_cfl(
                inputs / "cartesian_kspace",
                np.ones((2, 2, 2, 1, 1), np.complex64),
            )
            write_cfl(
                outputs / "image_nowave",
                np.ones((2, 2, 2, 1, 1), np.complex64),
            )
            (inputs / "manifest.json").write_text(
                json.dumps(
                    {
                        "reconstruction": {
                            "mode": "cartesian",
                            "tool": "pics",
                            "readout_oversampling_removed": True,
                        },
                        "echoes": [
                            {"echo": 1, "cartesian_kspace": "cartesian_kspace"}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            resolved = discover_bart_echoes(inputs, outputs)[0]
            self.assertEqual(resolved["mode"], "cartesian")
            self.assertEqual(resolved["image"].name, "image_nowave")
            self.assertEqual(resolved["kspace"].name, "cartesian_kspace")

    def test_bart_axes_map_to_validated_patient_directions(self) -> None:
        """Raw BART RO, LIN and PAR ramps should point S, A and patient left.

        Each ramp increases along one raw BART axis and is exported with the
        converter's default axis roles and array flips plus the BART affine
        flips. Distinct axis lengths also verify the RAS axis permutation.
        """

        parser = _build_parser()
        roles = tuple(parser.get_default("nifti_axis_roles"))
        array_flips = tuple(parser.get_default("nifti_axis_flips"))
        shape = (6, 5, 4)
        # Canonical RAS axis and slope sign expected for each raw BART axis.
        expected = {0: (2, 1.0), 1: (1, 1.0), 2: (0, -1.0)}
        for raw_axis, (ras_axis, sign) in expected.items():
            with self.subTest(raw_axis=raw_axis), tempfile.TemporaryDirectory() as folder:
                ramp_shape = [1, 1, 1]
                ramp_shape[raw_axis] = shape[raw_axis]
                ramp = np.arange(1, shape[raw_axis] + 1, dtype=np.float32)
                image = np.broadcast_to(ramp.reshape(ramp_shape), shape).astype(
                    np.complex64
                )
                with mock.patch(
                    "utils.nifti_export_twix._get_twix_scan_and_hdr",
                    return_value=_sagittal_twix_header(),
                ):
                    reconstruction.save_mprage_output_to_nifti(
                        image=image,
                        twix_file="unused.dat",
                        out_folder=folder,
                        nifti_sub="orientation",
                        twix_array_axis_roles=roles,
                        twix_array_axis_flips=array_flips,
                        twix_affine_axis_flips=BART_AFFINE_AXIS_FLIPS,
                    )

                nifti = nib.load(
                    Path(folder)
                    / "sub-orientation"
                    / "sub-orientation_part-mag_MPRAGE.nii.gz"
                )
                self.assertEqual(nib.aff2axcodes(nifti.affine), ("R", "A", "S"))
                data = np.asanyarray(nifti.dataobj)
                other_axes = tuple(axis for axis in range(3) if axis != ras_axis)
                profile = data.mean(axis=other_axes)
                self.assertEqual(profile.size, shape[raw_axis])
                self.assertGreater(sign * (profile[-1] - profile[0]), 0.0)

    def test_native_exports_use_bart_affine_flips(self) -> None:
        """Both native CG-SENSE writer calls should use the BART affine flips.

        Native and BART reconstructions share one logical image orientation, so
        neither native call may fall back to the writer's default affine flips.
        """

        self.assertEqual(
            reconstruction.RECON_NIFTI_AFFINE_AXIS_FLIPS, BART_AFFINE_AXIS_FLIPS
        )
        source = inspect.getsource(reconstruction)
        self.assertEqual(
            source.count("twix_affine_axis_flips=RECON_NIFTI_AFFINE_AXIS_FLIPS"), 2
        )


if __name__ == "__main__":
    unittest.main()
