from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

RECON_ROOT = Path(__file__).resolve().parents[1]
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from recon import recon_wave_mprage_from_twix_integrated_nifti as reconstruction


def _centered_fft(array: np.ndarray) -> np.ndarray:
    """Return the centered orthonormal FFT along readout axis zero."""

    return np.fft.fftshift(
        np.fft.fft(
            np.fft.ifftshift(array, axes=(0,)),
            axis=0,
            norm="ortho",
        ),
        axes=(0,),
    )


def _embed_center(array: np.ndarray, oversampling_factor: int) -> np.ndarray:
    """Embed logical image data in the center of an oversampled readout FOV."""

    shape = list(array.shape)
    shape[0] *= oversampling_factor
    embedded = np.zeros(shape, dtype=array.dtype)
    start = shape[0] // 2 - array.shape[0] // 2
    embedded[start : start + array.shape[0]] = array
    return embedded


class MprageCalibrationPipelineTests(unittest.TestCase):
    """Verify integration of corrected ACS processing with BART export."""

    def test_nifti_is_enabled_by_default_and_can_be_disabled(self) -> None:
        base = ["recon", "--twix", "input.dat", "--seq", "input.seq", "--out", "out"]
        with mock.patch.object(sys, "argv", base):
            self.assertTrue(reconstruction._parse_cli_args().save_nifti)
        with mock.patch.object(sys, "argv", base + ["--no-save-nifti"]):
            self.assertFalse(reconstruction._parse_cli_args().save_nifti)

    def test_resume_is_enabled_by_default_and_accepts_explicit_override(self) -> None:
        base = ["recon", "--twix", "input.dat", "--seq", "input.seq", "--out", "out"]
        with mock.patch.object(sys, "argv", base + ["--resume"]):
            self.assertTrue(reconstruction._parse_cli_args().resume)
        with mock.patch.object(sys, "argv", base + ["--no-resume"]):
            self.assertFalse(reconstruction._parse_cli_args().resume)

    def test_integrated_bart_runner_uses_wrapper_defaults_and_skips_nifti(self) -> None:
        """The default backend should invoke BART without running local CG-SENSE."""

        with mock.patch.object(reconstruction.subprocess, "run") as run:
            reconstruction._run_bart_reconstruction(
                bart_input_folder=Path("inputs"),
                bart_output_folder=Path("outputs"),
                twix_file=Path("input.dat"),
                seq_file=Path("input.seq"),
                save_nifti=False,
                save_nifti_phase=False,
                nifti_out_folder=Path("nifti"),
                nifti_sub=None,
                nifti_suffix="MPRAGE",
                nifti_axis_roles=("phase", "readout", "slice"),
                nifti_axis_flips=(True, False, False),
                twix_coord_system="LPS",
                twix_inplane_rot_sign=-1.0,
                twix_use_fov_for_voxel_size=False,
                file_tag="",
                yflip=-1,
                zflip=-1,
            )

        command = run.call_args.args[0]
        self.assertIn("--skip-nifti", command)
        self.assertNotIn("--wave-options", command)
        run.assert_called_once_with(command, check=True)

    def test_bart_calibration_uses_alias_free_logical_acs(self) -> None:
        """The exporter should crop image RO before compression and padding."""

        rng = np.random.default_rng(23)
        logical_image = (
            rng.standard_normal((4, 2, 2, 3))
            + 1j * rng.standard_normal((4, 2, 2, 3))
        ).astype(np.complex64)
        oversampled_image = _embed_center(logical_image, 2)
        oversampled_image[0] = 50.0
        raw_acs = _centered_fft(oversampled_image).astype(np.complex64)
        integrated_ref = torch.zeros((8, 2, 2, 5, 3), dtype=torch.complex64)
        integrated_ref[:, :, :, -1, :] = torch.from_numpy(raw_acs)
        compression = np.eye(3, dtype=np.complex64)[:, :2]

        with mock.patch.object(reconstruction, "load_ref", return_value=integrated_ref):
            actual = reconstruction._build_bart_calibration_kspace(
                mprage_data_file="unused.dat",
                Nx=4,
                Ny=4,
                Nz=4,
                os_factor=2,
                Nacs=2,
                Wcc=compression,
            )

        expected_acs = _centered_fft(logical_image)[..., :2]
        self.assertEqual(actual.shape, (4, 4, 4, 2))
        np.testing.assert_allclose(
            actual[:, 1:3, 1:3, :],
            expected_acs,
            rtol=2e-6,
            atol=2e-6,
        )
        self.assertEqual(np.count_nonzero(actual[:, :1]), 0)
        self.assertEqual(np.count_nonzero(actual[:, 3:]), 0)
        self.assertEqual(np.count_nonzero(actual[:, :, :1]), 0)
        self.assertEqual(np.count_nonzero(actual[:, :, 3:]), 0)

    def test_cache_tags_reject_stride_derived_calibration(self) -> None:
        """Corrected PCA and CSM caches should have distinct identities."""

        self.assertEqual(
            reconstruction._coil_compression_cache_tag("case"),
            "roimgcrop_case",
        )
        self.assertEqual(
            reconstruction._espirit_cache_tag("case", "3d"),
            "roimgcrop_case",
        )
        self.assertEqual(
            reconstruction._espirit_cache_tag("case", "slice2d"),
            "slice2d_sagmask_roimgcrop_case",
        )


if __name__ == "__main__":
    unittest.main()
