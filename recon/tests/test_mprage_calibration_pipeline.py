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

    def test_espirit_crop_is_accepted_by_the_integrated_cli(self) -> None:
        """The top-level crop option should configure either reconstruction backend."""
        argv = [
            "recon",
            "--twix",
            "input.dat",
            "--seq",
            "input.seq",
            "--out",
            "out",
            "--espirit-crop",
            "0.5",
        ]
        with mock.patch.object(sys, "argv", argv):
            self.assertEqual(reconstruction._parse_cli_args().espirit_crop, 0.5)

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
                espirit_crop=0.5,
            )

        command = run.call_args.args[0]
        self.assertIn("--skip-nifti", command)
        self.assertNotIn("--wave-options", command)
        ecalib_start = command.index("--ecalib-options")
        self.assertEqual(
            command[ecalib_start : ecalib_start + 4],
            ["--ecalib-options", "-c", "0.5", "--end-ecalib-options"],
        )
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
        expected_full = np.zeros((4, 4, 4, 2), dtype=np.complex64)
        expected_full[:, 1:3, 1:3, :] = expected_acs
        expected_full = np.roll(
            np.flip(expected_full, axis=(1, 2)),
            shift=(1, 1),
            axis=(1, 2),
        )
        self.assertEqual(actual.shape, (4, 4, 4, 2))
        np.testing.assert_allclose(
            actual,
            expected_full,
            rtol=2e-6,
            atol=2e-6,
        )
        self.assertEqual(np.count_nonzero(actual), np.count_nonzero(expected_full))
        self.assertGreater(np.count_nonzero(actual[:, 3]), 0)

    def test_cache_tags_reject_stride_derived_calibration(self) -> None:
        """Corrected PCA and CSM caches should have distinct identities."""

        self.assertEqual(
            reconstruction._coil_compression_cache_tag("case"),
            "roimgcrop_case",
        )
        self.assertEqual(
            reconstruction._espirit_cache_tag("case", "3d"),
            "pealign_roimgcrop_case",
        )
        self.assertEqual(
            reconstruction._espirit_cache_tag("case", "slice2d"),
            "slice2d_sagmask_pealign_roimgcrop_case",
        )

    def test_refscan_alignment_is_centered_lin_par_reversal(self) -> None:
        """Refscan alignment should reverse PE axes without shifting k-space zero."""

        source = torch.arange(2 * 4 * 6 * 2).reshape(2, 4, 6, 2)
        actual = reconstruction._align_refscan_acs_to_imaging_grid(source)

        expected = source[:, torch.tensor([0, 3, 2, 1])]
        expected = expected[:, :, torch.tensor([0, 5, 4, 3, 2, 1])]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[:, 0, 0], source[:, 0, 0])

    def test_psf_composition_maps_refscan_slopes_to_imaging_polarity(self) -> None:
        """Theory and fitted slopes should reverse with the refscan PE coordinates."""

        delta_ky = np.array([0.35, -0.2])
        delta_kz = np.array([-0.15, 0.4])
        y_norm = np.array([-0.5, 0.0, 0.25])
        z_norm = np.array([-0.5, 0.0])
        a_fit = np.array([0.7, -0.3])
        b_fit = np.array([-0.4, 0.2])
        c_fit = np.array([0.25, -0.1])

        calibrated, theory = reconstruction._compose_calibrated_psf_on_imaging_grid(
            delta_ky_idx=delta_ky,
            delta_kz_idx=delta_kz,
            y_norm=y_norm,
            z_norm=z_norm,
            a_fit=a_fit,
            b_fit=b_fit,
            c_fit=c_fit,
            yflip=1,
            zflip=1,
        )

        image_y, image_z = np.meshgrid(y_norm, z_norm, indexing="ij")
        expected_theory = np.exp(
            1j
            * 2.0
            * np.pi
            * (
                delta_ky[:, None, None] * image_y[None, ...]
                + delta_kz[:, None, None] * image_z[None, ...]
            )
        )
        expected_deviation = (
            -a_fit[:, None, None] * image_y[None, ...]
            - b_fit[:, None, None] * image_z[None, ...]
            + c_fit[:, None, None]
        )

        np.testing.assert_allclose(theory.numpy(), expected_theory, atol=1e-6)
        np.testing.assert_allclose(
            calibrated.numpy(),
            expected_theory * np.exp(1j * expected_deviation),
            atol=1e-6,
        )
        self.assertEqual(
            reconstruction._current_psf_composition_provenance(),
            {
                "version": 4,
                "calibration_to_imaging_pe_coordinate_sign": {
                    "LIN": -1,
                    "PAR": -1,
                },
                "constant_phase_sign": 1,
                "bart_forward_application": "direct",
            },
        )

    def test_psf_composition_preserves_constant_phase_sign(self) -> None:
        """Spatial phase conventions should not conjugate the constant term."""

        calibrated, theory = reconstruction._compose_calibrated_psf_on_imaging_grid(
            delta_ky_idx=np.zeros(1),
            delta_kz_idx=np.zeros(1),
            y_norm=np.array([-0.5, 0.0]),
            z_norm=np.array([-0.5, 0.0]),
            a_fit=np.zeros(1),
            b_fit=np.zeros(1),
            c_fit=np.array([0.4]),
            yflip=1,
            zflip=1,
        )

        np.testing.assert_allclose(theory.numpy(), 1.0, atol=1e-7)
        np.testing.assert_allclose(calibrated.numpy(), np.exp(0.4j), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
