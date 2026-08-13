#!/usr/bin/env python3
"""Convert a BART Wave-MPRAGE CFL image to a geometry-correct NIfTI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

RECON_ROOT = Path(__file__).resolve().parents[1]
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))

from bart.bart_utils.bart_io import read_cfl


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def discover_bart_echoes(input_dir: str | Path, output_dir: str | Path) -> list[dict[str, Any]]:
    """Resolve the manifest's single matched k-space and reconstructed image pair."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    manifest_path = input_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"BART input manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("echoes")
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError("Wave-MPRAGE BART conversion requires exactly one manifest echo entry.")
    entry = entries[0]
    if int(entry.get("echo", -1)) != 1:
        raise ValueError("Wave-MPRAGE BART manifest echo must be numbered 1.")
    kspace_name = str(entry.get("wave_kspace", ""))
    if kspace_name != "wave_kspace":
        raise ValueError(
            f"Invalid Wave-MPRAGE wave_kspace basename: {kspace_name!r}"
        )
    suffix = kspace_name[len("wave_kspace") :]
    kspace_base = input_path / kspace_name
    image_base = output_path / f"image_wave{suffix}"
    for base in (kspace_base, image_base):
        if not base.with_suffix(".hdr").is_file() or not base.with_suffix(".cfl").is_file():
            raise FileNotFoundError(f"Missing BART CFL pair: {base}.{{hdr,cfl}}")
    return [{"echo": 1, "wave_kspace": kspace_base, "image": image_base}]


def restore_bart_intensity(image: np.ndarray, wave_kspace: np.ndarray) -> tuple[np.ndarray, float]:
    """Undo the per-input k-space normalization performed by ``bart wave``."""

    scale = float(np.linalg.norm(np.asarray(wave_kspace, dtype=np.complex64)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"BART wave k-space norm must be positive and finite; got {scale}.")
    return np.asarray(image, dtype=np.complex64) * scale, scale


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a BART Wave-MPRAGE image using matching TWIX/Pulseq geometry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bart-input-dir", required=True)
    parser.add_argument("--bart-output-dir", required=True)
    parser.add_argument("--twix", required=True, help="Matching Siemens TWIX .dat file.")
    parser.add_argument("--seq", required=True, help="Matching Pulseq .seq file.")
    parser.add_argument("--out", required=True, help="NIfTI output directory.")
    parser.add_argument("--file-tag", default="")
    parser.add_argument("--save-phase", action="store_true")
    parser.add_argument("--nifti-sub", default=None)
    parser.add_argument("--nifti-suffix", default="MPRAGE")
    parser.add_argument(
        "--nifti-axis-roles",
        nargs=3,
        default=("phase", "readout", "slice"),
        metavar=("AXIS0", "AXIS1", "AXIS2"),
    )
    parser.add_argument(
        "--nifti-axis-flips",
        nargs=3,
        type=_parse_bool,
        default=(True, False, False),
        metavar=("FLIP0", "FLIP1", "FLIP2"),
    )
    parser.add_argument("--twix-coord-system", choices=("LPS", "RAS"), default="LPS")
    parser.add_argument("--twix-inplane-rot-sign", type=float, default=-1.0)
    parser.add_argument("--twix-use-fov-for-voxel-size", action="store_true")
    parser.add_argument("--yflip", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--zflip", type=int, choices=(-1, 1), default=-1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_dir = Path(args.bart_input_dir).expanduser().resolve()
    output_dir = Path(args.bart_output_dir).expanduser().resolve()
    twix_file = Path(args.twix).expanduser().resolve()
    seq_file = Path(args.seq).expanduser().resolve()
    nifti_out = Path(args.out).expanduser().resolve()
    for path, label in ((twix_file, "TWIX"), (seq_file, "Pulseq")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")

    import recon_wave_mprage_from_twix_integrated_nifti as native

    seq = native.pp.Sequence()
    seq.read(str(seq_file), remove_duplicates=False)
    defs = seq.definitions
    geom = native._derive_hardcoded_sag_logical_geometry(defs)
    native._assert_sag_geometry(defs)
    os_factor = int(defs.get("ReadoutOversamplingFactor", 4))
    expected_shape = (int(geom["Nro"]), int(geom["Nlin"]), int(geom["Npar"]))

    entry = discover_bart_echoes(input_dir, output_dir)[0]
    image = read_cfl(entry["image"])
    if image.ndim != 3:
        raise ValueError(
            f"BART image must reduce to one 3D map; got {image.shape}. "
            "Multiple ESPIRiT map sets are not supported."
        )
    if image.shape != expected_shape:
        raise ValueError(f"BART image has shape {image.shape}; expected {expected_shape}.")
    restored_image, kspace_norm = restore_bart_intensity(
        image, read_cfl(entry["wave_kspace"])
    )

    voxel_size_mm = native._derive_nifti_voxel_size_mm(defs, geom)
    geometry_diagnostics = native._report_seq_twix_geometry(
        twix_file=str(twix_file),
        geom=geom,
        received_image_shape=expected_shape,
        os_factor=1,
        voxel_size_mm=voxel_size_mm,
        twix_array_axis_roles=tuple(args.nifti_axis_roles),
        twix_array_axis_flips=tuple(args.nifti_axis_flips),
        twix_coord_system=args.twix_coord_system,
        twix_inplane_rot_sign=float(args.twix_inplane_rot_sign),
    )
    geometry_diagnostics["BARTOutputLogicalShape"] = list(expected_shape)
    geometry_diagnostics["BARTReadoutAlreadyDeoversampled"] = True

    ry = int(defs.get("MPRAGE_PE2_R", 2))
    rz = int(defs.get("MPRAGE_PE1_R", 3))
    ncalib = int(defs.get("Calibration_Ncalib1", 72))
    nacs = int(defs.get("Calibration_Nacs", 32))
    metadata = native._build_mprage_nifti_metadata(
        tag_wave="wave",
        file_tag=args.file_tag,
        defs=defs,
        geom=geom,
        os_factor=os_factor,
        Ry=ry,
        Rz=rz,
        ncalib=ncalib,
        nacs=nacs,
        yflip=int(args.yflip),
        zflip=int(args.zflip),
        voxel_size_mm=voxel_size_mm,
        geometry_diagnostics=geometry_diagnostics,
    )
    metadata.update(
        {
            "ReconstructionSoftware": "BART wave",
            "BARTImageInput": entry["image"].name,
            "BARTWaveKspaceInput": entry["wave_kspace"].name,
            "BARTWaveKspaceNormRestored": kspace_norm,
            "BARTInternalNormalizationRestored": True,
            "BARTOutputAlreadyReadoutDeoversampled": True,
        }
    )
    nifti_sub = args.nifti_sub or native._default_nifti_sub(
        "wave",
        geom["res_ro"],
        geom["res_lin"],
        geom["res_par"],
        ry,
        rz,
        args.file_tag,
    )
    native.save_mprage_output_to_nifti(
        image=restored_image,
        twix_file=str(twix_file),
        out_folder=nifti_out,
        nifti_sub=nifti_sub,
        suffix=native._sanitize_filename_component(args.nifti_suffix),
        tag_wave="wave",
        file_tag=args.file_tag,
        voxel_size_mm=voxel_size_mm,
        crop_readout_os=1,
        save_phase=bool(args.save_phase),
        twix_array_axis_roles=tuple(args.nifti_axis_roles),
        twix_array_axis_flips=tuple(args.nifti_axis_flips),
        twix_coord_system=args.twix_coord_system,
        twix_inplane_rot_sign=float(args.twix_inplane_rot_sign),
        twix_use_fov_for_voxel_size=bool(args.twix_use_fov_for_voxel_size),
        metadata=metadata,
    )
    print(f"Converted BART Wave-MPRAGE image to {nifti_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
