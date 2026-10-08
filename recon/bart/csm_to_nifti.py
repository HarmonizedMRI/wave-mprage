#!/usr/bin/env python3
"""Convert BART Wave-MPRAGE sensitivity maps to an RSS NIfTI volume."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np


RECON_ROOT = Path(__file__).resolve().parents[1]
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))


def _cfl_base(path: str | Path) -> Path:
    """Return a BART basename after removing an optional CFL/HDR suffix."""
    value = Path(path).expanduser().resolve()
    return value.with_suffix("") if value.suffix in {".cfl", ".hdr"} else value


def _read_bart_dimensions(header: Path) -> tuple[int, ...]:
    """Read BART dimensions and remove only trailing singleton axes."""
    dimension_line = next(
        (
            line.strip()
            for line in header.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ),
        None,
    )
    if dimension_line is None:
        raise ValueError(f"BART header contains no dimensions: {header}")
    try:
        dimensions = [int(value) for value in dimension_line.split()]
    except ValueError as exc:
        raise ValueError(
            f"Invalid BART dimensions in {header}: {dimension_line!r}"
        ) from exc
    if not dimensions or any(value < 1 for value in dimensions):
        raise ValueError(f"BART dimensions must be positive: {dimensions}")
    while len(dimensions) > 1 and dimensions[-1] == 1:
        dimensions.pop()
    return tuple(dimensions)


def compute_csm_rss(csm: str | Path, *, partition_chunk: int = 4) -> np.ndarray:
    """Compute coil RSS from a BART CSM pair with bounded working memory.

    Args:
        csm: BART CFL basename or its ``.hdr``/``.cfl`` path.
        partition_chunk: Number of partition planes processed per chunk.

    Returns:
        Float32 RSS volume in logical ``(RO, LIN, PAR)`` order.

    Raises:
        ValueError: If dimensions, byte count, chunk size, or values are invalid.
        FileNotFoundError: If either member of the BART pair is missing.
    """
    if partition_chunk < 1:
        raise ValueError("partition_chunk must be positive.")
    base = _cfl_base(csm)
    header = base.with_suffix(".hdr")
    data_file = base.with_suffix(".cfl")
    if not header.is_file() or not data_file.is_file():
        raise FileNotFoundError(f"Missing BART CFL pair: {base}.{{hdr,cfl}}")

    dimensions = _read_bart_dimensions(header)
    if len(dimensions) != 4:
        raise ValueError(
            "Expected BART sensitivity maps in (RO, LIN, PAR, coil) order; "
            f"received dimensions {dimensions}."
        )
    expected_bytes = (
        int(np.prod(dimensions, dtype=np.int64)) * np.dtype(np.complex64).itemsize
    )
    actual_bytes = data_file.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"CFL size mismatch for {data_file}: expected {expected_bytes} bytes, "
            f"found {actual_bytes}."
        )

    maps = np.memmap(
        data_file,
        dtype=np.complex64,
        mode="r",
        shape=dimensions,
        order="F",
    )
    rss = np.empty(dimensions[:3], dtype=np.float32)
    for start in range(0, dimensions[2], partition_chunk):
        stop = min(start + partition_chunk, dimensions[2])
        power = np.zeros(
            (dimensions[0], dimensions[1], stop - start), dtype=np.float32
        )
        for coil in range(dimensions[3]):
            block = np.asarray(maps[:, :, start:stop, coil])
            power += block.real * block.real + block.imag * block.imag
        np.sqrt(power, out=rss[:, :, start:stop])
    if not np.all(np.isfinite(rss)):
        raise ValueError("CSM RSS contains non-finite values.")
    return rss


def convert_csm_to_nifti(
    *,
    csm: str | Path,
    twix: str | Path,
    seq: str | Path,
    output: str | Path,
    reference_nifti: str | Path | None = None,
    partition_chunk: int = 4,
) -> Path:
    """Convert one BART CSM pair to a geometry-correct RSS NIfTI.

    Args:
        csm: BART sensitivity-map basename or CFL/HDR path.
        twix: Matching integrated MPRAGE TWIX file.
        seq: Matching Wave-MPRAGE Pulseq file.
        output: Exact output ``.nii`` or ``.nii.gz`` path.
        reference_nifti: Optional existing reconstruction used to validate shape
            and affine before saving.
        partition_chunk: Number of partition planes processed per CSM chunk.

    Returns:
        Resolved path of the newly written NIfTI.

    Raises:
        FileExistsError: If the output already exists.
        ValueError: If geometry or the optional reference does not match.
    """
    import pypulseq as pp

    import recon_wave_mprage_from_twix_integrated_nifti as native
    from utils.nifti_export_twix import (
        apply_array_axis_flips,
        canonicalize_arrays_to_ras,
        make_nifti_affine_from_twix,
    )

    twix_path = Path(twix).expanduser().resolve()
    seq_path = Path(seq).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    for path, label in ((twix_path, "TWIX"), (seq_path, "Pulseq")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    if output_path.name.endswith(".nii.gz"):
        pass
    elif output_path.suffix != ".nii":
        raise ValueError("output must end in .nii or .nii.gz")

    sequence = pp.Sequence()
    sequence.read(str(seq_path), remove_duplicates=False)
    geometry = native._derive_hardcoded_sag_logical_geometry(sequence.definitions)
    native._assert_sag_geometry(sequence.definitions)
    voxel_size_mm = native._derive_nifti_voxel_size_mm(
        sequence.definitions, geometry
    )

    rss = compute_csm_rss(csm, partition_chunk=partition_chunk)
    expected_shape = (
        int(geometry["Nro"]),
        int(geometry["Nlin"]),
        int(geometry["Npar"]),
    )
    if rss.shape != expected_shape:
        raise ValueError(f"CSM shape {rss.shape} does not match {expected_shape}.")

    # Apply the same logical-array and RAS transforms as reconstructed MPRAGE.
    rss = apply_array_axis_flips([rss], (True, False, False))[0]
    affine, _, _ = make_nifti_affine_from_twix(
        twix_file=str(twix_path),
        npy_shape=rss.shape,
        twix_array_axis_roles=("phase", "readout", "slice"),
        twix_array_axis_flips=(True, False, True),
        twix_coord_system="LPS",
        twix_inplane_rot_sign=-1.0,
        twix_use_fov_for_voxel_size=False,
        voxel_size_mm=voxel_size_mm,
    )
    (rss_ras,), affine_ras, _ = canonicalize_arrays_to_ras([rss], affine)

    if reference_nifti is not None:
        reference_path = Path(reference_nifti).expanduser().resolve()
        reference = nib.load(reference_path)
        if rss_ras.shape != reference.shape:
            raise ValueError(
                f"Canonical CSM shape {rss_ras.shape} does not match reference "
                f"shape {reference.shape}."
            )
        if not np.allclose(affine_ras, reference.affine, rtol=0.0, atol=1e-4):
            raise ValueError("Canonical CSM affine does not match the reference NIfTI.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.asarray(rss_ras, dtype=np.float32), affine_ras)
    image.set_qform(affine_ras, code=1)
    image.set_sform(affine_ras, code=1)
    image.header.set_xyzt_units("mm")
    image.header["descrip"] = b"RSS magnitude of BART ESPIRiT sensitivity maps"
    nib.save(image, output_path)

    saved = nib.load(output_path)
    if saved.shape != rss_ras.shape or not np.allclose(
        saved.affine, affine_ras, rtol=0.0, atol=1e-4
    ):
        raise RuntimeError("Saved NIfTI failed post-write geometry validation.")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert BART Wave-MPRAGE coil sensitivities to an RSS-magnitude "
            "NIfTI with matching TWIX geometry."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csm", required=True, help="BART CSM basename or CFL/HDR path.")
    parser.add_argument("--twix", required=True, help="Matching integrated TWIX file.")
    parser.add_argument("--seq", required=True, help="Matching Pulseq sequence file.")
    parser.add_argument("--out", required=True, help="Exact .nii or .nii.gz output path.")
    parser.add_argument(
        "--reference-nifti",
        default=None,
        help="Optional existing MPRAGE NIfTI for affine and shape validation.",
    )
    parser.add_argument(
        "--partition-chunk",
        type=int,
        default=4,
        help="Partition planes processed at once while calculating RSS.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run command-line CSM conversion and print the validated output geometry."""
    args = _build_parser().parse_args(argv)
    output = convert_csm_to_nifti(
        csm=args.csm,
        twix=args.twix,
        seq=args.seq,
        output=args.out,
        reference_nifti=args.reference_nifti,
        partition_chunk=args.partition_chunk,
    )
    saved = nib.load(output)
    print(f"Saved: {output}")
    print(f"Shape: {saved.shape}")
    print(f"Orientation: {nib.aff2axcodes(saved.affine)}")
    print(f"Voxel size [mm]: {saved.header.get_zooms()[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
