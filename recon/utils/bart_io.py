"""BART CFL export helpers for Wave-CAIPI reconstruction inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_cfl(path: str | Path, array: np.ndarray) -> Path:
    """Write a complex array as a column-major BART ``.hdr``/``.cfl`` pair."""

    base = Path(path)
    if base.suffix in {".hdr", ".cfl"}:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(array, dtype=np.complex64)
    if data.ndim < 1 or any(int(size) < 1 for size in data.shape):
        raise ValueError(f"BART CFL output must have non-empty dimensions: {data.shape}.")
    base.with_suffix(".hdr").write_text(
        "# Dimensions\n" + " ".join(str(int(size)) for size in data.shape) + "\n",
        encoding="utf-8",
    )
    with base.with_suffix(".cfl").open("wb") as stream:
        np.ravel(data, order="F").tofile(stream)
    return base


def _complex64(name: str, array: Any, ndim: int) -> np.ndarray:
    result = np.asarray(array, dtype=np.complex64)
    if result.ndim != ndim or any(int(size) < 1 for size in result.shape):
        raise ValueError(f"{name} must be a non-empty {ndim}D array; got {result.shape}.")
    return result


def export_wave_inputs(
    out_folder: str | Path,
    *,
    wave_kspace: np.ndarray,
    calibrated_psf: np.ndarray,
    coil_sens: np.ndarray,
    kspace_calib: np.ndarray,
) -> Path:
    """Export reconstruction-native arrays for BART ``ecalib`` and ``wave``."""

    destination = Path(out_folder)
    destination.mkdir(parents=True, exist_ok=True)
    kspace = _complex64("wave_kspace", wave_kspace, 5)
    psf = _complex64("calibrated_psf", calibrated_psf, 4)
    maps = _complex64("coil_sens", coil_sens, 4)
    calib = _complex64("kspace_calib", kspace_calib, 4)

    wx, sy, sz, necho, nc = map(int, kspace.shape)
    if psf.shape != (necho, wx, sy, sz):
        raise ValueError(
            f"calibrated_psf must have shape {(necho, wx, sy, sz)}; got {psf.shape}."
        )
    sx = int(maps.shape[1])
    if maps.shape != (nc, sx, sy, sz):
        raise ValueError(f"coil_sens must have shape {(nc, sx, sy, sz)}; got {maps.shape}.")
    if calib.shape != (sx, sy, sz, nc):
        raise ValueError(
            f"kspace_calib must have shape {(sx, sy, sz, nc)}; got {calib.shape}."
        )

    exported_maps = np.moveaxis(maps, 0, 3)[..., None]
    write_cfl(destination / "coil_sens", exported_maps)
    write_cfl(destination / "kspace_calib", calib)
    files: list[dict[str, Any]] = []
    for echo_index in range(necho):
        suffix = "" if necho == 1 else f"_echo-{echo_index + 1:02d}"
        kspace_name = f"wave_kspace{suffix}"
        psf_name = f"psf{suffix}"
        exported_kspace = kspace[:, :, :, echo_index, :, None]
        exported_psf = psf[echo_index, :, :, :, None, None]
        write_cfl(destination / kspace_name, exported_kspace)
        write_cfl(destination / psf_name, exported_psf)
        files.append(
            {
                "echo": echo_index + 1,
                "wave_kspace": kspace_name,
                "wave_kspace_shape": list(exported_kspace.shape),
                "psf": psf_name,
                "psf_shape": list(exported_psf.shape),
            }
        )

    manifest = {
        "format": "BART CFL",
        "dimension_order": ["READ", "PHS1", "PHS2", "COIL", "MAPS"],
        "coil_sens": "coil_sens",
        "coil_sens_shape": list(exported_maps.shape),
        "kspace_calib": "kspace_calib",
        "kspace_calib_shape": list(calib.shape),
        "echoes": files,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
