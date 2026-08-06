"""Encapsulated ESPIRiT calibration backends for Wave-MPRAGE reconstruction.

The public :func:`estimate_espirit_maps` entry point supports:

``3d``
    The existing native SigPy 3D ESPIRiT calibration. This remains the
    reference/default method and can run on either CPU or GPU.

``slice2d``
    CPU-only hybrid-space calibration. The logical readout axis is transformed
    to image space, independent 2D ESPIRiT calibrations are run across the two
    phase-encoding dimensions, and the resulting maps are stacked along the
    logical readout axis.

The slice-wise implementation deliberately lives outside the main reconstruction
script so that worker scheduling, axis handling, and map assembly remain an
implementation detail rather than being mixed into the reconstruction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sigpy as sp
import sigpy.mri as mr
from joblib import Parallel, cpu_count, delayed, parallel_config
from scipy.ndimage import binary_closing, binary_dilation, label


@dataclass(frozen=True)
class EspiritCalibrationInfo:
    """Execution details returned with an ESPIRiT map estimate."""

    mode: str
    cpu_workers: Optional[int]
    logical_ro_slices: int
    zero_input_slices: tuple[int, ...]
    masked_low_signal_slices: tuple[int, ...] = ()
    support_threshold: Optional[float] = None
    support_peak_rms: Optional[float] = None
    support_first_active: Optional[int] = None
    support_last_active: Optional[int] = None


def estimate_espirit_maps(
    kspace: np.ndarray,
    *,
    mode: str = "3d",
    device: Optional[sp.Device] = None,
    crop: float = 0.8,
    calib_width: int = 24,
    thresh: float = 0.02,
    kernel_width: int = 6,
    max_iter: int = 100,
    cpu_workers: Optional[int] = None,
    slice_support: str = "off",
    slice_support_noise_fraction: float = 0.10,
    slice_support_noise_multiplier: float = 3.0,
    slice_support_relative_floor: float = 1e-4,
    slice_support_padding: int = 3,
    slice_support_diagnostic_path: Optional[str] = None,
) -> tuple[np.ndarray, EspiritCalibrationInfo]:
    """Estimate coil sensitivity maps using a selected ESPIRiT backend.

    Parameters
    ----------
    kspace
        Coil-first logical k-space with shape ``(coil, RO, LIN, PAR)``. Readout
        oversampling must already have been removed before calling this function.
    mode
        ``"3d"`` for native 3D SigPy ESPIRiT or ``"slice2d"`` for parallel
        hybrid-space 2D ESPIRiT along logical RO.
    device
        SigPy device used by the 3D backend. The slice2d backend is CPU-only.
    crop, calib_width, thresh, kernel_width, max_iter
        Parameters passed directly to ``sigpy.mri.app.EspiritCalib``.
    cpu_workers
        Number of process workers for slice2d. ``None`` selects the available
        physical-core count, limited by the number of logical RO slices.
    slice_support
        ``"off"`` keeps the original slice2d behavior. ``"sag"`` enables a
        conservative whole-plane support guard for sagittal Wave-MPRAGE, where
        logical RO is physical superior-inferior. The guard rejects only
        low-signal RO planes before per-slice ESPIRiT and pads the detected
        support to preserve superior scalp and inferior jaw/neck anatomy.
    slice_support_noise_fraction, slice_support_noise_multiplier,
    slice_support_relative_floor, slice_support_padding
        Parameters for sagittal RO support detection. The defaults are
        deliberately conservative for SAG head imaging.
    slice_support_diagnostic_path
        Optional PNG path for the normalized RO-RMS support diagnostic. A
        matching ``.npz`` file is written beside it.
    """

    kspace = np.asarray(kspace, dtype=np.complex64)
    if kspace.ndim != 4:
        raise ValueError(
            "ESPIRiT input must have shape (coil, RO, LIN, PAR); "
            f"received {kspace.shape}."
        )
    if any(int(size) < 1 for size in kspace.shape):
        raise ValueError(f"ESPIRiT input contains an empty dimension: {kspace.shape}.")

    mode = str(mode).strip().lower()
    if mode not in ("3d", "slice2d"):
        raise ValueError("ESPIRiT calibration mode must be '3d' or 'slice2d'.")

    slice_support = str(slice_support).strip().lower()
    if slice_support not in ("off", "sag"):
        raise ValueError("slice_support must be 'off' or 'sag'.")

    crop = float(crop)
    if not np.isfinite(crop) or not 0.0 <= crop <= 1.0:
        raise ValueError("ESPIRiT crop must be a finite value between 0 and 1.")

    calib_width = _positive_int("calib_width", calib_width)
    kernel_width = _positive_int("kernel_width", kernel_width)
    max_iter = _positive_int("max_iter", max_iter)
    thresh = float(thresh)
    if not np.isfinite(thresh) or thresh < 0.0:
        raise ValueError("ESPIRiT thresh must be a finite non-negative value.")

    if mode == "3d":
        maps = _estimate_3d(
            kspace,
            device=sp.Device(-1) if device is None else device,
            crop=crop,
            calib_width=calib_width,
            thresh=thresh,
            kernel_width=kernel_width,
            max_iter=max_iter,
        )
        info = EspiritCalibrationInfo(
            mode="3d",
            cpu_workers=None,
            logical_ro_slices=int(kspace.shape[1]),
            zero_input_slices=(),
        )
        return maps, info

    if device is not None and int(device.id) != -1:
        raise ValueError(
            "slice2d ESPIRiT is CPU-only. Use a CPU SigPy device or select "
            "the 3d calibration mode for GPU execution."
        )

    return _estimate_slice2d(
        kspace,
        crop=crop,
        calib_width=calib_width,
        thresh=thresh,
        kernel_width=kernel_width,
        max_iter=max_iter,
        cpu_workers=cpu_workers,
        slice_support=slice_support,
        slice_support_noise_fraction=slice_support_noise_fraction,
        slice_support_noise_multiplier=slice_support_noise_multiplier,
        slice_support_relative_floor=slice_support_relative_floor,
        slice_support_padding=slice_support_padding,
        slice_support_diagnostic_path=slice_support_diagnostic_path,
    )


def _estimate_3d(
    kspace: np.ndarray,
    *,
    device: sp.Device,
    crop: float,
    calib_width: int,
    thresh: float,
    kernel_width: int,
    max_iter: int,
) -> np.ndarray:
    """Run the existing native 3D SigPy ESPIRiT calibration."""

    kspace_device = sp.to_device(kspace, device)
    maps_device = mr.app.EspiritCalib(
        kspace_device,
        calib_width=calib_width,
        thresh=thresh,
        kernel_width=kernel_width,
        crop=crop,
        max_iter=max_iter,
        device=device,
        show_pbar=True,
    ).run()
    maps = sp.to_device(maps_device, sp.Device(-1))
    maps = np.asarray(maps, dtype=np.complex64)
    _validate_output(maps, expected_shape=kspace.shape, label="3D ESPIRiT")
    return maps


def _estimate_slice2d(
    kspace: np.ndarray,
    *,
    crop: float,
    calib_width: int,
    thresh: float,
    kernel_width: int,
    max_iter: int,
    cpu_workers: Optional[int],
    slice_support: str,
    slice_support_noise_fraction: float,
    slice_support_noise_multiplier: float,
    slice_support_relative_floor: float,
    slice_support_padding: int,
    slice_support_diagnostic_path: Optional[str],
) -> tuple[np.ndarray, EspiritCalibrationInfo]:
    """Run parallel 2D ESPIRiT over logical-readout hybrid-space slices."""

    # Input ordering is (coil, RO, LIN, PAR). RO oversampling has already been
    # removed by the calling reconstruction before this transform.
    hybrid = sp.ifft(kspace, axes=(1,))
    hybrid = np.ascontiguousarray(hybrid, dtype=np.complex64)

    nro = int(hybrid.shape[1])

    active_ro = np.ones(nro, dtype=bool)
    slice_rms = None
    support_threshold = None
    support_peak = None
    support_first = None
    support_last = None
    if slice_support == "sag":
        (
            active_ro,
            slice_rms,
            support_threshold,
            support_peak,
        ) = _detect_sag_ro_support(
            hybrid,
            noise_fraction=slice_support_noise_fraction,
            noise_multiplier=slice_support_noise_multiplier,
            relative_floor=slice_support_relative_floor,
            padding_slices=slice_support_padding,
        )
        active_indices = np.flatnonzero(active_ro)
        support_first = int(active_indices[0])
        support_last = int(active_indices[-1])
        masked_count = int(np.count_nonzero(~active_ro))
        print(
            "SAG slice2d RO support guard (physical S-I): "
            f"active={active_indices.size}/{nro}, "
            f"masked={masked_count}, "
            f"range=[{support_first}, {support_last}], "
            f"threshold={support_threshold:.6g}."
        )
        if slice_support_diagnostic_path:
            _save_sag_ro_support_diagnostic(
                slice_rms=slice_rms,
                threshold=support_threshold,
                active_ro=active_ro,
                output_path=slice_support_diagnostic_path,
            )

    workers = _resolve_worker_count(cpu_workers, nro)
    print(
        "ESPIRiT slice2d backend: "
        f"{nro} logical-RO slices, {workers} CPU process worker(s)."
    )
    print(
        "Each worker calibrates one (coil, LIN, PAR) hybrid-space plane; "
        "native BLAS threads are limited to one per worker."
    )

    # The worker is module-level so that loky can serialize it reliably. Each
    # task receives only one small (coil, LIN, PAR) plane rather than the full
    # four-dimensional calibration array.
    with parallel_config(
        backend="loky",
        n_jobs=workers,
        inner_max_num_threads=1,
    ):
        results = Parallel(batch_size=1, verbose=10)(
            delayed(_calibrate_single_ro_slice)(
                ro_index,
                hybrid[:, ro_index, :, :],
                active=bool(active_ro[ro_index]),
                crop=crop,
                calib_width=calib_width,
                thresh=thresh,
                kernel_width=kernel_width,
                max_iter=max_iter,
            )
            for ro_index in range(nro)
        )

    # Parallel preserves input ordering. Stacking on axis 1 restores
    # (coil, RO, LIN, PAR) without a manual preallocation/indexing loop.
    maps = np.stack([result[1] for result in results], axis=1)
    maps = np.asarray(maps, dtype=np.complex64)

    # Enforce exact zero outside SAG whole-plane support. This second guard is
    # intentional: it prevents later RSS normalization from turning tiny
    # numerical values in masked planes into unit-norm sensitivity maps.
    maps[:, ~active_ro, :, :] = 0
    _validate_output(maps, expected_shape=kspace.shape, label="slice2d ESPIRiT")

    zero_slices = tuple(result[0] for result in results if result[2] == "zero")
    masked_slices = tuple(result[0] for result in results if result[2] == "masked")
    if zero_slices:
        print(
            "slice2d ESPIRiT skipped exactly-zero logical-RO planes: "
            + ", ".join(str(index) for index in zero_slices)
        )
    if masked_slices:
        print(
            "slice2d ESPIRiT masked low-signal SAG logical-RO planes: "
            + ", ".join(str(index) for index in masked_slices)
        )

    info = EspiritCalibrationInfo(
        mode="slice2d",
        cpu_workers=workers,
        logical_ro_slices=nro,
        zero_input_slices=zero_slices,
        masked_low_signal_slices=masked_slices,
        support_threshold=support_threshold,
        support_peak_rms=support_peak,
        support_first_active=support_first,
        support_last_active=support_last,
    )
    return maps, info


def _calibrate_single_ro_slice(
    ro_index: int,
    kspace_slice: np.ndarray,
    *,
    active: bool,
    crop: float,
    calib_width: int,
    thresh: float,
    kernel_width: int,
    max_iter: int,
) -> tuple[int, np.ndarray, str]:
    """Calibrate one logical-RO hybrid-space plane in a worker process."""

    kspace_slice = np.ascontiguousarray(kspace_slice, dtype=np.complex64)

    if not active:
        return ro_index, np.zeros_like(kspace_slice), "masked"

    # Retain the original exact-zero safeguard inside the active SAG support.
    if not np.any(kspace_slice):
        return ro_index, np.zeros_like(kspace_slice), "zero"

    maps = mr.app.EspiritCalib(
        kspace_slice,
        calib_width=calib_width,
        thresh=thresh,
        kernel_width=kernel_width,
        crop=crop,
        max_iter=max_iter,
        device=sp.Device(-1),
        show_pbar=False,
    ).run()
    maps = np.asarray(maps, dtype=np.complex64)
    _validate_output(
        maps,
        expected_shape=kspace_slice.shape,
        label=f"slice2d ESPIRiT RO slice {ro_index}",
    )
    return ro_index, maps, "calibrated"



def _detect_sag_ro_support(
    hybrid: np.ndarray,
    *,
    noise_fraction: float = 0.10,
    noise_multiplier: float = 3.0,
    relative_floor: float = 1e-4,
    padding_slices: int = 3,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Detect physical S-I whole-plane support for sagittal Wave-MPRAGE.

    ``hybrid`` must use ``(coil, logical_RO, LIN, PAR)`` ordering. In the
    validated SAG MPRAGE geometry, logical RO is physical z / superior-inferior.
    The detector is deliberately conservative: it selects the strongest
    contiguous signal component and pads both ends to retain superior scalp and
    inferior jaw/neck slices. It is not an in-plane anatomical mask.
    """

    if hybrid.ndim != 4:
        raise ValueError(
            "Expected hybrid calibration shape (coil, RO, LIN, PAR); "
            f"received {hybrid.shape}."
        )

    noise_fraction = float(noise_fraction)
    noise_multiplier = float(noise_multiplier)
    relative_floor = float(relative_floor)
    padding_slices = int(padding_slices)
    if not 0.0 < noise_fraction <= 0.5:
        raise ValueError("slice support noise_fraction must be in (0, 0.5].")
    if not np.isfinite(noise_multiplier) or noise_multiplier <= 0.0:
        raise ValueError("slice support noise_multiplier must be positive.")
    if not np.isfinite(relative_floor) or relative_floor < 0.0:
        raise ValueError("slice support relative_floor must be non-negative.")
    if padding_slices < 0:
        raise ValueError("slice support padding_slices must be non-negative.")

    slice_rms = np.sqrt(
        np.mean(np.abs(hybrid) ** 2, axis=(0, 2, 3), dtype=np.float64)
    )
    nro = int(slice_rms.size)
    if nro < 1:
        raise ValueError("Hybrid calibration contains no logical-RO slices.")

    noise_count = max(4, int(np.ceil(noise_fraction * nro)))
    noise_count = min(noise_count, nro)
    lowest = np.partition(slice_rms, noise_count - 1)[:noise_count]
    noise_floor = float(np.median(lowest))
    peak = float(np.max(slice_rms))
    threshold = max(noise_multiplier * noise_floor, relative_floor * peak)

    active = np.asarray(slice_rms > threshold, dtype=bool)
    if not np.any(active):
        print(
            "WARNING: SAG RO support detection found no active slices; "
            "leaving all logical-RO slices enabled."
        )
        return np.ones(nro, dtype=bool), slice_rms, threshold, peak

    # Fill isolated one-slice gaps, then keep the strongest contiguous S-I
    # component. A sagittal head/slab acquisition should form one contiguous
    # whole-plane support interval along physical z.
    active = binary_closing(active, structure=np.ones(3, dtype=bool))
    labels, component_count = label(active)
    if component_count > 1:
        component_scores = [
            float(np.sum(slice_rms[labels == component]))
            for component in range(1, component_count + 1)
        ]
        active = labels == (int(np.argmax(component_scores)) + 1)

    if padding_slices:
        active = binary_dilation(
            active,
            structure=np.ones(3, dtype=bool),
            iterations=padding_slices,
        )

    return np.asarray(active, dtype=bool), slice_rms, threshold, peak


def _save_sag_ro_support_diagnostic(
    *,
    slice_rms: np.ndarray,
    threshold: float,
    active_ro: np.ndarray,
    output_path: str,
) -> None:
    """Save SAG logical-RO support values and a compact diagnostic plot."""

    png_path = Path(output_path).expanduser()
    if png_path.suffix.lower() != ".png":
        png_path = png_path.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path = png_path.with_suffix(".npz")

    peak = max(float(np.max(slice_rms)), np.finfo(float).eps)
    np.savez(
        npz_path,
        slice_rms=np.asarray(slice_rms, dtype=np.float64),
        normalized_slice_rms=np.asarray(slice_rms / peak, dtype=np.float64),
        threshold=float(threshold),
        normalized_threshold=float(threshold / peak),
        active_ro=np.asarray(active_ro, dtype=bool),
        logical_ro_physical_axis="z",
        logical_ro_patient_axis="superior-inferior",
    )

    try:
        import matplotlib.pyplot as plt

        x = np.arange(slice_rms.size)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, slice_rms / peak, label="RO-plane RMS / peak")
        ax.axhline(
            threshold / peak,
            linestyle="--",
            label="support threshold",
        )
        masked = ~np.asarray(active_ro, dtype=bool)
        if np.any(masked):
            ax.scatter(
                x[masked],
                (slice_rms / peak)[masked],
                marker="x",
                label="masked whole planes",
            )
        ax.set_xlabel("Logical RO index (S-I for SAG)")
        ax.set_ylabel("Normalized hybrid-space RMS")
        ax.set_title("SAG slice2d ESPIRiT whole-plane RO support")
        ax.set_ylim(bottom=0)
        ax.legend()
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"Saved SAG RO support diagnostic: {png_path}")
        print(f"Saved SAG RO support values: {npz_path}")
    except Exception as exc:
        print(
            "WARNING: unable to save SAG RO support PNG "
            f"({type(exc).__name__}: {exc}); values were saved to {npz_path}."
        )

def _resolve_worker_count(requested: Optional[int], task_count: int) -> int:
    """Resolve automatic or user-requested worker count without hard-coding."""

    task_count = _positive_int("task_count", task_count)
    available = max(1, int(cpu_count(only_physical_cores=True)))

    if requested is None:
        return min(available, task_count)

    requested = _positive_int("cpu_workers", requested)
    if requested > available:
        print(
            f"Requested {requested} ESPIRiT CPU workers, but joblib reports "
            f"{available} available physical core(s); using {available}."
        )
    return min(requested, available, task_count)


def _validate_output(
    maps: np.ndarray,
    *,
    expected_shape: tuple[int, ...],
    label: str,
) -> None:
    """Validate map shape and numerical finiteness."""

    if tuple(maps.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"{label} returned shape {maps.shape}; expected {expected_shape}."
        )
    if not np.all(np.isfinite(maps)):
        raise FloatingPointError(
            f"{label} produced non-finite sensitivity-map values."
        )


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value
