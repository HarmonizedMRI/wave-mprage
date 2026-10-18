"""Receive-noise covariance estimation and prewhitening utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mapvbvd
import numpy as np


PREWHITENING_METHOD = {
    "enabled": True,
    "method": "embedded-noise-covariance-eigendecomposition",
    "version": 1,
    "covariance_normalization": "unit-mean-diagonal",
}


def prewhitening_provenance(enabled: bool) -> dict[str, Any]:
    """Return the stable manifest identity for the requested noise model.

    Args:
        enabled: Whether embedded-noise prewhitening is requested.

    Returns:
        JSON-compatible prewhitening identity used by cache and resume checks.
    """

    if not enabled:
        return {"enabled": False}
    return dict(PREWHITENING_METHOD)


def load_twix_noise_samples_coillast(filename: str | Path) -> np.ndarray:
    """Load an embedded TWIX noise scan as ``(sample, physical_coil)``.

    Args:
        filename: Siemens multi-RAID TWIX input containing a noise scan.

    Returns:
        Finite complex64 noise samples with receive channels last.

    Raises:
        ValueError: If no usable noise scan or channel dimension is present.
        RuntimeError: If mapVBVD returns an unexpected dimensional layout.
    """

    twix_obj = mapvbvd.mapVBVD(str(filename), quiet=True)
    scans = twix_obj if isinstance(twix_obj, list) else [twix_obj]
    noise_obj = next(
        (
            scan.noise
            for scan in scans
            if getattr(scan, "noise", None) is not None
        ),
        None,
    )
    if noise_obj is None:
        raise ValueError(f"TWIX file has no noise-adjustment scan: {filename}")
    noise_obj.flagRemoveOS = False
    data = np.asarray(noise_obj[""])
    data_dims = list(noise_obj.dataDims)
    full_size = tuple(int(value) for value in noise_obj.fullSize)
    retained_dims = [
        name for name, size in zip(data_dims, full_size) if size > 1
    ]
    data = np.squeeze(data)
    if "Cha" not in retained_dims:
        raise ValueError("TWIX noise scan has no non-singleton channel dimension.")
    if data.ndim != len(retained_dims):
        raise RuntimeError(
            "Unexpected noise array layout: "
            f"shape={data.shape}, retained_dimensions={retained_dims}."
        )
    channel_axis = retained_dims.index("Cha")
    data = np.moveaxis(data, channel_axis, -1)
    samples = np.ascontiguousarray(
        data.reshape(-1, data.shape[-1]), dtype=np.complex64
    )
    if samples.shape[0] < samples.shape[1] or not np.isfinite(samples).all():
        raise ValueError(f"Invalid TWIX noise samples with shape {samples.shape}.")
    return samples


def estimate_noise_covariance(samples_coillast: np.ndarray) -> np.ndarray:
    """Estimate a unit-mean-diagonal Hermitian noise covariance.

    Args:
        samples_coillast: Noise samples with physical receive channels last.

    Returns:
        Complex64 covariance normalized to remove arbitrary global noise scale.

    Raises:
        ValueError: If the samples or estimated variance are invalid.
    """

    samples = np.asarray(samples_coillast)
    if samples.ndim < 2:
        raise ValueError("Noise samples require a receive-coil dimension.")
    samples = samples.reshape(-1, samples.shape[-1]).astype(
        np.complex128, copy=False
    )
    if samples.shape[0] < samples.shape[1] or not np.isfinite(samples).all():
        raise ValueError(f"Invalid noise sample matrix with shape {samples.shape}.")
    samples = samples - samples.mean(axis=0, keepdims=True)
    covariance = samples.conj().T @ samples / samples.shape[0]
    covariance = 0.5 * (covariance + covariance.conj().T)
    mean_variance = float(np.trace(covariance).real / covariance.shape[0])
    if not np.isfinite(mean_variance) or mean_variance <= 0:
        raise ValueError("Noise covariance has a nonpositive mean variance.")
    covariance /= mean_variance
    return covariance.astype(np.complex64)


def make_noise_whitener(
    covariance: np.ndarray, *, eigenvalue_floor_relative: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a row-coil whitener ``A`` satisfying ``A.H Psi A = I``.

    Args:
        covariance: Square Hermitian physical-coil noise covariance.
        eigenvalue_floor_relative: Minimum eigenvalue relative to the largest.

    Returns:
        Complex64 whitener and ascending float64 covariance eigenvalues.

    Raises:
        ValueError: If the covariance is invalid or numerically singular.
    """

    covariance = np.asarray(covariance, dtype=np.complex128)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"Noise covariance must be square: {covariance.shape}.")
    if not np.isfinite(covariance).all():
        raise ValueError("Noise covariance contains non-finite values.")
    covariance = 0.5 * (covariance + covariance.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = float(eigenvalues[-1])
    floor = float(eigenvalue_floor_relative) * largest
    if largest <= 0 or float(eigenvalues[0]) <= floor:
        raise ValueError(
            "Noise covariance is singular or below the eigenvalue floor: "
            f"minimum={eigenvalues[0]:.6e}, floor={floor:.6e}."
        )
    whitener = (
        eigenvectors * (1.0 / np.sqrt(eigenvalues))[None, :]
    ) @ eigenvectors.conj().T
    return whitener.astype(np.complex64), eigenvalues.astype(np.float64)


def validate_noise_whitener(
    covariance: np.ndarray, whitener: np.ndarray, *, atol: float = 2e-5
) -> float:
    """Validate the row-coil whitening convention and return maximum error.

    Args:
        covariance: Physical-coil noise covariance.
        whitener: Physical-to-prewhitened coil transform.
        atol: Maximum accepted identity error.

    Returns:
        Maximum absolute error in ``whitener.H @ covariance @ whitener``.

    Raises:
        ValueError: If dimensions mismatch or whitening exceeds tolerance.
    """

    covariance = np.asarray(covariance, dtype=np.complex128)
    whitener = np.asarray(whitener, dtype=np.complex128)
    if whitener.shape != covariance.shape:
        raise ValueError(
            f"Whitener shape {whitener.shape} does not match {covariance.shape}."
        )
    whitened = whitener.conj().T @ covariance @ whitener
    error = float(np.max(np.abs(whitened - np.eye(covariance.shape[0]))))
    if not np.isfinite(error) or error > atol:
        raise ValueError(
            f"Noise whitener validation failed with maximum error {error:.6e}."
        )
    return error


def prepare_noise_prewhitener(
    filename: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load noise data, estimate a whitener, and return diagnostics.

    Args:
        filename: Siemens TWIX input containing the noise-adjustment scan.

    Returns:
        Covariance, row-coil whitener, eigenvalues, and summary diagnostics.
    """

    samples = load_twix_noise_samples_coillast(filename)
    covariance = estimate_noise_covariance(samples)
    whitener, eigenvalues = make_noise_whitener(covariance)
    identity_error = validate_noise_whitener(covariance, whitener)
    diagonal = np.real(np.diag(covariance))
    correlation = covariance / np.sqrt(diagonal[:, None] * diagonal[None, :])
    off_diagonal = np.abs(correlation - np.eye(covariance.shape[0]))
    diagnostics = {
        **prewhitening_provenance(True),
        "noise_samples": int(samples.shape[0]),
        "physical_coils": int(samples.shape[1]),
        "covariance_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "maximum_absolute_offdiagonal_correlation": float(off_diagonal.max()),
        "whitener_identity_error": identity_error,
    }
    return covariance, whitener, eigenvalues, diagnostics
