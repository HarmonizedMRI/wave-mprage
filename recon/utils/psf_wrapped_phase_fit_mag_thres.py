"""
psf_wrapped_phase_fit_mag_thres.py

Helper functions for fitting a linear phase model

    p(y, z) = a*y + b*z + c

when the measured phase p is wrapped to [-pi, pi).

Main function
-------------
fit_wrapped_phase_planes(...)

Recommended import
------------------
from wrapped_phase_fit import fit_wrapped_phase_planes
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import torch


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """
    Wrap phase values to [-pi, pi).

    Parameters
    ----------
    x : torch.Tensor
        Input phase tensor in radians.

    Returns
    -------
    torch.Tensor
        Wrapped phase tensor.
    """
    return torch.atan2(torch.sin(x), torch.cos(x))


def _weighted_lstsq(
    A: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    ridge: float = 1e-8,
) -> torch.Tensor:
    """
    Solve the weighted least-squares problem

        min_theta || sqrt(w) * (A theta - b) ||^2

    with small ridge stabilization.
    """
    w = torch.clamp(w, min=0.0)
    sqrt_w = torch.sqrt(w)

    Aw = A * sqrt_w[:, None]
    bw = b * sqrt_w

    ATA = Aw.T @ Aw
    ATb = Aw.T @ bw

    eye = torch.eye(ATA.shape[0], dtype=ATA.dtype, device=ATA.device)
    ridge_scaled = ridge * torch.trace(ATA).clamp_min(1.0) / ATA.shape[0]

    try:
        theta = torch.linalg.solve(ATA + ridge_scaled * eye, ATb)
    except RuntimeError:
        theta = torch.linalg.lstsq(Aw, bw).solution

    return theta


def _mean_over_dims(
    x: torch.Tensor,
    dims: Optional[Union[int, Sequence[int]]],
) -> torch.Tensor:
    """
    Mean over one or more dimensions while supporting negative indices.
    """
    if dims is None:
        return x

    if isinstance(dims, int):
        dims = (dims,)
    else:
        dims = tuple(dims)

    ndim = x.ndim
    dims = tuple(d if d >= 0 else ndim + d for d in dims)

    # Reduce highest dimensions first so indices remain valid.
    for d in sorted(dims, reverse=True):
        x = x.mean(dim=d)

    return x


def fit_wrapped_phase_planes(
    psf_diff: torch.Tensor,
    hyb_nowave: torch.Tensor,
    y_norm: Union[torch.Tensor, Sequence[float]],
    z_norm: Union[torch.Tensor, Sequence[float]],
    *,
    mag_threshold: float = 8e-10,
    max_masked_threshold: float = 0.94,
    n_irls: int = 10,
    huber_delta: float = 0.7,
    ridge: float = 1e-8,
    min_valid_pixels: int = 10,
    weight_clip: Optional[float] = 100.0,
    mag_average_dims: Optional[Union[int, Sequence[int]]] = -1,
    use_previous_fit_as_init: bool = True,
    initial_coefficients: Optional[torch.Tensor] = None,
    verbose: bool = True,
    return_wrapped_prediction: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Fit wrapped phase planes independently for each kx location.

    The model is

        p(kx, y, z) = a(kx)*y + b(kx)*z + c(kx)

    but the measured phase p is assumed to be wrapped to [-pi, pi).

    This function avoids global phase unwrapping. Instead, it uses
    model-based local unwrapping:

        residual = wrap_to_pi(p - A @ theta)
        p_model_unwrapped = A @ theta + residual

    and then updates theta using weighted robust least squares.

    Parameters
    ----------
    psf_diff : torch.Tensor
        Wrapped phase tensor with shape [Nx, Ny, Nz].
        Values should be in radians, usually in [-pi, pi).

    hyb_nowave : torch.Tensor
        Complex tensor used to estimate phase reliability.
        Common shape is [Nx, Ny, Nz, Nc].
        The magnitude-squared is averaged over `mag_average_dims`.

    y_norm : torch.Tensor or sequence
        y-coordinate vector with length Ny.

    z_norm : torch.Tensor or sequence
        z-coordinate vector with length Nz.

    mag_threshold : float
        Pixels with averaged |hyb_nowave|^2 below this value are masked out.

    max_masked_threshold : float
        Skip a kx location if this fraction of pixels is masked out.

    n_irls : int
        Number of robust IRLS iterations.

    huber_delta : float
        Huber threshold in radians. Smaller values reject outliers more strongly.

    ridge : float
        Ridge stabilization for least-squares solve.

    min_valid_pixels : int
        Minimum number of valid pixels required to perform a fit.

    weight_clip : float or None
        Optional upper clamp for normalized magnitude weights.
        Use None to disable clipping.

    mag_average_dims : int, sequence of int, or None
        Dimensions of `hyb_nowave.abs().square()` to average over.
        Original code used dim=3, which corresponds to -1 for shape [Nx, Ny, Nz, Nc].
        Use None if `hyb_nowave` already has shape [Nx, Ny, Nz].

    use_previous_fit_as_init : bool
        If True, initialize each kx fit using the previous successful kx fit.

    initial_coefficients : torch.Tensor or None
        Optional initial coefficients [a, b, c].
        Used for the first fit, or for every fit if `use_previous_fit_as_init=False`.

    verbose : bool
        If True, print progress.

    return_wrapped_prediction : bool
        If True, also return `psf_diff_pred_wrapped`.

    Returns
    -------
    dict
        Dictionary containing:

        - "a_fit_all": fitted a coefficients, shape [Nx]
        - "b_fit_all": fitted b coefficients, shape [Nx]
        - "c_fit_all": fitted c coefficients, shape [Nx]
        - "coefficients": stacked coefficients, shape [Nx, 3]
        - "psf_diff_pred": unwrapped fitted plane, shape [Nx, Ny, Nz]
        - "mask": validity mask, shape [Nx, Ny, Nz]
        - "wrapped_rms": wrapped RMS residual per kx, shape [Nx]
        - "valid_pixels": number of valid pixels per kx, shape [Nx]
        - "masked_ratio": masked fraction per kx, shape [Nx]
        - "skipped": boolean tensor, shape [Nx]
        - "psf_diff_pred_wrapped": wrapped fitted plane, shape [Nx, Ny, Nz]
          only included when return_wrapped_prediction=True
    """
    if psf_diff.ndim != 3:
        raise ValueError(
            f"psf_diff must have shape [Nx, Ny, Nz], but got shape {tuple(psf_diff.shape)}"
        )

    if not torch.is_floating_point(psf_diff):
        raise TypeError("psf_diff must be a real floating-point tensor containing phase in radians.")

    device = psf_diff.device
    dtype = psf_diff.dtype

    Nx, Ny, Nz = psf_diff.shape

    y_norm_tensor = torch.as_tensor(y_norm, dtype=dtype, device=device)
    z_norm_tensor = torch.as_tensor(z_norm, dtype=dtype, device=device)

    if y_norm_tensor.numel() != Ny:
        raise ValueError(
            f"len(y_norm) must match psf_diff.shape[1]. Got {y_norm_tensor.numel()} and {Ny}."
        )

    if z_norm_tensor.numel() != Nz:
        raise ValueError(
            f"len(z_norm) must match psf_diff.shape[2]. Got {z_norm_tensor.numel()} and {Nz}."
        )

    if hyb_nowave.shape[0] != Nx or hyb_nowave.shape[1] != Ny or hyb_nowave.shape[2] != Nz:
        raise ValueError(
            "The first three dimensions of hyb_nowave must match psf_diff. "
            f"Got hyb_nowave shape {tuple(hyb_nowave.shape)} and psf_diff shape {tuple(psf_diff.shape)}."
        )

    Y_grid, Z_grid = torch.meshgrid(
        y_norm_tensor,
        z_norm_tensor,
        indexing="ij",
    )

    y_flat = Y_grid.flatten()
    z_flat = Z_grid.flatten()
    ones_flat = torch.ones_like(y_flat)

    A_full = torch.stack([y_flat, z_flat, ones_flat], dim=1)

    # Reliability estimate: averaged |hyb_nowave|^2
    mag2 = hyb_nowave.abs().square()
    mag2 = _mean_over_dims(mag2, mag_average_dims)
    mag2 = mag2.to(dtype=dtype, device=device)

    if mag2.shape != psf_diff.shape:
        raise ValueError(
            "After averaging, mag2 must have the same shape as psf_diff. "
            f"Got mag2 shape {tuple(mag2.shape)} and psf_diff shape {tuple(psf_diff.shape)}. "
            "Check mag_average_dims."
        )

    mask = mag2 > mag_threshold

    a_fit_all = torch.full((Nx,), torch.nan, dtype=dtype, device=device)
    b_fit_all = torch.full((Nx,), torch.nan, dtype=dtype, device=device)
    c_fit_all = torch.full((Nx,), torch.nan, dtype=dtype, device=device)

    psf_diff_pred = torch.full_like(psf_diff, torch.nan)

    wrapped_rms = torch.full((Nx,), torch.nan, dtype=dtype, device=device)
    valid_pixels_all = torch.zeros((Nx,), dtype=torch.long, device=device)
    masked_ratio_all = torch.full((Nx,), torch.nan, dtype=dtype, device=device)
    skipped = torch.ones((Nx,), dtype=torch.bool, device=device)

    if initial_coefficients is not None:
        prev_coefficients = torch.as_tensor(
            initial_coefficients,
            dtype=dtype,
            device=device,
        ).reshape(3)
    else:
        prev_coefficients = None

    eps = torch.finfo(dtype).eps

    with torch.no_grad():
        for kx_loc in range(Nx):
            p_flat = psf_diff[kx_loc].flatten()
            mask_flat = mask[kx_loc].flatten()
            mag2_flat = mag2[kx_loc].flatten()

            total_pixels = mask_flat.numel()
            valid_pixels = int(mask_flat.sum().item())
            masked_pixels = total_pixels - valid_pixels
            masked_ratio = masked_pixels / total_pixels

            valid_pixels_all[kx_loc] = valid_pixels
            masked_ratio_all[kx_loc] = masked_ratio

            if masked_ratio > max_masked_threshold or valid_pixels < min_valid_pixels:
                if verbose:
                    print(
                        f"Skipped {kx_loc}/{Nx}: "
                        f"masked_ratio={masked_ratio:.3f}, valid={valid_pixels}"
                    )
                continue

            A = A_full[mask_flat]
            p = p_flat[mask_flat]

            # Magnitude reliability weights.
            w_mag = mag2_flat[mask_flat]

            median_w = w_mag.median().clamp_min(eps)
            w_mag = w_mag / median_w

            if weight_clip is not None:
                w_mag = torch.clamp(w_mag, max=float(weight_clip))

            # Initialization.
            if use_previous_fit_as_init and prev_coefficients is not None:
                coefficients = prev_coefficients.clone()
            elif initial_coefficients is not None:
                coefficients = torch.as_tensor(
                    initial_coefficients,
                    dtype=dtype,
                    device=device,
                ).reshape(3).clone()
            else:
                # Fallback initialization. This can be imperfect if the data is heavily wrapped,
                # but the model-based IRLS usually improves it.
                coefficients = _weighted_lstsq(A, p, w_mag, ridge=ridge)

            # Robust model-based wrapped IRLS.
            for _ in range(n_irls):
                pred = A @ coefficients

                # Wrapped residual relative to current model.
                r = wrap_to_pi(p - pred)

                # Local model-based unwrapping.
                p_model_unwrapped = pred + r

                # Huber robust weights.
                abs_r = torch.abs(r)
                w_huber = torch.ones_like(abs_r)
                large = abs_r > huber_delta
                w_huber[large] = huber_delta / abs_r[large].clamp_min(eps)

                w_total = w_mag * w_huber

                new_coefficients = _weighted_lstsq(
                    A,
                    p_model_unwrapped,
                    w_total,
                    ridge=ridge,
                )

                denom = torch.norm(coefficients).clamp_min(1.0)
                if torch.norm(new_coefficients - coefficients) < 1e-6 * denom:
                    coefficients = new_coefficients
                    break

                coefficients = new_coefficients

            a_fit, b_fit, c_fit = coefficients

            a_fit_all[kx_loc] = a_fit
            b_fit_all[kx_loc] = b_fit
            c_fit_all[kx_loc] = c_fit

            if use_previous_fit_as_init:
                prev_coefficients = coefficients.clone()

            psf_diff_pred_flat = A_full @ coefficients
            psf_diff_pred[kx_loc] = psf_diff_pred_flat.view(psf_diff[kx_loc].shape)

            final_resid = wrap_to_pi(p - A @ coefficients)
            wrapped_rms[kx_loc] = torch.sqrt(torch.mean(final_resid.square()))

            skipped[kx_loc] = False

            if verbose:
                print(
                    f"Finished {kx_loc}/{Nx}: "
                    f"a={a_fit.item():.4g}, "
                    f"b={b_fit.item():.4g}, "
                    f"c={c_fit.item():.4g}, "
                    f"wrapped RMS={wrapped_rms[kx_loc].item():.4g} rad"
                )

    coefficients_all = torch.stack([a_fit_all, b_fit_all, c_fit_all], dim=1)

    result = {
        "a_fit_all": a_fit_all,
        "b_fit_all": b_fit_all,
        "c_fit_all": c_fit_all,
        "coefficients": coefficients_all,
        "psf_diff_pred": psf_diff_pred,
        "mask": mask,
        "wrapped_rms": wrapped_rms,
        "valid_pixels": valid_pixels_all,
        "masked_ratio": masked_ratio_all,
        "skipped": skipped,
    }

    if return_wrapped_prediction:
        result["psf_diff_pred_wrapped"] = wrap_to_pi(psf_diff_pred)

    return result
