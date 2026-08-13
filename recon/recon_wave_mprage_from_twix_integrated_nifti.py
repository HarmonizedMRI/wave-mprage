#!/usr/bin/env python3
"""Integrated Wave-MPRAGE reconstruction from Siemens TWIX data.

Author: Yiyun Dong
Affiliation: Athinoula A. Martinos Center for Biomedical Imaging
License: MIT License

Description:
    Reconstruct Wave-MPRAGE or no-wave MPRAGE data from an integrated
    Wave-MPRAGE + FLASH-calibration Siemens TWIX acquisition. The same TWIX
    file and Pulseq sequence are used for the image data, integrated refscan
    ACS data, and integrated wave PSF calibration projections.

    The reconstruction pipeline uses coil compression, ESPIRiT coil
    sensitivity estimation, and CG-SENSE. For wave data, the calibrated PSF is
    estimated from the integrated FLASH-calibration refscan blocks and then
    used in the wave CG-SENSE operator.

Notes:
    - This script assumes the integrated refscan layout uses the first four
      refscan sets for nowave/wave sin/cos PSF calibration projections and the
      last refscan set for the ACS block used by coil compression/ESPIRiT.
    - Future updates may add retro low-resolution reconstruction.
    - Optional NIfTI export can save cropped-readout magnitude and phase
      images using orientation from the input MPRAGE TWIX file.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy import io
from scipy.optimize import least_squares
from scipy.signal import lombscargle
import pypulseq as pp

import platform
import os
import argparse
from pathlib import Path

try:
    import cupy as cp
except Exception as exc:
    cp = None
    _CUPY_IMPORT_ERROR = exc
else:
    _CUPY_IMPORT_ERROR = None
import sigpy as sp
import sigpy.mri as mr
import gc

from scipy.ndimage import zoom

from utils.twix_import import *
from utils.coil_compression_kspace import *
from utils.plot_coil_sens import *
from bart.bart_utils.bart_io import export_wave_inputs
from utils.espirit_calibration import estimate_espirit_maps

from utils.psf_wrapped_phase_fit import fit_wrapped_phase_planes
from utils.psf_wrapped_phase_fit import smooth_1d_nan

from utils.wave_cg_sense_precondition import cg_sense_wave, fft3call, ifft3call, fftc_dim, ifftc_dim

# Global plotting style: increase font sizes for all figures in this notebook/script.
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
})


def main():
    cfg = _collect_runtime_config()
    data_folder = cfg["data_folder"]
    out_folder = cfg["out_folder"]
    mprage_data_file = cfg["mprage_data_file"]
    mprage_seq_file = cfg["mprage_seq_file"]
    file_tag = cfg["file_tag"]
    tag_wave = cfg["tag_wave"]
    reuse_coil_calib = cfg["reuse_coil_calib"]
    espirit_device = cfg["espirit_device"]
    espirit_gpu_index = cfg["espirit_gpu_index"]
    espirit_crop = cfg["espirit_crop"]
    espirit_calib_mode = cfg["espirit_calib_mode"]
    espirit_cpu_workers = cfg["espirit_cpu_workers"]
    yflip = cfg["yflip"]
    zflip = cfg["zflip"]
    save_nifti = cfg["save_nifti"]
    save_nifti_phase = cfg["save_nifti_phase"]
    nifti_out_folder = cfg["nifti_out_folder"]
    nifti_sub = cfg["nifti_sub"]
    nifti_suffix = cfg["nifti_suffix"]
    nifti_axis_roles = cfg["nifti_axis_roles"]
    nifti_axis_flips = cfg["nifti_axis_flips"]
    twix_coord_system = cfg["twix_coord_system"]
    twix_inplane_rot_sign = cfg["twix_inplane_rot_sign"]
    twix_use_fov_for_voxel_size = cfg["twix_use_fov_for_voxel_size"]
    psf_coefficient_processing = cfg["psf_coefficient_processing"]
    psf_fit_kx_min = cfg["psf_fit_kx_min"]
    psf_fit_kx_max = cfg["psf_fit_kx_max"]
    save_bart_inputs = cfg["save_bart_inputs"]

    seq = pp.Sequence()
    seq.read(mprage_seq_file, remove_duplicates=False)
    defs = seq.definitions
    os_factor = int(defs.get('ReadoutOversamplingFactor', 4))

    # Current integrated sagittal Wave-MPRAGE convention is hard-coded here:
    #   sequence physical definitions: defs["Nx"], defs["Ny"], defs["Nz"] = physical x/y/z matrix
    #   acquisition axes: ax.d1 = 'z' readout, ax.d2 = 'x' PAR/inner PE, ax.d3 = 'y' LIN/outer PE
    # Therefore the logical reconstruction array is (readout, LIN, PAR) = (physical z, physical y, physical x).
    geom = _derive_hardcoded_sag_logical_geometry(defs)
    Nx = geom["Nro"]       # logical readout dimension = physical z dimension
    Ny = geom["Nlin"]      # logical LIN/outer-PE dimension = physical y dimension
    Nz = geom["Npar"]      # logical PAR/inner-PE dimension = physical x dimension
    res_x = geom["res_ro"]
    res_y = geom["res_lin"]
    res_z = geom["res_par"]
    Ry = int(defs.get('MPRAGE_PE2_R', 2))
    Rz = int(defs.get('MPRAGE_PE1_R', 3))
    Nx_os = Nx * os_factor
    ncalib = int(defs.get('Calibration_Ncalib1', 72))
    nacs = int(defs.get('Calibration_Nacs', 32))
    slice_orientation = _assert_sag_geometry(defs)
    nifti_voxel_size_mm = _derive_nifti_voxel_size_mm(defs, geom)

    print("Integrated Wave-MPRAGE reconstruction")
    print(f"  Physical matrix: Nx_phys={geom['Nxyz'][0]}, Ny_phys={geom['Nxyz'][1]}, Nz_phys={geom['Nxyz'][2]}")
    print(f"  Logical matrix:  Nro={Nx}, Nlin={Ny}, Npar={Nz}, Nro_os={Nx_os}")
    print(f"  Logical axes:    RO={geom['readout_axis']}, LIN={geom['lin_axis']}, PAR={geom['par_axis']}")
    print(f"  Physical FOV:    Fx={geom['FOVxyz'][0]:g}, Fy={geom['FOVxyz'][1]:g}, Fz={geom['FOVxyz'][2]:g} m")
    print(f"  Logical res:     RO={res_x:g}, LIN={res_y:g}, PAR={res_z:g} m")
    print(f"  Acceleration:    Ry={Ry}, Rz={Rz}")
    print(f"  Integrated calib: Ncalib={ncalib}, Nacs={nacs}")
    if save_nifti:
        print(f"  NIfTI export:    enabled -> {nifti_out_folder}")
        print(
            "  NIfTI voxel size: "
            f"{nifti_voxel_size_mm[0]:g} x {nifti_voxel_size_mm[1]:g} x "
            f"{nifti_voxel_size_mm[2]:g} mm"
        )

    print("Importing image data from integrated TWIX file...")
    img = load_img(mprage_data_file)
    if not torch.is_tensor(img):
        img = torch.as_tensor(img)
    _check_image_data_shape(img, Nx_os, Ny, Nz)
    geometry_diagnostics = _report_seq_twix_geometry(
        twix_file=mprage_data_file,
        geom=geom,
        received_image_shape=tuple(int(v) for v in img.shape),
        os_factor=os_factor,
        voxel_size_mm=nifti_voxel_size_mm,
        twix_array_axis_roles=nifti_axis_roles,
        twix_array_axis_flips=nifti_axis_flips,
        twix_coord_system=twix_coord_system,
        twix_inplane_rot_sign=twix_inplane_rot_sign,
    )
    Ncoil = int(img.shape[-1])

    print("Preparing coil compression matrix and coil sensitivity maps...")
    Wcc, csm_full_cc_np, Ncoil_ref = load_or_generate_coil_sens(
        mprage_data_file=mprage_data_file,
        Ny=Ny,
        Nz=Nz,
        os_factor=os_factor,
        out_folder=out_folder,
        file_tag=file_tag,
        Nacs=nacs,
        reuse_coil_calib=reuse_coil_calib,
        espirit_device=espirit_device,
        espirit_gpu_index=espirit_gpu_index,
        espirit_crop=espirit_crop,
        espirit_calib_mode=espirit_calib_mode,
        espirit_cpu_workers=espirit_cpu_workers,
    )
    if Ncoil_ref != Ncoil:
        raise ValueError(
            f"Image/refscan coil-count mismatch: image data has {Ncoil} coils, "
            f"but refscan/ACS data has {Ncoil_ref} coils."
        )

    print(f'Importing data, Ry={Ry}, Rz={Rz}')
    kspace_echo = torch.zeros((Nx_os, Ny, Nz, Ncoil), dtype=torch.cfloat)
    kspace_echo[:, :img.shape[1], :img.shape[2], :] = img

    kspace_cc_echo = apply_cc_coillast_torch(kspace_echo, Wcc, x_chunk=8)
    kspace_cc_file = (
        out_folder + 'kspace_' + tag_wave + '_cc_' +
        str(res_x) + 'x' + str(res_y) + 'x' + str(res_z) +
        '_Ry' + str(Ry) + '_Rz' + str(Rz) + '_' + file_tag
    )
    _save_npy(kspace_cc_file, kspace_cc_echo, 'coil-compressed k-space')

    # Use ESPIRiT maps estimated above.
    ncc = int(csm_full_cc_np.shape[0])
    _check_csm_shape(csm_full_cc_np, Nx, Ny, Nz)
    sens = torch.zeros((ncc, Nx_os, Ny, Nz), dtype=torch.complex64)
    x0 = Nx_os // 2 - csm_full_cc_np.shape[1] // 2
    x1 = x0 + csm_full_cc_np.shape[1]
    sens[:, x0:x1] = torch.from_numpy(csm_full_cc_np).contiguous()

    # Sampling mask M. kspace_cc_echo shape: (Nx_os, Ny, Nz, ncc).
    mask_2d = torch.sum(torch.abs(kspace_cc_echo) ** 2, dim=(0, 3)) > 0
    mask_2d = mask_2d.cpu().numpy().astype(np.float32)
    mask_t = torch.from_numpy(mask_2d).view(1, 1, *mask_2d.shape)  # broadcast to (ncoil, Nx_os, Ny, Nz)

    if tag_wave == 'wave':
        print("Processing Wave Data...")
        y_meas = kspace_cc_echo.permute(3, 0, 1, 2)  # (ncoil, Nx_os, Ny, Nz)

        print("Generating calibrated PSF from integrated refscan calibration blocks...")
        psf_calib, psf_theory = generate_calibrated_psf(
            mprage_data_file=mprage_data_file,
            mprage_seq_file=mprage_seq_file,
            out_folder=out_folder,
            Nx_os=Nx_os,
            Ny=Ny,
            Nz=Nz,
            file_tag=file_tag,
            yflip=yflip,
            zflip=zflip,
            Ncalib=ncalib,
            Nacs=nacs,
            slice_orientation=slice_orientation,
            coefficient_processing=psf_coefficient_processing,
            fit_kx_min=psf_fit_kx_min,
            fit_kx_max=psf_fit_kx_max,
        )
        print("Generated calibrated PSF")

        if save_bart_inputs:
            bart_tag = _sanitize_filename_component(file_tag) if file_tag else ""
            bart_folder = Path(out_folder) / (
                "bart_inputs" + (f"_{bart_tag}" if bart_tag else "")
            )
            kspace_calib = _build_bart_calibration_kspace(
                mprage_data_file=mprage_data_file,
                Nx=Nx,
                Ny=Ny,
                Nz=Nz,
                os_factor=os_factor,
                Nacs=nacs,
                Wcc=Wcc,
            )
            manifest_path = export_wave_inputs(
                bart_folder,
                wave_kspace=kspace_cc_echo.unsqueeze(3).numpy(),
                calibrated_psf=psf_calib.unsqueeze(0).numpy(),
                coil_sens=csm_full_cc_np,
                kspace_calib=kspace_calib,
            )
            print(f"Saved BART Wave-CAIPI inputs: {manifest_path}")

        psf_to_use = psf_calib.clone()
        # psf_to_use = psf_theory.clone()

        img_pcg_wave = cg_sense_wave(
            y=y_meas,
            sens=sens,
            psf_to_use=psf_to_use,
            mask_t=mask_t,
            n_iter=50,
            tol=1e-6,
            init="zero",
            use_preconditioner=True,
            use_direct_if_full=True,
        )

        image_wave_file = (
            out_folder + 'image_cg_wave_integrated_calib_' +
            str(res_x) + 'x' + str(res_y) + 'x' + str(res_z) +
            '_Ry' + str(Ry) + '_Rz' + str(Rz) + '_' + file_tag
        )
        _save_npy(image_wave_file, img_pcg_wave, 'wave CG-SENSE image')
        if save_nifti:
            save_mprage_output_to_nifti(
                image=img_pcg_wave,
                twix_file=mprage_data_file,
                out_folder=nifti_out_folder,
                nifti_sub=nifti_sub or _default_nifti_sub(tag_wave, res_x, res_y, res_z, Ry, Rz, file_tag),
                suffix=nifti_suffix,
                tag_wave=tag_wave,
                file_tag=file_tag,
                voxel_size_mm=nifti_voxel_size_mm,
                crop_readout_os=os_factor,
                save_phase=save_nifti_phase,
                twix_array_axis_roles=nifti_axis_roles,
                twix_array_axis_flips=nifti_axis_flips,
                twix_coord_system=twix_coord_system,
                twix_inplane_rot_sign=twix_inplane_rot_sign,
                twix_use_fov_for_voxel_size=twix_use_fov_for_voxel_size,
                metadata=_build_mprage_nifti_metadata(
                    tag_wave=tag_wave,
                    file_tag=file_tag,
                    defs=defs,
                    geom=geom,
                    os_factor=os_factor,
                    Ry=Ry,
                    Rz=Rz,
                    ncalib=ncalib,
                    nacs=nacs,
                    yflip=yflip,
                    zflip=zflip,
                    voxel_size_mm=nifti_voxel_size_mm,
                    geometry_diagnostics=geometry_diagnostics,
                    psf_coefficient_processing=psf_coefficient_processing,
                    psf_fit_kx_range=(psf_fit_kx_min, psf_fit_kx_max),
                ),
            )

    elif tag_wave == 'nowave':
        # Perform CG SENSE for no wave. Keep the reconstruction operator unchanged.
        def E(x):
            """Forward operator: x (Nx, Ny, Nz) -> k-space coils (nc, Nx, Ny, Nz)."""
            img_coils = sens * x.unsqueeze(0)                      # S
            kspace = fft3call(img_coils, dim=(1, 2, 3))            # F
            return kspace * mask_t                                # M

        def EH(k):
            """Adjoint operator: k-space coils -> image."""
            img_coils = ifft3call(k * mask_t, dim=(1, 2, 3))       # F^H
            return (torch.conj(sens) * img_coils).sum(dim=0)       # S^H

        def cg_sense(y, n_iter=50, tol=1e-6):
            """Solve (E^H E)x = E^H y with conjugate gradient."""
            x = torch.zeros((Nx_os, Ny, Nz), dtype=torch.complex64)
            b = EH(y)
            r = b.clone()
            p = r.clone()
            rr = torch.vdot(r.reshape(-1), r.reshape(-1)).real
            bb = torch.vdot(b.reshape(-1), b.reshape(-1)).real

            for i in range(n_iter):
                print(f'{i}/{n_iter}')
                Ap = EH(E(p))
                pAp = torch.vdot(p.reshape(-1), Ap.reshape(-1)).real
                alpha = rr / pAp
                x = x + alpha * p
                r = r - alpha * Ap
                rr_new = torch.vdot(r.reshape(-1), r.reshape(-1)).real

                rel = torch.sqrt(rr_new / bb)
                if rel < tol:
                    print(f"CG converged at iter {i + 1}, rel-res={rel.item():.2e}")
                    return x

                beta = rr_new / rr
                p = r + beta * p
                rr = rr_new

            print(f"CG reached max_iter={n_iter}, final rel-res={torch.sqrt(rr / bb).item():.2e}")
            return x

        print("Processing No-Wave Data...")
        y_meas = kspace_cc_echo.permute(3, 0, 1, 2)  # (ncoil, Nx_os, Ny, Nz)

        img_cg_nowave = cg_sense(y_meas, n_iter=50, tol=1e-6)
        image_nowave_file = (
            out_folder + 'image_cg_nowave_' +
            str(res_x) + 'x' + str(res_y) + 'x' + str(res_z) +
            '_Ry' + str(Ry) + '_Rz' + str(Rz) + '_' + file_tag
        )
        _save_npy(image_nowave_file, img_cg_nowave, 'no-wave CG-SENSE image')
        if save_nifti:
            save_mprage_output_to_nifti(
                image=img_cg_nowave,
                twix_file=mprage_data_file,
                out_folder=nifti_out_folder,
                nifti_sub=nifti_sub or _default_nifti_sub(tag_wave, res_x, res_y, res_z, Ry, Rz, file_tag),
                suffix=nifti_suffix,
                tag_wave=tag_wave,
                file_tag=file_tag,
                voxel_size_mm=nifti_voxel_size_mm,
                crop_readout_os=os_factor,
                save_phase=save_nifti_phase,
                twix_array_axis_roles=nifti_axis_roles,
                twix_array_axis_flips=nifti_axis_flips,
                twix_coord_system=twix_coord_system,
                twix_inplane_rot_sign=twix_inplane_rot_sign,
                twix_use_fov_for_voxel_size=twix_use_fov_for_voxel_size,
                metadata=_build_mprage_nifti_metadata(
                    tag_wave=tag_wave,
                    file_tag=file_tag,
                    defs=defs,
                    geom=geom,
                    os_factor=os_factor,
                    Ry=Ry,
                    Rz=Rz,
                    ncalib=ncalib,
                    nacs=nacs,
                    yflip=yflip,
                    zflip=zflip,
                    voxel_size_mm=nifti_voxel_size_mm,
                    geometry_diagnostics=geometry_diagnostics,
                    psf_coefficient_processing=None,
                    psf_fit_kx_range=None,
                ),
            )


# -----------------------------------------------------------------------------
# MPRAGE NIfTI export
# -----------------------------------------------------------------------------


def save_mprage_output_to_nifti(
    image,
    twix_file,
    out_folder,
    nifti_sub,
    suffix="MPRAGE",
    tag_wave="wave",
    file_tag="",
    voxel_size_mm=(1.0, 1.0, 1.0),
    crop_readout_os=1,
    save_phase=False,
    twix_array_axis_roles=("phase", "readout", "slice"),
    twix_array_axis_flips=(True, False, False),
    twix_coord_system="LPS",
    twix_inplane_rot_sign=-1.0,
    twix_use_fov_for_voxel_size=False,
    metadata=None,
):
    """Save one MPRAGE reconstruction as cropped-readout NIfTI files.

    The reconstruction image is expected to be in logical array order
    (readout, LIN/phase, PAR/partition). Native output supplies oversampled
    readout with ``crop_readout_os > 1``; BART output supplies the logical
    readout grid with ``crop_readout_os=1``.
    """
    from utils.nifti_export_twix import (
        apply_array_axis_flips,
        crop_readout_oversampling,
        make_nifti_affine_from_twix,
        normalize_magnitude,
        prepare_image_array,
        save_nifti_with_json,
    )

    if torch.is_tensor(image):
        img_np = image.detach().cpu().numpy()
    else:
        img_np = np.asarray(image)

    if img_np.ndim != 3:
        raise ValueError(f"Expected 3D MPRAGE image for NIfTI export, got shape {img_np.shape}.")

    img_crop = crop_readout_oversampling(img_np, crop_readout_os=crop_readout_os)
    print(f"NIfTI readout crop: {img_np.shape} -> {img_crop.shape} using crop_readout_os={crop_readout_os}")
    readout_processing = (
        "after readout-oversampling crop"
        if int(crop_readout_os) > 1
        else "without an additional readout crop"
    )

    magnitude = prepare_image_array(img_crop, part="mag")
    magnitude, magnitude_normalization = normalize_magnitude(
        magnitude,
        percentile=99.0,
    )

    print(
        "NIfTI magnitude normalization: "
        f"positive-voxel p{magnitude_normalization['Percentile']:g} "
        f"{magnitude_normalization['InputPercentileValue']:.6g} -> 1.0 "
        "(no clipping)"
    )

    outputs = [("mag", magnitude)]
    if save_phase:
        outputs.append(
            ("phase", prepare_image_array(img_crop, part="phase"))
        )

    images_to_flip = [arr for _, arr in outputs]
    images_to_flip = apply_array_axis_flips(images_to_flip, twix_array_axis_flips)
    outputs = [(part, arr) for (part, _), arr in zip(outputs, images_to_flip)]
    print(f"Applied NIfTI physical array flips: {tuple(bool(x) for x in twix_array_axis_flips)}")

    affine, voxel_size_from_affine, twix_info = make_nifti_affine_from_twix(
        twix_file=twix_file,
        npy_shape=outputs[0][1].shape,
        twix_array_axis_roles=twix_array_axis_roles,
        # Flips were applied to image data above, so do not apply them again to the affine.
        twix_array_axis_flips=(False, False, False),
        twix_coord_system=twix_coord_system,
        twix_inplane_rot_sign=twix_inplane_rot_sign,
        twix_use_fov_for_voxel_size=twix_use_fov_for_voxel_size,
        voxel_size_mm=voxel_size_mm,
    )

    out_dir = Path(out_folder)
    sub_folder = str(nifti_sub)
    if not sub_folder.startswith("sub-"):
        sub_folder = "sub-" + sub_folder
    sub_out_dir = out_dir / sub_folder
    sub_out_dir.mkdir(parents=True, exist_ok=True)

    base_metadata = dict(metadata or {})
    base_metadata.update({
        "NIfTISourceImageShapeBeforeReadoutCrop": [int(v) for v in img_np.shape],
        "NIfTIImageShapeAfterReadoutCrop": [int(v) for v in outputs[0][1].shape],
        "NIfTIVoxelSizeMm": [float(v) for v in voxel_size_from_affine],
        "NIfTITwixArrayAxisRoles": list(twix_array_axis_roles),
        "NIfTIPhysicalArrayFlipsApplied": [bool(x) for x in twix_array_axis_flips],
        "NIfTITwixCoordinateSystemAssumption": twix_coord_system,
        "NIfTITwixInplaneRotationSign": float(twix_inplane_rot_sign),
        "NIfTITwixUseFovForVoxelSize": bool(twix_use_fov_for_voxel_size),
        "NIfTIOrientation": {
            "OrientationSource": "TwixMeasYaps",
            "TwixOrientation": twix_info,
        },
    })

    for part, arr in outputs:
        basename = f"{sub_folder}_part-{part}_{suffix}"
        nii_path = sub_out_dir / f"{basename}.nii.gz"
        json_path = sub_out_dir / f"{basename}.json"

        sidecar = dict(base_metadata)
        sidecar["Part"] = part

        if part == "phase":
            sidecar["Units"] = "rad"
            sidecar["ImageProcessing"] = (
                f"angle(complex_image), {readout_processing}"
            )
        else:
            sidecar["Units"] = "relative"
            sidecar["MagnitudeNormalization"] = magnitude_normalization
            sidecar["ImageProcessing"] = (
                f"abs(complex_image), {readout_processing}; "
                "scaled so the 99th percentile of positive finite voxels "
                "equals 1.0; not clipped"
            )
        save_nifti_with_json(arr, affine, nii_path, json_path, metadata=sidecar)


def _default_nifti_sub(tag_wave, res_x, res_y, res_z, Ry, Rz, file_tag):
    """Build a compact MPRAGE NIfTI subject/folder name."""
    res_mm = (res_x * 1e3, res_y * 1e3, res_z * 1e3)
    tag = f"MPRAGE_{tag_wave}_{res_mm[0]:g}x{res_mm[1]:g}x{res_mm[2]:g}mm_Ry{Ry}_Rz{Rz}"
    if file_tag:
        tag += f"_{file_tag}"
    return _sanitize_filename_component(tag)


def _build_mprage_nifti_metadata(
    tag_wave,
    file_tag,
    defs,
    geom,
    os_factor,
    Ry,
    Rz,
    ncalib,
    nacs,
    yflip,
    zflip,
    voxel_size_mm,
    geometry_diagnostics=None,
    psf_coefficient_processing=None,
    psf_fit_kx_range=None,
):
    """Create metadata without feeding sidecar values back into reconstruction."""
    metadata = {
        "Modality": "MR",
        "PulseSequenceType": "MPRAGE",
        "PulseSequenceDetails": "Custom Pulseq integrated Wave-MPRAGE with FLASH PSF calibration",
        "MRAcquisitionType": "3D",
        "Reconstruction": "CG-SENSE Wave" if tag_wave == "wave" else "CG-SENSE no-wave",
        "WaveReconstructionTag": tag_wave,
        "FileTag": file_tag,
        "ReadoutOversamplingFactor": int(os_factor),
        "Acceleration": {"Ry": int(Ry), "Rz": int(Rz)},
        "IntegratedCalibration": True,
        "Ncalib": int(ncalib),
        "Nacs": int(nacs),
        "PSFSignConvention": {"yflip": int(yflip), "zflip": int(zflip)},
        "LogicalReconAxisOrder": [
            "readout_physical_z",
            "LIN_outerPE_physical_y",
            "PAR_innerPE_physical_x",
        ],
        "PhysicalMatrixXYZ": [int(v) for v in geom["Nxyz"]],
        "LogicalMatrixRO_LIN_PAR": [int(geom["Nro"]), int(geom["Nlin"]), int(geom["Npar"])],
        "PhysicalFOVMXYZ": [float(v) for v in geom["FOVxyz"]],
        "PhysicalFOVMmXYZ": [float(v) * 1e3 for v in geom["FOVxyz"]],
        "LogicalFOVMRO_LIN_PAR": [float(geom["FOVro"]), float(geom["FOVlin"]), float(geom["FOVpar"])],
        "LogicalFOVMmRO_LIN_PAR": [
            float(geom["FOVro"]) * 1e3,
            float(geom["FOVlin"]) * 1e3,
            float(geom["FOVpar"]) * 1e3,
        ],
        "VoxelSizeMmRO_LIN_PAR": [float(v) for v in voxel_size_mm],
        "NIfTIVoxelSizeSource": "Pulseq definitions, checked against TWIX geometry",
    }

    # Metadata-only lookups. Current sequence timing definitions are seconds;
    # FlipAngle is degrees. Missing optional fields are omitted, not defaulted.
    repetition_time = _first_finite_definition(defs, "MPRAGE_TRout", "TR")
    if repetition_time is not None:
        metadata["RepetitionTime"] = repetition_time
        metadata["RepetitionTimeUnits"] = "s"

    echo_time = _first_finite_definition(defs, "TE")
    if echo_time is not None:
        metadata["EchoTime"] = echo_time
        metadata["EchoTimeUnits"] = "s"

    inversion_time = _first_finite_definition(defs, "MPRAGE_TI", "TI")
    if inversion_time is not None:
        metadata["InversionTime"] = inversion_time
        metadata["InversionTimeUnits"] = "s"

    inner_tr = _first_finite_definition(defs, "MPRAGE_TRinner")
    if inner_tr is not None:
        metadata["MPRAGEInnerRepetitionTime"] = inner_tr
        metadata["MPRAGEInnerRepetitionTimeUnits"] = "s"

    flip_angle = _first_finite_definition(defs, "FlipAngle")
    if flip_angle is not None:
        metadata["FlipAngle"] = flip_angle
        metadata["FlipAngleUnits"] = "degree"

    scalar_defs = {
        "Name": "PulseqSequenceName",
        "OrientationMapping": "PulseqOrientationMapping",
        "ReadoutAxis": "PulseqReadoutPhysicalAxis",
        "InnerPEAxis": "PulseqInnerPhaseEncodingPhysicalAxis",
        "OuterPEAxis": "PulseqOuterPhaseEncodingPhysicalAxis",
        "MPRAGE_ETL_Target": "EchoTrainLength",
        "Calibration_RFType": "CalibrationRFType",
        "Calibration_SlabAxis": "CalibrationSlabPhysicalAxis",
    }
    for seq_key, meta_key in scalar_defs.items():
        if seq_key in defs:
            metadata[meta_key] = _json_scalar(defs.get(seq_key))

    numeric_calibration_defs = {
        "Calibration_TRinner": ("CalibrationRepetitionTime", "s"),
        "Calibration_TE": ("CalibrationEchoTime", "s"),
        "Calibration_RFDuration": ("CalibrationRFDuration", "s"),
        "Calibration_RFTBW": ("CalibrationRFTBW", None),
        "Calibration_SlabThickness": ("CalibrationSlabThickness", "m"),
        "Calibration_ReadoutDuration": ("CalibrationReadoutDuration", "s"),
        "Calibration_ReadoutSamples": ("CalibrationReadoutSamples", None),
        "Calibration_WaveAmplitude_mTm": ("CalibrationWaveAmplitude", "mT/m"),
        "Calibration_WaveCycles": ("CalibrationWaveCycles", None),
        "Calibration_Ncalib1": ("CalibrationNcalib1", None),
        "Calibration_Ncalib2": ("CalibrationNcalib2", None),
        "Calibration_Nacs": ("CalibrationNacs", None),
    }
    for seq_key, (meta_key, unit) in numeric_calibration_defs.items():
        value = _first_finite_definition(defs, seq_key)
        if value is not None:
            metadata[meta_key] = value
            if unit is not None:
                metadata[meta_key + "Units"] = unit

    if geometry_diagnostics is not None:
        metadata["GeometryDiagnostics"] = geometry_diagnostics
        metadata["PhaseEncodingDirections"] = geometry_diagnostics.get("Directions", {})

    if tag_wave == "wave" and psf_coefficient_processing is not None:
        metadata["PSFCoefficientProcessing"] = str(psf_coefficient_processing)
        if psf_coefficient_processing == "sine-line" and psf_fit_kx_range is not None:
            metadata["PSFFitKxRange"] = [int(psf_fit_kx_range[0]), int(psf_fit_kx_range[1])]
            metadata["PSFFitKxRangeConvention"] = "half-open [min, max)"
            metadata["PSFFitModel"] = "A*sin(w*kx+phi)+C1*kx+C2"
    return metadata


def _json_scalar(value):
    """Convert a Pulseq scalar to a JSON-safe Python scalar."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _first_finite_definition(defs, *keys):
    """Return the first finite scalar definition, without inserting a default."""
    for key in keys:
        if key not in defs:
            continue
        try:
            value = float(defs.get(key))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None

def _sanitize_filename_component(value):
    """Return a filesystem-friendly component for generated NIfTI names."""
    value = str(value).strip()
    allowed = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            allowed.append(ch)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "mprage"


# -----------------------------------------------------------------------------
# Integrated refscan ACS / coil-sensitivity handling
# -----------------------------------------------------------------------------

def _cuda_device_count():
    """Return visible CUDA device count and an optional diagnostic message."""
    if cp is None:
        detail = "CuPy is not installed"
        if _CUPY_IMPORT_ERROR is not None:
            detail += f" ({type(_CUPY_IMPORT_ERROR).__name__}: {_CUPY_IMPORT_ERROR})"
        return 0, detail

    try:
        return int(cp.cuda.runtime.getDeviceCount()), None
    except Exception as exc:
        return 0, f"CuPy/CUDA initialization failed ({type(exc).__name__}: {exc})"


def _cuda_device_name(index):
    """Return a readable CUDA device name."""
    props = cp.cuda.runtime.getDeviceProperties(index)
    name = props.get("name", f"GPU {index}")
    return name.decode() if isinstance(name, bytes) else str(name)


def _select_espirit_device(mode="auto", gpu_index=0):
    """Select a SigPy device for ESPIRiT with an explicit CPU fallback."""
    mode = str(mode).strip().lower()
    if mode not in ("auto", "cpu", "gpu"):
        raise ValueError("espirit_device must be 'auto', 'cpu', or 'gpu'.")

    gpu_index = int(gpu_index)
    if gpu_index < 0:
        raise ValueError("espirit_gpu_index must be a non-negative integer.")

    if mode == "cpu":
        print("ESPIRiT calibration device: CPU (explicit request)")
        return sp.Device(-1), False

    count, diagnostic = _cuda_device_count()
    if count > gpu_index:
        print(f"CuPy version: {cp.__version__}")
        print(f"Visible CUDA device count: {count}")
        for index in range(count):
            try:
                print(f"  GPU {index}: {_cuda_device_name(index)}")
            except Exception as exc:
                print(f"  GPU {index}: unable to read device properties ({exc})")
        print(f"ESPIRiT calibration device: GPU {gpu_index}")
        return sp.Device(gpu_index), True

    if count > 0:
        diagnostic = (
            f"requested GPU index {gpu_index}, but only {count} CUDA device(s) are visible"
        )
    elif diagnostic is None:
        diagnostic = "no CUDA devices are visible"

    if mode == "gpu":
        raise RuntimeError(
            "GPU ESPIRiT was requested, but no usable requested GPU was found: "
            f"{diagnostic}. Use --espirit-device auto or cpu to allow CPU calibration."
        )

    print(f"ESPIRiT GPU unavailable: {diagnostic}.")
    print("ESPIRiT calibration device: CPU fallback")
    return sp.Device(-1), False


def load_or_generate_coil_sens(
    mprage_data_file,
    Ny,
    Nz,
    os_factor,
    out_folder,
    file_tag,
    Nacs=32,
    reuse_coil_calib=False,
    espirit_device="auto",
    espirit_gpu_index=0,
    espirit_crop=0.8,
    espirit_calib_mode="3d",
    espirit_cpu_workers=None,
):
    """Load cached Wcc/CSM or generate them from the integrated ACS refscan set."""
    csm_tag = _espirit_cache_tag(file_tag, espirit_calib_mode)
    wcc_file = _npy_output_path(out_folder + 'coil_compression_energy_' + file_tag)
    csm_file = _npy_output_path(out_folder + 'csm_full_' + csm_tag)

    if reuse_coil_calib and os.path.isfile(wcc_file) and os.path.isfile(csm_file):
        print("Reusing existing coil compression matrix and coil sensitivity maps.")
        print(
            f"ESPIRiT crop threshold {espirit_crop:g} is not reapplied "
            "because cached sensitivity maps are being reused."
        )
        print(f"Loading Wcc from: {wcc_file}")
        Wcc = np.load(wcc_file)
        print(f"Loading CSM from: {csm_file}")
        csm_full_cc_np = np.load(csm_file)
        ref_data = load_ref(mprage_data_file)
        _check_integrated_refscan_shape(ref_data, Nacs=Nacs, Ncalib=None)
        Ncoil = int(ref_data.shape[-1])
        return Wcc, csm_full_cc_np, Ncoil

    if reuse_coil_calib:
        print("Requested reuse of coil calibration, but cached files were not found. Recomputing from ACS.")

    return generate_coil_sens(
        mprage_data_file=mprage_data_file,
        Ny=Ny,
        Nz=Nz,
        os_factor=os_factor,
        out_folder=out_folder,
        file_tag=file_tag,
        Nacs=Nacs,
        espirit_device=espirit_device,
        espirit_gpu_index=espirit_gpu_index,
        espirit_crop=espirit_crop,
        espirit_calib_mode=espirit_calib_mode,
        espirit_cpu_workers=espirit_cpu_workers,
    )


def generate_coil_sens(
    mprage_data_file,
    Ny,
    Nz,
    os_factor,
    out_folder,
    file_tag,
    Nacs=32,
    espirit_device="auto",
    espirit_gpu_index=0,
    espirit_crop=0.8,
    espirit_calib_mode="3d",
    espirit_cpu_workers=None,
):
    """Generate Wcc and ESPIRiT CSMs from the integrated sequence ACS refscan."""
    espirit_crop = float(espirit_crop)
    if not np.isfinite(espirit_crop) or not 0.0 <= espirit_crop <= 1.0:
        raise ValueError("espirit_crop must be a finite value between 0 and 1.")
    espirit_calib_mode = str(espirit_calib_mode).strip().lower()
    if espirit_calib_mode not in ("3d", "slice2d"):
        raise ValueError("espirit_calib_mode must be '3d' or 'slice2d'.")
    csm_tag = _espirit_cache_tag(file_tag, espirit_calib_mode)
    print(f"ESPIRiT calibration mode: {espirit_calib_mode}")

    print(f"ESPIRiT crop threshold: {espirit_crop:g}")
    if espirit_calib_mode == "slice2d":
        if str(espirit_device).strip().lower() == "gpu":
            raise ValueError(
                "--espirit-calib-mode slice2d is CPU-only and cannot be used "
                "with --espirit-device gpu."
            )
        print(
            "slice2d ESPIRiT is CPU-only; logical readout oversampling is "
            "removed before the hybrid-space transform."
        )
        device = sp.Device(-1)
        using_gpu = False
    else:
        device, using_gpu = _select_espirit_device(
            mode=espirit_device,
            gpu_index=espirit_gpu_index,
        )

    # The integrated refscan layout uses the last set for ACS.
    data_ref = load_ref(mprage_data_file)
    _check_integrated_refscan_shape(data_ref, Nacs=Nacs, Ncalib=None)
    kspace_nowave_acs = data_ref[:, :Nacs, :Nacs, -1, :]
    Nx_os, Ny_acs, Nz_acs, Ncoil = kspace_nowave_acs.shape
    Nx = Nx_os // os_factor

    print(f"Integrated ACS shape: {tuple(kspace_nowave_acs.shape)}")

    # calculate coil compression energy
    Wcc, cc_svals, cc_energy = estimate_cc_matrix_coillast(
        kspace_nowave_acs,
        ncc=12,
        acs=min(Ny_acs, Nz_acs),
        x_step=os_factor,
    )
    print("Wcc:", Wcc.shape)
    print("Energy retained by 12 coils:", cc_energy[11])

    # For ESPIRiT, only make low-res CPU array first.
    # Convert to coil-first: (32, x, y, z)
    kspace_nowave_np = (
        kspace_nowave_acs
        .permute(3, 0, 1, 2)[:, ::os_factor]
        .contiguous()
        .numpy()
        .astype(np.complex64, copy=False)
    )

    # Low-resolution crop before ESPIRiT device transfer.
    low_shape = (kspace_nowave_np.shape[0], Nx, 32, 32)
    kspace_low_np = sp.resize(kspace_nowave_np, low_shape).astype(np.complex64, copy=False)

    # Coil compression: 32 -> 12
    kspace_low_cc_np = apply_cc_coilfirst_np(kspace_low_np, Wcc)
    print("kspace_low_cc_np:", kspace_low_cc_np.shape)

    if using_gpu:
        cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    # The slice2d backend receives logical k-space only after the existing
    # readout-oversampling removal above. It transforms logical RO to hybrid
    # space and preserves the joint LIN/PAR calibration plane.
    csm_low_cc_np, espirit_info = estimate_espirit_maps(
        kspace_low_cc_np,
        mode=espirit_calib_mode,
        device=device,
        crop=espirit_crop,
        calib_width=24,
        thresh=0.02,
        kernel_width=6,
        max_iter=100,
        cpu_workers=espirit_cpu_workers,
        # SAG-specific whole-plane guard: logical RO is physical z / S-I.
        # These conservative defaults retain superior scalp and inferior
        # jaw/neck through three-slice padding while rejecting noise-only
        # whole RO planes. Native 3D ESPIRiT is unaffected.
        slice_support="sag" if espirit_calib_mode == "slice2d" else "off",
        slice_support_noise_fraction=0.03,
        slice_support_noise_multiplier=1.5,
        slice_support_relative_floor=1e-5,
        slice_support_padding=20,
        slice_support_diagnostic_path=(
            out_folder + "espirit_slice2d_sag_ro_support_" + csm_tag + ".png"
            if espirit_calib_mode == "slice2d"
            else None
        ),
    )
    print("ESPIRiT execution info:", espirit_info)
    print("csm_low_cc_np:", csm_low_cc_np.shape)

    # zoom to full res coil sensitivity maps
    target_img_shape = (Nx_os, Ny, Nz)
    zoom_factors = (
        1,
        target_img_shape[0] / os_factor / csm_low_cc_np.shape[1],
        target_img_shape[1] / csm_low_cc_np.shape[2],
        target_img_shape[2] / csm_low_cc_np.shape[3],
    )

    csm_full_cc_np = (
        zoom(csm_low_cc_np.real, zoom_factors, order=1)
        + 1j * zoom(csm_low_cc_np.imag, zoom_factors, order=1)
    ).astype(np.complex64)

    if espirit_calib_mode == "slice2d" and espirit_info.masked_low_signal_slices:
        # Logical RO already has the final non-oversampled Nx length, so the
        # low-resolution SAG support indices map directly to full CSM axis 1.
        csm_full_cc_np[
            :,
            list(espirit_info.masked_low_signal_slices),
            :,
            :,
        ] = 0

    # Normalize RSS across coils without converting masked zero planes into
    # unit-norm maps.
    rss = np.sqrt(np.sum(np.abs(csm_full_cc_np) ** 2, axis=0, keepdims=True))
    csm_full_cc_np = np.divide(
        csm_full_cc_np,
        rss,
        out=np.zeros_like(csm_full_cc_np),
        where=rss > 1e-8,
    )

    # Save the Wcc once, but keep sensitivity-map products mode-specific so
    # --reuse-coil-calib cannot silently mix 3d and slice2d estimates.
    _save_npy(out_folder + 'coil_compression_energy_' + file_tag, Wcc, 'coil compression matrix')
    _save_npy(out_folder + 'csm_acs_' + csm_tag, csm_low_cc_np, 'low-resolution ESPIRiT CSM')
    _save_npy(out_folder + 'csm_full_' + csm_tag, csm_full_cc_np, 'full-resolution ESPIRiT CSM')
    plot_csm_magnitude_grid(csm_full_cc_np, z=csm_full_cc_np.shape[-1] // 2)
    mag_png = out_folder + 'csm_full_mag_' + csm_tag + '.png'
    print(f"Saving CSM magnitude plot to: {mag_png}")
    plt.savefig(mag_png, dpi=150)
    plot_csm_phase_grid(csm_full_cc_np, z=csm_full_cc_np.shape[-1] // 2)
    phase_png = out_folder + 'csm_full_phase_' + csm_tag + '.png'
    print(f"Saving CSM phase plot to: {phase_png}")
    plt.savefig(phase_png, dpi=150)
    plt.close('all')
    print(csm_full_cc_np.shape)

    return Wcc, csm_full_cc_np, Ncoil


def _build_bart_calibration_kspace(
    *,
    mprage_data_file,
    Nx,
    Ny,
    Nz,
    os_factor,
    Nacs,
    Wcc,
):
    """Return compressed integrated ACS on BART's full logical image grid."""

    data_ref = load_ref(mprage_data_file)
    _check_integrated_refscan_shape(data_ref, Nacs=Nacs, Ncalib=None)
    kspace_acs = data_ref[:, :Nacs, :Nacs, -1, :]
    kspace_acs_cc = apply_cc_coillast_torch(kspace_acs, Wcc, x_chunk=8)
    kspace_acs_cc = kspace_acs_cc[::os_factor]
    ncc = int(Wcc.shape[1])
    expected = (Nx, Nacs, Nacs, ncc)
    if tuple(kspace_acs_cc.shape) != expected:
        raise ValueError(
            "Unexpected compressed BART calibration shape: "
            f"received {tuple(kspace_acs_cc.shape)}, expected {expected}."
        )
    if Nacs > Ny or Nacs > Nz:
        raise ValueError(f"ACS size {Nacs} does not fit BART grid {(Nx, Ny, Nz)}.")

    full = torch.zeros((Nx, Ny, Nz, ncc), dtype=torch.complex64)
    y0 = (Ny - Nacs) // 2
    z0 = (Nz - Nacs) // 2
    full[:, y0 : y0 + Nacs, z0 : z0 + Nacs, :] = kspace_acs_cc
    return full.numpy()


# -----------------------------------------------------------------------------
# Integrated PSF calibration handling
# -----------------------------------------------------------------------------

def generate_theoretical_wave_trajectory(fn_seq, Nx_os, Nacs_total, slice_orientation='SAG'):
    """Generate full-imaging theoretical wave trajectory from an integrated sequence.

    The final integrated calibration/ACS tail is excluded before reshaping ADC
    trajectory samples into readout lines.
    """
    seq = pp.Sequence()
    seq.read(fn_seq, remove_duplicates=False)
    defs = seq.definitions

    ktraj_adc, ktraj, t_exc, t_ref, t_adc = seq.calculate_kspace()
    N_total = int(ktraj_adc.shape[1])
    if N_total <= Nacs_total:
        raise ValueError(
            f"Sequence ADC trajectory has {N_total} samples, but the integrated "
            f"calibration/ACS tail requires {Nacs_total} samples. Check Ncalib/Nacs/Nx_os."
        )
    n_img_samples = N_total - Nacs_total
    if n_img_samples % Nx_os != 0:
        raise ValueError(
            f"Imaging trajectory sample count ({n_img_samples}) is not divisible by Nx_os ({Nx_os}). "
            f"Check the integrated sequence layout and Ncalib/Nacs settings."
        )

    k_adc = np.asarray(ktraj_adc[:, :n_img_samples], dtype=np.float64).reshape(3, -1, Nx_os)
    ky_adc = k_adc[1]
    kz_adc = k_adc[0]

    line_means = ky_adc[:, 0]
    center_line_idx = int(np.argmin(np.abs(line_means)))
    delta_ky = ky_adc[center_line_idx]

    line_means = kz_adc[:, 0]
    center_line_idx = int(np.argmin(np.abs(line_means)))
    delta_kz = kz_adc[center_line_idx]

    fov_y, fov_z = _get_fov_yz(defs, slice_orientation=slice_orientation)
    delta_ky_idx = delta_ky * fov_y
    delta_kz_idx = delta_kz * fov_z

    return delta_ky_idx, delta_kz_idx


def _resolve_mprage_wave_mode(
    requested_mode,
    mprage_seq_file,
    Nx_os,
    Ncalib,
    Nacs,
    slice_orientation="SAG",
    activity_threshold=1e-4,
):
    """Resolve auto/wave/nowave from the MPRAGE imaging trajectory.

    The integrated FLASH calibration and ACS tail is excluded by
    generate_theoretical_wave_trajectory(), so only the MPRAGE imaging
    trajectory is inspected.
    """
    requested_mode = str(requested_mode).strip().lower()
    if requested_mode not in ("auto", "wave", "nowave"):
        raise ValueError(
            "wave mode must be 'auto', 'wave', or 'nowave'."
        )

    delta_ky_idx, delta_kz_idx = generate_theoretical_wave_trajectory(
        fn_seq=mprage_seq_file,
        Nx_os=Nx_os,
        Nacs_total=int(Nx_os * (4 * Ncalib + Nacs * Nacs)),
        slice_orientation=slice_orientation,
    )

    def waveform_excursion(values):
        values = np.asarray(values, dtype=np.float64)
        values = values - np.mean(values)
        return float(np.max(np.abs(values)))

    y_excursion = waveform_excursion(delta_ky_idx)
    z_excursion = waveform_excursion(delta_kz_idx)

    y_active = y_excursion > activity_threshold
    z_active = z_excursion > activity_threshold

    print(
        "MPRAGE wave detection: "
        f"ky excursion={y_excursion:.6g}, "
        f"kz excursion={z_excursion:.6g}"
    )

    if y_active != z_active:
        active_axis = "sine/LIN only" if y_active else "cosine/PAR only"
        raise ValueError(
            "One-axis Wave-MPRAGE is not supported by this reconstruction. "
            f"Trajectory inspection detected {active_axis}. "
            "Use both wave axes or disable both."
        )

    detected_mode = "wave" if y_active and z_active else "nowave"

    if requested_mode == "auto":
        print(f"Detected MPRAGE wave mode: {detected_mode}")
        return detected_mode

    if requested_mode != detected_mode:
        raise ValueError(
            f"Requested --wave-mode/--mode={requested_mode!r}, but "
            f"trajectory inspection detected {detected_mode!r}."
        )

    print(f"Verified MPRAGE wave mode: {requested_mode}")
    return requested_mode


def fit_wave_psf_deviation_from_projection(mprage_data_file, mprage_seq_file, out_folder, file_tag,
                                           yflip=1, zflip=1, Ncalib=72, Nacs=32,
                                           slice_orientation='SAG'):
    """Fit PSF deviation from integrated refscan projection calibration blocks."""
    data_ref = load_ref(mprage_data_file)
    _check_integrated_refscan_shape(data_ref, Nacs=Nacs, Ncalib=Ncalib)
    Nx_os, _, _, _, Ncoil = data_ref.shape

    seq = pp.Sequence()
    seq.read(mprage_seq_file, remove_duplicates=False)
    defs = seq.definitions
    ktraj_adc, ktraj, t_exc, t_ref, t_adc = seq.calculate_kspace()
    Nacs_total = int(Nx_os * (Ncalib * 4 + Nacs * Nacs))
    if ktraj_adc.shape[1] < Nacs_total:
        raise ValueError(
            f"Sequence ADC trajectory has {ktraj_adc.shape[1]} samples, but integrated "
            f"calibration/ACS tail requires {Nacs_total} samples. Check Ncalib={Ncalib}, "
            f"Nacs={Nacs}, and Nx_os={Nx_os}."
        )
    k_adc = np.asarray(ktraj_adc[:, -Nacs_total:], dtype=np.float64).reshape(3, -1, Nx_os)

    fov_y, fov_z = _get_fov_yz(defs, slice_orientation=slice_orientation)

    a_fit_all = []
    b_fit_all = []
    c_fit_all = []
    mask_all = []

    # Integrated refscan set convention:
    #   set 0: nowave sin projection
    #   set 1: wave sin projection
    #   set 2: nowave cos projection
    #   set 3: wave cos projection
    #   last set: ACS block
    for wave_mode in ['sin', 'cos']:
        print(f'Calibrating wave trajectory of {wave_mode}')
        if wave_mode == 'sin':
            kspace_nowave_echo = data_ref[:, :Ncalib, :1, 0, :]
            kspace_wave_echo = data_ref[:, :Ncalib, :1, 1, :]
            ky_adc_sin = k_adc[1, Ncalib * 1:Ncalib * 2]
            delta_ky = ky_adc_sin[Ncalib // 2]
            delta_ky_idx = delta_ky * fov_y
            y_norm_lr = (np.arange(Ncalib) - (Ncalib / 2.0)) / Ncalib
            z_norm_lr = np.array([0.0])
            psf_np_sin = np.exp(-1j * yflip * 2.0 * np.pi * delta_ky_idx[:, None] * y_norm_lr[None, :]).astype(np.complex64)
            psf_theory = torch.from_numpy(psf_np_sin[..., np.newaxis])

        elif wave_mode == 'cos':
            kspace_nowave_echo = data_ref[:, :1, :Ncalib, 2, :]
            kspace_wave_echo = data_ref[:, :1, :Ncalib, 3, :]
            kz_adc_cos = k_adc[0, Ncalib * 3:Ncalib * 4]
            delta_kz = kz_adc_cos[Ncalib // 2]
            delta_kz_idx = delta_kz * fov_z
            y_norm_lr = np.array([0.0])
            z_norm_lr = (np.arange(Ncalib) - (Ncalib / 2.0)) / Ncalib
            psf_np_cos = np.ones((Nx_os, 1), dtype=np.complex64)
            psf_np_cos = psf_np_cos[..., np.newaxis] * np.exp(
                -1j * zflip * 2.0 * np.pi * delta_kz_idx[:, None, None] * z_norm_lr[None, None, :]
            ).astype(np.complex64)
            psf_theory = torch.from_numpy(psf_np_cos)

        # Convert both to image domain first, then to hybrid domain (FFT along readout only).
        img_nowave = ifft3call(kspace_nowave_echo)
        img_wave = ifft3call(kspace_wave_echo)

        hyb_nowave = fftc_dim(img_nowave, dim=0)
        hyb_wave = fftc_dim(img_wave, dim=0)

        # Average cross-power phase over coils to estimate PSF phase term.
        cross = hyb_wave * torch.conj(hyb_nowave) / (1e-8 + hyb_nowave * torch.conj(hyb_nowave))
        psf_real = torch.exp(1j * torch.angle(cross.mean(dim=-1)))

        psf_diff_lr = torch.angle(torch.conj(psf_theory) * psf_real)
        hyb_nowave_lr = hyb_nowave.clone()

        result = fit_wrapped_phase_planes(
            psf_diff=psf_diff_lr,
            hyb_nowave=hyb_nowave_lr,
            y_norm=y_norm_lr,
            z_norm=z_norm_lr,
            mask_mode="combined",
            mag_abs_floor=0.0,
            local_window_size=5,
            coherence_threshold=0.75,
            use_phase_coherence_weight=True,
            phase_weight_power=2.0,
            use_residual_coherence_refinement=True,
            residual_window_size=5,
            residual_coherence_threshold=0.75,
            use_residual_coherence_weight=True,
            residual_weight_power=2.0,
            n_irls=10,
            huber_delta=0.7,
            return_quality_maps=True,
            verbose=False,
        )

        a_fit_all.append(result["a_fit_all"])
        b_fit_all.append(result["b_fit_all"])
        c_fit_all.append(result["c_fit_all"])
        mask_all.append(result["mask"])

        if wave_mode == 'sin':
            tag = f'projy_{Ncalib}kyline'
        elif wave_mode == 'cos':
            tag = f'projz_{Ncalib}kzline'
        else:
            tag = ''
        _save_npy(out_folder + 'a_fit_all_' + str(tag) + '_' + file_tag, result["a_fit_all"], f'a(t) fit {tag}')
        _save_npy(out_folder + 'b_fit_all_' + str(tag) + '_' + file_tag, result["b_fit_all"], f'b(t) fit {tag}')
        _save_npy(out_folder + 'c_fit_all_' + str(tag) + '_' + file_tag, result["c_fit_all"], f'c(t) fit {tag}')

    a_fit = a_fit_all[0]
    b_fit = b_fit_all[1]
    c_fit = c_fit_all[0] + c_fit_all[1]

    return a_fit, b_fit, c_fit, Nacs_total


def _sine_line_model(t, A, w, phi, C1, C2):
    """Evaluate A*sin(w*t + phi) + C1*t + C2."""
    return A * np.sin(w * t + phi) + C1 * t + C2


def _fit_sine_plus_line(t, values):
    """Fit a sine plus linear trend to finite samples in one coefficient."""
    t = np.asarray(t, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    valid = np.isfinite(t) & np.isfinite(values)
    t = t[valid]
    values = values[valid]
    if t.size < 6:
        raise ValueError("At least 6 finite coefficient samples are required for sine-line PSF processing.")
    order = np.argsort(t)
    t = t[order]
    values = values[order]
    if np.ptp(t) == 0:
        raise ValueError("PSF fit kx coordinates must contain more than one distinct value.")

    t_ref = float(np.mean(t))
    x = t - t_ref
    span = float(np.ptp(x))
    unique_dt = np.diff(np.unique(t))
    median_dt = float(np.median(unique_dt))
    w_min = 2.0 * np.pi / span
    w_max = np.pi / median_dt

    C1_initial, C2_ref_initial = np.polyfit(x, values, 1)
    detrended = values - (C1_initial * x + C2_ref_initial)
    detrended -= np.mean(detrended)
    w_grid = np.linspace(w_min, w_max, 10000)
    power = lombscargle(x, detrended, w_grid, precenter=False, normalize=True)
    w_initial = float(w_grid[int(np.argmax(power))])

    design = np.column_stack([
        np.sin(w_initial * x),
        np.cos(w_initial * x),
        x,
        np.ones_like(x),
    ])
    sine_coef, cosine_coef, C1_initial, C2_ref_initial = np.linalg.lstsq(
        design, values, rcond=None
    )[0]
    A_initial = float(np.hypot(sine_coef, cosine_coef))
    phi_ref_initial = float(np.arctan2(cosine_coef, sine_coef))
    initial = np.array([
        max(A_initial, np.finfo(float).eps),
        w_initial,
        phi_ref_initial,
        C1_initial,
        C2_ref_initial,
    ])

    def residuals(parameters):
        A, w, phi_ref, C1, C2_ref = parameters
        return A * np.sin(w * x + phi_ref) + C1 * x + C2_ref - values

    result = least_squares(
        residuals,
        initial,
        bounds=(
            [0.0, w_min, -np.inf, -np.inf, -np.inf],
            [np.inf, w_max, np.inf, np.inf, np.inf],
        ),
        method="trf",
        x_scale="jac",
        loss="linear",
    )
    A, w, phi_ref, C1, C2_ref = result.x
    phi = (phi_ref - w * t_ref + np.pi) % (2.0 * np.pi) - np.pi
    C2 = C2_ref - C1 * t_ref
    return {
        "A": float(A),
        "w": float(w),
        "phi": float(phi),
        "C1": float(C1),
        "C2": float(C2),
        "success": bool(result.success),
        "message": str(result.message),
        "n_samples": int(t.size),
    }


def _process_psf_coefficients(
    a_raw,
    b_raw,
    c_raw,
    Nx_os,
    coefficient_processing="smooth",
    fit_kx_min=None,
    fit_kx_max=None,
    out_folder=None,
    file_tag="",
):
    """Apply mutually exclusive smoothing or sine-line coefficient processing."""
    mode = str(coefficient_processing).strip().lower()
    if mode == "smooth":
        return (
            smooth_1d_nan(a_raw, window=9),
            smooth_1d_nan(b_raw, window=9),
            smooth_1d_nan(c_raw, window=9),
        )
    if mode != "sine-line":
        raise ValueError("coefficient_processing must be 'smooth' or 'sine-line'.")
    if fit_kx_min is None or fit_kx_max is None:
        raise ValueError("sine-line PSF processing requires fit_kx_min and fit_kx_max.")
    fit_kx_min = int(fit_kx_min)
    fit_kx_max = int(fit_kx_max)
    if not (0 <= fit_kx_min < fit_kx_max <= int(Nx_os)):
        raise ValueError(
            f"PSF fit range must satisfy 0 <= min < max <= Nx_os; got "
            f"[{fit_kx_min}, {fit_kx_max}) with Nx_os={Nx_os}."
        )

    kx_fit = np.arange(fit_kx_min, fit_kx_max, dtype=float)
    kx_all = np.arange(int(Nx_os), dtype=float)
    outputs = []
    diagnostics = {}
    for name, raw in (("a", a_raw), ("b", b_raw), ("c", c_raw)):
        # Keep this branch tensor-native so dtype/device information remains
        # available when the full fitted curve is converted back to PyTorch.
        raw_1d = torch.as_tensor(raw).detach().squeeze()
        if raw_1d.ndim != 1:
            raise ValueError(
                f"{name}_raw should reduce to a 1D vector after squeeze; "
                f"got shape {tuple(raw_1d.shape)}"
            )

        params = _fit_sine_plus_line(
            kx_fit,
            raw_1d[fit_kx_min:fit_kx_max].cpu().numpy(),
        )
        fitted = _sine_line_model(
            kx_all,
            params["A"],
            params["w"],
            params["phi"],
            params["C1"],
            params["C2"],
        )
        outputs.append(torch.as_tensor(fitted, dtype=raw_1d.dtype, device=raw_1d.device))
        diagnostics[name] = params

    if out_folder is not None:
        diag_path = os.path.join(out_folder, f"psf_sine_line_fit_{file_tag}.json")
        with open(diag_path, "w") as f:
            json.dump(
                {
                    "model": "A*sin(w*kx+phi)+C1*kx+C2",
                    "kx_range": [fit_kx_min, fit_kx_max],
                    "kx_range_convention": "half-open",
                    "coefficients": diagnostics,
                },
                f,
                indent=2,
            )
        print(f"Saved sine-line PSF fit diagnostics: {diag_path}")
    return tuple(outputs)

def generate_calibrated_psf(mprage_data_file, mprage_seq_file, out_folder, Nx_os, Ny, Nz, file_tag,
                            yflip=-1, zflip=-1, Ncalib=72, Nacs=32,
                            slice_orientation='SAG', psf_plot=True,
                            coefficient_processing='smooth', fit_kx_min=None, fit_kx_max=None):
    """Generate calibrated wave PSF from the integrated calibration module."""
    a_fit_all, b_fit_all, c_fit_all, Nacs_total = fit_wave_psf_deviation_from_projection(
        mprage_data_file=mprage_data_file,
        mprage_seq_file=mprage_seq_file,
        out_folder=out_folder,
        file_tag=file_tag,
        yflip=yflip,
        zflip=zflip,
        Ncalib=Ncalib,
        Nacs=Nacs,
        slice_orientation=slice_orientation,
    )

    a_smooth, b_smooth, c_smooth = _process_psf_coefficients(
        a_fit_all,
        b_fit_all,
        c_fit_all,
        Nx_os=Nx_os,
        coefficient_processing=coefficient_processing,
        fit_kx_min=fit_kx_min,
        fit_kx_max=fit_kx_max,
        out_folder=out_folder,
        file_tag=file_tag,
    )

    a_fit = _squeeze_fit_vector(a_smooth, name='a_smooth')
    b_fit = _squeeze_fit_vector(b_smooth, name='b_smooth')
    c_fit = _squeeze_fit_vector(c_smooth, name='c_smooth')

    if psf_plot:
        plt.figure(figsize=(6, 4))
        plt.plot(a_fit, label="a(t)")
        plt.plot(b_fit, label="b(t)")
        plt.plot(c_fit, label="c(t)")
        plt.axvline(a_fit.shape[-1] // 2, linestyle='--', color='k')
        plt.axhline(0, linestyle='--', color='k')
        plt.legend()
        plt.ylim([-3, 3])
        plt.xlim([0, Nx_os])
        plt.title(f'Integrated PSF calibration fit ({coefficient_processing})')
        fig_path = out_folder + 'psf_integrated_calib_fit_' + file_tag + '.png'
        print(f"Saving PSF calibration fit plot to: {fig_path}")
        plt.savefig(fig_path)
        plt.close('all')

    # generate theoretical psf on the final [Ny, Nz] grid
    delta_ky_idx, delta_kz_idx = generate_theoretical_wave_trajectory(
        fn_seq=mprage_seq_file,
        Nx_os=Nx_os,
        Nacs_total=Nacs_total,
        slice_orientation=slice_orientation,
    )
    y_norm = (np.arange(Ny) - (Ny / 2.0)) / Ny
    z_norm = (np.arange(Nz) - (Nz / 2.0)) / Nz
    psf_np = np.exp(-1j * yflip * 2.0 * np.pi * delta_ky_idx[:, None] * y_norm[None, :]).astype(np.complex64)
    psf_np = psf_np[..., np.newaxis] * np.exp(
        -1j * zflip * 2.0 * np.pi * delta_kz_idx[:, None, None] * z_norm[None, None, :]
    ).astype(np.complex64)
    psf_theory = torch.from_numpy(psf_np)

    # generate calibrated psf
    psf_diff_pred_new = torch.zeros_like(torch.angle(psf_theory))
    Nx_os_psf = psf_theory.shape[0]
    if a_fit.shape[0] != Nx_os_psf or b_fit.shape[0] != Nx_os_psf or c_fit.shape[0] != Nx_os_psf:
        raise ValueError(
            f"PSF fit length mismatch: a={a_fit.shape[0]}, b={b_fit.shape[0]}, c={c_fit.shape[0]}, "
            f"but psf_theory has Nx_os={Nx_os_psf}."
        )

    y_norm_tensor = torch.from_numpy(y_norm)
    z_norm_tensor = torch.from_numpy(z_norm)
    Y_grid, Z_grid = torch.meshgrid(y_norm_tensor, z_norm_tensor, indexing='ij')

    y_flat = Y_grid.flatten()
    z_flat = Z_grid.flatten()

    for kx_loc in range(Nx_os_psf):
        ones = torch.ones_like(y_flat)
        A_full = torch.stack([y_flat, z_flat, ones], dim=1)
        coefficients = torch.Tensor((a_fit[kx_loc], b_fit[kx_loc], c_fit[kx_loc]))
        coefficients = coefficients.to(dtype=A_full.dtype)

        psf_diff_pred_flat = A_full @ coefficients
        psf_diff_pred_new[kx_loc] = psf_diff_pred_flat.view(psf_theory[kx_loc].shape)

    psf_diff_pred_new = torch.nan_to_num(psf_diff_pred_new.clone(), nan=0.0)
    psf_calib = psf_theory * torch.exp(1j * psf_diff_pred_new)
    return psf_calib, psf_theory


# -----------------------------------------------------------------------------
# CLI / I/O helpers
# -----------------------------------------------------------------------------

def _parse_cli_args():
    """Parse optional command-line arguments while tolerating notebook extras."""
    parser = argparse.ArgumentParser(
        description="Reconstruct Wave-MPRAGE/no-wave MPRAGE data from an integrated Siemens TWIX file."
    )
        # Canonical direct-path interface, aligned with the GRE reconstruction.
    #
    # The older MPRAGE-specific argument names remain accepted as aliases,
    # but --twix, --seq, --out, and --wave-mode are the preferred names.
    parser.add_argument(
        "--twix",
        "--mprage-data-file",
        dest="twix",
        required=True,
        help="Integrated Wave-MPRAGE + calibration Siemens TWIX .dat file.",
    )
    parser.add_argument(
        "--seq",
        "--mprage-seq-file",
        dest="seq",
        required=True,
        help="Matching integrated Wave-MPRAGE + calibration Pulseq .seq file.",
    )
    parser.add_argument(
        "--out",
        "--out-folder",
        dest="out",
        required=True,
        help="Output directory for reconstruction and calibration files.",
    )
    parser.add_argument(
        "--wave-mode",
        "--mode",
        "--tag-wave",
        dest="mode",
        choices=("auto", "wave", "nowave"),
        default="auto",
        help=(
            "Reconstruction mode. 'auto' determines wave versus no-wave "
            "from the MPRAGE imaging trajectory. --mode and --tag-wave "
            "are compatibility aliases."
        ),
    )

    parser.add_argument(
        "--file-tag",
        default=None,
        help="Suffix tag used in output filenames.",
    )
    parser.add_argument("--reuse-coil-calib", action="store_true",
                        help="Reuse the existing coil-compression matrix and the CSM cache for the selected ESPIRiT calibration mode when both are present.")
    parser.add_argument(
        "--espirit-device",
        choices=("auto", "cpu", "gpu"),
        default=None,
        help="ESPIRiT device: auto uses GPU when available and otherwise CPU. Default: auto.",
    )
    parser.add_argument(
        "--espirit-gpu-index",
        type=int,
        default=None,
        help="CUDA device index used when ESPIRiT selects a GPU. Default: 0.",
    )
    parser.add_argument(
        "--espirit-crop",
        type=float,
        default=None,
        help=(
            "ESPIRiT eigenvalue crop threshold. Lower values generally retain "
            "a larger sensitivity-map support region. Default: 0.8."
        ),
    )
    parser.add_argument(
        "--espirit-calib-mode",
        choices=("3d", "slice2d"),
        default=None,
        help=(
            "ESPIRiT calibration backend. '3d' is the native reference method "
            "and remains the default. 'slice2d' performs CPU-parallel 2D "
            "calibration over logical-RO hybrid-space slices and applies a "
            "conservative SAG whole-plane RO support guard (physical S-I)."
        ),
    )
    parser.add_argument(
        "--espirit-cpu-workers",
        type=int,
        default=None,
        help=(
            "CPU process workers for --espirit-calib-mode slice2d. By default "
            "the available physical-core count is used, limited by the number "
            "of logical readout slices."
        ),
    )
    parser.add_argument("--yflip", type=int, default=None,
                        help="Sign convention for y wave PSF calibration. Default: -1.")
    parser.add_argument("--zflip", type=int, default=None,
                        help="Sign convention for z wave PSF calibration. Default: -1.")
    parser.add_argument("--save-nifti", action="store_true",
                        help="Also save the reconstructed image as NIfTI after center-cropping readout oversampling.")
    parser.add_argument("--save-nifti-phase", action="store_true",
                        help="When --save-nifti is used, also save phase in radians. Magnitude is always saved.")
    parser.add_argument(
        "--save-bart-inputs",
        action="store_true",
        help=(
            "Export BART CFL inputs under <out>/bart_inputs[_tag]. This is "
            "available for wave acquisitions only."
        ),
    )
    parser.add_argument("--nifti-out-folder", default=None,
                        help="Folder for NIfTI outputs. Default: <out-folder>/nifti/.")
    parser.add_argument("--nifti-sub", default=None,
                        help="Subject/folder name for NIfTI outputs. Default is generated from recon settings.")
    parser.add_argument("--nifti-suffix", default=None,
                        help="NIfTI filename suffix. Default: MPRAGE.")
    parser.add_argument(
        "--nifti-axis-roles",
        nargs=3,
        default=("phase", "readout", "slice"),
        metavar=("AXIS0", "AXIS1", "AXIS2"),
        help=(
            "Twix physical roles of the reconstructed MPRAGE array axes. "
            "The sagittal MPRAGE reconstruction is ordered as "
            "(physical z, physical y, physical x), corresponding to "
            "(phase, readout, slice) in the Twix geometry."
        ),
    )
    parser.add_argument("--nifti-axis-flips", default=None,
                        help="Comma-separated booleans for physical array flips before NIfTI. Default: true,false,false.")
    parser.add_argument("--twix-coord-system", default=None, choices=("LPS", "RAS"),
                        help="Coordinate-system assumption for Twix Sag/Cor/Tra vectors. Default: LPS.")
    parser.add_argument("--twix-inplane-rot-sign", type=float, default=None,
                        help="Sign applied to Twix in-plane rotation. Default: -1.0.")
    parser.add_argument("--twix-use-fov-for-voxel-size", action="store_true",
                        help="Infer NIfTI voxel sizes from Twix FOV instead of reconstruction voxel size.")
    parser.add_argument(
        "--psf-coefficient-processing",
        choices=("smooth", "sine-line"),
        default="smooth",
        help=(
            "Post-process fitted PSF coefficients with the existing NaN-aware smoothing "
            "or replace smoothing with a sine-plus-line fit. Default: smooth."
        ),
    )
    parser.add_argument(
        "--psf-fit-kx-min",
        type=int,
        default=None,
        help="Inclusive first oversampled-readout index for sine-line PSF fitting.",
    )
    parser.add_argument(
        "--psf-fit-kx-max",
        type=int,
        default=None,
        help="Exclusive final oversampled-readout index for sine-line PSF fitting.",
    )

    return parser.parse_args()


def _prompt_for_value(name, prompt_text, default=None, required=True):
    """Read a value from globals, otherwise from stdin or a default."""
    if name in globals() and globals()[name] not in (None, ""):
        value = globals()[name]
        print(f"Using {name}: {value}")
        return value

    if default is not None:
        print(f"Using {name}: {default}")
        return default

    try:
        value = input(f"{prompt_text}: ").strip()
    except EOFError as exc:
        if required:
            raise ValueError(
                f"Missing required input '{name}'. Provide it as a global variable, "
                f"a command-line argument, or run interactively to enter it at the prompt."
            ) from exc
        return None

    if required and value == "":
        raise ValueError(f"Missing required input '{name}'.")
    return value


def _normalize_folder(folder):
    """Expand, create, and return a folder path with a trailing separator."""
    folder = os.path.abspath(os.path.expanduser(os.path.expandvars(str(folder))))
    os.makedirs(folder, exist_ok=True)
    return folder if folder.endswith(os.sep) else folder + os.sep


def _resolve_input_path(path_value, data_folder, label):
    """Resolve an input file path relative to data_folder and verify it exists."""
    path_value = os.path.expanduser(os.path.expandvars(str(path_value)))
    if data_folder and not os.path.isabs(path_value):
        path_value = os.path.join(data_folder, path_value)
    path_value = os.path.abspath(path_value)
    if not os.path.isfile(path_value):
        raise FileNotFoundError(f"{label} not found: {path_value}")
    print(f"Using {label}: {path_value}")
    return path_value


def _parse_wave_tag(value):
    """Normalize wave/no-wave user input."""
    value = str(value).strip().lower()
    if value in ("wave", "w", "yes", "y", "true", "1"):
        return "wave"
    if value in ("nowave", "no-wave", "no_wave", "nw", "no", "n", "false", "0"):
        return "nowave"
    raise ValueError("tag_wave must be 'wave' or 'nowave' (yes/no is also accepted interactively).")


def _npy_output_path(path_without_ext):
    """Return the exact .npy filename that will be written."""
    path_without_ext = str(path_without_ext)
    return path_without_ext if path_without_ext.endswith(".npy") else path_without_ext + ".npy"


def _espirit_cache_tag(file_tag, mode):
    """Return a mode-specific CSM tag for the SAG MPRAGE implementation."""
    mode = str(mode).strip().lower()
    file_tag = str(file_tag)
    if mode == "3d":
        return file_tag
    if mode == "slice2d":
        # The SAG whole-plane RO guard changes the estimator output, so use a
        # distinct cache tag rather than reusing pre-guard slice2d CSMs.
        prefix = "slice2d_sagmask"
        return prefix if file_tag == "" else prefix + "_" + file_tag
    raise ValueError("ESPIRiT calibration mode must be '3d' or 'slice2d'.")


def _save_npy(path_without_ext, array, label):
    """Save a NumPy/PyTorch object and print the exact output path."""
    out_path = _npy_output_path(path_without_ext)
    print(f"Saving {label} to: {out_path}")
    np.save(out_path, array)
    return out_path


def _collect_runtime_config():
    """Collect runtime paths/tags from CLI args, existing globals, or prompts."""
    cli = _parse_cli_args()

        # Direct input paths: no shared --data-folder is required.
    mprage_data_value = _resolve_input_path(
        cli.twix,
        data_folder=None,
        label="integrated MPRAGE data file",
    )
    mprage_seq_value = _resolve_input_path(
        cli.seq,
        data_folder=None,
        label="integrated MPRAGE sequence file",
    )
    out_folder_value = _normalize_folder(cli.out)

    # Retain this internal value because the existing return dictionary and
    # main() still contain a data_folder entry. It is not used to resolve the
    # input files anymore.
    data_folder_value = os.path.dirname(mprage_data_value)

    file_tag_value = "" if cli.file_tag is None else str(cli.file_tag)

    # Read enough sequence information to distinguish wave from no-wave while
    # excluding the appended integrated calibration/ACS trajectory.
    mode_seq = pp.Sequence()
    mode_seq.read(mprage_seq_value, remove_duplicates=False)
    mode_defs = mode_seq.definitions

    mode_geom = _derive_hardcoded_sag_logical_geometry(mode_defs)
    mode_os_factor = int(
        mode_defs.get("ReadoutOversamplingFactor", 4)
    )
    mode_Nx_os = int(mode_geom["Nro"] * mode_os_factor)
    mode_ncalib = int(
        mode_defs.get("Calibration_Ncalib1", 72)
    )
    mode_nacs = int(
        mode_defs.get("Calibration_Nacs", 32)
    )
    mode_orientation = mode_defs.get(
        "OrientationMapping",
        "SAG",
    )

    tag_wave_value = _resolve_mprage_wave_mode(
        requested_mode=cli.mode,
        mprage_seq_file=mprage_seq_value,
        Nx_os=mode_Nx_os,
        Ncalib=mode_ncalib,
        Nacs=mode_nacs,
        slice_orientation=mode_orientation,
    )
    save_bart_inputs_value = bool(
        cli.save_bart_inputs or globals().get("save_bart_inputs", False)
    )
    if save_bart_inputs_value and tag_wave_value != "wave":
        raise ValueError("--save-bart-inputs requires a wave acquisition.")

    reuse_coil_calib_value = bool(cli.reuse_coil_calib or globals().get("reuse_coil_calib", False))
    espirit_device_value = cli.espirit_device
    if espirit_device_value is None:
        espirit_device_value = globals().get("espirit_device", "auto")
    espirit_device_value = str(espirit_device_value).strip().lower()
    if espirit_device_value not in ("auto", "cpu", "gpu"):
        raise ValueError("espirit_device must be 'auto', 'cpu', or 'gpu'.")

    if cli.espirit_gpu_index is not None:
        espirit_gpu_index_value = int(cli.espirit_gpu_index)
    else:
        espirit_gpu_index_value = int(globals().get("espirit_gpu_index", 0))
    if espirit_gpu_index_value < 0:
        raise ValueError("espirit_gpu_index must be a non-negative integer.")

    if cli.espirit_crop is not None:
        espirit_crop_value = float(cli.espirit_crop)
    else:
        espirit_crop_value = float(globals().get("espirit_crop", 0.8))

    if not np.isfinite(espirit_crop_value) or not 0.0 <= espirit_crop_value <= 1.0:
        raise ValueError("--espirit-crop must be a finite value between 0 and 1.")

    espirit_calib_mode_value = cli.espirit_calib_mode
    if espirit_calib_mode_value is None:
        espirit_calib_mode_value = globals().get("espirit_calib_mode", "3d")
    espirit_calib_mode_value = str(espirit_calib_mode_value).strip().lower()
    if espirit_calib_mode_value not in ("3d", "slice2d"):
        raise ValueError("espirit_calib_mode must be '3d' or 'slice2d'.")

    if cli.espirit_cpu_workers is not None:
        espirit_cpu_workers_value = int(cli.espirit_cpu_workers)
    else:
        global_workers = globals().get("espirit_cpu_workers", None)
        espirit_cpu_workers_value = None if global_workers in (None, "") else int(global_workers)
    if espirit_cpu_workers_value is not None and espirit_cpu_workers_value < 1:
        raise ValueError("--espirit-cpu-workers must be a positive integer.")
    if espirit_calib_mode_value == "slice2d" and espirit_device_value == "gpu":
        raise ValueError(
            "--espirit-calib-mode slice2d is CPU-only; use --espirit-device cpu "
            "or auto, or select --espirit-calib-mode 3d."
        )

    yflip_value = _get_optional_int("yflip", cli.yflip, default=-1, allowed_values=(-1, 1))
    zflip_value = _get_optional_int("zflip", cli.zflip, default=-1, allowed_values=(-1, 1))

    save_nifti_value = bool(cli.save_nifti or globals().get("save_nifti", False))
    save_nifti_phase_value = bool(cli.save_nifti_phase or globals().get("save_nifti_phase", False))

    nifti_out_folder_value = cli.nifti_out_folder
    if nifti_out_folder_value is None and "nifti_out_folder" in globals() and globals()["nifti_out_folder"] not in (None, ""):
        nifti_out_folder_value = globals()["nifti_out_folder"]
    if nifti_out_folder_value is None:
        nifti_out_folder_value = os.path.join(out_folder_value, "nifti")
    nifti_out_folder_value = _normalize_folder(nifti_out_folder_value)

    nifti_sub_value = cli.nifti_sub
    if nifti_sub_value is None and "nifti_sub" in globals() and globals()["nifti_sub"] not in (None, ""):
        nifti_sub_value = str(globals()["nifti_sub"])

    nifti_suffix_value = cli.nifti_suffix
    if nifti_suffix_value is None and "nifti_suffix" in globals() and globals()["nifti_suffix"] not in (None, ""):
        nifti_suffix_value = str(globals()["nifti_suffix"])
    if nifti_suffix_value is None:
        nifti_suffix_value = "MPRAGE"
    nifti_suffix_value = _sanitize_filename_component(nifti_suffix_value)

    axis_roles_source = cli.nifti_axis_roles
    if axis_roles_source is None and "nifti_axis_roles" in globals() and globals()["nifti_axis_roles"] not in (None, ""):
        axis_roles_source = globals()["nifti_axis_roles"]
    nifti_axis_roles_value = _parse_axis_roles(axis_roles_source, default=("phase", "readout", "slice"))

    axis_flips_source = cli.nifti_axis_flips
    if axis_flips_source is None and "nifti_axis_flips" in globals() and globals()["nifti_axis_flips"] not in (None, ""):
        axis_flips_source = globals()["nifti_axis_flips"]
    nifti_axis_flips_value = _parse_bool_tuple(axis_flips_source, default=(True, False, False))

    twix_coord_system_value = cli.twix_coord_system
    if twix_coord_system_value is None and "twix_coord_system" in globals() and globals()["twix_coord_system"] not in (None, ""):
        twix_coord_system_value = str(globals()["twix_coord_system"]).upper()
    if twix_coord_system_value is None:
        twix_coord_system_value = "LPS"
    if twix_coord_system_value not in ("LPS", "RAS"):
        raise ValueError("twix_coord_system must be 'LPS' or 'RAS'.")

    if cli.twix_inplane_rot_sign is not None:
        twix_inplane_rot_sign_value = float(cli.twix_inplane_rot_sign)
    elif "twix_inplane_rot_sign" in globals() and globals()["twix_inplane_rot_sign"] not in (None, ""):
        twix_inplane_rot_sign_value = float(globals()["twix_inplane_rot_sign"])
    else:
        twix_inplane_rot_sign_value = -1.0

    psf_coefficient_processing_value = str(cli.psf_coefficient_processing).strip().lower()
    psf_fit_kx_min_value = cli.psf_fit_kx_min
    psf_fit_kx_max_value = cli.psf_fit_kx_max
    if psf_coefficient_processing_value == "sine-line":
        if psf_fit_kx_min_value is None or psf_fit_kx_max_value is None:
            raise ValueError(
                "--psf-coefficient-processing sine-line requires both "
                "--psf-fit-kx-min and --psf-fit-kx-max."
            )
        if psf_fit_kx_min_value < 0 or psf_fit_kx_max_value <= psf_fit_kx_min_value:
            raise ValueError(
                "PSF fit indices must satisfy 0 <= --psf-fit-kx-min < --psf-fit-kx-max."
            )
    elif psf_fit_kx_min_value is not None or psf_fit_kx_max_value is not None:
        raise ValueError(
            "--psf-fit-kx-min/--psf-fit-kx-max are only valid with "
            "--psf-coefficient-processing sine-line."
        )

    twix_use_fov_for_voxel_size_value = bool(
        cli.twix_use_fov_for_voxel_size or globals().get("twix_use_fov_for_voxel_size", False)
    )

    mprage_data_value = _resolve_input_path(mprage_data_value, data_folder_value, "integrated MPRAGE data file")
    mprage_seq_value = _resolve_input_path(mprage_seq_value, data_folder_value, "integrated MPRAGE sequence file")

    print("Runtime configuration summary:")
    print(f"  twix: {mprage_data_value}")
    print(f"  seq: {mprage_seq_value}")
    print(f"  out: {out_folder_value}")
    print(f"  file_tag:          {file_tag_value}")
    print(f"  reconstruction:    {tag_wave_value}")
    print(f"  reuse_coil_calib:  {reuse_coil_calib_value}")
    print("  coil compression: CPU")
    print(f"  ESPIRiT request:  {espirit_device_value} (GPU index {espirit_gpu_index_value})")
    print(f"  ESPIRiT mode:     {espirit_calib_mode_value}")
    print(f"  ESPIRiT crop:     {espirit_crop_value:g}")
    if espirit_calib_mode_value == "slice2d":
        print(
            "  ESPIRiT workers:  "
            + ("auto" if espirit_cpu_workers_value is None else str(espirit_cpu_workers_value))
        )
        print("  SAG RO support:   auto whole-plane guard, S-I, padding=3")
    else:
        print("  ESPIRiT workers:  n/a (native 3D backend)")
    print("  CG-SENSE:         CPU")
    print(f"  yflip/zflip:       {yflip_value}/{zflip_value}")
    print(f"  save_nifti:        {save_nifti_value}")
    print(f"  save_bart_inputs:  {save_bart_inputs_value}")
    if save_nifti_value:
        print(f"  nifti_out_folder:  {nifti_out_folder_value}")
        print(f"  nifti_sub:         {nifti_sub_value if nifti_sub_value else '<auto>'}")
        print(f"  nifti_suffix:      {nifti_suffix_value}")
        print(f"  nifti_axis_roles:  {nifti_axis_roles_value}")
        print(f"  nifti_axis_flips:  {nifti_axis_flips_value}")
        print(f"  twix_coord_system: {twix_coord_system_value}")
        print(f"  twix_rot_sign:     {twix_inplane_rot_sign_value}")

    return {
        "data_folder": data_folder_value,
        "out_folder": out_folder_value,
        "mprage_data_file": mprage_data_value,
        "mprage_seq_file": mprage_seq_value,
        "file_tag": file_tag_value,
        "tag_wave": tag_wave_value,
        "reuse_coil_calib": reuse_coil_calib_value,
        "espirit_device": espirit_device_value,
        "espirit_gpu_index": espirit_gpu_index_value,
        "espirit_crop": espirit_crop_value,
        "espirit_calib_mode": espirit_calib_mode_value,
        "espirit_cpu_workers": espirit_cpu_workers_value,
        "yflip": yflip_value,
        "zflip": zflip_value,
        "save_nifti": save_nifti_value,
        "save_nifti_phase": save_nifti_phase_value,
        "save_bart_inputs": save_bart_inputs_value,
        "nifti_out_folder": nifti_out_folder_value,
        "nifti_sub": nifti_sub_value,
        "nifti_suffix": nifti_suffix_value,
        "nifti_axis_roles": nifti_axis_roles_value,
        "nifti_axis_flips": nifti_axis_flips_value,
        "twix_coord_system": twix_coord_system_value,
        "twix_inplane_rot_sign": twix_inplane_rot_sign_value,
        "twix_use_fov_for_voxel_size": twix_use_fov_for_voxel_size_value,
        "psf_coefficient_processing": psf_coefficient_processing_value,
        "psf_fit_kx_min": psf_fit_kx_min_value,
        "psf_fit_kx_max": psf_fit_kx_max_value,
    }


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _get_optional_int(name, cli_value, default, allowed_values=None):
    """Get an optional integer from CLI, globals, or a default.

    By default, values must be positive. If allowed_values is provided,
    the value only needs to be one of those explicit integer choices.
    This is used for sign flags such as yflip/zflip, where -1 is valid.
    """
    if cli_value is not None:
        value = int(cli_value)
    elif name in globals() and globals()[name] not in (None, ""):
        value = int(globals()[name])
    else:
        value = int(default)

    if allowed_values is not None:
        allowed_values = tuple(int(v) for v in allowed_values)
        if value not in allowed_values:
            raise ValueError(f"{name} must be one of {allowed_values}.")
    elif value < 1:
        raise ValueError(f"{name} must be a positive integer.")

    print(f"Using {name}: {value}")
    return value


def _as_float_list(value, default):
    """Convert a Pulseq definition value to a float list."""
    if value is None:
        value = default
    if isinstance(value, str):
        value = value.replace("[", "").replace("]", "").replace(",", " ").split()
    if np.isscalar(value):
        value = [float(value)] * 3
    value = [float(v) for v in value]
    if len(value) == 1:
        value = value * 3
    if len(value) < 3:
        raise ValueError(f"FOV definition must contain at least 3 values; got {value}")
    return value


def _derive_hardcoded_sag_logical_geometry(defs):
    """Return logical recon geometry for current sagittal integrated MPRAGE.

    The sequence does not save ax.d1/ax.d2/ax.d3 in definitions. Current
    integrated Wave-MPRAGE uses ax.d1='z' for readout, ax.d2='x' for PAR,
    and ax.d3='y' for LIN. Therefore logical reconstruction axes are:
        array axis 0 / RO  = physical z = defs['Nz']
        array axis 1 / LIN = physical y = defs['Ny']
        array axis 2 / PAR = physical x = defs['Nx']
    """
    fov_xyz = _as_float_list(defs.get('FOV', [0.224, 0.224, 0.224]), default=[0.224, 0.224, 0.224])
    n_xyz = [
        int(defs.get('Nx', 256)),
        int(defs.get('Ny', 256)),
        int(defs.get('Nz', 192)),
    ]

    n_ro = n_xyz[2]
    n_lin = n_xyz[1]
    n_par = n_xyz[0]
    fov_ro = float(fov_xyz[2])
    fov_lin = float(fov_xyz[1])
    fov_par = float(fov_xyz[0])

    return {
        "Nxyz": tuple(n_xyz),
        "FOVxyz": tuple(float(v) for v in fov_xyz),
        "Nro": n_ro,
        "Nlin": n_lin,
        "Npar": n_par,
        "FOVro": fov_ro,
        "FOVlin": fov_lin,
        "FOVpar": fov_par,
        "res_ro": fov_ro / n_ro,
        "res_lin": fov_lin / n_lin,
        "res_par": fov_par / n_par,
        "readout_axis": "z",
        "lin_axis": "y",
        "par_axis": "x",
    }


def _assert_sag_geometry(defs):
    """Keep the current SAG-only reconstruction convention explicit."""
    orientation = str(defs.get("OrientationMapping", "SAG")).strip().upper()
    if orientation != "SAG":
        raise ValueError(
            f"This reconstruction is validated only for SAG geometry; got OrientationMapping={orientation!r}."
        )
    expected_axes = {
        "ReadoutAxis": "z",
        "InnerPEAxis": "x",
        "OuterPEAxis": "y",
    }
    for key, expected in expected_axes.items():
        if key not in defs:
            print(f"Geometry diagnostic: .seq definition {key} is absent; using asserted SAG mapping {expected}.")
            continue
        actual = str(defs.get(key)).strip().lower()
        if actual != expected:
            raise ValueError(
                f"SAG geometry assertion failed: {key}={actual!r}, expected {expected!r}."
            )
    return orientation


def _derive_nifti_voxel_size_mm(defs, geom):
    """Return logical RO/LIN/PAR spacing in mm without changing reconstruction geometry."""
    from_fov_mm = np.asarray(
        [geom["res_ro"], geom["res_lin"], geom["res_par"]], dtype=float
    ) * 1e3
    direct_values = []
    for key in ("ResolutionZ_mm", "ResolutionY_mm", "ResolutionX_mm"):
        value = _first_finite_definition(defs, key)
        if value is None:
            direct_values = []
            break
        direct_values.append(value)

    if direct_values:
        voxel_size_mm = np.asarray(direct_values, dtype=float)
        if not np.allclose(voxel_size_mm, from_fov_mm, rtol=1e-3, atol=1e-4):
            print(
                "WARNING: .seq Resolution*_mm definitions disagree with FOV/matrix-derived spacing.\n"
                f"  Resolution*_mm logical RO/LIN/PAR: {voxel_size_mm.tolist()} mm\n"
                f"  FOV/matrix logical RO/LIN/PAR:    {from_fov_mm.tolist()} mm\n"
                "  NIfTI export will use the explicit Resolution*_mm definitions."
            )
    else:
        voxel_size_mm = from_fov_mm
        print("NIfTI spacing: Resolution*_mm definitions unavailable; using FOV/matrix converted from m to mm.")

    if not np.all(np.isfinite(voxel_size_mm)) or np.any(voxel_size_mm <= 0):
        raise ValueError(f"Invalid NIfTI voxel size derived from .seq: {voxel_size_mm.tolist()} mm")
    if np.any(voxel_size_mm < 0.05) or np.any(voxel_size_mm > 20.0):
        raise ValueError(
            "Implausible NIfTI voxel size in millimetres: "
            f"{voxel_size_mm.tolist()}. This commonly indicates an m/mm/um conversion error."
        )
    return tuple(float(v) for v in voxel_size_mm)


def _coerce_twix_fov_mm(raw_value, expected_mm):
    """Choose whether a TWIX FOV is already mm or is represented in metres."""
    if raw_value is None:
        return None, "missing"
    raw = float(raw_value)
    candidates = ((raw, "raw-as-mm"), (raw * 1e3, "raw-as-m-converted-to-mm"))
    value, interpretation = min(
        candidates,
        key=lambda item: abs(item[0] - expected_mm) / max(abs(expected_mm), 1e-12),
    )
    return float(value), interpretation


def _direction_patient_string(vector_ras):
    """Describe the positive direction of a RAS vector as an anatomical arrow."""
    v = np.asarray(vector_ras, dtype=float)
    if v.shape != (3,) or not np.all(np.isfinite(v)) or np.linalg.norm(v) == 0:
        return "unknown"
    axis = int(np.argmax(np.abs(v)))
    positive = v[axis] >= 0
    if axis == 0:
        return "L->R" if positive else "R->L"
    if axis == 1:
        return "P->A" if positive else "A->P"
    return "I->S" if positive else "S->I"


def _report_seq_twix_geometry(
    twix_file,
    geom,
    received_image_shape,
    os_factor,
    voxel_size_mm,
    twix_array_axis_roles,
    twix_array_axis_flips,
    twix_coord_system,
    twix_inplane_rot_sign,
):
    """Print a warning-only .seq/TWIX geometry and PE-direction diagnostic."""
    from utils.nifti_export_twix import make_nifti_affine_from_twix

    logical_shape = (int(geom["Nro"]), int(geom["Nlin"]), int(geom["Npar"]))
    try:
        _, _, twix_info = make_nifti_affine_from_twix(
            twix_file=twix_file,
            npy_shape=logical_shape,
            twix_array_axis_roles=twix_array_axis_roles,
            twix_array_axis_flips=(False, False, False),
            twix_coord_system=twix_coord_system,
            twix_inplane_rot_sign=twix_inplane_rot_sign,
            twix_use_fov_for_voxel_size=False,
            voxel_size_mm=voxel_size_mm,
        )
    except Exception as exc:
        message = f"Unable to read TWIX geometry ({type(exc).__name__}: {exc})"
        print("Sequence/TWIX geometry diagnostics (warning-only)")
        print(f"  WARNING: {message}")
        print("  Reconstruction will continue.")
        return {
            "Status": "warning",
            "Passed": False,
            "SequenceOrientation": "SAG",
            "Error": message,
            "FOVChecks": {},
            "MatrixChecks": {},
            "Directions": {
                "ReadoutPhysicalAxis": "z",
                "OuterPhaseEncodingPhysicalAxis": "y",
                "InnerPhaseEncodingPhysicalAxis": "x",
            },
        }

    expected_fov_by_axis_mm = np.asarray(
        [geom["FOVro"], geom["FOVlin"], geom["FOVpar"]], dtype=float
    ) * 1e3
    axis_for_role = {role: axis for axis, role in enumerate(twix_array_axis_roles)}
    twix_fov_raw = twix_info.get("FOVRaw", twix_info.get("FOV", {}))
    fov_checks = {}
    all_match = True
    for role in ("readout", "phase", "slice"):
        axis = axis_for_role[role]
        expected_mm = float(expected_fov_by_axis_mm[axis])
        raw = twix_fov_raw.get(role)
        observed_mm, interpretation = _coerce_twix_fov_mm(raw, expected_mm)
        match = observed_mm is not None and np.isclose(observed_mm, expected_mm, rtol=0.01, atol=0.5)
        all_match = all_match and bool(match)
        fov_checks[role] = {
            "SequenceMm": expected_mm,
            "TwixRaw": None if raw is None else float(raw),
            "TwixInterpretedMm": observed_mm,
            "TwixUnitInterpretation": interpretation,
            "Match": bool(match),
        }

    received = tuple(int(v) for v in received_image_shape)
    matrix_checks = {
        "ExpectedLogicalRO": int(geom["Nro"]),
        "ExpectedReadoutOversampled": int(geom["Nro"] * os_factor),
        "ExpectedLogicalLIN": int(geom["Nlin"]),
        "ExpectedLogicalPAR": int(geom["Npar"]),
        "ReceivedReadoutSamples": received[0],
        "ReceivedLINExtent": received[1],
        "ReceivedPARExtent": received[2],
        "ReadoutSamplesMatch": received[0] == int(geom["Nro"] * os_factor),
        "ReceivedLINWithinSequenceMatrix": received[1] <= int(geom["Nlin"]),
        "ReceivedPARWithinSequenceMatrix": received[2] <= int(geom["Npar"]),
        "TwixHeaderMatrix": twix_info.get("Matrix", {}),
    }
    all_match = all_match and matrix_checks["ReadoutSamplesMatch"]
    all_match = all_match and matrix_checks["ReceivedLINWithinSequenceMatrix"]
    all_match = all_match and matrix_checks["ReceivedPARWithinSequenceMatrix"]

    direction_by_role = {
        "readout": twix_info.get("ReadoutDirectionRAS"),
        "phase": twix_info.get("PhaseDirectionRAS"),
        "slice": twix_info.get("SliceDirectionRAS"),
    }
    stored_direction_by_role = {role: list(vector) if vector is not None else None
                                for role, vector in direction_by_role.items()}
    for axis, role in enumerate(twix_array_axis_roles):
        if bool(twix_array_axis_flips[axis]) and stored_direction_by_role[role] is not None:
            stored_direction_by_role[role] = (-np.asarray(stored_direction_by_role[role], dtype=float)).tolist()
    directions = {
        "ReadoutPhysicalAxis": "z",
        "OuterPhaseEncodingPhysicalAxis": "y",
        "InnerPhaseEncodingPhysicalAxis": "x",
        "ReadoutAcquisitionDirectionPatient": _direction_patient_string(direction_by_role["phase"]),
        "OuterPhaseEncodingAcquisitionDirectionPatient": _direction_patient_string(direction_by_role["readout"]),
        "InnerPhaseEncodingAcquisitionDirectionPatient": _direction_patient_string(direction_by_role["slice"]),
        "ReadoutStoredPositiveDirectionPatient": _direction_patient_string(stored_direction_by_role["phase"]),
        "OuterPhaseEncodingStoredPositiveDirectionPatient": _direction_patient_string(stored_direction_by_role["readout"]),
        "InnerPhaseEncodingStoredPositiveDirectionPatient": _direction_patient_string(stored_direction_by_role["slice"]),
        "ReadoutDirectionRAS": direction_by_role["phase"],
        "OuterPhaseEncodingDirectionRAS": direction_by_role["readout"],
        "InnerPhaseEncodingDirectionRAS": direction_by_role["slice"],
        "ReadoutStoredDirectionRAS": stored_direction_by_role["phase"],
        "OuterPhaseEncodingStoredDirectionRAS": stored_direction_by_role["readout"],
        "InnerPhaseEncodingStoredDirectionRAS": stored_direction_by_role["slice"],
    }

    print("Sequence/TWIX geometry diagnostics (warning-only)")
    print("  Orientation assertion: SAG")
    for role, check in fov_checks.items():
        status = "MATCH" if check["Match"] else "WARNING"
        print(
            f"  FOV {role:7s}: seq={check['SequenceMm']:g} mm, "
            f"twix={check['TwixInterpretedMm']} mm "
            f"({check['TwixUnitInterpretation']}) [{status}]"
        )
    print(
        "  Matrix received: "
        f"RO_os={received[0]}, LIN_extent={received[1]}, PAR_extent={received[2]}"
    )
    print(f"  Readout acquisition direction:        {directions['ReadoutAcquisitionDirectionPatient']}")
    print(f"  Outer PE / LIN acquisition direction: {directions['OuterPhaseEncodingAcquisitionDirectionPatient']}")
    print(f"  Inner PE / PAR acquisition direction: {directions['InnerPhaseEncodingAcquisitionDirectionPatient']}")
    print(f"  Readout stored positive direction:     {directions['ReadoutStoredPositiveDirectionPatient']}")
    print(f"  Outer PE / LIN stored direction:       {directions['OuterPhaseEncodingStoredPositiveDirectionPatient']}")
    print(f"  Inner PE / PAR stored direction:       {directions['InnerPhaseEncodingStoredPositiveDirectionPatient']}")
    if not all_match:
        print("WARNING: one or more .seq/TWIX geometry diagnostics did not match; reconstruction will continue.")

    return {
        "Status": "match" if all_match else "warning",
        "Passed": bool(all_match),
        "SequenceOrientation": "SAG",
        "FOVChecks": fov_checks,
        "MatrixChecks": matrix_checks,
        "Directions": directions,
    }

def _parse_axis_roles(value, default=("readout", "phase", "slice")):
    """Parse comma-separated NIfTI/Twix axis roles."""
    if value is None:
        roles = tuple(default)
    elif isinstance(value, str):
        roles = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    else:
        roles = tuple(str(item).strip().lower() for item in value)

    valid = {"readout", "phase", "slice"}
    if len(roles) != 3 or set(roles) != valid:
        raise ValueError(
            "nifti_axis_roles must contain readout, phase, and slice exactly once, "
            f"got {roles}."
        )
    return roles


def _parse_bool_tuple(value, default=(True, False, False)):
    """Parse comma-separated booleans such as 'false,true,false'."""
    if value is None:
        return tuple(bool(x) for x in default)
    if isinstance(value, str):
        items = [item.strip().lower() for item in value.split(",") if item.strip()]
    else:
        items = list(value)
    if len(items) != 3:
        raise ValueError(f"Expected three boolean values, got {items}.")

    out = []
    for item in items:
        if isinstance(item, (bool, np.bool_)):
            out.append(bool(item))
        elif str(item).lower() in ("1", "true", "t", "yes", "y"):
            out.append(True)
        elif str(item).lower() in ("0", "false", "f", "no", "n"):
            out.append(False)
        else:
            raise ValueError(f"Cannot parse boolean value {item!r}.")
    return tuple(out)


def _get_fov_yz(defs, slice_orientation='SAG'):
    """Return physical y/z FOVs for PSF phase scaling."""
    fov_def = _as_float_list(defs.get('FOV', [0.224, 0.224, 0.224]), default=[0.224, 0.224, 0.224])
    if slice_orientation == 'SAG':
        fov_y = float(fov_def[1])
        fov_z = float(fov_def[0])
    elif slice_orientation == 'TRA':
        fov_y = float(fov_def[1])
        fov_z = float(fov_def[2])
    else:
        raise ValueError(f'Unsupported slice orientation {slice_orientation}')
    return fov_y, fov_z


def _check_image_data_shape(img, Nx_os, Ny, Nz):
    """Validate loaded image k-space dimensions before zero filling."""
    if img.ndim != 4:
        raise ValueError(f"Expected image data shape (Nx_os, Ny_acq, Nz_acq, Ncoil); got {tuple(img.shape)}")
    if img.shape[0] > Nx_os:
        raise ValueError(f"Image readout dimension {img.shape[0]} exceeds expected Nx_os={Nx_os}.")
    if img.shape[1] > Ny or img.shape[2] > Nz:
        raise ValueError(
            f"Image PE dimensions {tuple(img.shape[1:3])} exceed expected Ny/Nz={(Ny, Nz)}."
        )
    print(f"Loaded image data shape: {tuple(img.shape)}")


def _check_integrated_refscan_shape(data_ref, Nacs=32, Ncalib=None):
    """Validate the integrated refscan layout."""
    if data_ref.ndim != 5:
        raise ValueError(
            f"Expected integrated refscan shape (Nx_os, Ny_ref, Nz_ref, set, Ncoil); got {tuple(data_ref.shape)}"
        )
    if data_ref.shape[3] < 5:
        raise ValueError(
            f"Integrated refscan must contain at least 5 sets: 4 PSF calibration sets + 1 ACS set. "
            f"Got {data_ref.shape[3]} sets."
        )
    if data_ref.shape[1] < Nacs or data_ref.shape[2] < Nacs:
        raise ValueError(
            f"Requested Nacs={Nacs}, but refscan PE dimensions are {tuple(data_ref.shape[1:3])}."
        )
    if Ncalib is not None and (data_ref.shape[1] < Ncalib or data_ref.shape[2] < Ncalib):
        raise ValueError(
            f"Requested Ncalib={Ncalib}, but refscan PE dimensions are {tuple(data_ref.shape[1:3])}."
        )
    print(f"Loaded integrated refscan shape: {tuple(data_ref.shape)}")


def _check_csm_shape(csm_full_cc_np, Nx, Ny, Nz):
    """Validate ESPIRiT CSM dimensions before inserting into oversampled readout grid."""
    if csm_full_cc_np.ndim != 4:
        raise ValueError(f"Expected CSM shape (ncc, Nx, Ny, Nz); got {tuple(csm_full_cc_np.shape)}")
    if csm_full_cc_np.shape[1] != Nx or csm_full_cc_np.shape[2] != Ny or csm_full_cc_np.shape[3] != Nz:
        raise ValueError(
            f"CSM shape {tuple(csm_full_cc_np.shape)} does not match expected (ncc, {Nx}, {Ny}, {Nz})."
        )


def _squeeze_fit_vector(arr, name='fit'):
    """Return a 1D fit vector, preserving the prototype behavior while checking shape."""
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(f"{name} should reduce to a 1D vector after squeeze; got shape {arr.shape}")
    return arr


if __name__ == "__main__":
    main()
