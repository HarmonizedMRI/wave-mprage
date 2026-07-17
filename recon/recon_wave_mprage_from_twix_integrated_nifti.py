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

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy import io
import pypulseq as pp

import platform
import os
import argparse
from pathlib import Path

import cupy as cp
import sigpy as sp
import sigpy.mri as mr
import gc

from scipy.ndimage import zoom

from utils.twix_import import *
from utils.coil_compression_kspace import *
from utils.plot_coil_sens import *

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
    slice_orientation = defs.get('OrientationMapping', 'SAG')

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

    print("Importing image data from integrated TWIX file...")
    img = load_img(mprage_data_file)
    if not torch.is_tensor(img):
        img = torch.as_tensor(img)
    _check_image_data_shape(img, Nx_os, Ny, Nz)
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
        )
        print("Generated calibrated PSF")

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
                voxel_size_mm=(res_x * 1e3, res_y * 1e3, res_z * 1e3),
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
                voxel_size_mm=(res_x * 1e3, res_y * 1e3, res_z * 1e3),
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
    twix_array_axis_roles=("readout", "phase", "slice"),
    twix_array_axis_flips=(False, True, False),
    twix_coord_system="LPS",
    twix_inplane_rot_sign=-1.0,
    twix_use_fov_for_voxel_size=False,
    metadata=None,
):
    """Save one MPRAGE reconstruction as cropped-readout NIfTI files.

    The reconstruction image is expected to be in logical array order
    (readout_os, LIN/phase, PAR/partition). Readout oversampling is cropped
    only for the NIfTI export; the original .npy output remains unchanged.
    """
    from utils.nifti_export_twix import (
        apply_array_axis_flips,
        crop_readout_oversampling,
        make_nifti_affine_from_twix,
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

    outputs = [("mag", prepare_image_array(img_crop, part="mag"))]
    if save_phase:
        outputs.append(("phase", prepare_image_array(img_crop, part="phase")))

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
        sidecar["Units"] = "rad" if part == "phase" else "arbitrary"
        sidecar["ImageProcessing"] = (
            "angle(complex_image), after readout-oversampling crop"
            if part == "phase" else
            "abs(complex_image), after readout-oversampling crop"
        )
        save_nifti_with_json(arr, affine, nii_path, json_path, metadata=sidecar)


def _default_nifti_sub(tag_wave, res_x, res_y, res_z, Ry, Rz, file_tag):
    """Build a compact MPRAGE NIfTI subject/folder name."""
    res_mm = (res_x * 1e3, res_y * 1e3, res_z * 1e3)
    tag = f"MPRAGE_{tag_wave}_{res_mm[0]:g}x{res_mm[1]:g}x{res_mm[2]:g}mm_Ry{Ry}_Rz{Rz}"
    if file_tag:
        tag += f"_{file_tag}"
    return _sanitize_filename_component(tag)


def _build_mprage_nifti_metadata(tag_wave, file_tag, defs, geom, os_factor, Ry, Rz, ncalib, nacs, yflip, zflip):
    """Create MPRAGE-specific JSON sidecar metadata."""
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
        "LogicalReconAxisOrder": ["readout_physical_z", "LIN_outerPE_physical_y", "PAR_innerPE_physical_x"],
        "PhysicalMatrixXYZ": [int(v) for v in geom["Nxyz"]],
        "LogicalMatrixRO_LIN_PAR": [int(geom["Nro"]), int(geom["Nlin"]), int(geom["Npar"])],
        "PhysicalFovMXYZ": [float(v) for v in geom["FOVxyz"]],
        "LogicalFovMRO_LIN_PAR": [float(geom["FOVro"]), float(geom["FOVlin"]), float(geom["FOVpar"])],
    }

    optional_defs = {
        "TR": "RepetitionTime",
        "TE": "EchoTime",
        "TI": "InversionTime",
        "FlipAngle": "FlipAngle",
        "Name": "PulseqSequenceName",
        "OrientationMapping": "PulseqOrientationMapping",
    }
    for seq_key, meta_key in optional_defs.items():
        if seq_key in defs:
            value = defs.get(seq_key)
            try:
                if np.isscalar(value) and seq_key != "Name" and seq_key != "OrientationMapping":
                    value = float(value)
            except Exception:
                pass
            metadata[meta_key] = value

    return metadata


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

def load_or_generate_coil_sens(mprage_data_file, Ny, Nz, os_factor, out_folder, file_tag, Nacs=32, reuse_coil_calib=False):
    """Load cached Wcc/CSM or generate them from the integrated ACS refscan set."""
    wcc_file = _npy_output_path(out_folder + 'coil_compression_energy_' + file_tag)
    csm_file = _npy_output_path(out_folder + 'csm_full_' + file_tag)

    if reuse_coil_calib and os.path.isfile(wcc_file) and os.path.isfile(csm_file):
        print("Reusing existing coil compression matrix and coil sensitivity maps.")
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
    )


def generate_coil_sens(mprage_data_file, Ny, Nz, os_factor, out_folder, file_tag, Nacs=32):
    """Generate Wcc and ESPIRiT CSMs from the integrated sequence ACS refscan."""
    # assign sigpy operator to GPU
    print("CuPy version:", cp.__version__)
    print("GPU count:", cp.cuda.runtime.getDeviceCount())

    for i in range(cp.cuda.runtime.getDeviceCount()):
        props = cp.cuda.runtime.getDeviceProperties(i)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        print(i, name)

    device = sp.Device(0)   # first visible GPU
    print(device)

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

    # Low-res crop before GPU
    low_shape = (kspace_nowave_np.shape[0], Nx, 32, 32)
    kspace_low_np = sp.resize(kspace_nowave_np, low_shape).astype(np.complex64, copy=False)

    # Coil compression: 32 -> 12
    kspace_low_cc_np = apply_cc_coilfirst_np(kspace_low_np, Wcc)
    print("kspace_low_cc_np:", kspace_low_cc_np.shape)

    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    kspace_low_cc_sp = sp.to_device(kspace_low_cc_np, device)

    # Generate low-res coil sensitivity maps
    csm_low_cc = mr.app.EspiritCalib(
        kspace_low_cc_sp,
        calib_width=24,
        device=device,
        crop=0.8,
        show_pbar=True,
    ).run()

    csm_low_cc_np = sp.to_device(csm_low_cc, sp.Device(-1))
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

    # Normalize RSS across coils.
    rss = np.sqrt(np.sum(np.abs(csm_full_cc_np) ** 2, axis=0, keepdims=True))
    csm_full_cc_np /= np.maximum(rss, 1e-8)

    # save
    _save_npy(out_folder + 'coil_compression_energy_' + file_tag, Wcc, 'coil compression matrix')
    _save_npy(out_folder + 'csm_acs_' + file_tag, csm_low_cc_np, 'low-resolution ESPIRiT CSM')
    _save_npy(out_folder + 'csm_full_' + file_tag, csm_full_cc_np, 'full-resolution ESPIRiT CSM')

    plot_csm_magnitude_grid(csm_full_cc_np, z=csm_full_cc_np.shape[-1] // 2)
    mag_png = out_folder + f'csm_full_mag_' + file_tag + '.png'
    print(f"Saving CSM magnitude plot to: {mag_png}")
    plt.savefig(mag_png, dpi=150)
    plot_csm_phase_grid(csm_full_cc_np, z=csm_full_cc_np.shape[-1] // 2)
    phase_png = out_folder + f'csm_full_phase_' + file_tag + '.png'
    print(f"Saving CSM phase plot to: {phase_png}")
    plt.savefig(phase_png, dpi=150)
    plt.close('all')
    print(csm_full_cc_np.shape)

    return Wcc, csm_full_cc_np, Ncoil


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


def generate_calibrated_psf(mprage_data_file, mprage_seq_file, out_folder, Nx_os, Ny, Nz, file_tag,
                            yflip=-1, zflip=-1, Ncalib=72, Nacs=32,
                            slice_orientation='SAG', psf_plot=True):
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

    a_smooth = smooth_1d_nan(a_fit_all, window=9)
    b_smooth = smooth_1d_nan(b_fit_all, window=9)
    c_smooth = smooth_1d_nan(c_fit_all, window=9)

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
        plt.title('Integrated PSF calibration fit')
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
    parser.add_argument("--data-folder", default=None, help="Folder containing input .dat/.seq files.")
    parser.add_argument("--out-folder", default=None, help="Folder where output .npy/.png files are saved.")
    parser.add_argument("--mprage-data-file", default=None, help="Integrated Wave-MPRAGE + calibration TWIX .dat file.")
    parser.add_argument("--mprage-seq-file", default=None, help="Integrated Wave-MPRAGE + calibration Pulseq .seq file.")
    parser.add_argument("--calib-data-file", default=None, help="Deprecated compatibility alias. Integrated mode uses --mprage-data-file.")
    parser.add_argument("--calib-seq-file", default=None, help="Deprecated compatibility alias. Integrated mode uses --mprage-seq-file.")
    parser.add_argument("--file-tag", default=None, help="Suffix tag used in output filenames.")
    parser.add_argument("--tag-wave", choices=("wave", "nowave"), default=None,
                        help="Reconstruction type: 'wave' or 'nowave'.")
    parser.add_argument("--reuse-coil-calib", action="store_true",
                        help="Reuse existing coil_compression_energy_<tag>.npy and csm_full_<tag>.npy if present.")
    parser.add_argument("--yflip", type=int, default=None,
                        help="Sign convention for y wave PSF calibration. Default: -1.")
    parser.add_argument("--zflip", type=int, default=None,
                        help="Sign convention for z wave PSF calibration. Default: -1.")
    parser.add_argument("--save-nifti", action="store_true",
                        help="Also save the reconstructed image as NIfTI after center-cropping readout oversampling.")
    parser.add_argument("--save-nifti-phase", action="store_true",
                        help="When --save-nifti is used, also save phase in radians. Magnitude is always saved.")
    parser.add_argument("--nifti-out-folder", default=None,
                        help="Folder for NIfTI outputs. Default: <out-folder>/nifti/.")
    parser.add_argument("--nifti-sub", default=None,
                        help="Subject/folder name for NIfTI outputs. Default is generated from recon settings.")
    parser.add_argument("--nifti-suffix", default=None,
                        help="NIfTI filename suffix. Default: MPRAGE.")
    parser.add_argument("--nifti-axis-roles", default=None,
                        help="Comma-separated Twix roles for output axes. Default: readout,phase,slice.")
    parser.add_argument("--nifti-axis-flips", default=None,
                        help="Comma-separated booleans for physical array flips before NIfTI. Default: false,true,false.")
    parser.add_argument("--twix-coord-system", default=None, choices=("LPS", "RAS"),
                        help="Coordinate-system assumption for Twix Sag/Cor/Tra vectors. Default: LPS.")
    parser.add_argument("--twix-inplane-rot-sign", type=float, default=None,
                        help="Sign applied to Twix in-plane rotation. Default: -1.0.")
    parser.add_argument("--twix-use-fov-for-voxel-size", action="store_true",
                        help="Infer NIfTI voxel sizes from Twix FOV instead of reconstruction voxel size.")
    args, _ = parser.parse_known_args()
    return args


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


def _save_npy(path_without_ext, array, label):
    """Save a NumPy/PyTorch object and print the exact output path."""
    out_path = _npy_output_path(path_without_ext)
    print(f"Saving {label} to: {out_path}")
    np.save(out_path, array)
    return out_path


def _collect_runtime_config():
    """Collect runtime paths/tags from CLI args, existing globals, or prompts."""
    cli = _parse_cli_args()

    data_folder_value = _prompt_for_value(
        "data_folder",
        "Folder containing the integrated TWIX .dat and Pulseq .seq files",
        default=cli.data_folder,
    )
    data_folder_value = os.path.abspath(os.path.expanduser(os.path.expandvars(str(data_folder_value))))

    out_folder_value = _prompt_for_value(
        "out_folder",
        "Folder for output .npy/.png files",
        default=cli.out_folder,
    )
    out_folder_value = _normalize_folder(out_folder_value)

    mprage_data_default = cli.mprage_data_file
    if mprage_data_default is None and cli.calib_data_file is not None:
        mprage_data_default = cli.calib_data_file
    mprage_data_value = _prompt_for_value(
        "mprage_data_file",
        "Integrated Wave-MPRAGE + calibration TWIX .dat file",
        default=mprage_data_default,
    )

    mprage_seq_default = cli.mprage_seq_file
    if mprage_seq_default is None and cli.calib_seq_file is not None:
        mprage_seq_default = cli.calib_seq_file
    mprage_seq_value = _prompt_for_value(
        "mprage_seq_file",
        "Integrated Wave-MPRAGE + calibration Pulseq .seq file",
        default=mprage_seq_default,
    )

    file_tag_value = _prompt_for_value(
        "file_tag",
        "Output filename suffix/file tag",
        default=cli.file_tag,
        required=False,
    )
    file_tag_value = "" if file_tag_value is None else str(file_tag_value)

    tag_default = cli.tag_wave
    if tag_default is None and "tag_wave" in globals() and globals()["tag_wave"] not in (None, ""):
        tag_default = globals()["tag_wave"]
    if tag_default is None:
        tag_default = _prompt_for_value(
            "tag_wave",
            "Reconstruct wave data? Enter yes/wave or no/nowave",
            default=None,
        )
    tag_wave_value = _parse_wave_tag(tag_default)
    print(f"Using tag_wave: {tag_wave_value}")

    reuse_coil_calib_value = bool(cli.reuse_coil_calib or globals().get("reuse_coil_calib", False))

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
    nifti_axis_roles_value = _parse_axis_roles(axis_roles_source, default=("readout", "phase", "slice"))

    axis_flips_source = cli.nifti_axis_flips
    if axis_flips_source is None and "nifti_axis_flips" in globals() and globals()["nifti_axis_flips"] not in (None, ""):
        axis_flips_source = globals()["nifti_axis_flips"]
    nifti_axis_flips_value = _parse_bool_tuple(axis_flips_source, default=(False, True, False))

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

    twix_use_fov_for_voxel_size_value = bool(
        cli.twix_use_fov_for_voxel_size or globals().get("twix_use_fov_for_voxel_size", False)
    )

    mprage_data_value = _resolve_input_path(mprage_data_value, data_folder_value, "integrated MPRAGE data file")
    mprage_seq_value = _resolve_input_path(mprage_seq_value, data_folder_value, "integrated MPRAGE sequence file")

    print("Runtime configuration summary:")
    print(f"  data_folder:       {data_folder_value}")
    print(f"  out_folder:        {out_folder_value}")
    print(f"  file_tag:          {file_tag_value}")
    print(f"  reconstruction:    {tag_wave_value}")
    print(f"  reuse_coil_calib:  {reuse_coil_calib_value}")
    print(f"  yflip/zflip:       {yflip_value}/{zflip_value}")
    print(f"  save_nifti:        {save_nifti_value}")
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
        "yflip": yflip_value,
        "zflip": zflip_value,
        "save_nifti": save_nifti_value,
        "save_nifti_phase": save_nifti_phase_value,
        "nifti_out_folder": nifti_out_folder_value,
        "nifti_sub": nifti_sub_value,
        "nifti_suffix": nifti_suffix_value,
        "nifti_axis_roles": nifti_axis_roles_value,
        "nifti_axis_flips": nifti_axis_flips_value,
        "twix_coord_system": twix_coord_system_value,
        "twix_inplane_rot_sign": twix_inplane_rot_sign_value,
        "twix_use_fov_for_voxel_size": twix_use_fov_for_voxel_size_value,
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


def _parse_bool_tuple(value, default=(False, True, False)):
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
