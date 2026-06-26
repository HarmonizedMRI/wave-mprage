import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy import io
import pypulseq as pp

import platform
import os

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

# Global plotting style: increase font sizes for all figures in this notebook
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
    # Check if data_folder, out_folder, mprage_data_file, mprage_seq_file, calib_data_file, calib_seq_file, file_tag exist as variable names. If not
    # To-Do: prompt user for data_folder that contains the twix .dat data
    # To-Do: prompt user for out_folder that saves the output .npy
    # To-Do: prompt user for mprage_data_file for the wave MPRAGE data
    # To-Do: prompt user for mprage_seq_file for the wave MPRAGE sequence
    # To-Do: prompt user for calib_data_file for the wave calibration data
    # To-Do: prompt user for calib_seq_file for the wave calibration sequence
    # To-Do: prompt user for file_tag for the special file name suffix
    # To-Do: prompt user for whether this is wave recon [yes for wave, no for nowave] so that [tag_wave = 'wave' or 'nowave']

    seq = pp.Sequence()
    seq.read(mprage_seq_file, remove_duplicates=False)
    defs = seq.definitions
    fov_def = defs.get('FOV', [0.224, 0.224, 0.224])
    Nx = defs.get('Nx', 256)
    Ny = defs.get('Ny', 256)
    Nz = defs.get('Nz', 192)
    os_factor = defs.get('ro_os', 4)
    res_x = fov_def[0] / Nx
    res_y = fov_def[1] / Ny
    res_z = fov_def[2] / Nz
    Ry = defs.get('R1', 2)
    Rz = defs.get('R2', 3)

    # generate coil compression energy
    print("Generate coil compression energy and coil sensitivity maps...")
    Wcc, csm_full_cc_np, Ncoil = generate_coil_sens(calib_data_file, Ny, Nz, os_factor, out_folder, file_tag)
    
    print(f'Importing data, Ry={Ry}, Rz={Rz}')
    Nx_os = Nx * os_factor
    img = load_img(mprage_data_file)
    kspace_echo = torch.zeros((Nx_os, Ny, Nz, Ncoil), dtype=torch.cfloat)
    kspace_echo[:, :img.shape[1], :img.shape[2], :, :] = img

    kspace_cc_echo = apply_cc_coillast_torch(kspace_echo, Wcc, x_chunk=8,)
    np.save(out_folder + 'kspace_' + tag_wave + '_cc_'+str(res_x)+'x'+str(res_y)+'x'+str(res_z)+'_Ry'+str(Ry)+'_Rz'+str(Rz)+'_' + file_tag, kspace_cc_echo)

    # Use ESPIRiT maps estimated above
    sens = torch.zeros((12, Nx_os, Ny, Nz), dtype=torch.complex64)
    sens[:, Nx_os//2 - Nx//2:Nx_os//2 + Nx//2] = torch.from_numpy(csm_full_cc_np).contiguous()

    # Sampling mask M
    mask_2d = torch.sum(torch.abs(kspace_cc_echo) ** 2, dim=(0, 3, 4)) > 0
    mask_2d = mask_2d.cpu().numpy().astype(np.float32)
    mask_t = torch.from_numpy(mask_2d).view(1, 1, *mask_2d.shape)  # broadcast to (ncoil, Nx_os, Ny, Nz)

    # Reconstruction
    if tag_wave == 'wave':  # reconstruct wave image
        print(f"Processing Wave Data...")
        # Data consistency term uses the measured wave k-space for one echo
        y_meas = kspace_cc_echo.permute(3,0,1,2)  # (ncoil, Nx_os, Ny, Nz)

        # generate calibrated psf
        print("Generated calibrated psf")
        psf_calib, psf_theory = generate_calibrated_psf(calib_data_file, calib_seq_file, out_folder, Nx_os, Ny, Nz, yflip = -1, zflip = -1)  # Use yflip, zflip = -1 for 'SAG', 1 for 'TRA'

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

        # To-Do: Print saving to filename 'xxx' (further development will provide option to save to nifti)
        np.save(out_folder + 'image_cg_wave_calib_' +str(res_x)+'x'+str(res_y)+'x'+str(res_z)+'_Ry'+str(Ry)+'_Rz'+str(Rz) + '_' + file_tag, img_pcg_wave)
    
    elif tag_wave == 'nowave':  # reconstruct nowave image 
        # Perform CG SENSE for no wave
        def E(x):
            """Forward operator: x (Nx, Ny) -> k-space coils (nc, Nx, Ny, Nz)."""
            img_coils = sens * x.unsqueeze(0)                      # S
            kspace = fft3call(img_coils, dim=(1,2,3))                           # F
            return kspace * mask_t                                 # M

        def EH(k):
            """Adjoint operator: k-space coils -> image."""
            img_coils = ifft3call(k * mask_t, dim=(1,2,3))                      # F^H
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

                beta = rr_new / (rr)
                p = r + beta * p
                rr = rr_new

            print(f"CG reached max_iter={n_iter}, final rel-res={torch.sqrt(rr / bb).item():.2e}")
            return x
        
        print(f"Processing No-Wave Data...")
        # Data consistency term uses the measured wave k-space for one echo
        y_meas = kspace_cc_echo.permute(3,0,1,2)  # (ncoil, Nx_os, Ny, Nz)

        img_cg_nowave = cg_sense(y_meas, n_iter=50, tol=1e-6)
        # To-Do: Print saving to filename 'xxx' (further development will provide option to save to nifti)
        np.save(out_folder + 'image_cg_nowave_' +str(res_x)+'x'+str(res_y)+'x'+str(res_z)+'_Ry'+str(Ry)+'_Rz'+str(Rz)+'_' + file_tag, img_cg_nowave)


def generate_coil_sens(calib_data_file, Ny, Nz, os_factor, out_folder, file_tag):
    # assign sigpy operator to GPU
    print("CuPy version:", cp.__version__)
    print("GPU count:", cp.cuda.runtime.getDeviceCount())

    for i in range(cp.cuda.runtime.getDeviceCount()):
        props = cp.cuda.runtime.getDeviceProperties(i)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        print(i, name)

    device = sp.Device(0)   # first visible GPU
    print(device)

    # import acs
    kspace_nowave_acs = load_ref(calib_data_file)[:,:,:,4]
    Nx_os, Ny_acs, Nz_acs, Ncoil = kspace_nowave_acs.shape
    Nx = Nx_os // os_factor

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
    target_img_shape = (Nx_os, Ny, Nz)  # (256, 256, 192)
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
    np.save(out_folder + 'coil_compression_energy_' + file_tag, Wcc)
    np.save(out_folder + 'csm_full_' + file_tag, csm_full_cc_np)
    plot_csm_magnitude_grid(csm_full_cc_np, z=csm_full_cc_np.shape[-1] // 2)
    plt.savefig(out_folder + f'csm_full_mag_' + file_tag + '.png', dpi=150)
    plot_csm_phase_grid(csm_full_cc_np, z=csm_full_cc_np.shape[-1] // 2)
    plt.savefig(out_folder + f'csm_full_phase_' + file_tag + '.png', dpi=150)
    print(csm_full_cc_np.shape)

    return Wcc, csm_full_cc_np, Ncoil
    

def generate_theoretical_wave_trajectory(seq, Nx_os, Ncalib, slice_orientation):
    defs = seq.definitions
    # Build wave PSF from the sequence trajectory
    ktraj_adc, ktraj, t_exc, t_ref, t_adc = seq.calculate_kspace()
    
    k_adc = np.asarray(ktraj_adc, dtype=np.float64).reshape(3, -1, Nx_os)
    ky_adc_sin = k_adc[1, Ncalib*1:Ncalib*2]
    kz_adc_cos = k_adc[0, Ncalib*3:Ncalib*4]

    delta_ky = ky_adc_sin[Ncalib//2]  # should have the k space center zero
    delta_kz = kz_adc_cos[Ncalib//2]  # should have the k space center zero

    fov_def = defs.get('FOV', [0.224, 0.224, 0.224])
    if isinstance(fov_def, (list, tuple, np.ndarray)):
        if slice_orientation == 'SAG':
            fov_y = float(fov_def[1]) if len(fov_def) > 1 else float(fov_def[0])
            fov_z = float(fov_def[0]) if len(fov_def) > 1 else float(fov_def[0])  # SAG
        elif slice_orientation == 'TRA':
            fov_y = float(fov_def[1]) if len(fov_def) > 1 else float(fov_def[0])
            fov_z = float(fov_def[2]) if len(fov_def) > 1 else float(fov_def[0])
        else:
            raise ValueError(f'Unsupported slice orientation {slice_orientation}')
    else:
        fov_y = float(fov_def)
        fov_z = float(fov_def)
    delta_ky_idx = delta_ky * fov_y
    delta_kz_idx = delta_kz * fov_z

    return delta_ky_idx, delta_kz_idx


def fit_wave_psf_deviation_from_projection(kspace_nowave_echo, kspace_wave_echo, delta_ky_idx, delta_kz_idx, out_folder, file_tag, wave_dim = 'y', yflip = -1, zflip = -1):
    if wave_dim is None:
        raise ValueError('Wave dimension is not specified. Please specify "x", "y" or "z".')

    Nx_os, Ny_meas, Nz_meas, Ncoil = kspace_nowave_echo.shape

    # Convert both to image domain first, then to hybrid domain (FFT along readout only)
    img_nowave = ifft3call(kspace_nowave_echo)
    img_wave = ifft3call(kspace_wave_echo)

    hyb_nowave = fftc_dim(img_nowave, dim=0)
    hyb_wave = fftc_dim(img_wave, dim=0)

    # Average cross-power phase over coils to estimate PSF phase term
    cross = hyb_wave * torch.conj(hyb_nowave) / (1e-8 + hyb_nowave * torch.conj(hyb_nowave))             # (ncoil, Nx_os, Ny)
    psf_real = torch.exp(1j * torch.angle(cross.mean(dim=-1))) # (Nx_os, Ny), unit magnitude

    # generate theoretical psf
    y_norm = (np.arange(Ny_meas) - (Ny_meas / 2.0)) / Ny_meas
    z_norm = (np.arange(Nz_meas) - (Nz_meas / 2.0)) / Nz_meas
    if wave_dim == 'y':
        z_norm = np.array([0.0])
    elif wave_dim == 'z':
        y_norm = np.array([0.0])
    psf_np = np.exp(-1j * yflip * 2.0 * np.pi * delta_ky_idx[:, None] * y_norm[None, :]).astype(np.complex64)
    psf_np = psf_np[..., np.newaxis] * np.exp(-1j * zflip * 2.0 * np.pi * delta_kz_idx[:, None, None] * z_norm[None, None, :]).astype(np.complex64)
    psf_theory = torch.from_numpy(psf_np)

    # generate and fit the psf deviation
    psf_diff = torch.angle(torch.conj(psf_theory) * psf_real)

    result = fit_wrapped_phase_planes(
        psf_diff=psf_diff,
        hyb_nowave=hyb_nowave,
        y_norm=y_norm,
        z_norm=z_norm,
        mask_mode="combined",
        mag_abs_floor=0.0,              # or a very weak floor like 1e-14

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

    a_fit_all = result["a_fit_all"]
    b_fit_all = result["b_fit_all"]
    c_fit_all = result["c_fit_all"]
    mask = result["mask"]

    if wave_dim == 'y':
        tag = 'projy_72kyline'
    elif wave_dim == 'z':
        tag = 'projz_72kzline'
    else:
        tag = ''
    np.save(out_folder + 'a_fit_all_' + str(tag) + '_' + file_tag, a_fit_all)
    np.save(out_folder + 'b_fit_all_' + str(tag) + '_' + file_tag, b_fit_all)
    np.save(out_folder + 'c_fit_all_' + str(tag) + '_' + file_tag, c_fit_all)

    return a_fit_all, b_fit_all, c_fit_all


def generate_calibrated_psf(calib_data_file, calib_seq_file, out_folder, Nx_os, Ny, Nz, file_tag = None, yflip = -1, zflip = -1, Ncalib = 72, slice_orientation = 'SAG', psf_plot = True):
    kspace_calib_data = load_img(calib_data_file)
    kspace_nowave_sin = kspace_calib_data[:,:,:1,0]
    kspace_wave_sin = kspace_calib_data[:,:,:1,1]
    kspace_nowave_cos = kspace_calib_data[:,:1,:,2]
    kspace_wave_cos = kspace_calib_data[:,:1,:,3]

    # To-Do: check if kspace_calib_data.shape[0] matches Nx_os, if not, raise Error that the oversampled readout dimension doesn't match

    # generate theoretical wave trajectory
    seq = pp.Sequence()
    seq.read(calib_seq_file, remove_duplicates=False)
    delta_ky_idx, delta_kz_idx = generate_theoretical_wave_trajectory(seq, Nx_os, Ncalib, slice_orientation)

    a_fit_sin, b_fit_sin, c_fit_sin = fit_wave_psf_deviation_from_projection(kspace_nowave_sin, kspace_wave_sin, delta_ky_idx, delta_kz_idx, out_folder, file_tag, wave_dim = 'y', yflip = yflip, zflip = zflip)
    a_fit_cos, b_fit_cos, c_fit_cos = fit_wave_psf_deviation_from_projection(kspace_nowave_cos, kspace_wave_cos, delta_ky_idx, delta_kz_idx, out_folder, file_tag, wave_dim = 'z', yflip = yflip, zflip = zflip)
    
    a_fit_merged, b_fit_merged, c_fit_merged = a_fit_sin, b_fit_cos, c_fit_sin + c_fit_cos
    a_smooth = smooth_1d_nan(a_fit_merged, window=9)
    b_smooth = smooth_1d_nan(b_fit_merged, window=9)
    c_smooth = smooth_1d_nan(c_fit_merged, window=9)

    if psf_plot:
        plt.figure(figsize = (6,4))
        plt.plot(a_smooth, label="a(t)")
        plt.plot(b_smooth, label="b(t)")
        plt.plot(c_smooth, label="c(t)")
        plt.axvline(a_smooth.shape[-1]//2, linestyle = '--', color = 'k')
        plt.axhline(0, linestyle = '--', color = 'k')
        plt.legend()
        plt.ylim([-3,3])
        plt.xlim([0, Nx_os])
        plt.title(f'PSF calibration fit')
        plt.savefig(out_folder + 'psf_calib_fit_' + file_tag + '.png')
        plt.close('all')

    a_fit = a_smooth[0]
    b_fit = b_smooth[0]
    c_fit = c_smooth[0]

    # generate theoretical psf snapped to [Ny, Nz] grid
    y_norm = (np.arange(Ny) - (Ny / 2.0)) / Ny
    z_norm = (np.arange(Nz) - (Nz / 2.0)) / Nz
    psf_np = np.exp(-1j * yflip * 2.0 * np.pi * delta_ky_idx[:, None] * y_norm[None, :]).astype(np.complex64)
    psf_np = psf_np[..., np.newaxis] * np.exp(-1j * zflip * 2.0 * np.pi * delta_kz_idx[:, None, None] * z_norm[None, None, :]).astype(np.complex64)
    psf_theory = torch.from_numpy(psf_np)

    # generate calibrated psf
    psf_diff_pred_new = torch.zeros_like(torch.angle(psf_theory))

    y_norm_tensor = torch.from_numpy(y_norm)
    z_norm_tensor = torch.from_numpy(z_norm)
    Y_grid, Z_grid = torch.meshgrid(y_norm_tensor, z_norm_tensor, indexing='ij')

    y_flat = Y_grid.flatten()
    z_flat = Z_grid.flatten()

    for kx_loc in range(Nx_os):
        ones = torch.ones_like(y_flat)
        A_full = torch.stack([y_flat, z_flat, ones], dim=1)
        coefficients = torch.Tensor((a_fit[kx_loc], b_fit[kx_loc], c_fit[kx_loc]))
        coefficients = coefficients.to(dtype=A_full.dtype)

        psf_diff_pred_flat = A_full @ coefficients
        psf_diff_pred_new[kx_loc] = psf_diff_pred_flat.view(psf_theory[kx_loc].shape)

    psf_diff_pred_new = torch.nan_to_num(psf_diff_pred_new.clone(), nan=0.0)
    psf_calib = psf_theory * torch.exp(1j*psf_diff_pred_new)
    return psf_calib, psf_theory


if __name__ == "__main__":
    main()