# Reconstruction

## Entry point

```text
recon/recon_wave_mprage_from_twix_integrated_nifti.py
```

Run it from the repository root so that the relative `recon/utils/` imports are resolved consistently:

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py --help
```

## Inputs

The current script expects:

- one integrated Siemens TWIX `.dat` file
- the matching integrated Pulseq `.seq` file
- output folder
- reconstruction mode: `wave` or `nowave`

The TWIX containers are expected to be:

```text
image   -> MPRAGE k-space
refscan -> FLASH projection calibration and ACS
```

The integrated refscan SET convention is:

| SET | Purpose |
|---:|---|
| 0 | no-wave sine-projection calibration |
| 1 | sine-wave projection calibration |
| 2 | no-wave cosine-projection calibration |
| 3 | cosine-wave projection calibration |
| last / 4 | no-wave ACS for coil compression and ESPIRiT |

Calibration dimensions are read from `Calibration_Ncalib1` and `Calibration_Nacs` in the sequence definitions, with defaults of 72 and 32.

## Pipeline

1. Read geometry and acceleration definitions from the `.seq` file.
2. Load MPRAGE k-space from the TWIX `image` container.
3. Load the integrated ACS block from the final `refscan` SET.
4. Estimate a 32-to-12 coil-compression matrix on CPU.
5. Apply coil compression on CPU.
6. Estimate low-resolution ESPIRiT maps on the selected SigPy device.
7. Interpolate and normalize the sensitivity maps.
8. For wave data, fit the FLASH projection phase deviation and construct the calibrated wave PSF.
9. Run wave or no-wave CG-SENSE on CPU.
10. Save `.npy` arrays, diagnostic plots, and optional NIfTI outputs.

## CPU and GPU behavior

The current implementation does not move the full reconstruction to GPU.

| Operation | Current device |
|---|---|
| Coil-compression estimation | CPU / NumPy and SciPy |
| Coil-compression application | CPU / PyTorch tensor |
| ESPIRiT calibration | selectable SigPy CPU or GPU |
| Wave and no-wave CG-SENSE | CPU / PyTorch tensor |

### ESPIRiT selection

```text
--espirit-device auto   default; use a visible compatible GPU, otherwise CPU
--espirit-device cpu    always use SigPy CPU device
--espirit-device gpu    require a usable CuPy/CUDA device
--espirit-gpu-index N   select GPU index, default 0
```

`auto` catches missing CuPy, CUDA initialization failures, and an unavailable requested GPU index, then reports the reason and uses CPU. Explicit `gpu` mode raises an error for those conditions.

CuPy is therefore optional for CPU reconstruction. Install the `gpu` dependency group only on a CUDA 12 system where GPU-assisted ESPIRiT is desired.

## Examples

### Wave reconstruction, automatic device

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py \
  --data-folder /path/to/data \
  --out-folder /path/to/output \
  --mprage-data-file meas_integrated_wave_mprage.dat \
  --mprage-seq-file mprage_3d_flashcalib_wave.seq \
  --tag-wave wave \
  --file-tag test01 \
  --espirit-device auto
```

### CPU-only reconstruction

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py \
  --data-folder /path/to/data \
  --out-folder /path/to/output \
  --mprage-data-file meas_integrated_wave_mprage.dat \
  --mprage-seq-file mprage_3d_flashcalib_wave.seq \
  --tag-wave wave \
  --espirit-device cpu
```

### Require GPU 1

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py \
  --data-folder /path/to/data \
  --out-folder /path/to/output \
  --mprage-data-file meas_integrated_wave_mprage.dat \
  --mprage-seq-file mprage_3d_flashcalib_wave.seq \
  --tag-wave wave \
  --espirit-device gpu \
  --espirit-gpu-index 1
```

### Reuse cached coil calibration

```text
--reuse-coil-calib
```

This reuses `coil_compression_energy_<tag>.npy` and `csm_full_<tag>.npy` when both exist. Use cached files only when the acquisition geometry, coil configuration, ACS data, and reconstruction dimensions match.

## Sagittal logical-axis convention

The current integrated sequence uses:

```text
RO  = physical z = defs["Nz"]
LIN = physical y = defs["Ny"]
PAR = physical x = defs["Nx"]
```

The reconstructed array is stored as:

```text
axis 0 = oversampled readout / physical z
axis 1 = LIN / physical y
axis 2 = PAR / physical x
```

FOV and resolution are remapped in the same order.

## NIfTI export

Enable magnitude export with:

```text
--save-nifti
```

Also export phase in radians with:

```text
--save-nifti-phase
```

The helper reads orientation from the MPRAGE TWIX MeasYaps geometry, center-crops readout oversampling only for the NIfTI output, and writes `.nii.gz` plus JSON sidecars.

Important defaults:

```text
output folder:        <out_folder>/nifti/
axis roles:           readout,phase,slice
axis flips:           false,true,false
Twix coordinate mode: LPS
in-plane rotation:    sign -1.0
```

Relevant options include:

```text
--nifti-out-folder
--nifti-sub
--nifti-suffix
--nifti-axis-roles
--nifti-axis-flips
--twix-coord-system
--twix-inplane-rot-sign
--twix-use-fov-for-voxel-size
```

## Outputs

The output folder may contain:

- coil-compression matrix
- low- and full-resolution sensitivity maps
- CSM magnitude and phase plots
- coil-compressed MPRAGE k-space
- PSF phase-fit arrays and plots
- wave or no-wave CG-SENSE image as `.npy`
- optional magnitude and phase NIfTI files and JSON sidecars
