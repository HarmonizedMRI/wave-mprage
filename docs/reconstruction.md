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

The script requires:

- one integrated Siemens TWIX `.dat` file
- the matching integrated Pulseq `.seq` file
- an output directory
- a wave-mode selection: `auto`, `wave`, or `nowave`

The preferred command-line arguments are:

```text
--twix PATH       integrated Wave-MPRAGE + calibration TWIX .dat file
--seq PATH        matching integrated Wave-MPRAGE + calibration .seq file
--out PATH        reconstruction output directory
--wave-mode MODE  auto, wave, or nowave; default auto
```

The paths passed to `--twix` and `--seq` may be absolute or relative to the current working directory. A separate shared data-folder argument is not required.

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

## Wave-mode selection

`--wave-mode` accepts:

```text
auto    inspect the MPRAGE imaging trajectory and select wave or nowave
wave    require a two-axis wave imaging trajectory
nowave  require a no-wave imaging trajectory
```

`auto` is the default. The mode detector excludes the appended integrated FLASH calibration and ACS trajectory before inspecting the MPRAGE imaging trajectory.

The supported imaging configurations are:

| Detected imaging trajectory | Result |
|---|---|
| sine and cosine wave axes both active | `wave` |
| sine and cosine wave axes both inactive | `nowave` |
| sine only | rejected |
| cosine only | rejected |

When the user explicitly selects `wave` or `nowave`, the script verifies that the requested mode agrees with the detected trajectory. A mismatch raises an error rather than running the wrong forward model.

## Compatibility aliases

The concise direct-path arguments are preferred, but the earlier MPRAGE-specific names remain accepted:

| Preferred argument | Compatibility aliases | Parsed destination |
|---|---|---|
| `--twix` | `--mprage-data-file` | `twix` |
| `--seq` | `--mprage-seq-file` | `seq` |
| `--out` | `--out-folder` | `out` |
| `--wave-mode` | `--mode`, `--tag-wave` | `mode` |

For example, these are equivalent:

```bash
--wave-mode auto
--mode auto
```

The old `--data-folder` argument is not part of the direct-path interface. Supply the complete relative or absolute path to each input instead.

## Pipeline

1. Read geometry and acceleration definitions from the `.seq` file.
2. Inspect the MPRAGE imaging trajectory and resolve the requested wave mode.
3. Load MPRAGE k-space from the TWIX `image` container.
4. Load the integrated ACS block from the final `refscan` SET.
5. Estimate a 32-to-12 coil-compression matrix on CPU.
6. Apply coil compression on CPU.
7. Estimate low-resolution ESPIRiT maps with the selected `3d` or `slice2d` calibration backend.
8. Interpolate and normalize the sensitivity maps.
9. For wave data, fit the FLASH projection phase deviation and construct the calibrated wave PSF.
10. Run wave or no-wave CG-SENSE on CPU.
11. Save `.npy` arrays, diagnostic plots, and optional NIfTI outputs.

## BART Wave-CAIPI input export

Add `--save-bart-inputs` to a wave reconstruction to write BART-compatible
`.hdr`/`.cfl` pairs under `<out>/bart_inputs` (or `bart_inputs_<tag>` when
`--file-tag` is set).

| Basename | BART shape | Contents |
|---|---|---|
| `wave_kspace` | `(Nx_os, Ny, Nz, Ncc, 1)` | Coil-compressed acquired k-space |
| `psf` | `(Nx_os, Ny, Nz, 1, 1)` | Calibrated wave PSF |
| `coil_sens` | `(Nx, Ny, Nz, Ncc, 1)` | Sensitivity maps from this reconstruction |
| `kspace_calib` | `(Nx, Ny, Nz, Ncc)` | Coil-compressed, centered integrated ACS |

The companion script runs BART ESPIRiT calibration, Wave-CAIPI reconstruction,
and NIfTI conversion:

```bash
recon/bart/run_wave_recon.sh \
  --bart-input /path/to/output/bart_inputs \
  --bart-output /path/to/output/bart_reconstruction \
  --maps-source bart \
  --twix /path/to/meas_integrated_wave_mprage.dat \
  --seq /path/to/matching_wave_mprage.seq \
  --save-phase \
  --ecalib-options -c 0.8 --end-ecalib-options \
  --wave-options -w -r 0.001 -f -i 100 -t 1e-6 --end-wave-options
```

Options inside the `--ecalib-options` and `--wave-options` sections are passed
unchanged to the corresponding BART commands. The helper prints each complete
command before running it. If `--nifti-output` is omitted, converted files are
written to `BART_OUTPUT/nifti`; pass the option only to override that location.
Conversion uses `python` from the active Conda environment or virtual
environment. Set `PYTHON_BIN` to select a different interpreter. To skip
`ecalib`, use existing maps explicitly:

```bash
recon/bart/run_wave_recon.sh \
  --bart-input /path/to/output/bart_inputs \
  --bart-output /path/to/output/bart_reconstruction \
  --maps-source existing \
  --existing-maps /path/to/output/bart_inputs/coil_sens \
  --twix /path/to/meas_integrated_wave_mprage.dat \
  --seq /path/to/matching_wave_mprage.seq \
  --nifti-output /path/to/custom/nifti \
  --wave-options -l -r 0.002 -b 8 -f -i 100 --end-wave-options
```

The final converter uses matching TWIX geometry and Pulseq metadata, restores
the k-space norm removed internally by `bart wave`, and does not crop BART's
already de-oversampled readout. Use `--help` for the complete interface.

## PSF coefficient processing

For wave reconstruction, the projection calibration first estimates the readout-dependent phase-plane coefficients `a(kx)`, `b(kx)`, and `c(kx)`. The final coefficient-processing method is selected with:

```text
--psf-coefficient-processing smooth      default
--psf-coefficient-processing sine-line
```

### `smooth`

The default path applies NaN-aware one-dimensional smoothing to the directly estimated coefficients. This preserves the established reconstruction behavior and should be used for routine data when the fitted coefficient curves remain stable across readout.

### `sine-line`

The optional path fits each coefficient within a user-selected high-fidelity readout interval to:

```text
A * sin(w * kx + phi) + C1 * kx + C2
```

The fitted model is then evaluated over the complete oversampled readout. In this mode, the sine-plus-line model **replaces** the normal smoothing step; the fitted curves are not smoothed again.

Use this option when the direct PSF coefficient fit is reliable over a central or otherwise trusted region but blows up, becomes discontinuous, or is contaminated outside that region. Both range arguments are required:

```text
--psf-fit-kx-min INTEGER
--psf-fit-kx-max INTEGER
```

The selected interval follows the half-open Python convention `[kx_min, kx_max)` and must satisfy:

```text
0 <= kx_min < kx_max <= Nx_os
```

Example:

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py \
  --twix /path/to/data/meas_integrated_wave_mprage.dat \
  --seq /path/to/data/mprage_3d_flashcalib_wave.seq \
  --out /path/to/output \
  --wave-mode wave \
  --psf-coefficient-processing sine-line \
  --psf-fit-kx-min 200 \
  --psf-fit-kx-max 512
```

Choose the high-fidelity interval from the saved coefficient and PSF diagnostic plots. It should exclude visibly corrupted readout regions while retaining enough finite samples and oscillatory structure for a stable fit. The script reports an error when `sine-line` is selected without both bounds.

## CPU and GPU behavior

The current implementation does not move the full reconstruction to GPU.

| Operation | Current device |
|---|---|
| Coil-compression estimation | CPU / NumPy and SciPy |
| Coil-compression application | CPU / PyTorch tensor |
| Native `3d` ESPIRiT calibration | selectable SigPy CPU or GPU |
| Parallel `slice2d` ESPIRiT calibration | CPU process workers |
| Wave and no-wave CG-SENSE | CPU / PyTorch tensor |

### ESPIRiT calibration mode

```text
--espirit-calib-mode 3d       default; native joint 3D SigPy ESPIRiT
--espirit-calib-mode slice2d  CPU-parallel hybrid-space 2D ESPIRiT
```

`3d` remains the reference mode. It estimates one joint 3D calibration and can use either CPU or GPU.

`slice2d` is an optional CPU-only acceleration path. The reconstruction first performs its existing readout-oversampling removal and coil compression, yielding logical k-space ordered as `(coil, RO, LIN, PAR)`. It then inverse-transforms logical RO and runs independent 2D ESPIRiT calibrations over the joint LIN-PAR plane. This preserves calibration coupling across both accelerated phase-encoding dimensions while removing coupling only along the fully sampled readout direction.

Because `slice2d` estimates logical-RO positions independently, it is not mathematically identical to joint 3D ESPIRiT. Validate representative datasets by comparing CSM support, magnitude and phase continuity along logical RO, low-SNR anterior anatomy, final reconstruction differences, runtime, and peak memory.

### ESPIRiT crop threshold

```text
--espirit-crop FLOAT  eigenvalue support threshold; default 0.8
```

The crop threshold is passed directly to SigPy in both `3d` and `slice2d` modes. Higher values apply a stricter support mask and set more low-eigenvalue locations to zero; lower values retain broader support. In `slice2d`, the crop is applied independently within each logical-RO plane.

Testing with the current Wave-MPRAGE implementation found `0.8–0.9` to be a reasonable practical range:

- start with `0.8` when preserving low-SNR anatomy and broader map support is important;
- use `0.9` when a somewhat stricter, cleaner support mask is preferred;
- inspect the saved CSM magnitude and phase plots rather than selecting the value from background suppression alone.

Changing `--espirit-crop` requires recomputing sensitivity maps. When `--reuse-coil-calib` loads an existing CSM cache, the new crop value is not reapplied.

### Slice2d CPU workers

```text
--espirit-cpu-workers N  process workers used only by slice2d
```

When this argument is omitted, joblib selects the available physical-core count and the implementation caps it by the number of logical-RO slices. Each worker calibrates one `(coil, LIN, PAR)` plane, and native BLAS threads are limited to one per worker to avoid nested oversubscription.

Automatic worker selection is a useful first choice on a dedicated system. Set an explicit value when sharing a node, following a scheduler allocation, controlling RAM, or benchmarking. More workers are not always faster because the independent SVDs compete for memory bandwidth and create process overhead. See the troubleshooting guide for `lscpu`, `nproc`, CPU-affinity checks, and worker-count tuning.

### ESPIRiT device selection

```text
--espirit-device auto   default; use a visible compatible GPU, otherwise CPU
--espirit-device cpu    always use SigPy CPU device
--espirit-device gpu    require a usable CuPy/CUDA device
--espirit-gpu-index N   select GPU index, default 0
```

This `auto` setting is independent of `--wave-mode auto`:

- `--wave-mode auto` selects the reconstruction forward model from the sequence trajectory.
- `--espirit-device auto` selects CPU or GPU for ESPIRiT calibration.

ESPIRiT device auto-selection catches missing CuPy, CUDA initialization failures, and an unavailable requested GPU index, then reports the reason and uses CPU. Explicit `gpu` mode raises an error for those conditions.

Device selection applies directly to the native `3d` backend. The `slice2d` backend always runs on CPU: `--espirit-device cpu` and `--espirit-device auto` are accepted, while `--espirit-device gpu` with `--espirit-calib-mode slice2d` is rejected.

For acquisitions with more than 32 receive channels, consider CPU calibration when CPU memory and runtime are more suitable than the available GPU resources, or when GPU ESPIRiT is unstable. The ESPIRiT input is coil-compressed, but high-channel-count acquisitions still increase TWIX loading and coil-compression preparation demands.

CuPy is therefore optional for CPU reconstruction. Install the `gpu` dependency group only on a CUDA 12 system where GPU-assisted native 3D ESPIRiT is desired.

## Examples

### Automatic wave-mode and ESPIRiT device selection

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py \
  --twix /path/to/data/meas_integrated_wave_mprage.dat \
  --seq /path/to/data/mprage_3d_flashcalib_wave.seq \
  --out /path/to/output \
  --wave-mode auto \
  --file-tag test01 \
  --espirit-device auto
```

### Explicit wave reconstruction on CPU

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py \
  --twix /path/to/data/meas_integrated_wave_mprage.dat \
  --seq /path/to/data/mprage_3d_flashcalib_wave.seq \
  --out /path/to/output \
  --wave-mode wave \
  --espirit-device cpu
```

### CPU-parallel slice2d ESPIRiT

Let the implementation select the available physical-core count automatically:

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py \
  --twix /path/to/data/meas_integrated_wave_mprage.dat \
  --seq /path/to/data/mprage_3d_flashcalib_wave.seq \
  --out /path/to/output \
  --wave-mode wave \
  --espirit-device cpu \
  --espirit-calib-mode slice2d \
  --espirit-crop 0.8
```

Limit the run to a selected number of CPU workers:

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py \
  --twix /path/to/data/meas_integrated_wave_mprage.dat \
  --seq /path/to/data/mprage_3d_flashcalib_wave.seq \
  --out /path/to/output \
  --wave-mode wave \
  --espirit-device cpu \
  --espirit-calib-mode slice2d \
  --espirit-crop 0.9 \
  --espirit-cpu-workers 16
```

### Explicit no-wave reconstruction

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py \
  --twix /path/to/data/meas_integrated_nowave_mprage.dat \
  --seq /path/to/data/mprage_3d_flashcalib_nowave.seq \
  --out /path/to/output \
  --wave-mode nowave \
  --espirit-device auto
```

### Require GPU 1

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py \
  --twix /path/to/data/meas_integrated_wave_mprage.dat \
  --seq /path/to/data/mprage_3d_flashcalib_wave.seq \
  --out /path/to/output \
  --wave-mode auto \
  --espirit-device gpu \
  --espirit-gpu-index 1
```

### Compatibility aliases

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py \
  --mprage-data-file /path/to/data/meas_integrated_wave_mprage.dat \
  --mprage-seq-file /path/to/data/mprage_3d_flashcalib_wave.seq \
  --out-folder /path/to/output \
  --tag-wave wave
```

### Reuse cached coil calibration

```text
--reuse-coil-calib
```

This reuses the coil-compression matrix and the CSM cache for the selected ESPIRiT calibration mode when both exist. It is useful when ESPIRiT completed successfully but reconstruction failed later: rerun with the same output directory, `--file-tag`, and `--espirit-calib-mode`, then add `--reuse-coil-calib`.

For `--file-tag test01`, the full-resolution map names are:

```text
3d:       csm_full_test01.npy
slice2d:  csm_full_slice2d_test01.npy
```

The mode-specific names prevent a `slice2d` run from silently reusing a 3D CSM, or vice versa. The coil-compression matrix remains shared because calibration mode does not change coil compression. Use cached files only when the TWIX measurement, acquisition geometry, coil configuration, ACS data, compression settings, reconstruction dimensions, ESPIRiT mode, and intended crop setting match. A different `--espirit-crop` value is not applied to a reused map.

The supported option name is `--reuse-coil-calib`.

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
output folder:        <out>/nifti/
axis roles:           phase,readout,slice
axis flips:           true,false,false
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
