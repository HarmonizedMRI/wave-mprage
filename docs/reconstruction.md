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
7. Estimate low-resolution ESPIRiT maps on the selected SigPy device.
8. Interpolate and normalize the sensitivity maps.
9. For wave data, fit the FLASH projection phase deviation and construct the calibrated wave PSF.
10. Run wave or no-wave CG-SENSE on CPU.
11. Save `.npy` arrays, diagnostic plots, and optional NIfTI outputs.

## CPU and GPU behavior

The current implementation does not move the full reconstruction to GPU.

| Operation | Current device |
|---|---|
| Coil-compression estimation | CPU / NumPy and SciPy |
| Coil-compression application | CPU / PyTorch tensor |
| ESPIRiT calibration | selectable SigPy CPU or GPU |
| Wave and no-wave CG-SENSE | CPU / PyTorch tensor |

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

CuPy is therefore optional for CPU reconstruction. Install the `gpu` dependency group only on a CUDA 12 system where GPU-assisted ESPIRiT is desired.

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
output folder:        <out>/nifti/
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
