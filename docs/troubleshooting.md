# Troubleshooting

## Required argument is missing

The reconstruction now uses a strict direct-path command-line interface. The following arguments are required:

```text
--twix
--seq
--out
```

For example:

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py \
  --twix /path/to/data/scan.dat \
  --seq /path/to/data/scan.seq \
  --out /path/to/output \
  --wave-mode auto
```

The old `--data-folder` argument is not needed. Pass a complete relative or absolute path to `--twix` and `--seq`.

## An older argument name is used

The following compatibility aliases remain valid:

```text
--mprage-data-file  -> --twix
--mprage-seq-file   -> --seq
--out-folder        -> --out
--mode              -> --wave-mode
--tag-wave          -> --wave-mode
```

Prefer the concise names in new scripts and documentation.

## Wave-mode auto-detection disagrees with the requested mode

When `--wave-mode wave` or `--wave-mode nowave` is supplied, the script compares the request with the MPRAGE imaging trajectory. The integrated FLASH calibration and ACS tail is excluded from this detection.

A mismatch usually means one of the following:

- the `.seq` file does not match the TWIX measurement
- `wave` was requested for a no-wave sequence
- `nowave` was requested for a wave sequence
- the sequence trajectory is malformed or uses an unsupported configuration

Use the matching `.seq` file or use `--wave-mode auto` to let the script select the mode.

## One-axis Wave-MPRAGE is rejected

The current reconstruction supports only:

```text
both sine and cosine wave axes active -> wave
both sine and cosine wave axes inactive -> nowave
```

Sine-only and cosine-only MPRAGE imaging trajectories are rejected because the reconstruction does not provide a validated one-axis wave path. Regenerate the sequence with both wave axes enabled or both disabled.

## `pip install --group` is not recognized

Dependency groups require pip 25.1 or newer:

```bash
python -m pip install --upgrade "pip>=25.1"
```

Then run:

```bash
python -m pip install --group recon
```

## `uv sync --locked` reports an outdated lockfile

Make sure `pyproject.toml` and `uv.lock` came from the same repository revision. To intentionally update dependencies, run `uv lock` and review the resulting lockfile change before committing it.

## CuPy is not installed

This is valid for a CPU installation. Use:

```text
--espirit-device auto
```

or:

```text
--espirit-device cpu
```

Do not install the `gpu` dependency group unless the machine has a compatible NVIDIA driver and CUDA 12 environment.

## ESPIRiT automatic device selection falls back to CPU

This section refers to `--espirit-device auto`, not `--wave-mode auto`.

The script prints the reason. Common causes are:

- CuPy is not installed
- the NVIDIA driver is unavailable
- CUDA initialization failed
- no GPU is visible in the current job/container
- `--espirit-gpu-index` is outside the visible device range

This fallback affects only ESPIRiT calibration. Coil compression and CG-SENSE already run on CPU.

## Explicit GPU mode fails

`--espirit-device gpu` intentionally does not fall back. Check:

```bash
python - <<'PY'
import cupy as cp
print(cp.__version__)
print(cp.cuda.runtime.getDeviceCount())
for i in range(cp.cuda.runtime.getDeviceCount()):
    print(i, cp.cuda.runtime.getDeviceProperties(i)["name"])
PY
```

Use the CUDA 12 dependency group only when the installed driver supports it:

```bash
uv sync --locked --group gpu
```

## CPU ESPIRiT is slow

CPU fallback prioritizes portability rather than speed. The input is already coil-compressed and spatially reduced before ESPIRiT, but a CPU run can still take substantially longer than GPU calibration. Reuse validated cached calibration files with `--reuse-coil-calib` when appropriate.

## CUDA, CuPy, and driver mismatch

Typical symptoms include CUDA library-load errors, driver-version errors, or CuPy initialization failures. `uv.lock` controls Python packages but cannot install or lock the host NVIDIA driver. Confirm the driver on the execution machine and avoid installing multiple CuPy variants in the same environment.

## Image/refscan coil-count mismatch

The integrated TWIX `image` and `refscan` containers must use the same receive-coil configuration. Regenerate calibration data or choose a matching measurement if the counts differ.

## Integrated refscan layout error

The script requires at least five SETs:

```text
0 no-wave sine projection
1 sine-wave projection
2 no-wave cosine projection
3 cosine-wave projection
4/last ACS
```

It also checks that the refscan PE dimensions can contain `Calibration_Ncalib1` and `Calibration_Nacs`.

## Missing or incorrect sequence definitions

Use the matching integrated `.seq` file. The reconstruction relies on definitions including:

```text
Nx, Ny, Nz, FOV
ReadoutOversamplingFactor
MPRAGE_PE2_R
MPRAGE_PE1_R
Calibration_Ncalib1
Calibration_Nacs
OrientationMapping
```

Regenerate the sequence after adding or changing those definitions.

## Cached coil calibration has the wrong dimensions

Delete or disable the cached files when geometry, readout oversampling, ACS size, receive coils, or compression settings have changed:

```text
coil_compression_energy_<tag>.npy
csm_full_<tag>.npy
```

Then rerun without `--reuse-coil-calib`.

## Memory errors

The full CG-SENSE reconstruction is CPU-based and can require substantial RAM. Reduce concurrent jobs, close other large processes, or run on a node with more memory. The GPU memory requirement is limited mainly to the reduced ESPIRiT problem.

## NIfTI is mirrored or rotated

Compare against a trusted DICOM/NIfTI reference and adjust:

```text
--nifti-axis-flips
--twix-inplane-rot-sign
--twix-coord-system
```

The current defaults are `false,true,false`, rotation sign `-1`, and `LPS`.

## MATLAB cannot find Pulseq

Provide either the Pulseq repository root or its `matlab/` directory when prompted. The path helper must be able to find the `+mr` package.

## PNS/CNS or forbidden-frequency checks are unavailable

Those checks are optional. Supply Safe PNS Prediction, a scanner `.asc` file, and `forbiddenFreqCheck.m` when needed, or skip the checks. The `.seq` file is written before optional post-write checks.
