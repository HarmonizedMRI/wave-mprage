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

## PSF coefficient curves blow up

Inspect the saved `a(kx)`, `b(kx)`, `c(kx)`, `psf_real`, and `psf_theory` diagnostics. When the direct coefficient fit is trustworthy only within a limited readout region and becomes unstable outside it, rerun with the sine-plus-line coefficient model:

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py \
  --twix /path/to/data/scan.dat \
  --seq /path/to/data/scan.seq \
  --out /path/to/output \
  --wave-mode wave \
  --psf-coefficient-processing sine-line \
  --psf-fit-kx-min 200 \
  --psf-fit-kx-max 512
```

Replace the example bounds with a high-fidelity interval identified from your calibration diagnostics. The interval is `[kx_min, kx_max)`. Both bounds are mandatory in `sine-line` mode.

The sine-plus-line model is intended as a controlled substitution for the default `smooth` processing. It fits `A*sin(w*kx+phi) + C1*kx + C2` over the trusted interval, evaluates the model across the full readout, and does not apply the normal smoothing afterward. It cannot recover calibration information when the selected interval itself is aliased or contaminated.

## Reconstruction failed after ESPIRiT completed

When the coil-compression matrix and full-resolution ESPIRiT sensitivity maps were saved before a later failure, rerun the same acquisition with:

```text
--reuse-coil-calib
```

Use the same `--out` directory, `--file-tag`, and `--espirit-calib-mode`. The script uses:

```text
coil_compression_energy_<tag>.npy
csm_full_<tag>.npy                 # 3d
csm_full_slice2d_<tag>.npy         # slice2d
```

The supported option is `--reuse-coil-calib` rather than `--reuse-exist-calib`. Reuse these files only when the TWIX measurement, receive-coil selection, geometry, ACS dimensions, oversampling, compression settings, reconstruction dimensions, calibration mode, and intended crop value are unchanged. A newly supplied `--espirit-crop` value is not reapplied to an existing cached map. If any relevant setting differs, rerun without the reuse option.

## Residual wave aliasing or contaminated PSF calibration

Some failures originate in the scan prescription and cannot be fully corrected during reconstruction:

- Turn off neck-coil elements to reduce neck and shoulder signal entering the projection calibration.
- Prescribe the FOV box to cover the entire signal-producing volume relevant to the scan. Signal originating outside the encoded FOV can wrap into the acquisition, and wave-induced aliasing from that unencoded volume cannot be fully resolved.
- Confirm that the matching `.seq` file and sagittal protocol were used.

When out-of-FOV signal contaminates only part of the fitted coefficient curves, the optional `sine-line` model may help if a clearly high-fidelity interval remains. It is not a replacement for an adequate FOV or appropriate receive-coil selection.

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

The default `--espirit-calib-mode 3d` performs one joint 3D SigPy calibration. On CPU this can remain slow even though the input has already been coil-compressed and spatially reduced.

When native 3D CPU ESPIRiT is the bottleneck, try the optional CPU-parallel backend:

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py \
  --twix /path/to/data/scan.dat \
  --seq /path/to/data/scan.seq \
  --out /path/to/output \
  --wave-mode auto \
  --espirit-device cpu \
  --espirit-calib-mode slice2d \
  --espirit-crop 0.8
```

The implementation removes readout oversampling before transforming logical RO to hybrid space. Do not apply the slice-wise method directly to the raw fourfold-oversampled RO array.

Native 3D remains the reference estimator. Compare representative 3D and slice2d results before adopting slice2d routinely. Reuse validated mode-specific calibration files with `--reuse-coil-calib` when appropriate.

## Choose the slice2d CPU-worker count on Linux

Inspect the machine topology with:

```bash
lscpu
```

Important fields include:

```text
CPU(s)               logical CPUs visible to the system
Thread(s) per core   hardware threads per physical core
Core(s) per socket   physical cores per socket
Socket(s)            CPU sockets
```

Also check the CPUs available to the current process or job:

```bash
nproc
grep Cpus_allowed_list /proc/self/status
```

On a scheduler-managed system, use the CPUs allocated to the job rather than the full node count. `lscpu` reports topology; it does not guarantee that every listed CPU is available to your process or currently idle.

When `--espirit-cpu-workers` is omitted, joblib selects the available physical-core count and the implementation caps it by the number of logical-RO slices. This automatic value is a reasonable first choice on a dedicated machine. On a shared node or when memory bandwidth is limiting, set a lower value explicitly and benchmark, for example:

```text
--espirit-cpu-workers 8
--espirit-cpu-workers 16
--espirit-cpu-workers 24
```

Do not assume that using every logical CPU is fastest. Each worker performs an SVD and uses memory bandwidth; excessive workers can increase process overhead and RAM pressure. Increase the worker count only while runtime continues to improve meaningfully.

## Slice2d is slow or uses too much memory

Try the following:

- reduce `--espirit-cpu-workers`;
- confirm the runtime summary reports `ESPIRiT mode: slice2d` and the expected worker count;
- avoid running several reconstructions concurrently on the same node;
- check available memory and CPU load before launching the calibration;
- compare a small set of worker counts rather than immediately using all logical CPUs.

Worker count changes execution only and should not materially change the maps. Crop and calibration mode do change the estimator or support and require separate validation.

## Slice2d rejects GPU mode

`slice2d` is intentionally CPU-only. This combination is invalid:

```text
--espirit-calib-mode slice2d --espirit-device gpu
```

Use one of:

```text
--espirit-calib-mode slice2d --espirit-device cpu
--espirit-calib-mode slice2d --espirit-device auto
--espirit-calib-mode 3d --espirit-device gpu
```

## ESPIRiT crop removes low-SNR anatomy

`--espirit-crop` remains active in both `3d` and `slice2d` modes. Higher values create a stricter support mask; lower values retain broader support. In slice2d mode the threshold is applied independently to each logical-RO plane.

Testing with the current Wave-MPRAGE implementation found `0.8–0.9` to be a reasonable practical range:

- use `0.8` as the first choice when low-SNR anterior anatomy is being removed;
- use `0.9` when the broader support from `0.8` includes too much unreliable background;
- inspect both CSM magnitude and phase plots and the final reconstruction.

Recompute the maps after changing crop. Do not use `--reuse-coil-calib` for that comparison, because a cached CSM has already had its crop mask applied. If important anatomy remains absent at `0.8`, the limitation may come from the calibration subspace or local SNR rather than crop alone.

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

Delete or disable the cached files when geometry, readout oversampling, ACS size, receive coils, compression settings, ESPIRiT mode, or the intended crop value has changed:

```text
coil_compression_energy_<tag>.npy
csm_full_<tag>.npy                 # 3d
csm_full_slice2d_<tag>.npy         # slice2d
```

Then rerun without `--reuse-coil-calib`.

## Memory errors

The full CG-SENSE reconstruction is CPU-based and can require substantial RAM. Reduce concurrent jobs, close other large processes, or run on a node with more memory. For `slice2d` ESPIRiT, also reduce `--espirit-cpu-workers` because multiple worker processes run calibrations concurrently. The GPU memory requirement is limited mainly to the reduced native 3D ESPIRiT problem.

## NIfTI is mirrored or rotated

Compare against a trusted DICOM/NIfTI reference and adjust:

```text
--nifti-axis-flips
--twix-inplane-rot-sign
--twix-coord-system
```

The current MPRAGE defaults are axis roles `phase,readout,slice`, axis flips `true,false,false`, rotation sign `-1`, and `LPS`.

## MATLAB cannot find Pulseq

Provide either the Pulseq repository root or its `matlab/` directory when prompted. The path helper must be able to find the `+mr` package.

## PNS/CNS or forbidden-frequency checks are unavailable

Those checks are optional. Supply Safe PNS Prediction, a scanner `.asc` file, and `forbiddenFreqCheck.m` when needed, or skip the checks. The `.seq` file is written before optional post-write checks.
