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

Inspect the saved `a(kx)`, `b(kx)`, `c(kx)`, `psf_real`, and `psf_theory` diagnostics. The reconstruction continues to use `smooth` by default. To opt into automatic sine-plus-line processing, omit both manual bounds:

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py \
  --twix /path/to/data/scan.dat \
  --seq /path/to/data/scan.seq \
  --out /path/to/output \
  --wave-mode wave \
  --psf-coefficient-processing sine-line
```

Automatic mode normally uses a near-global fit after removing readout edge guards. If center-referenced slope and variance checks detect sustained coefficient corruption, it selects a stable center-containing interval instead. Rejected coefficient or projection-quality samples inside the chosen interval do not enter the fit. Inspect `psf_integrated_calib_fit_<tag>.png` and `psf_sine_line_fit_<tag>.json` for the selected interval, algorithm version, rejection thresholds, fit residuals, conditioning, and extrapolation-stability checks.

For a reproducible manual override, add both bounds:

```text
--psf-fit-kx-min 200
--psf-fit-kx-max 512
```

Replace these example values with a high-fidelity interval identified from the calibration diagnostics. The interval is `[kx_min, kx_max)`. Providing only one bound is an error.

The sine-plus-line model is an explicit alternative to the default `smooth` processing. It fits `A*sin(w*kx+phi) + C1*kx + C2` over the selected interval and evaluates the model across the full readout. Automatic mode smooths quality-masked samples before fitting; manual mode fits raw finite samples. The command fails rather than silently reverting to `smooth` when range selection or fit validation is inadequate. It cannot recover calibration information when the accepted support itself is aliased or contaminated.

## Reconstruction failed after ESPIRiT completed

When the coil-compression matrix and full-resolution ESPIRiT sensitivity maps were saved before a later failure, rerun the same acquisition with:

```text
--reuse-coil-calib
```

Use the same `--out` directory, `--file-tag`, and `--espirit-calib-mode`. The script uses:

```text
coil_compression_energy_<tag>.npy
csm_full_<tag>.npy                         # native 3d
csm_full_slice2d_sagmask_<tag>.npy         # SAG slice2d with RO support guard
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

For sagittal MPRAGE, the slice2d path also uses a superior-inferior whole-RO-plane support guard. Inspect its diagnostic output before accepting the maps, particularly at the top of the scalp and the inferior jaw/neck.

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

## SAG slice2d RO mask removes superior or inferior anatomy

This section applies only to sagittal Wave-MPRAGE reconstructed with:

```text
--espirit-calib-mode slice2d
```

For the validated SAG geometry, logical RO is physical `z`, corresponding to the superior-inferior direction. The slice2d support guard examines each complete logical-RO plane and sets low-signal planes to zero before the per-plane ESPIRiT calibration. It is intended to reject noise-only planes at the two ends of the superior-inferior FOV.

The guard and `--espirit-crop` act at different stages:

```text
SAG RO support guard
    removes or retains complete superior-inferior planes

--espirit-crop
    controls the in-plane ESPIRiT eigenvalue support within retained planes
```

Therefore, changing `--espirit-crop` cannot restore a complete RO plane that the SAG support guard has already masked.

### Identify which mask caused the clipping

Inspect the saved diagnostic products:

```text
espirit_slice2d_sag_ro_support_slice2d_sagmask_<tag>.png
espirit_slice2d_sag_ro_support_slice2d_sagmask_<tag>.npz
```

The PNG shows the normalized hybrid-space RO-plane RMS, the support threshold, and the planes rejected by the guard.

- If the missing superior or inferior plane is marked as rejected, adjust the SAG support parameters.
- If the plane is retained but pixels within that plane are absent from the CSM, adjust `--espirit-crop`.
- If the plane and its in-plane CSM support are both present but anatomy remains absent in the reconstruction, investigate calibration SNR, receive-coil coverage, and the reconstruction operator rather than either mask.

### Recommended tuning order

Keep the ESPIRiT crop fixed at `0.8` while tuning the whole-plane guard. Change one group of parameters at a time in this order:

1. Increase `slice_support_padding`.
2. Lower `slice_support_noise_multiplier`.
3. Lower `slice_support_relative_floor`.
4. Reduce `slice_support_noise_fraction` when only a few truly empty edge planes exist.
5. Adjust `--espirit-crop` only after the superior-inferior plane support is correct.

The support parameters are currently named arguments in the `estimate_espirit_maps(...)` call in the MPRAGE reconstruction script.

### Top-of-head or inferior anatomy is clipped

Small superior head-cap planes and inferior jaw/neck planes can have low whole-plane RMS because the signal occupies only a small fraction of the LIN-PAR plane. Start by increasing the safety padding:

```python
slice_support_padding=8
```

For approximately 1 mm logical-RO resolution, this preserves about 8 mm beyond each detected support boundary. If necessary, try:

```python
slice_support_padding=10
```

or:

```python
slice_support_padding=12
```

Padding is the safest first adjustment because it restores an anatomical margin without lowering the support threshold throughout the full FOV.

If padding alone is insufficient, reduce the required signal above the estimated noise floor:

```python
slice_support_noise_multiplier=1.5
```

A practical testing range is:

```text
1.5 -> 2.0 -> 2.5
```

Lower values preserve weaker planes. Higher values reject edge noise more aggressively.

Next, reduce the threshold relative to the strongest RO plane:

```python
slice_support_relative_floor=1e-5
```

If weak superior or inferior anatomy remains clipped, test:

```python
slice_support_relative_floor=0.0
```

With a value of zero, the noise-floor criterion alone determines the threshold.

When the acquired FOV contains only a small number of genuinely empty RO planes, reduce the fraction used to estimate the noise floor:

```python
slice_support_noise_fraction=0.05
```

or, more conservatively:

```python
slice_support_noise_fraction=0.03
```

A value that is too large can include weak anatomical planes in the noise estimate and make the threshold overly aggressive.

### Recommended conservative SAG settings

A reasonable first adjustment for superior or inferior clipping is:

```python
slice_support_noise_fraction=0.05
slice_support_noise_multiplier=1.5
slice_support_relative_floor=1e-5
slice_support_padding=10
```

Keep:

```text
--espirit-crop 0.8
```

during this comparison.

If anatomy remains clipped:

```python
slice_support_noise_fraction=0.03
slice_support_noise_multiplier=1.5
slice_support_relative_floor=0.0
slice_support_padding=12
```

If lateral or edge noise-only planes return, increase only the noise multiplier first:

```python
slice_support_noise_multiplier=2.0
```

### Edge noise remains after relaxing the mask

If noise-only superior or inferior planes return:

- increase `slice_support_noise_multiplier` gradually;
- reduce excessive padding;
- keep `slice_support_relative_floor` small unless the noise-floor estimate is unreliable;
- confirm the diagnostic curve shows a clear flat edge-noise plateau.

Do not raise `--espirit-crop` specifically to reject complete noise-only RO planes. Crop acts within each retained plane and may remove valid low-SNR anatomy without reliably solving a whole-plane failure.

### Recompute after every support-mask change

The support decision is already embedded in the saved CSM. Do not use:

```text
--reuse-coil-calib
```

when comparing support-mask settings. Delete the existing SAG slice2d CSM cache or use a new `--file-tag`:

```text
csm_acs_slice2d_sagmask_<tag>.npy
csm_full_slice2d_sagmask_<tag>.npy
```

The coil-compression matrix may remain reusable when all coil-compression inputs are unchanged, but the current reconstruction recomputes the calibration products together when cache reuse is disabled.

### Limitation of the current score

The guard uses one RMS score per complete LIN-PAR plane. This is intentionally simple, but it can undervalue valid superior slices where the head occupies only a small area. If extensive tuning is required across subjects, consider changing the score to a robust high-percentile or top-fraction signal statistic rather than continuing to lower global thresholds.

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
csm_full_<tag>.npy                         # native 3d
csm_full_slice2d_sagmask_<tag>.npy         # SAG slice2d with RO support guard
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
