# Wave MPRAGE

Pulseq-based 3D Wave-MPRAGE sequence generation with an integrated FLASH wave-calibration module and Python reconstruction utilities.

The repository has two intentionally separate parts:

- MATLAB generates the integrated MPRAGE + FLASH calibration sequence.
- Python reconstructs Wave-MPRAGE or no-wave MPRAGE data from the resulting Siemens TWIX measurement.

`pyproject.toml` is used only to define the Python reconstruction environment. This repository is not configured as an installable Python package and does not need to be published to PyPI.

## Repository layout

```text
.
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── docs/
│   ├── sequence.md
│   ├── reconstruction.md
│   └── troubleshooting.md
├── seq/
│   ├── mprage_3d_wave_with_flash_calibration.m
│   └── utils/
├── recon/
│   ├── recon_wave_mprage_from_twix_integrated_nifti.py
│   └── utils/
└── external/
```

The recommended entry points are:

```text
seq/mprage_3d_wave_with_flash_calibration.m
recon/recon_wave_mprage_from_twix_integrated_nifti.py
```

## Requirements

### Sequence generation

- MATLAB
- Pulseq MATLAB toolbox

Optional scanner-safety checks can use Safe PNS Prediction, a scanner `.asc` file, and an existing `forbiddenFreqCheck.m` helper.

### Reconstruction

- Python 3.11
- CPU reconstruction environment defined in `pyproject.toml`
- Optional NVIDIA GPU and CUDA 12-compatible CuPy for faster ESPIRiT calibration

The current execution model is:

| Step | Device |
|---|---|
| Coil-compression matrix estimation and application | CPU |
| ESPIRiT sensitivity-map calibration | Native 3D on CPU/GPU, or optional CPU-parallel `slice2d` |
| Wave/no-wave CG-SENSE | CPU |

A GPU is optional. The default `--espirit-calib-mode 3d` uses the native joint 3D SigPy calibration; `--espirit-device auto` uses a compatible GPU when CuPy can access one and otherwise falls back to CPU. The optional `--espirit-calib-mode slice2d` backend is CPU-only and parallelizes independent hybrid-space 2D calibrations across logical readout positions. For acquisitions with more than 32 receive channels, consider CPU calibration when the available CPU memory is more suitable than the GPU resources or when GPU calibration is unstable.

## Clone

```bash
git clone --recurse-submodules https://github.com/HarmonizedMRI/wave-mprage.git
cd wave-mprage
```

## Install the reconstruction environment

### Recommended: uv

Install `uv`, then create the locked CPU-capable environment:

```bash
uv sync --locked
```

Run commands inside the environment with `uv run`:

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py --help
```

To include CUDA 12 CuPy for GPU-assisted ESPIRiT:

```bash
uv sync --locked --group gpu
```

The committed `uv.lock` records the exact resolved Python dependency versions. The host NVIDIA driver and GPU remain system requirements and are not contained in the lockfile.

### Alternative: pip

`pip` 25.1 or newer can install the same standardized dependency groups directly from `pyproject.toml`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group recon
```

For CUDA 12 CuPy:

```bash
python -m pip install --group gpu
```

The `pip` route resolves compatible versions at installation time. Use `uv sync --locked` when exact lockfile reproduction is important.

## Generate the integrated sequence

Open MATLAB and run:

```matlab
cd seq
mprage_3d_wave_with_flash_calibration
```

On the first run, enter the requested paths. The helper stores machine-specific path settings in a local JSON file. Leaving the output path blank uses MATLAB's current folder.

Generated sequence files are written under:

```text
generated_seq_v141/
generated_seq_v151/
```

See [Sequence generation](docs/sequence.md) for acquisition order, TWIX routing, calibration SET layout, path handling, and validation.

Before scanning, turn off the neck-coil elements and prescribe an FOV that covers the complete signal-producing volume. Out-of-FOV neck or shoulder signal can contaminate the projection calibration, and wave-induced aliasing from anatomy outside the encoded FOV cannot be fully resolved.

## Reconstruct an integrated acquisition

The integrated measurement is expected to contain:

```text
image   -> MPRAGE k-space
refscan -> four FLASH PSF-calibration sets plus the ACS set
```

Review all arguments:

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py --help
```

The preferred input interface uses direct paths:

```text
--twix       integrated Siemens TWIX .dat file
--seq        matching integrated Pulseq .seq file
--out        reconstruction output directory
--wave-mode  auto, wave, or nowave; default auto
```

`--wave-mode auto` inspects the MPRAGE imaging trajectory after excluding the appended FLASH calibration and ACS trajectory. It selects `wave` when both wave axes are active and `nowave` when both are inactive. One-axis wave acquisitions are rejected.

Example reconstruction with automatic wave-mode detection, automatic ESPIRiT device selection, and NIfTI export:

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py \
  --twix /path/to/data/meas_integrated_wave_mprage.dat \
  --seq /path/to/data/mprage_3d_flashcalib_wave.seq \
  --out /path/to/output \
  --wave-mode auto \
  --file-tag test01 \
  --espirit-device auto \
  --save-nifti \
  --save-nifti-phase
```

Add `--save-bart-inputs` to export the calibrated PSF, coil-compressed
k-space, sensitivity maps, and integrated ACS as BART CFL pairs. The companion
`recon/run_bart_wave_recon.sh` script runs BART `ecalib` and `wave`; see
[Reconstruction](docs/reconstruction.md#bart-wave-caipi-input-export) for the
exact dimensions and command.

Force CPU ESPIRiT while explicitly requiring wave reconstruction:

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py \
  --twix /path/to/data/meas_integrated_wave_mprage.dat \
  --seq /path/to/data/mprage_3d_flashcalib_wave.seq \
  --out /path/to/output \
  --wave-mode wave \
  --espirit-device cpu
```

Use `--espirit-device gpu --espirit-gpu-index 0` to require a specific GPU. The script raises a clear error instead of silently falling back when GPU mode is explicitly requested.

### ESPIRiT calibration mode, crop, and CPU workers

The calibration backend is selected independently of the Wave-MPRAGE forward model:

| Argument | Behavior |
|---|---|
| `--espirit-calib-mode 3d` | Native joint 3D SigPy ESPIRiT; default and reference mode; CPU or GPU |
| `--espirit-calib-mode slice2d` | CPU-parallel 2D ESPIRiT over logical-RO hybrid-space slices |

The `slice2d` backend receives the logical low-resolution ACS after the existing readout-oversampling removal. It inverse-transforms logical RO and calibrates each joint LIN-PAR plane independently, preserving both accelerated phase-encoding dimensions within every 2D calibration.

A practical CPU-parallel example is:

```bash
uv run python recon/recon_wave_mprage_from_twix_integrated_nifti.py \
  --twix /path/to/data/meas_integrated_wave_mprage.dat \
  --seq /path/to/data/mprage_3d_flashcalib_wave.seq \
  --out /path/to/output \
  --wave-mode auto \
  --espirit-device cpu \
  --espirit-calib-mode slice2d \
  --espirit-crop 0.8
```

`--espirit-crop` is valid for both calibration modes. Testing with the current Wave-MPRAGE implementation found `0.8–0.9` to be a reasonable practical range: `0.8` retains somewhat broader low-SNR support, while `0.9` applies a stricter support mask. Inspect the saved CSM magnitude and phase plots for each acquisition.

When `--espirit-cpu-workers` is omitted, `slice2d` automatically uses the available physical-core count, capped by the number of logical-RO slices. Set it explicitly when sharing a machine, limiting memory use, or benchmarking, for example:

```text
--espirit-cpu-workers 16
```

See [Troubleshooting](docs/troubleshooting.md) for Linux CPU inspection with `lscpu`, worker-count guidance, and crop-related checks.

The following compatibility aliases remain accepted:

| Preferred argument | Compatibility aliases |
|---|---|
| `--twix` | `--mprage-data-file` |
| `--seq` | `--mprage-seq-file` |
| `--out` | `--out-folder` |
| `--wave-mode` | `--mode`, `--tag-wave` |

The old shared `--data-folder` argument is no longer needed because `--twix` and `--seq` accept direct absolute or relative paths.

See [Reconstruction](docs/reconstruction.md) for the pipeline, ESPIRiT calibration modes, crop and CPU-worker behavior, input assumptions, complete command-line interface, output files, and NIfTI conventions. The reconstruction guide also documents the optional `smooth` and `sine-line` PSF coefficient-processing modes; troubleshooting guidance explains CPU sizing with `lscpu`, crop/support checks, sine-line use, and safe reuse of an existing mode-specific coil calibration.

## Documentation

- [Sequence generation](docs/sequence.md)
- [Reconstruction](docs/reconstruction.md)
- [Troubleshooting](docs/troubleshooting.md)

## License

MIT License. See `LICENSE`.
