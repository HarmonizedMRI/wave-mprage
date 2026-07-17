# Wave MPRAGE

Pulseq-based 3D Wave MPRAGE sequence generation with integrated FLASH wave calibration and Python reconstruction utilities.

The MPRAGE sequence is built based on Maxim's Pulseq MPRAGE demo:
`matlab/demoSeq/writeMPRAGE.m` from the Pulseq MATLAB toolbox.

## Repository layout

```text
.
├── README.md
├── .gitignore
├── seq/
│   ├── mprage_3d_wave_with_flash_calibration.m
│   └── utils/
│       ├── configurePathSettings.m
│       ├── defineCosineWaveGradient.m
│       ├── defineSineWaveGradient.m
│       └── ...
└── recon/
    ├── recon_wave_mprage_from_twix_integrated_nifti.py
    └── utils/
        ├── nifti_export_twix.py
        └── ...
```

The repository may also retain earlier standalone MPRAGE and FLASH calibration scripts for development or comparison. The recommended sequence generator is:

```text
seq/mprage_3d_wave_with_flash_calibration.m
```

## Integrated MPRAGE + FLASH sequence

### Acquisition order

The integrated sequence contains two acquisitions in this order:

1. MPRAGE image acquisition
2. FLASH calibration acquisition, including dummy scans and per-part settling

The MPRAGE acquisition is placed first so that the continuous FLASH calibration train does not alter the longitudinal magnetization immediately before the first MPRAGE inversion block.

### Shared settings

The MPRAGE and FLASH calibration sections share the following settings:

- flip angle `alpha`
- readout duration `ro_dur`
- readout oversampling `ro_os`
- readout spoiling `ro_spoil`
- RF-spoiling increment `rfSpoilingInc`
- RF duration `rfLen`
- sagittal axis definition and axis order
- FOV and matrix size
- wave amplitude `gwave_max`
- wave slew limit `swave_max`
- number of wave cycles `Ncycles`
- scanner and Pulseq system limits

The integrated script currently supports:

```matlab
slOrientation = 'SAG';
ax.d1 = 'z';  % readout
ax.d2 = 'x';  % inner PE / PAR
ax.d3 = 'y';  % outer PE / LIN
```

The MPRAGE-specific flags default to:

```matlab
isUseWave_cos = true;
isUseWave_sin = true;
```

The FLASH calibration always generates its required no-wave, sine-wave, and cosine-wave acquisitions from the same shared wave-event library.

### TWIX routing

The combined measurement stores MPRAGE and FLASH calibration data in different TWIX containers:

| Acquisition | `REF` | `IMA` | `SET` | TWIX destination |
|---|---:|---:|---:|---|
| MPRAGE image | `false` | `false` | `0` | `image` |
| FLASH calibration | `true` | `false` | `0`–`4` | `refscan` |

MPRAGE does not acquire a separate ACS block. Its acquired k-space is stored only in `image`.

Every acquired FLASH calibration ADC is marked as a reference scan and is stored in `refscan`.

### FLASH calibration SET layout

The calibration uses compact local `LIN` and `PAR` indices inside each `SET`:

| SET | Acquisition | Local size |
|---:|---|---|
| 0 | no-wave, ky-wide / kz-narrow | `Ncalib1 × Ncalib2` |
| 1 | sine-wave, ky-wide / kz-narrow | `Ncalib1 × Ncalib2` |
| 2 | no-wave, kz-wide / ky-narrow | `Ncalib2 × Ncalib1` |
| 3 | cosine-wave, kz-wide / ky-narrow | `Ncalib2 × Ncalib1` |
| 4 | no-wave ACS | `Nacs × Nacs` |

With the default settings:

```matlab
Ncalib1 = 72;
Ncalib2 = 1;
Nacs    = 32;
```

the expected logical refscan extent is:

```text
LIN × PAR × SET = 72 × 72 × 5
```

Unacquired positions remain zero. The ACS data occupies local indices:

```text
LIN = 0:31
PAR = 0:31
SET = 4
```

### Validation performed by the script

Before writing the sequence, the script checks that:

- MPRAGE has no separate ACS acquisition
- MPRAGE ADCs are routed exclusively to `image`
- all five FLASH calibration sets are routed exclusively to `refscan`
- MPRAGE uses `SET=0`, `REF=false`, and `IMA=false`
- calibration uses `REF=true` and `IMA=false`
- calibration `SET`, `PAR`, and `LIN` order matches the requested acquisition table
- ACS occupies the expected local block in `SET=4`
- MPRAGE PE sampling matches the requested acceleration pattern
- combined `SET/PAR/LIN/REF/IMA` label evolution is correct
- sequence timing passes `seq.checkTiming`

The sequence is written before the optional PNS/CNS and forbidden-frequency checks.

## Path settings

### Local JSON configuration

Path settings are managed by:

```text
seq/utils/configurePathSettings.m
```

The following settings are saved locally:

- `pulseq_path`
- `safe_pns_prediction_path`
- `out_path`
- `system_asc_file`

The settings are written beside the main MATLAB script as:

```text
mprage_flash_path_settings.json
```

On the first run, the script prompts for all path settings and saves them.

On later runs, the script asks whether to use the saved settings. When saved settings are reused, it also asks whether any path should be changed. One, multiple, or all path entries can be selected for editing.

For optional settings:

```text
-
```

clears the saved path.

For the target output path, pressing Enter without entering a path uses MATLAB's current folder:

```matlab
pwd
```

Existing MATLAB workspace values are shown as defaults when path settings are configured.

Long saved paths are printed separately from `input()`. The interactive prompts use short lines such as `Path:`, `File:`, and `Choice:` to avoid MATLAB Command Window wrapping and cursor misalignment.

### Generated sequence folders

Generated files are written under the selected output root:

```text
generated_seq_v141/
generated_seq_v151/
```

The behavior is controlled by:

```matlab
write_v141_format
```

- `true`: write both legacy Pulseq v1.4.1 and current-format sequence files
- `false`: write only the current-format sequence file

When `write_v141_format=true`, output is organized as:

```text
<out_path>/
├── generated_seq_v141/
│   └── <sequence_name>_v141.seq
└── generated_seq_v151/
    └── <sequence_name>.seq
```

### Git exclusions

The following local or generated content should be ignored by Git:

```gitignore
mprage_flash_path_settings.json
generated_seq_v141/
generated_seq_v151/
```

Merge these entries into the repository `.gitignore` if it already contains other rules.

## Sequence utilities

All local functions used by the original MPRAGE and FLASH calibration scripts were moved into `seq/utils/`.

Shared waveform helpers include:

- `defineCosineWaveGradient.m`
- `defineSineWaveGradient.m`
- `makeFixedDurationPreRamp.m`
- `makeExtendedTrapezoidAndWaveform.m`

MPRAGE scheduling helpers include fixed-ETL planning, segmented block construction, center-slot enforcement, PE sampling, and label validation utilities.

The repository intentionally does not provide a replacement for:

```text
forbiddenFreqCheck.m
```

The integrated script expects the existing helper to be available under `seq/utils/` or elsewhere on the MATLAB path.

## Reconstruction

### Recommended integrated reconstruction script

The recommended reconstruction entry point for the integrated MPRAGE + FLASH calibration acquisition is:

```text
recon/recon_wave_mprage_from_twix_integrated_nifti.py
```

This script is designed for a single integrated Siemens TWIX measurement and its matching integrated Pulseq `.seq` file. The integrated measurement contains:

```text
image   -> MPRAGE k-space
refscan -> FLASH calibration and ACS k-space
```

The reconstruction workflow includes:

- loading Wave MPRAGE or no-wave MPRAGE data from the `image` container
- loading integrated FLASH calibration and ACS data from the `refscan` container
- estimating coil-compression weights from the integrated ACS refscan set
- estimating ESPIRiT coil sensitivity maps
- generating a calibrated wave PSF from the integrated FLASH calibration refscan sets
- reconstructing wave data with wave CG-SENSE
- reconstructing no-wave data with standard CG-SENSE
- saving intermediate coil-compressed k-space, coil maps, PSF diagnostics, and reconstructed images as `.npy` and `.png` outputs
- optionally exporting the final image to NIfTI using the orientation stored in the input MPRAGE TWIX `.dat` file

The older `recon_wave_mprage_from_twix.py` script may be retained for compatibility or development history, but the integrated NIfTI-capable script should be used for the current integrated acquisition.

### Integrated refscan layout used by reconstruction

The integrated reconstruction assumes the following `refscan` `SET` layout:

| SET | Purpose |
|---:|---|
| 0 | no-wave sine-projection calibration |
| 1 | sine-wave projection calibration |
| 2 | no-wave cosine-projection calibration |
| 3 | cosine-wave projection calibration |
| 4 / last | no-wave ACS block for coil compression and ESPIRiT |

The default calibration sizes are:

```text
Ncalib = 72
Nacs   = 32
```

Ncalib and Nacs are read from the Pulseq sequence definitions. The integrated sequence should save these values using seq.setDefinition, for example:
  seq.setDefinition('Ncalib', Ncalib1);
  seq.setDefinition('Nacs', Nacs);
  
If reconstruction raises an error that Ncalib or Nacs cannot be found, update the MATLAB sequence script to write these values into the .seq definitions, then regenerate the .seq file.

### Hard-coded sagittal logical-axis convention

The current integrated Wave-MPRAGE sequence is sagittal and uses this logical axis order during sequence generation:

```matlab
ax.d1 = 'z';  % readout
ax.d2 = 'x';  % inner PE / PAR
ax.d3 = 'y';  % outer PE / LIN
```

Because this `ax` structure is not saved in the `.seq` definitions, the reconstruction script hard-codes the corresponding remapping from physical sequence definitions to logical reconstruction dimensions:

```text
logical readout Nro = defs["Nz"]
logical phase   Ny  = defs["Ny"]
logical PAR     Nz  = defs["Nx"]
```

The FOV and voxel-size calculations are remapped in the same order:

```text
logical readout FOV = FOVz
logical phase FOV   = FOVy
logical PAR FOV     = FOVx
```

For example, if the sequence definitions contain:

```text
defs["Nx"] = 192
defs["Ny"] = 256
defs["Nz"] = 256
ro_os      = 4
```

then reconstruction uses:

```text
Nro    = 256
Ny     = 256
Nz     = 192
Nro_os = 1024
```

The reconstructed array is stored in logical order:

```text
axis 0 = readout, physical z
axis 1 = LIN / phase, physical y
axis 2 = PAR / partition, physical x
```

### NIfTI export

The integrated reconstruction script can optionally export the final reconstructed image to NIfTI:

```bash
--save-nifti
```

Magnitude NIfTI export is enabled whenever `--save-nifti` is used. Phase export can also be enabled:

```bash
--save-nifti-phase
```

NIfTI orientation is derived from the input MPRAGE TWIX `.dat` file using MeasYaps geometry. The script uses the helper module:

```text
recon/utils/nifti_export_twix.py
```

NIfTI export center-crops readout oversampling before writing the NIfTI files. This crop is applied only to the NIfTI output; the `.npy` reconstruction output remains in the oversampled readout grid.

Default NIfTI behavior:

```text
output folder:        <out_folder>/nifti/
magnitude export:     enabled with --save-nifti
phase export:         enabled with --save-nifti-phase
axis roles:           readout,phase,slice
axis flips:           false,true,false
Twix coordinate mode: LPS
in-plane rot sign:    -1.0
```

Useful NIfTI-related command-line options:

```bash
--save-nifti
--save-nifti-phase
--nifti-out-folder /path/to/nifti_output
--nifti-sub sub-name-or-label
--nifti-suffix MPRAGE
--nifti-axis-roles readout,phase,slice
--nifti-axis-flips false,true,false
--twix-coord-system LPS
--twix-inplane-rot-sign -1
--twix-use-fov-for-voxel-size
```

If the exported image appears mirrored or rotated relative to a trusted DICOM/NIfTI reference, adjust the axis flips or in-plane rotation sign and re-export.

### Reconstruction outputs

The reconstruction script writes outputs to the selected output folder, including:

- coil-compression matrix and energy files
- coil sensitivity maps
- coil sensitivity magnitude and phase plots
- coil-compressed MPRAGE k-space
- PSF calibration fits and diagnostic plots for wave reconstruction
- reconstructed wave or no-wave image arrays as `.npy`
- optional magnitude NIfTI and JSON sidecar files
- optional phase NIfTI and JSON sidecar files

Output filenames include the resolution, acceleration factors, reconstruction mode, and user-provided file tag when available.

## Prerequisites

### Sequence generation

Required:

- MATLAB
- Pulseq MATLAB toolbox

Optional:

- [Safe PNS prediction](https://github.com/filip-szczepankiewicz/safe_pns_prediction), for PNS/CNS checks
- scanner `.asc` file, for PNS/CNS and forbidden-frequency checks
- an existing `forbiddenFreqCheck.m` helper

The optional checks can be skipped when running the sequence script.

### Reconstruction

Required:

- Python 3.9 or newer recommended
- NumPy
- SciPy
- Matplotlib
- PyTorch
- CuPy
- SigPy
- pypulseq
- an NVIDIA GPU visible to CuPy, SigPy, and PyTorch
- the repository `recon/utils/` modules available on the Python path

Required for optional NIfTI export:

- nibabel
- mapvbvd, for reading Siemens TWIX MeasYaps geometry from the input `.dat` file

Optional:

- pydicom, if using DICOM-derived affine helper functions retained in `nifti_export_twix.py`

The reconstruction code uses the GPU for coil-compression-related processing and ESPIRiT calibration through CuPy and SigPy. NIfTI export uses the input MPRAGE TWIX `.dat` file to derive orientation and writes `.nii.gz` plus JSON sidecar outputs.

Before running the reconstruction, make sure the following components are mutually compatible:

- Linux `glibc` version
- NVIDIA GPU model and compute capability
- NVIDIA driver version
- CUDA runtime or toolkit version
- CuPy package and CUDA build, such as the correct `cupy-cudaXX` package
- SigPy version
- PyTorch version and PyTorch CUDA build

Version mismatches commonly appear as missing `GLIBC_x.y` errors, CUDA library-load failures, GPU initialization failures, or CuPy/PyTorch CUDA-version conflicts. A clean conda or virtual environment is recommended.

## Installing dependencies

### Pulseq

Pulseq is required, but this repository does not vendor Pulseq as a Git submodule.

Install or clone Pulseq separately, then provide its repository root or MATLAB folder when prompted. The integrated script accepts either:

```text
/path/to/pulseq
```

or:

```text
/path/to/pulseq/matlab
```

The script detects the location of the `+mr` package and adds the appropriate folder to the MATLAB path.

If the public Pulseq repository is unavailable from the local environment, use an existing local installation, a lab-maintained mirror or fork, or a downloaded source snapshot.

### Safe PNS prediction

Safe PNS prediction is optional. Leave the path empty or enter `-` to clear it when PNS/CNS checks are not needed.

### Python reconstruction environment

Create a dedicated Python environment. Exact package versions depend on the local GPU, driver, CUDA, and `glibc` setup.

Example:

```bash
conda create -n wave-mprage-recon python=3.10
conda activate wave-mprage-recon

pip install numpy scipy matplotlib pypulseq sigpy torch nibabel mapvbvd
# Install the CuPy package matching the local CUDA environment:
# pip install cupy-cuda11x
# or
# pip install cupy-cuda12x
# Optional if using DICOM affine utilities:
# pip install pydicom
```

Verify the GPU stack:

```bash
python - <<'PY'
import torch
import cupy as cp
import sigpy as sp

print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())
print('cupy:', cp.__version__, 'gpu count:', cp.cuda.runtime.getDeviceCount())
print('sigpy:', sp.__version__)
PY
```

## Basic usage

### Generate the integrated sequence

Open MATLAB and run the integrated script from the `seq/` folder:

```matlab
cd seq
mprage_3d_wave_with_flash_calibration
```

On the first run, configure the requested paths. The settings are saved locally for later runs.

On subsequent runs:

1. choose whether to reuse the saved JSON settings
2. choose whether any path setting should be changed
3. select the individual entries to update when needed

Leaving the output-path prompt blank uses the current MATLAB folder as the output root.

The generated sequence filename includes the MPRAGE wave mode, matrix, resolution, ETL plan, acceleration factors, calibration size, ACS size, readout oversampling, wave settings, orientation, and scanner type.

### Optional workspace defaults

The path helper reads existing values from the MATLAB base workspace and presents them as defaults:

```matlab
pulseq_path = '/path/to/pulseq';
safe_pns_prediction_path = '/path/to/safe_pns_prediction';  % optional
out_path = '/path/to/output/root';
system_asc_file = '/path/to/scanner.asc';                   % optional

cd seq
mprage_3d_wave_with_flash_calibration
```

The interactive JSON workflow is still used, allowing saved or workspace paths to be reviewed and changed.

### Reconstruct Wave MPRAGE data

Run the integrated reconstruction script from the repository root or the `recon/` folder.

The integrated sequence produces one TWIX measurement with MPRAGE in `image` and FLASH calibration/ACS data in `refscan`. Reconstruction reads both containers from the same measurement.

Review command-line options with:

```bash
python recon_wave_mprage_from_twix_integrated_nifti.py --help
```

Example wave reconstruction with NIfTI magnitude and phase export:

```bash
python recon_wave_mprage_from_twix_integrated_nifti.py \
  --data-folder /path/to/data \
  --out-folder /path/to/output \
  --mprage-data-file meas_integrated_wave_mprage.dat \
  --mprage-seq-file mprage_3d_flashcalib_wave.seq \
  --tag-wave wave \
  --file-tag test01 \
  --save-nifti \
  --save-nifti-phase
```

Example no-wave reconstruction without NIfTI export:

```bash
python recon_wave_mprage_from_twix_integrated_nifti.py \
  --data-folder /path/to/data \
  --out-folder /path/to/output \
  --mprage-data-file meas_integrated_nowave_mprage.dat \
  --mprage-seq-file mprage_3d_flashcalib_nowave.seq \
  --tag-wave nowave \
  --file-tag test01
```

### Reconstruction inputs

For the integrated acquisition, the core inputs are:

- combined MPRAGE + FLASH calibration TWIX `.dat` file
- matching integrated Pulseq `.seq` file
- output folder
- output filename tag
- reconstruction mode: `wave` or `nowave`

The MPRAGE and calibration portions automatically share geometry, FOV, matrix size, orientation, readout oversampling, readout duration, RF settings, wave amplitude, and number of wave cycles because they are generated by the same sequence script.

For NIfTI export, the same MPRAGE TWIX `.dat` file is also used as the orientation source.
