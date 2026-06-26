# Wave MPRAGE

Pulseq-based 3D Wave MPRAGE sequence development and reconstruction code.

The MPRAGE sequence is built based on Maxim's Pulseq MPRAGE demo:
`matlab/demoSeq/writeMPRAGE.m` from the Pulseq MATLAB toolbox.

## Repository layout

```text
.
├── README.md
├── seq/
│   ├── mprage_3d_wave.m              # Wave MPRAGE sequence
│   ├── flash_wave_calibration.m       # FLASH partial-k-space sequence for wave PSF calibration
│   └── utils/                         # Sequence utility functions
└── recon/
    ├── recon_wave_mprage_from_twix.py # Wave/no-wave MPRAGE reconstruction from Siemens TWIX data
    └── utils/                         # Reconstruction utility functions
```

### `seq/mprage_3d_wave.m`

Main Wave MPRAGE sequence generator.

The script supports interactive path setup for:
- Pulseq MATLAB toolbox
- optional Safe PNS prediction toolbox
- output directory
- optional scanner `.asc` file for PNS/CNS and forbidden-frequency checks

The sequence writes the `.seq` file before optional safety/frequency checks, so sequence generation can still proceed if the optional checking dependencies are not available.

### `seq/flash_wave_calibration.m`

FLASH calibration sequence for wave PSF calibration.

Make sure the following settings are consistent with `mprage_3d_wave.m`:
- geometry
- FOV
- matrix size
- orientation
- readout duration and oversampling
- wave amplitude
- number of wave cycles
- sine/cosine wave settings

### `seq/utils/`

Utility functions used by the sequence scripts, including helper code for optional forbidden-frequency checks.

### `recon/recon_wave_mprage_from_twix.py`

Python reconstruction script for Siemens TWIX data.

The reconstruction workflow includes:
- loading Wave MPRAGE or no-wave MPRAGE TWIX data
- loading FLASH wave-calibration TWIX data
- estimating coil-compression weights from calibration data
- estimating ESPIRiT coil sensitivity maps
- generating a calibrated wave PSF from the FLASH calibration data
- reconstructing wave data with wave CG-SENSE
- reconstructing no-wave data with standard CG-SENSE
- saving intermediate coil-compressed k-space, coil maps, PSF diagnostics, and reconstructed images as `.npy`/`.png` outputs

The current reconstruction output is saved as NumPy files. NIfTI export can be added downstream if needed.

### `recon/utils/`

Utility functions used by the reconstruction script, including TWIX import, coil compression, coil sensitivity plotting, PSF phase fitting, and wave CG-SENSE helper functions.

## Prerequisites

### Sequence generation

Required:
- MATLAB
- Pulseq MATLAB toolbox

Optional:
- [Safe PNS prediction](https://github.com/filip-szczepankiewicz/safe_pns_prediction), for PNS/CNS checks
- Scanner `.asc` file, for PNS/CNS and forbidden-frequency checks

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
- an NVIDIA GPU visible to CuPy/SigPy/PyTorch
- the repository `recon/utils/` modules available on the Python path

The reconstruction code uses the GPU for coil-compression-related processing and ESPIRiT calibration through CuPy/SigPy. Before running the reconstruction, make sure the following components are mutually compatible:
- Linux `glibc` version
- NVIDIA GPU model and compute capability
- NVIDIA driver version
- CUDA runtime/toolkit version
- CuPy package/build, for example the correct `cupy-cudaXX` package
- SigPy version
- PyTorch version and PyTorch CUDA build

Version mismatches commonly appear as errors such as missing `GLIBC_x.y`, CUDA runtime/library load failures, GPU device initialization failures, or CuPy/PyTorch CUDA-version conflicts. Use a clean conda or virtualenv environment and install CuPy/PyTorch builds that match the CUDA version supported by your driver and system.

## Installing dependencies

### Pulseq

Pulseq is required, but this repository does not vendor Pulseq as a Git submodule.

Install or clone Pulseq separately, then provide its path when running the sequence script. The script expects the Pulseq MATLAB folder to be available as:

```matlab
addpath(fullfile(pulseq_path, 'matlab'));
```

For example:

```matlab
pulseq_path = '/path/to/pulseq';
```

If the public Pulseq GitHub repository is unavailable from your environment, use one of the following approaches:
- use an existing local Pulseq installation
- use a lab-maintained mirror or fork
- download a released/source snapshot manually
- provide `pulseq_path` interactively when prompted

### Safe PNS prediction

Safe PNS prediction is optional. If it is not available, leave the path empty and skip PNS/CNS checks when prompted.

Basic usage:

```matlab
safe_pns_prediction_path = fullfile(pwd, 'external', 'safe_pns_prediction');
```

### Python reconstruction environment

Create a dedicated Python environment for reconstruction. The exact package versions depend on your local GPU, driver, CUDA, and `glibc` setup.

Example using conda and pip:

```bash
conda create -n wave-mprage-recon python=3.10
conda activate wave-mprage-recon

pip install numpy scipy matplotlib pypulseq sigpy torch
# Install the CuPy package that matches your CUDA environment, for example:
# pip install cupy-cuda11x
# or
# pip install cupy-cuda12x
```

After installation, verify that the GPU stack is visible from Python:

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

### Generate the sequence

Clone this repository, open MATLAB, and run the sequence script from the `seq/` folder:

```matlab
cd seq
mprage_3d_wave
```

The script will prompt for missing paths if they are not already defined in the MATLAB workspace.

To avoid repeated prompts, define paths before running the script:

```matlab
pulseq_path = '/path/to/pulseq';
safe_pns_prediction_path = '/path/to/safe_pns_prediction';  % optional
out_path = '/path/to/output/folder/';
system_asc_file = '/path/to/scanner.asc';                   % optional

cd seq
mprage_3d_wave
```

### Reconstruct Wave MPRAGE data

Run the reconstruction script from the repository root or from the `recon/` folder. Provide the data folder, output folder, MPRAGE TWIX/sequence files, FLASH calibration TWIX/sequence files, output tag, and whether the dataset is wave or no-wave.

Example wave reconstruction:

```bash
cd recon
python recon_wave_mprage_from_twix.py \
  --data-folder /path/to/data \
  --out-folder /path/to/recon_output \
  --mprage-data-file wave_mprage.dat \
  --mprage-seq-file wave_mprage.seq \
  --calib-data-file flash_wave_calibration.dat \
  --calib-seq-file flash_wave_calibration.seq \
  --file-tag sub01_run01 \
  --tag-wave wave
```

Example no-wave reconstruction:

```bash
cd recon
python recon_wave_mprage_from_twix.py \
  --data-folder /path/to/data \
  --out-folder /path/to/recon_output \
  --mprage-data-file nowave_mprage.dat \
  --mprage-seq-file nowave_mprage.seq \
  --calib-data-file flash_wave_calibration.dat \
  --calib-seq-file flash_wave_calibration.seq \
  --file-tag sub01_run01 \
  --tag-wave nowave
```

The script can also prompt interactively for missing paths and tags.

### Reconstruction inputs

Required inputs:
- Wave or no-wave MPRAGE TWIX `.dat` file
- matching MPRAGE Pulseq `.seq` file
- FLASH wave-calibration TWIX `.dat` file
- matching FLASH wave-calibration Pulseq `.seq` file
- output folder
- output filename tag
- reconstruction mode: `wave` or `nowave`

The MPRAGE and calibration data should use compatible geometry, FOV, matrix size, orientation, readout oversampling, readout duration, and wave settings.

### Reconstruction outputs

The reconstruction script writes outputs to `out_folder`, including:
- coil-compression matrix/energy files
- coil sensitivity maps
- coil sensitivity magnitude and phase plots
- coil-compressed MPRAGE k-space
- PSF calibration fit files and diagnostic plots for wave reconstruction
- reconstructed image array for wave or no-wave reconstruction

Output filenames include the resolution, acceleration factors, reconstruction mode, and `file_tag` when available.
