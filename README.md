# Wave MPRAGE

Pulseq-based 3D Wave MPRAGE sequence development code.

The MPRAGE sequence is built based on Maxim's Pulseq MPRAGE demo:
`matlab/demoSeq/writeMPRAGE.m` from the Pulseq MATLAB toolbox.

## Repository layout

```text
.
├── README.md
└── seq/
    ├── mprage_3d_wave.m              # Wave MPRAGE sequence
    ├── flash_wave_calibration.m       # FLASH partial-k-space sequence for wave PSF calibration
    └── utils/                         # Utility functions
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

## Prerequisites

Required:
- MATLAB
- Pulseq MATLAB toolbox

Optional:
- [Safe PNS prediction](https://github.com/filip-szczepankiewicz/safe_pns_prediction), for PNS/CNS checks
- Scanner `.asc` file, for PNS/CNS and forbidden-frequency checks

The optional checks can be skipped when running the sequence script.

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

## Basic usage

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


