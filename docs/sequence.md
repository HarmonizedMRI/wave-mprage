# Sequence generation

## Primary entry point

Use:

```text
seq/mprage_3d_wave_with_flash_calibration.m
```

The script combines the MPRAGE image acquisition and FLASH wave calibration into one Pulseq sequence. MATLAB remains the authoritative sequence implementation; no Python sequence port is required for the current repository.

## Prerequisites

Required:

- MATLAB
- Pulseq MATLAB toolbox

Optional:

- Safe PNS Prediction for PNS/CNS checks
- scanner `.asc` file
- an existing `forbiddenFreqCheck.m` helper

The optional checks may be skipped. The sequence is written before those checks are run.

## Basic use

```matlab
cd seq
mprage_3d_wave_with_flash_calibration
```

The path helper accepts a Pulseq repository root or its `matlab/` directory and finds the `+mr` package automatically.

## Path settings

`seq/utils/configurePathSettings.m` manages:

- `pulseq_path`
- `safe_pns_prediction_path`
- `out_path`
- `system_asc_file`

The values are saved beside the main sequence script in a local file matching `*path_settings.json`. The file is ignored by Git because paths are machine-specific.

On later runs, the script can reuse the saved settings and selectively update one or more entries. Enter `-` to clear an optional path. Leaving `out_path` blank uses MATLAB's current folder.

Workspace variables with the same names may be supplied as prompt defaults.

## Generated files

Depending on `write_v141_format`, outputs are written to one or both folders:

```text
<out_path>/generated_seq_v141/
<out_path>/generated_seq_v151/
```

These generated folders are ignored by Git.

## Acquisition order

The integrated sequence contains:

1. MPRAGE image acquisition
2. FLASH calibration acquisition, including dummy scans and settling scans

MPRAGE is placed first so that the continuous FLASH train does not alter longitudinal magnetization immediately before the first MPRAGE inversion block.

## Shared geometry and wave settings

The MPRAGE and FLASH modules share the acquisition geometry and important readout/wave settings, including FOV, matrix size, readout duration, readout oversampling, RF spoiling, RF duration, wave amplitude, wave slew limit, wave cycles, orientation, and scanner limits.

The current integrated implementation uses the sagittal convention:

```matlab
slOrientation = 'SAG';
ax.d1 = 'z';  % readout
ax.d2 = 'x';  % inner PE / PAR
ax.d3 = 'y';  % outer PE / LIN
```

The reconstruction code mirrors this logical-axis mapping because the `ax` structure itself is not stored in the Pulseq definitions.

## TWIX routing

| Acquisition | `REF` | `IMA` | `SET` | TWIX container |
|---|---:|---:|---:|---|
| MPRAGE image | false | false | 0 | `image` |
| FLASH calibration | true | false | 0-4 | `refscan` |

MPRAGE has no separate ACS acquisition. The FLASH module includes the calibration projections and the ACS block in `refscan`.

## FLASH calibration SET layout

| SET | Acquisition | Local extent |
|---:|---|---|
| 0 | no-wave, ky-wide / kz-narrow | `Ncalib1 x Ncalib2` |
| 1 | sine-wave, ky-wide / kz-narrow | `Ncalib1 x Ncalib2` |
| 2 | no-wave, kz-wide / ky-narrow | `Ncalib2 x Ncalib1` |
| 3 | cosine-wave, kz-wide / ky-narrow | `Ncalib2 x Ncalib1` |
| 4 | no-wave ACS | `Nacs x Nacs` |

Current defaults are:

```matlab
Ncalib1 = 72;
Ncalib2 = 1;
Nacs    = 32;
```

The sequence definitions used by the current reconstruction include:

```text
Calibration_Ncalib1
Calibration_Nacs
ReadoutOversamplingFactor
MPRAGE_PE2_R
MPRAGE_PE1_R
OrientationMapping
Nx, Ny, Nz, FOV
```

## Validation

Before writing, the integrated script validates the expected image/refscan routing, SET/PAR/LIN label evolution, calibration ordering, ACS location, MPRAGE PE sampling, and Pulseq timing.

The optional post-write checks cover PNS/CNS and forbidden frequencies when their dependencies are available.

## Utilities

Local sequence helpers live under `seq/utils/`, including shared sine/cosine wave generation, ramps, scheduling, PE planning, label validation, and path configuration.

`forbiddenFreqCheck.m` is not reimplemented by this repository. Put an available copy on the MATLAB path when that check is needed.
