% mprage_3d_wave_with_flash_calibration.m
% Author: Yiyun Dong
% Affiliation: Athinoula A. Martinos Center for Biomedical Imaging
% Date: 2026-07-15
%
% Integrated Wave-MPRAGE + FLASH wave-calibration sequence.
%
% TWIX routing:
%   image:
%       MPRAGE imaging data only, with REF=false, IMA=false, SET=0.
%   refscan:
%       FLASH calibration data only, with REF=true and IMA=false:
%         SET 0: no-wave, ky-wide / kz-narrow
%         SET 1: sin-wave, ky-wide / kz-narrow
%         SET 2: no-wave, kz-wide / ky-narrow
%         SET 3: cos-wave, kz-wide / ky-narrow
%         SET 4: no-wave ACS, stored at local LIN/PAR indices 0:(Nacs-1)
%
% With Ncalib1=72 and Nacs=32, the logical refscan extent is
% [LIN=72, PAR=72, SET=5], and the ACS occupies the first 32x32 block of
% SET 4. Unacquired entries are zero-filled by the TWIX loader.
%
% Acquisition order is MPRAGE first, followed by FLASH calibration. The
% calibration includes its own dummy RF-spoiled GRE train.
%
% Wave-gradient and scheduling helpers are stored in ./utils/. The existing
% forbiddenFreqCheck.m is expected in ./utils/ or elsewhere on the MATLAB path.

% Do not call clear/clear all here: users may predefine path variables in the
% MATLAB workspace before running this script.
close all; clc
format long

%% Path
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end

% Add extracted helper functions before the first helper call below.
utils_path = fullfile(script_dir, 'utils');
if exist(utils_path, 'dir')
    addpath(utils_path);
else
    error('Required local utils folder not found: %s', utils_path);
end

% Path settings are stored locally beside this script in
% mprage_flash_path_settings.json. When that file already exists, the user
% can reuse all saved paths or choose specific entries to update.
pathSettings = configurePathSettings(script_dir);
pulseq_path = pathSettings.pulseq_path;
safe_pns_prediction_path = pathSettings.safe_pns_prediction_path;
out_path = ensureTrailingFilesep(pathSettings.out_path);
system_asc_file = pathSettings.system_asc_file;

pulseq_matlab_path = fullfile(pulseq_path, 'matlab');
if exist(fullfile(pulseq_matlab_path, '+mr'), 'dir')
    addpath(pulseq_matlab_path);
elseif exist(fullfile(pulseq_path, '+mr'), 'dir')
    addpath(pulseq_path);
else
    warning(['Could not find +mr under the provided Pulseq path. ', ...
             'Continuing after adding the provided path.']);
    addpath(pulseq_path);
end

if ~isempty(safe_pns_prediction_path)
    addpath(safe_pns_prediction_path);
end

%% Shared parameters: MPRAGE and FLASH calibration
% Write options:
%   false -> current Pulseq format only
%   true  -> write both legacy v1.4.1 and current format
write_v141_format = true;

alpha    = 7;
ro_dur   = 5120e-6;
ro_os    = 4;
ro_spoil = 3;
rfSpoilingInc = 50;
rfLen         = 100e-6;

% The integrated sequence supports the sagittal geometry only.
slOrientation = 'SAG';
fov = [192 256 256]*1e-3;        % [x y z], m
res = [1.0 1.0 1.0];             % requested [x y z], mm
N = 2 * round((fov(:).' * 1e3 ./ res) / 2);
actualRes = fov(:).' ./ N * 1e3;

ax = struct;
ax.d1 = 'z';                      % readout
ax.d2 = 'x';                      % inner PE / PAR
ax.d3 = setdiff('xyz', [ax.d1 ax.d2]); % outer PE / LIN
ax.n1 = strfind('xyz', ax.d1);
ax.n2 = strfind('xyz', ax.d2);
ax.n3 = strfind('xyz', ax.d3);

assert(strcmp(slOrientation, 'SAG'), ...
    'This integrated sequence currently supports slOrientation=''SAG'' only.');
fprintf(['Requested resolution [x y z] = [%.4g %.4g %.4g] mm. ', ...
         'Derived N = [%d %d %d]. Actual resolution = [%.4g %.4g %.4g] mm.\n'], ...
    res(1), res(2), res(3), N(1), N(2), N(3), ...
    actualRes(1), actualRes(2), actualRes(3));

% Shared wave parameters.
gwave_max = 8;                    % mT/m
swave_max = 200;                  % T/m/s
Ncycles   = 10;
tag_wave_details = ['_amp' num2str(gwave_max) ...
    '_cycles' num2str(Ncycles) '_' slOrientation];

%% FLASH calibration-only parameters
Ndummy = 300;
NsettlePerPart = 10;
Ncalib1 = 72;
Ncalib2 = 1;
Nacs    = 32;

assert(Ncalib1 == round(Ncalib1) && Ncalib1 > 0, ...
    'Ncalib1 must be a positive integer.');
assert(Ncalib2 == round(Ncalib2) && Ncalib2 > 0, ...
    'Ncalib2 must be a positive integer.');
assert(Nacs == round(Nacs) && Nacs > 0, ...
    'Nacs must be a positive integer.');
assert(Ncalib1 > Ncalib2, 'Require Ncalib1 > Ncalib2.');
assert(Ncalib1 <= N(ax.n2) && Ncalib1 <= N(ax.n3), ...
    'Ncalib1 exceeds a PE dimension.');
assert(Ncalib2 <= N(ax.n2) && Ncalib2 <= N(ax.n3), ...
    'Ncalib2 exceeds a PE dimension.');
assert(Nacs <= Ncalib1, 'Nacs must not exceed Ncalib1.');
assert(Ndummy >= 0 && Ndummy == round(Ndummy), ...
    'Ndummy must be a nonnegative integer.');
assert(NsettlePerPart >= 0 && NsettlePerPart == round(NsettlePerPart), ...
    'NsettlePerPart must be a nonnegative integer.');

%% MPRAGE-only parameters
TI    = 1.1;
TRout = 2.5;
R1 = 2;                           % acceleration along ax.d2 / PAR
R2 = 3;                           % acceleration along ax.d3 / LIN
ETLtarget = 192;

etlSeg = struct;
etlSeg.sMin      = 16;
etlSeg.KMax      = 12;
etlSeg.PMax      = 16;
etlSeg.fillerMax = 0.10;
etlSeg.savedMin  = 16;

% These flags affect MPRAGE only. Calibration always acquires its required
% no-wave, sine-wave, and cosine-wave parts from the shared event library.
isUseWave_cos = true;
isUseWave_sin = true;

assert(R1 >= 1 && R1 == round(R1), 'R1 must be a positive integer.');
assert(R2 >= 1 && R2 == round(R2), 'R2 must be a positive integer.');
assert(ETLtarget >= 1 && ETLtarget == round(ETLtarget), ...
    'ETLtarget must be a positive integer.');

%% System limits
% sys = mr.opts('MaxGrad',28,'GradUnit','mT/m',...
    % 'MaxSlew',150,'SlewUnit','T/m/s',...
    % 'rfRingdownTime', 20e-6, 'rfDeadtime', 100e-6, 'adcDeadTime', 10e-6);

% For Siemens scanner
sys_type_options          = {'prisma', 'skyra', 'Connectome2', 'C2_simulate_prisma', 'trio', 'prisma_XA30A', 'premier', 'CimaX', 'TerraX'};
sys_type                  = selectStringOption('sys_type', 'Select scanner/system name', sys_type_options, 'prisma');
slew_safety_magrin        = 0.7;
grad_safety_magrin        = 0.9;
lowPNS_slew_safety_margin = 0.4;
lowPNS_grad_safety_margin = grad_safety_magrin;
diff_slew_safety_margin   = 0.45; % decrease this to reduce PNS, this would not lengthen TE too much
diff_grad_safety_margin   = 0.97;

if strcmp(sys_type,'prisma') || strcmp(sys_type,'C2_simulate_prisma') || strcmp(sys_type,'prisma_XA30A')
    physical_slew_max = 200;
    physical_grad_max = 80;
    B0=2.89; % 1.5 2.89 3.0
elseif strcmp(sys_type,'premier')
    physical_slew_max = 200;
    physical_grad_max = 70;%80;
    B0=3;
elseif strcmp(sys_type,'Connectome2')
    physical_slew_max = 598.802;
    physical_grad_max = 500;
    B0=2.89;
elseif strcmp(sys_type,'skyra')
    physical_slew_max = 180;
    physical_grad_max = 43;
    B0=2.89;
elseif strcmp(sys_type,'trio')
    physical_slew_max = 170;
    physical_grad_max = 38;
    B0=2.89;
elseif strcmp(sys_type,'CimaX')
    physical_slew_max = 200;
    physical_grad_max = 200;
    B0=2.89;
elseif strcmp(sys_type,'TerraX')
    physical_slew_max = 250;
    physical_grad_max = 135;
    B0=2.89;
else
    error('Undefined')
end

isGEscanner = strcmp(sys_type,'premier');
if ~isGEscanner
    pislquant = 0;
end
if isGEscanner
    % RF/gradient delay (sec).
    % Conservative choice that should work across all GE scanners.
    psd_rf_wait = 200e-6;  % section 5.4 in PulseqOnGE_v1.0.pdf
    
    rfDeadTime =  100e-6;
    rfRingdownTime = 60e-6 + psd_rf_wait;
    adcDeadTime = 20e-6;
    adcRasterTime = 2e-6;
    rfRasterTime = 2e-6;
    gradRasterTime = 4e-6;
    blockDurationRaster = 4e-6;
else % is siemens
    rfDeadTime =  100e-6;
    rfRingdownTime = 100e-6;
    adcDeadTime = 20e-6;
    %     adcRasterTime = 2e-6;
    adcRasterTime = 100e-9;
    rfRasterTime = 1e-6;
    gradRasterTime = 10e-6;
    blockDurationRaster = 10e-6;
end
sys = mr.opts('MaxGrad',physical_grad_max*grad_safety_magrin,'GradUnit','mT/m',...
    'MaxSlew',physical_slew_max*slew_safety_magrin,'SlewUnit','T/m/s',...
    'rfDeadTime', rfDeadTime, ...
    'rfRingdownTime', rfRingdownTime, ...
    'adcDeadTime', adcDeadTime,...
    'adcRasterTime', adcRasterTime,...
    'rfRasterTime', rfRasterTime,...
    'gradRasterTime', gradRasterTime,...
    'blockDurationRaster', blockDurationRaster,...
    'B0',B0);
sys_lowPNS = mr.opts('MaxGrad',physical_grad_max*lowPNS_grad_safety_margin,'GradUnit','mT/m',...
    'MaxSlew',physical_slew_max*lowPNS_slew_safety_margin,'SlewUnit','T/m/s',...
    'rfDeadtime', rfDeadTime, ...
    'rfRingdownTime', rfRingdownTime, ...
    'adcDeadTime', adcDeadTime,...
    'adcRasterTime', adcRasterTime,...
    'rfRasterTime', rfRasterTime,...
    'gradRasterTime', gradRasterTime,...
    'blockDurationRaster', blockDurationRaster,...
    'B0',B0);
sys_diff = mr.opts('MaxGrad',physical_grad_max*diff_grad_safety_margin,'GradUnit','mT/m',...
    'MaxSlew',physical_slew_max*diff_slew_safety_margin,'SlewUnit','T/m/s',...
    'rfDeadtime', rfDeadTime, ...
    'rfRingdownTime', rfRingdownTime, ...
    'adcDeadTime', adcDeadTime,...
    'adcRasterTime', adcRasterTime,...
    'rfRasterTime', rfRasterTime,...
    'gradRasterTime', gradRasterTime,...
    'blockDurationRaster', blockDurationRaster,...
    'B0',B0);
lims = sys;

% Create the single integrated sequence object.
seq = mr.Sequence(sys);

%% Shared RF/readout/wave event library
rf = mr.makeBlockPulse(alpha*pi/180, sys, ...
    'Duration', rfLen, 'SliceThickness', fov(ax.n2), 'use', 'excitation');
rf180 = mr.makeAdiabaticPulse('hypsec', sys, ...
    'Duration', 10.24e-3, ...
    'dwell', 1e-5, ...
    'use', 'inversion', ...
    'pythonCmd', '/opt/homebrew/Caskroom/miniforge/base/envs/ptx314/bin/python');

deltak = 1./fov;
dwell = round((ro_dur / N(ax.n1) / ro_os) / sys.adcRasterTime) ...
    * sys.adcRasterTime;
Tread = dwell * N(ax.n1) * ro_os;
Nx_os = N(ax.n1) * ro_os;
fprintf('Readout: Nro=%d, ro_os=%d, Nx_os=%d, Tread=%.6f ms, dwell=%.6f us\n', ...
    N(ax.n1), ro_os, Nx_os, Tread*1e3, dwell*1e6);

gro = mr.makeTrapezoid(ax.d1, ...
    'Amplitude', N(ax.n1)*deltak(ax.n1)/ro_dur, ...
    'FlatTime', ceil((ro_dur+sys.adcDeadTime)/sys.gradRasterTime) ...
        * sys.gradRasterTime, ...
    'system', sys);
adc = mr.makeAdc(Nx_os, 'Duration', ro_dur, ...
    'Delay', gro.riseTime, 'system', sys);
assert(adc.numSamples == Nx_os, 'ADC sample count mismatch.');

groPre = mr.makeTrapezoid(ax.d1, ...
    'Area', -gro.amplitude * ...
        (adc.dwell*(adc.numSamples/2+0.5) + 0.5*gro.riseTime), ...
    'system', sys_lowPNS);
gpe1 = mr.makeTrapezoid(ax.d2, ...
    'Area', -deltak(ax.n2)*(N(ax.n2)/2), 'system', sys_lowPNS);
gpe2 = mr.makeTrapezoid(ax.d3, ...
    'Area', -deltak(ax.n3)*(N(ax.n3)/2), 'system', sys_lowPNS);
gslSp = mr.makeTrapezoid(ax.d3, ...
    'Area', max(deltak.*N)*4, 'Duration', 10e-3, 'system', sys_lowPNS);

[gro1, groSp] = mr.splitGradientAt(gro, gro.riseTime+gro.flatTime);
if ro_spoil > 0
    groSp = mr.makeExtendedTrapezoidArea(gro.channel, gro.amplitude, 0, ...
        deltak(ax.n1)/2*N(ax.n1)*ro_spoil, sys_lowPNS);
end

rf.delay = mr.calcDuration(groSp, gpe1, gpe2);
gPre_dur = max([mr.calcDuration(groPre), ...
    mr.calcDuration(gpe1), mr.calcDuration(gpe2)]);
gPre_dur = ceil(gPre_dur/sys.gradRasterTime)*sys.gradRasterTime;
groPre  = mr.makeTrapezoid(ax.d1, 'Area', groPre.area, ...
    'Duration', gPre_dur, 'system', sys_lowPNS);
gpe1Pre = mr.makeTrapezoid(ax.d2, 'Area', gpe1.area, ...
    'Duration', gPre_dur, 'system', sys_lowPNS);
gpe2Pre = mr.makeTrapezoid(ax.d3, 'Area', gpe2.area, ...
    'Duration', gPre_dur, 'system', sys_lowPNS);

gro1.delay = mr.calcDuration(groPre);
adc.delay = gro1.delay + gro.riseTime;
gro1 = mr.addGradients({gro1, groPre}, 'system', sys);

pe1Steps = ((0:N(ax.n2)-1)-N(ax.n2)/2)/N(ax.n2)*2;
pe2Steps = ((0:N(ax.n3)-1)-N(ax.n3)/2)/N(ax.n3)*2;

% Mode IDs for calibration.
MODE_NOWAVE = 1;
MODE_SIN    = 2;
MODE_COS    = 3;
modeNames = {'nowave', 'sin', 'cos'};

% PE1/PAR: cosine wave for sagittal geometry.
gpe1Pre_nowave  = cell(1, N(ax.n2));
gpe1Post_nowave = cell(1, N(ax.n2));
gpe1Pre_cos     = cell(1, N(ax.n2));
gpe1Post_cos    = cell(1, N(ax.n2));

% PE2/LIN: sine wave for sagittal geometry.
gpe2Pre_nowave  = cell(1, N(ax.n3));
gpe2Post_nowave = cell(1, N(ax.n3));
gpe2Pre_sin     = cell(1, N(ax.n3));
gpe2Post_sin    = cell(1, N(ax.n3));

allPostDur = [];
for i = 1:N(ax.n2)
    gpe1Pre_i = mr.scaleGrad(gpe1Pre, pe1Steps(i));
    gpe1Pre_nowave{i}  = gpe1Pre_i;
    gpe1Post_nowave{i} = mr.scaleGrad(gpe1, -pe1Steps(i));
    gpe1Pre_nowave{i}.id  = seq.registerGradEvent(gpe1Pre_nowave{i});
    gpe1Post_nowave{i}.id = seq.registerGradEvent(gpe1Post_nowave{i});

    debugFlag = (i == 1);
    [gpe1Pre_cos{i}, gpe1Post_cos{i}] = defineCosineWaveGradient( ...
        Tread, sys, sys_lowPNS, Ncycles, gwave_max, swave_max, ...
        gpe1Pre_i, gro, adc, physical_slew_max, debugFlag, debugFlag);
    gpe1Pre_cos{i}.id  = seq.registerGradEvent(gpe1Pre_cos{i});
    gpe1Post_cos{i}.id = seq.registerGradEvent(gpe1Post_cos{i});

    allPostDur = [allPostDur, ... %#ok<AGROW>
        mr.calcDuration(gpe1Post_nowave{i}), ...
        mr.calcDuration(gpe1Post_cos{i})];
end

for j = 1:N(ax.n3)
    gpe2Pre_j = mr.scaleGrad(gpe2Pre, pe2Steps(j));
    gpe2Pre_nowave{j}  = gpe2Pre_j;
    gpe2Post_nowave{j} = mr.scaleGrad(gpe2, -pe2Steps(j));
    gpe2Pre_nowave{j}.id  = seq.registerGradEvent(gpe2Pre_nowave{j});
    gpe2Post_nowave{j}.id = seq.registerGradEvent(gpe2Post_nowave{j});

    debugFlag = (j == 1);
    [gpe2Pre_sin{j}, gpe2Post_sin{j}] = defineSineWaveGradient( ...
        Tread, sys, sys_lowPNS, Ncycles, gwave_max, swave_max, ...
        gpe2Pre_j, gro, adc, physical_slew_max, debugFlag, debugFlag);
    gpe2Pre_sin{j}.id  = seq.registerGradEvent(gpe2Pre_sin{j});
    gpe2Post_sin{j}.id = seq.registerGradEvent(gpe2Post_sin{j});

    allPostDur = [allPostDur, ... %#ok<AGROW>
        mr.calcDuration(gpe2Post_nowave{j}), ...
        mr.calcDuration(gpe2Post_sin{j})];
end

% Common RF delay and inner TR cover every waveform mode used anywhere in
% the integrated sequence.
rf.delay = max([mr.calcDuration(groSp), allPostDur]);
TRinner = mr.calcDuration(rf) + mr.calcDuration(gro1);
TE = mr.calcDuration(rf) - (rf.delay + mr.calcRfCenter(rf)) ...
    + adc.delay + adc.dwell*(adc.numSamples/2+0.5);
fprintf('Shared FLASH train timing: TRinner=%.6f ms, TE=%.6f ms\n', ...
    TRinner*1e3, TE*1e3);

% MPRAGE chooses wave/no-wave on each PE axis independently.
if isUseWave_cos
    gpe1Pre_mpr  = gpe1Pre_cos;
    gpe1Post_mpr = gpe1Post_cos;
else
    gpe1Pre_mpr  = gpe1Pre_nowave;
    gpe1Post_mpr = gpe1Post_nowave;
end
if isUseWave_sin
    gpe2Pre_mpr  = gpe2Pre_sin;
    gpe2Post_mpr = gpe2Post_sin;
else
    gpe2Pre_mpr  = gpe2Pre_nowave;
    gpe2Post_mpr = gpe2Post_nowave;
end

% Calibration mode tables. A mode that does not belong to an axis uses that
% axis's no-wave event.
gpe1PreByMode  = {gpe1Pre_nowave,  gpe1Pre_nowave,  gpe1Pre_cos};
gpe1PostByMode = {gpe1Post_nowave, gpe1Post_nowave, gpe1Post_cos};
gpe2PreByMode  = {gpe2Pre_nowave,  gpe2Pre_sin,     gpe2Pre_nowave};
gpe2PostByMode = {gpe2Post_nowave, gpe2Post_sin,    gpe2Post_nowave};

% Register common invariant objects.
gslSp.id = seq.registerGradEvent(gslSp);
groSp.id = seq.registerGradEvent(groSp);
gro1.id  = seq.registerGradEvent(gro1);
[~, rf.shapeIDs] = seq.registerRfEvent(rf);
[rf180.id, rf180.shapeIDs] = seq.registerRfEvent(rf180);

%% MPRAGE image-only sampling and fixed-ETL schedule
[PE1_img, centerPE1LineIdx] = ...
    makeAcceleratedPESamplingPattern(N(ax.n2), R1);
[PE2_img, centerPE2LineIdx] = ...
    makeAcceleratedPESamplingPattern(N(ax.n3), R2);
nPE1Img = numel(PE1_img);
nPE2Img = numel(PE2_img);

if nPE1Img > ETLtarget
    error('Sampled MPRAGE PE1 count (%d) exceeds ETLtarget (%d).', ...
        nPE1Img, ETLtarget);
end

etlPlan_img = chooseFixedETLPlan(nPE1Img, ETLtarget, etlSeg);
imgPairsGlobal = makePEPairList(PE1_img, PE2_img, []);
centerSlotTarget = floor(ETLtarget/2) + 1;
imgBlocks = buildSegmentedFixedETLBlocks(PE1_img, PE2_img, ...
    ETLtarget, etlPlan_img, centerPE1LineIdx, centerPE2LineIdx, ...
    centerSlotTarget);
[nImgRealSlots, nImgDummySlots] = countFixedETLBlocks(imgBlocks);
assertGlobalCenterAtTarget(imgBlocks, centerPE1LineIdx, ...
    centerPE2LineIdx, centerSlotTarget, 'MPRAGE IMG');

inv180TailToEnd = mr.calcDuration(rf180) ...
    - mr.calcRfCenter(rf180) - rf180.delay;
rfStartToCenter = rf.delay + mr.calcRfCenter(rf);

fprintf(['MPRAGE: R1=%d -> %d/%d PE1 lines; R2=%d -> %d/%d PE2 lines; ', ...
         'ETL=%d.\n'], ...
    R1, nPE1Img, N(ax.n2), R2, nPE2Img, N(ax.n3), ETLtarget);
fprintf(['MPRAGE ETL plan: mode=%s, segment=%d, segments/PE2=%d, ', ...
         'segments/block=%d, filler/PE2=%d, efficiency=%.4f.\n'], ...
    etlPlan_img.mode, etlPlan_img.s, etlPlan_img.K, etlPlan_img.P, ...
    etlPlan_img.F, etlPlan_img.efficiency);
fprintf('MPRAGE: %d inversion block(s), real ADCs=%d, dummy slots=%d.\n', ...
    numel(imgBlocks), nImgRealSlots, nImgDummySlots);

% MPRAGE labels use full 0-based matrix coordinates.
lblLIN_img = cell(1, N(ax.n3));
for iY = 1:N(ax.n3)
    lblLIN_img{iY} = mr.makeLabel('SET', 'LIN', iY-1);
end
lblPAR_img = cell(1, N(ax.n2));
for iZ = 1:N(ax.n2)
    lblPAR_img{iZ} = mr.makeLabel('SET', 'PAR', iZ-1);
end
lblECO_img = mr.makeLabel('SET', 'ECO', 0);
lblSET_img = mr.makeLabel('SET', 'SET', 0);
lblRefOff  = mr.makeLabel('SET', 'REF', false);
lblImaOff  = mr.makeLabel('SET', 'IMA', false);

expectedPAR_img = [];
expectedLIN_img = [];

%% Add MPRAGE image acquisition
fprintf('Adding MPRAGE image acquisition...\n');
tic;
for jBlock = 1:numel(imgBlocks)
    block = imgBlocks(jBlock);
    nSlotsThisBlock = numel(block.isAcquire);
    if nSlotsThisBlock ~= ETLtarget
        error('MPRAGE block %d has %d slots; expected %d.', ...
            jBlock, nSlotsThisBlock, ETLtarget);
    end
    if block.centerSlot ~= centerSlotTarget
        error('MPRAGE block %d center slot is %d; expected %d.', ...
            jBlock, block.centerSlot, centerSlotTarget);
    end

    nAcqBeforeCenter = block.centerSlot - 1;
    TIdelay = round((TI - nAcqBeforeCenter*TRinner ...
        - inv180TailToEnd - rfStartToCenter) ...
        / sys.blockDurationRaster) * sys.blockDurationRaster;
    TRoutDelay = TRout - TRinner*nSlotsThisBlock ...
        - TIdelay - mr.calcDuration(rf180);

    if TIdelay < 0
        error(['Negative MPRAGE TIdelay in block %d. Reduce ETLtarget, ', ...
               'reduce TI, or increase TRout.'], jBlock);
    end
    if TIdelay < mr.calcDuration(gslSp)
        warning(['MPRAGE TIdelay %.3f ms is shorter than inversion-spoiler ', ...
                 'duration %.3f ms in block %d.'], ...
            TIdelay*1e3, mr.calcDuration(gslSp)*1e3, jBlock);
    end
    if TRoutDelay < 0
        error(['Negative MPRAGE TRoutDelay in block %d. Increase TRout or ', ...
               'reduce ETLtarget.'], jBlock);
    end

    seq.addBlock(rf180);
    seq.addBlock(TIdelay, gslSp);
    rf_phase = 0;
    rf_inc = 0;
    isFirstSlotInBlock = true;
    prevI = [];
    prevJ = [];

    for slotIdx = 1:nSlotsThisBlock
        iGlobal = block.iGlobal(slotIdx);
        jGlobal = block.jGlobal(slotIdx);

        rf.phaseOffset = rf_phase/180*pi;
        adc.phaseOffset = rf_phase/180*pi;
        rf_inc = mod(rf_inc + rfSpoilingInc, 360.0);
        rf_phase = mod(rf_phase + rf_inc, 360.0);

        if isFirstSlotInBlock
            seq.addBlock(rf);
            isFirstSlotInBlock = false;
        else
            seq.addBlock(rf, groSp, ...
                gpe1Post_mpr{prevI}, gpe2Post_mpr{prevJ});
        end

        if block.isAcquire(slotIdx)
            seq.addBlock(adc, gro1, ...
                gpe1Pre_mpr{iGlobal}, gpe2Pre_mpr{jGlobal}, ...
                lblPAR_img{iGlobal}, lblLIN_img{jGlobal}, lblECO_img, ...
                lblSET_img, lblRefOff, lblImaOff);
            expectedPAR_img(end+1,1) = iGlobal-1; %#ok<SAGROW>
            expectedLIN_img(end+1,1) = jGlobal-1; %#ok<SAGROW>
        else
            seq.addBlock(gro1, ...
                gpe1Pre_mpr{iGlobal}, gpe2Pre_mpr{jGlobal});
        end

        prevI = iGlobal;
        prevJ = jGlobal;
    end

    seq.addBlock(groSp, gpe1Post_mpr{prevI}, ...
        gpe2Post_mpr{prevJ}, mr.makeDelay(TRoutDelay));
end
fprintf('MPRAGE blocks added in %g seconds.\n', toc);
nMprageAdc = numel(expectedLIN_img);
assert(nMprageAdc == nImgRealSlots, ...
    'MPRAGE expected ADC count does not match scheduler count.');

%% FLASH calibration acquisition table
ky_calib1 = centerBlockIndices(N(ax.n3), Ncalib1);
ky_calib2 = centerBlockIndices(N(ax.n3), Ncalib2);
ky_acs    = centerBlockIndices(N(ax.n3), Nacs);
kz_calib1 = centerBlockIndices(N(ax.n2), Ncalib1);
kz_calib2 = centerBlockIndices(N(ax.n2), Ncalib2);
kz_acs    = centerBlockIndices(N(ax.n2), Nacs);

calParts = struct('id', {}, 'name', {}, 'mode', {}, ...
    'kyList', {}, 'kzList', {}, 'isACS', {});
calParts(1).id = 0;
calParts(1).name = 'nowave_kywide_kznarrow';
calParts(1).mode = MODE_NOWAVE;
calParts(1).kyList = ky_calib1;
calParts(1).kzList = kz_calib2;
calParts(1).isACS = false;

calParts(2).id = 1;
calParts(2).name = 'sin_kywide_kznarrow';
calParts(2).mode = MODE_SIN;
calParts(2).kyList = ky_calib1;
calParts(2).kzList = kz_calib2;
calParts(2).isACS = false;

calParts(3).id = 2;
calParts(3).name = 'nowave_kzwide_kynarrow';
calParts(3).mode = MODE_NOWAVE;
calParts(3).kyList = ky_calib2;
calParts(3).kzList = kz_calib1;
calParts(3).isACS = false;

calParts(4).id = 3;
calParts(4).name = 'cos_kzwide_kynarrow';
calParts(4).mode = MODE_COS;
calParts(4).kyList = ky_calib2;
calParts(4).kzList = kz_calib1;
calParts(4).isACS = false;

calParts(5).id = 4;
calParts(5).name = 'acs_nowave_center';
calParts(5).mode = MODE_NOWAVE;
calParts(5).kyList = ky_acs;
calParts(5).kzList = kz_acs;
calParts(5).isACS = true;

calAcqTable = struct('partArrayIdx', {}, 'partID', {}, 'mode', {}, ...
    'isACS', {}, 'iPhys', {}, 'jPhys', {}, 'iLocal', {}, 'jLocal', {});
calPartStart = zeros(1, numel(calParts));
calPartStop  = zeros(1, numel(calParts));
for p = 1:numel(calParts)
    calPartStart(p) = numel(calAcqTable)+1;
    kyList = calParts(p).kyList;
    kzList = calParts(p).kzList;
    for jLocal = 1:numel(kyList)
        jPhys = kyList(jLocal);
        for iLocal = 1:numel(kzList)
            iPhys = kzList(iLocal);
            row.partArrayIdx = p;
            row.partID = calParts(p).id;
            row.mode = calParts(p).mode;
            row.isACS = calParts(p).isACS;
            row.iPhys = iPhys;
            row.jPhys = jPhys;
            row.iLocal = iLocal;
            row.jLocal = jLocal;
            calAcqTable(end+1) = row; %#ok<SAGROW>
        end
    end
    calPartStop(p) = numel(calAcqTable);
end

nCalAdcExpected = 4*Ncalib1*Ncalib2 + Nacs*Nacs;
assert(numel(calAcqTable) == nCalAdcExpected, ...
    'Calibration acquisition-table length mismatch.');
for p = 1:numel(calParts)
    fprintf('Calibration SET %d: %-24s mode=%s, LIN=%d, PAR=%d, ADCs=%d\n', ...
        calParts(p).id, calParts(p).name, ...
        modeNames{calParts(p).mode}, numel(calParts(p).kyList), ...
        numel(calParts(p).kzList), ...
        numel(calParts(p).kyList)*numel(calParts(p).kzList));
end

% Compact local labels create a dense 72x72x5 logical refscan extent.
maxLocalLin = max(arrayfun(@(p) numel(p.kyList), calParts));
maxLocalPar = max(arrayfun(@(p) numel(p.kzList), calParts));
lblLIN_cal = cell(1, maxLocalLin);
for iY = 1:maxLocalLin
    lblLIN_cal{iY} = mr.makeLabel('SET', 'LIN', iY-1);
end
lblPAR_cal = cell(1, maxLocalPar);
for iZ = 1:maxLocalPar
    lblPAR_cal{iZ} = mr.makeLabel('SET', 'PAR', iZ-1);
end
lblSET_cal = cell(1, numel(calParts));
for p = 1:numel(calParts)
    lblSET_cal{p} = mr.makeLabel('SET', 'SET', calParts(p).id);
end
lblECO_cal = mr.makeLabel('SET', 'ECO', 0);
lblRefOn = mr.makeLabel('SET', 'REF', true);

%% Add FLASH calibration to refscan
% All five calibration SETs are marked REF=true. IMA remains false.
fprintf('Adding FLASH calibration reference acquisition...\n');
rf_phase = 0;
rf_inc = 0;
prevMode = [];
prevI = [];
prevJ = [];
dummyTableIdx = mod((-Ndummy:-1), numel(calAcqTable)) + 1;
tic;

for kk = 1:numel(dummyTableIdx)
    row = calAcqTable(dummyTableIdx(kk));
    mode = row.mode;
    iPhys = row.iPhys;
    jPhys = row.jPhys;

    rf.phaseOffset = rf_phase/180*pi;
    adc.phaseOffset = rf_phase/180*pi;
    rf_inc = mod(rf_inc + rfSpoilingInc, 360.0);
    rf_phase = mod(rf_phase + rf_inc, 360.0);

    if isempty(prevMode)
        seq.addBlock(rf);
    else
        seq.addBlock(rf, groSp, ...
            gpe1PostByMode{prevMode}{prevI}, ...
            gpe2PostByMode{prevMode}{prevJ});
    end
    seq.addBlock(gro1, ...
        gpe1PreByMode{mode}{iPhys}, gpe2PreByMode{mode}{jPhys});
    prevMode = mode;
    prevI = iPhys;
    prevJ = jPhys;
end

for p = 1:numel(calParts)
    if NsettlePerPart > 0
        partRows = calAcqTable(calPartStart(p):calPartStop(p));
        settleIdx = mod((-NsettlePerPart:-1), numel(partRows)) + 1;
        for kk = 1:numel(settleIdx)
            row = partRows(settleIdx(kk));
            mode = row.mode;
            iPhys = row.iPhys;
            jPhys = row.jPhys;

            rf.phaseOffset = rf_phase/180*pi;
            adc.phaseOffset = rf_phase/180*pi;
            rf_inc = mod(rf_inc + rfSpoilingInc, 360.0);
            rf_phase = mod(rf_phase + rf_inc, 360.0);

            if isempty(prevMode)
                seq.addBlock(rf);
            else
                seq.addBlock(rf, groSp, ...
                    gpe1PostByMode{prevMode}{prevI}, ...
                    gpe2PostByMode{prevMode}{prevJ});
            end
            seq.addBlock(gro1, ...
                gpe1PreByMode{mode}{iPhys}, ...
                gpe2PreByMode{mode}{jPhys});
            prevMode = mode;
            prevI = iPhys;
            prevJ = jPhys;
        end
    end

    for kk = calPartStart(p):calPartStop(p)
        row = calAcqTable(kk);
        mode = row.mode;
        iPhys = row.iPhys;
        jPhys = row.jPhys;

        rf.phaseOffset = rf_phase/180*pi;
        adc.phaseOffset = rf_phase/180*pi;
        rf_inc = mod(rf_inc + rfSpoilingInc, 360.0);
        rf_phase = mod(rf_phase + rf_inc, 360.0);

        if isempty(prevMode)
            seq.addBlock(rf);
        else
            seq.addBlock(rf, groSp, ...
                gpe1PostByMode{prevMode}{prevI}, ...
                gpe2PostByMode{prevMode}{prevJ});
        end

        seq.addBlock(adc, gro1, ...
            gpe1PreByMode{mode}{iPhys}, gpe2PreByMode{mode}{jPhys}, ...
            lblPAR_cal{row.iLocal}, lblLIN_cal{row.jLocal}, ...
            lblSET_cal{p}, lblECO_cal, lblRefOn, lblImaOff);

        prevMode = mode;
        prevI = iPhys;
        prevJ = jPhys;
    end
end
seq.addBlock(groSp, ...
    gpe1PostByMode{prevMode}{prevI}, gpe2PostByMode{prevMode}{prevJ});

fprintf('FLASH calibration blocks added in %g seconds.\n', toc);
fprintf('Calibration RF excitations: %d dummy + %d settling + %d acquired.\n', ...
    Ndummy, NsettlePerPart*numel(calParts), numel(calAcqTable));

%% Combined label and TWIX-routing validation
adc_lbl = seq.evalLabels('evolution', 'adc');
assert(isfield(adc_lbl, 'SET'), 'SET label missing from ADC evolution.');
assert(isfield(adc_lbl, 'REF'), 'REF label missing from ADC evolution.');
assert(isfield(adc_lbl, 'IMA'), 'IMA label missing from ADC evolution.');

expectedSET_cal = [calAcqTable.partID]';
expectedPAR_cal = [calAcqTable.iLocal]' - 1;
expectedLIN_cal = [calAcqTable.jLocal]' - 1;

expectedSET_all = [zeros(nMprageAdc,1); expectedSET_cal];
expectedPAR_all = [expectedPAR_img; expectedPAR_cal];
expectedLIN_all = [expectedLIN_img; expectedLIN_cal];
expectedREF_all = [false(nMprageAdc,1); true(numel(calAcqTable),1)];
expectedIMA_all = false(size(expectedREF_all));

assert(numel(adc_lbl.LIN) == numel(expectedLIN_all), ...
    'Unexpected total number of ADC events.');
assert(all(adc_lbl.SET(:) == expectedSET_all), 'Combined SET order mismatch.');
assert(all(adc_lbl.PAR(:) == expectedPAR_all), 'Combined PAR order mismatch.');
assert(all(adc_lbl.LIN(:) == expectedLIN_all), 'Combined LIN order mismatch.');
assert(all(logical(adc_lbl.REF(:)) == expectedREF_all), ...
    'Combined REF routing mismatch.');
assert(all(logical(adc_lbl.IMA(:)) == expectedIMA_all), ...
    'Combined IMA/PATRefAndIma routing mismatch.');

% MPRAGE image checks.
imgRange = 1:nMprageAdc;
imgSET = adc_lbl.SET(imgRange); imgSET = imgSET(:);
imgPAR = adc_lbl.PAR(imgRange); imgPAR = imgPAR(:);
imgLIN = adc_lbl.LIN(imgRange); imgLIN = imgLIN(:);
imgREF = adc_lbl.REF(imgRange); imgREF = imgREF(:);
imgIMA = adc_lbl.IMA(imgRange); imgIMA = imgIMA(:);
assert(all(imgREF == 0), ...
    'MPRAGE contains ADCs marked as refscan.');
assert(all(imgIMA == 0), ...
    'MPRAGE contains ADCs marked PATRefAndIma.');
assert(all(imgSET == 0), ...
    'MPRAGE image ADCs must use SET=0.');
imgLabelPairs = [imgPAR, imgLIN];
imgExpectedPairs0 = imgPairsGlobal - 1;
assert(size(unique(imgLabelPairs, 'rows'),1) == size(imgExpectedPairs0,1), ...
    'MPRAGE contains duplicate image PAR/LIN labels.');
assert(isempty(setdiff(imgExpectedPairs0, imgLabelPairs, 'rows')) ...
    && isempty(setdiff(imgLabelPairs, imgExpectedPairs0, 'rows')), ...
    'MPRAGE labels do not match the requested accelerated image mask.');

% Calibration refscan checks.
calRange = nMprageAdc + (1:numel(calAcqTable));
calSET = adc_lbl.SET(calRange); calSET = calSET(:);
calPAR = adc_lbl.PAR(calRange); calPAR = calPAR(:);
calLIN = adc_lbl.LIN(calRange); calLIN = calLIN(:);
calREF = adc_lbl.REF(calRange); calREF = calREF(:);
calIMA = adc_lbl.IMA(calRange); calIMA = calIMA(:);
assert(all(calREF ~= 0), ...
    'Every calibration ADC must be stored in refscan.');
assert(all(calIMA == 0), ...
    'Calibration must not set PATRefAndIma.');
calTriples = [calSET, calPAR, calLIN];
assert(size(unique(calTriples, 'rows'),1) == numel(calAcqTable), ...
    'Duplicate calibration [SET,PAR,LIN] labels found.');

for p = 1:numel(calParts)
    nThis = sum(calSET == calParts(p).id);
    nExpected = numel(calParts(p).kyList)*numel(calParts(p).kzList);
    assert(nThis == nExpected, ...
        'Unexpected number of calibration ADCs in SET %d.', calParts(p).id);
end

assert(max(calLIN) == Ncalib1-1, ...
    'Refscan LIN extent is not Ncalib1.');
assert(max(calPAR) == Ncalib1-1, ...
    'Refscan PAR extent is not Ncalib1.');
assert(max(calSET) == 4, ...
    'Refscan SET extent is not five sets (0:4).');

acsMask = (calSET == 4);
acsPairs = [calPAR(acsMask), calLIN(acsMask)];
[acsParExpected, acsLinExpected] = ndgrid(0:Nacs-1, 0:Nacs-1);
acsPairsExpected = [acsParExpected(:), acsLinExpected(:)];
assert(size(acsPairs,1) == Nacs*Nacs, ...
    'Calibration ACS SET has the wrong ADC count.');
assert(isempty(setdiff(acsPairsExpected, acsPairs, 'rows')) ...
    && isempty(setdiff(acsPairs, acsPairsExpected, 'rows')), ...
    'ACS is not stored at local PAR/LIN indices 0:(Nacs-1).');

fprintf(['Combined routing validated: MPRAGE image ADCs=%d; ', ...
         'calibration refscan ADCs=%d.\n'], ...
    nMprageAdc, numel(calAcqTable));
fprintf('Expected refscan extent: LIN=%d, PAR=%d, SET=%d. ACS: SET=4, local 0:%d x 0:%d.\n', ...
    Ncalib1, Ncalib1, numel(calParts), Nacs-1, Nacs-1);

if R1 == 1 && R2 == 1
    assert(nMprageAdc == N(ax.n2)*N(ax.n3), ...
        'Full-k-space MPRAGE acquisition-count check failed.');
end

%% Check timing
[ok, error_report] = seq.checkTiming;
if ok
    fprintf('Timing check passed successfully.\n');
else
    fprintf('Timing check failed! Error listing follows:\n');
    fprintf([error_report{:}]);
    fprintf('\n');
end

%% Sequence metadata and output filename
seq.setDefinition('FOV', fov);
seq.setDefinition('SliceThickness', fov(ax.n2)/N(ax.n2));
seq.setDefinition('TR', TRout);
seq.setDefinition('TE', TE);
seq.setDefinition('FlipAngle', alpha);
seq.setDefinition('Nx', N(1));
seq.setDefinition('Ny', N(2));
seq.setDefinition('Nz', N(3));
res_mm = fov(:).' ./ N(:).' * 1e3;
seq.setDefinition('RequestedResolutionX_mm', res(1));
seq.setDefinition('RequestedResolutionY_mm', res(2));
seq.setDefinition('RequestedResolutionZ_mm', res(3));
seq.setDefinition('ResolutionX_mm', res_mm(1));
seq.setDefinition('ResolutionY_mm', res_mm(2));
seq.setDefinition('ResolutionZ_mm', res_mm(3));
seq.setDefinition('ro_os', ro_os);
seq.setDefinition('Nx_os', Nx_os);
seq.setDefinition('OrientationMapping', slOrientation);
seq.setDefinition('ReadoutAxis', ax.d1);
seq.setDefinition('InnerPEAxis', ax.d2);
seq.setDefinition('OuterPEAxis', ax.d3);
seq.setDefinition('ReceiverGainHigh', 1);
seq.setDefinition('ReadoutOversamplingFactor', ro_os);

phaseResolution = fov(ax.n1)/N(ax.n1) / (fov(ax.n3)/N(ax.n3));
seq.setDefinition('kSpaceCenterLine', centerPE2LineIdx-1);
seq.setDefinition('kSpaceCenterPartition', centerPE1LineIdx-1);
seq.setDefinition('PhaseResolution', phaseResolution);

% MPRAGE definitions. There is deliberately no MPRAGE ACS acquisition.
seq.setDefinition('MPRAGE_TI', TI);
seq.setDefinition('MPRAGE_TRout', TRout);
seq.setDefinition('MPRAGE_TRinner', TRinner);
seq.setDefinition('MPRAGE_PE1_R', R1);
seq.setDefinition('MPRAGE_PE2_R', R2);
seq.setDefinition('MPRAGE_PE1_ImgLines', nPE1Img);
seq.setDefinition('MPRAGE_PE2_ImgLines', nPE2Img);
seq.setDefinition('MPRAGE_ImageADCs', nMprageAdc);
seq.setDefinition('MPRAGE_ETL_Target', ETLtarget);
seq.setDefinition('MPRAGE_ETL_CenterSlot0', centerSlotTarget-1);
seq.setDefinition('MPRAGE_ETL_Mode', etlPlan_img.mode);
seq.setDefinition('MPRAGE_ETL_SegLen', etlPlan_img.s);
seq.setDefinition('MPRAGE_ETL_SegmentsPerKy', etlPlan_img.K);
seq.setDefinition('MPRAGE_ETL_SegmentsPerBlock', etlPlan_img.P);
seq.setDefinition('MPRAGE_ETL_FillerPerKy', etlPlan_img.F);
seq.setDefinition('MPRAGE_ETL_Efficiency', etlPlan_img.efficiency);
seq.setDefinition('MPRAGE_ETL_Blocks', numel(imgBlocks));
seq.setDefinition('MPRAGE_ETL_RealSlots', nImgRealSlots);
seq.setDefinition('MPRAGE_ETL_DummySlots', nImgDummySlots);
seq.setDefinition('MPRAGE_UseWaveCos', double(isUseWave_cos));
seq.setDefinition('MPRAGE_UseWaveSin', double(isUseWave_sin));
seq.setDefinition('MPRAGE_HasSeparateACS', 0);

% Calibration/refscan definitions.
seq.setDefinition('Calibration_TRinner', TRinner);
seq.setDefinition('Calibration_TE', TE);
seq.setDefinition('Calibration_Ndummy', Ndummy);
seq.setDefinition('Calibration_NsettlePerPart', NsettlePerPart);
seq.setDefinition('Calibration_Ncalib1', Ncalib1);
seq.setDefinition('Calibration_Ncalib2', Ncalib2);
seq.setDefinition('Calibration_Nacs', Nacs);
seq.setDefinition('Calibration_NParts', numel(calParts));
seq.setDefinition('Calibration_RefscanADCs', numel(calAcqTable));
seq.setDefinition('Calibration_AllSetsInRefscan', 1);
seq.setDefinition('Calibration_RefscanNLin', Ncalib1);
seq.setDefinition('Calibration_RefscanNPar', Ncalib1);
seq.setDefinition('Calibration_RefscanNSets', numel(calParts));
seq.setDefinition('Calibration_ACSSetID', 4);
seq.setDefinition('Calibration_ACSLocalStart0', 0);
seq.setDefinition('Calibration_ACSLocalStop0', Nacs-1);

for p = 1:numel(calParts)
    prefix = ['CalPart' num2str(calParts(p).id) '_'];
    seq.setDefinition([prefix 'Name'], calParts(p).name);
    seq.setDefinition([prefix 'Mode'], modeNames{calParts(p).mode});
    seq.setDefinition([prefix 'SetID'], calParts(p).id);
    seq.setDefinition([prefix 'IsACS'], double(calParts(p).isACS));
    seq.setDefinition([prefix 'NLinLocal'], numel(calParts(p).kyList));
    seq.setDefinition([prefix 'NParLocal'], numel(calParts(p).kzList));
    seq.setDefinition([prefix 'KyPhysStart0'], calParts(p).kyList(1)-1);
    seq.setDefinition([prefix 'KyPhysStop0'], calParts(p).kyList(end)-1);
    seq.setDefinition([prefix 'KzPhysStart0'], calParts(p).kzList(1)-1);
    seq.setDefinition([prefix 'KzPhysStop0'], calParts(p).kzList(end)-1);
end

if isUseWave_sin || isUseWave_cos
    tag_wave = '_wave';
    if isUseWave_sin, tag_wave = [tag_wave '_sin']; end
    if isUseWave_cos, tag_wave = [tag_wave '_cos']; end
else
    tag_wave = '_nowave';
end

tag_res = ['_res', strrep(num2str(res_mm(1), '%.3g'), '.', 'p'), 'x', ...
    strrep(num2str(res_mm(2), '%.3g'), '.', 'p'), 'x', ...
    strrep(num2str(res_mm(3), '%.3g'), '.', 'p'), 'mm'];
tag_etl = ['_ETL' num2str(ETLtarget) '_' etlPlan_img.mode];
seqFilename = ['mprage_3d_flashcalib', tag_wave, '_', ...
    num2str(N(1)), 'x', num2str(N(2)), 'x', num2str(N(3)), ...
    tag_res, tag_etl, '_R1_', num2str(R1), '_R2_', num2str(R2), ...
    '_cal', num2str(Ncalib1), 'x', num2str(Ncalib2), ...
    '_acs', num2str(Nacs), '_os', num2str(ro_os), ...
    tag_wave_details, '_', sys_type];
seq.setDefinition('Name', seqFilename);

%% Write sequence
% Save the sequence before optional PNS/CNS and forbidden-frequency checks.
% If no output path was entered during path setup, out_path is MATLAB's
% current folder at setup time. Generated sequence folders are git-ignored.
outDir_v141 = fullfile(out_path, 'generated_seq_v141');
outDir_v151 = fullfile(out_path, 'generated_seq_v151');
if write_v141_format && ~exist(outDir_v141, 'dir'), mkdir(outDir_v141); end
if ~exist(outDir_v151, 'dir'), mkdir(outDir_v151); end

if write_v141_format
    seqFile_v141 = fullfile(outDir_v141, [seqFilename '_v141.seq']);
    seq.write_v141(seqFile_v141);    % Write to pulseq file (legacy v1.4.1 format)
    fprintf('Write to file (v141): %s\n', seqFile_v141);

    seqFile_v151 = fullfile(outDir_v151, [seqFilename '.seq']);
    seq.write(seqFile_v151);         % Also write current format
    fprintf('Write to file (v151): %s\n', seqFile_v151);
else
    seqFile_v151 = fullfile(outDir_v151, [seqFilename '.seq']);
    seq.write(seqFile_v151);         % Write to pulseq file (current format)
    fprintf('Write to file (v151 only): %s\n', seqFile_v151);
end

%% PNS/CNS check
% mr:restoreShape warnings are off during this optional check by default
% because non-Cartesian waveforms can trigger many restoreShape warnings.
% Comment out the warning('off',...) / warning('on',...) lines if you want
% to show those warnings.
do_pns_check = promptYesNoFromWorkspace('do_pns_check', 'Perform PNS/CNS check?', false);

if do_pns_check
    if isempty(safe_pns_prediction_path) || ~exist(safe_pns_prediction_path, 'dir')
        fprintf('Skipping PNS/CNS check: safe_pns_prediction_path was not provided or is invalid.\n');
    elseif isempty(system_asc_file) || ~exist(system_asc_file, 'file')
        fprintf('Skipping PNS/CNS check: system_asc_file was not provided or is invalid.\n');
    else
        warning('off', 'mr:restoreShape');
        try
            isHasCNS = strcmp(sys_type, 'CimaX') || strcmp(sys_type, 'TerraX');
            doPlots = true;
            [pns,tpns] = seq.calcPNS(system_asc_file, doPlots, 0); %#ok<ASGLU>
            if ~isGEscanner && max(tpns) > 0.95
                warning('PNS=%.2f too high, the sequence may not run on the scanner', max(tpns));
            end
            if isHasCNS
                [pns,tpns] = seq.calcPNS(system_asc_file, doPlots, 1); %#ok<ASGLU>
                if ~isGEscanner && max(tpns) > 0.95
                    warning('CNS=%.2f too high, the sequence may not run on the scanner', max(tpns));
                end
            end
        catch ME
            warning('PNS/CNS check failed: %s', ME.message);
        end
        warning('on', 'mr:restoreShape');
    end
else
    fprintf('Skipping PNS/CNS check by user choice.\n');
end

%% Forbidden-frequency check
% mr:restoreShape warnings are off during this optional check by default
% because non-Cartesian waveforms can trigger many restoreShape warnings.
% Comment out the warning('off',...) / warning('on',...) lines if you want
% to show those warnings.
do_forbidden_frequency_check = promptYesNoFromWorkspace('do_forbidden_frequency_check', ...
    'Perform forbidden-frequency check?', false);

if do_forbidden_frequency_check
    if isGEscanner
        fprintf('Skipping forbidden-frequency check: this helper is configured for Siemens-style .asc files, not GE/premier.\n');
    else
        if isempty(system_asc_file) || ~exist(system_asc_file, 'file')
            system_asc_file = getFileFromWorkspaceOrPrompt('system_asc_file', ...
                'System .asc file path for forbidden-frequency check (press Enter to skip)', true);
            system_asc_file = normalizeUserPath(system_asc_file);
        end

        if isempty(system_asc_file) || ~exist(system_asc_file, 'file')
            fprintf('Skipping forbidden-frequency check: system_asc_file was not provided or is invalid.\n');
        elseif exist('forbiddenFreqCheck', 'file') ~= 2
            fprintf('Skipping forbidden-frequency check: forbiddenFreqCheck.m was not found. Expected it under ./utils/ or on the MATLAB path.\n');
        else
            warning('off', 'mr:restoreShape');
            try
                tic;
                fprintf('Checking forbidden frequencies... ');
                forbiddenFreqCheck(seq, sys, system_asc_file);
                toc;
            catch ME
                warning('Forbidden-frequency check failed: %s', ME.message);
            end
            warning('on', 'mr:restoreShape');
        end
    end
else
    fprintf('Skipping forbidden-frequency check by user choice.\n');
end


%% Optional plotting/reporting
% seq.plot('TimeRange', [0 TRout*2], 'label', 'par,lin,set,ref,ima');
% rep = seq.testReport; fprintf([rep{:}]);

return;
