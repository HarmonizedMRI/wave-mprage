% flash_wave_calibration.m
% Author: Yiyun Dong
% Affiliation: Athinoula A. Martinos Center for Biomedical Imaging
% Date: 2026-06-26
%
% Build based on the Turbo-FLASH part of Maxim's MPRAGE Pulseq demo:
% https://github.com/pulseq-admin/pulseq/blob/master/matlab/demoSeq/writeMPRAGE.m

% Acquisition parts stored compactly using SET as the part dimension:
%   SET 0: no-wave, Ncalib1 center ky lines x Ncalib2 center kz lines
%   SET 1: sin-wave only, same ky/kz block as SET 0
%   SET 2: no-wave, Ncalib2 center ky lines x Ncalib1 center kz lines
%   SET 3: cos-wave only, same ky/kz block as SET 2
%   SET 4: ACS no-wave, Nacs center ky lines x Nacs center kz lines
%
% LIN/PAR labels are compact local indices inside each SET to avoid creating
% a huge sparse Set x Par x Lin array in TWIX/recon loaders. Physical ky/kz
% locations are encoded by the gradients and documented in sequence definitions.
% No AVG dimension is used.

% Do not call clear/clear all here: users may predefine path variables in the
% MATLAB workspace before running this script.
close all; clc
format long

%% Path

% Required unless already defined in the workspace:
%   pulseq_path : Pulseq repository root, or the Pulseq matlab folder.
%   out_path    : target output folder for generated .seq files.
%
% Optional at startup:
%   safe_pns_prediction_path : folder needed by seq.calcPNS/safe PNS helpers.
%   system_asc_file          : scanner .asc file used for PNS/CNS and/or
%                              forbidden-frequency checks. You can leave this
%                              empty and provide it later for the checks.
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end

pulseq_path = getDirectoryFromWorkspaceOrPrompt('pulseq_path', ...
    'Pulseq path (repository root or matlab folder)', false, false);
pulseq_path = normalizeUserPath(pulseq_path);

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

safe_pns_prediction_path = getDirectoryFromWorkspaceOrPrompt('safe_pns_prediction_path', ...
    'safe_pns_prediction path (optional; press Enter to skip)', true, false);
safe_pns_prediction_path = normalizeUserPath(safe_pns_prediction_path);
if ~isempty(safe_pns_prediction_path)
    addpath(safe_pns_prediction_path);
end

% Forbidden-frequency helper code is expected under ./utils/ relative to this
% script. Use fullfile so this works on Windows/macOS/Linux MATLAB.
utils_path = fullfile(script_dir, 'utils');
if exist(utils_path, 'dir')
    addpath(utils_path);
else
    warning('Local utils folder not found: %s. Forbidden-frequency check will be skipped unless forbiddenFreqCheck is already on the MATLAB path.', utils_path);
end

out_path = getDirectoryFromWorkspaceOrPrompt('out_path', ...
    'Target output path for generated .seq files', false, true);
out_path = ensureTrailingFilesep(normalizeUserPath(out_path));

system_asc_file = getFileFromWorkspaceOrPrompt('system_asc_file', ...
    'System .asc file path (optional now; press Enter to skip)', true);
system_asc_file = normalizeUserPath(system_asc_file);

%% Parameters

% Write options:
%   write_v141_format = false  -> use current format (v1.5.x)  [default]
%   write_v141_format = true   -> use legacy format (v1.4.1), required by
%                                 some simulators (MR0, Koma.jl as of 2025)
%   Note: arbitrary gradients with oversampling are incompatible with v1.4.1
write_v141_format = true;

% sequence parameters
alpha    = 7;
ro_dur   = 5120e-6;  % also Tread for wave
ro_os    = 4;        % readout oversampling. With Nx=256, ADC samples = 1024.
ro_spoil = 3;        % inherited readout spoiler setting from MPRAGE/GRE witness
Ndummy   = 300;      % dummy RF/readout TRs before the acquired data, no ADC
NsettlePerPart = 10;  % optional no-ADC TRs before each acquired part using that part's waveform mode

% calibration sizes. Sanity check: Ncalib1 must be larger than Ncalib2.
Ncalib1 = 72;
Ncalib2 = 1;
Nacs    = 32;

% RF spoiling and excitation
rfSpoilingInc = 50;              % RF spoiling increment
rfLen         = 100e-6;
ax            = struct; % encoding axes

% geometry: sagittal convention inherited from mprage_3d_wave.m
% provided geometry must be consistent with wave MPRAGE
slOrientation = 'SAG';  % 'SAG', 'COR', 'TRA' (To-Do: Support 'Cor')
if strcmp(slOrientation, 'SAG')
    fov = [192 256 256]*1e-3;         % [x y z] FOV in meters
    N   = [192 256 256];              % [x y z] matrix sizes
    ax.d1 = 'z';                      % readout axis; N(ax.n1)=256 -> Nx_os=1024 with ro_os=4
    ax.d2 = 'x';                      % inner PE / kz-like calibration dimension / PAR label
elseif strcmp(slOrientation, 'TRA')
    fov = [220 220 256]*1e-3;         % [x y z] FOV in meters
    N   = [256 256 72];              % [x y z] matrix sizes
    ax.d1 = 'x';                      % readout axis; N(ax.n1)=256 -> Nx_os=1024 with ro_os=4
    ax.d2 = 'z';                      % inner PE / kz-like calibration dimension / PAR label
else
    return;
end
ax.d3 = setdiff('xyz',[ax.d1 ax.d2]); % outer PE / ky-like calibration dimension / LIN label
ax.n1 = strfind('xyz',ax.d1);
ax.n2 = strfind('xyz',ax.d2);
ax.n3 = strfind('xyz',ax.d3);

% wave parameters (must be identical to wave MPRAGE)
% In SAG mode, the existing code convention is:
%   cos wave -> PE1/PAR axis gpe1, used for the kz-wide block
%   sin wave -> PE2/LIN axis gpe2, used for the ky-wide block
% In COR mode, the helper choice follows the original mprage_3d_wave.m swap.
gwave_max = 8;   % mT/m
swave_max = 200;  % T/m/s
Ncycles   = 10;
tag_wave_details = ['_amp' num2str(gwave_max) '_cycles' num2str(Ncycles) '_' slOrientation];

% input sanity checks
assert(ro_os == round(ro_os) && ro_os >= 1, 'ro_os must be a positive integer.');
assert(Ncalib1 == round(Ncalib1) && Ncalib1 > 0, 'Ncalib1 must be a positive integer.');
assert(Ncalib2 == round(Ncalib2) && Ncalib2 > 0, 'Ncalib2 must be a positive integer.');
assert(Nacs    == round(Nacs)    && Nacs    > 0, 'Nacs must be a positive integer.');
assert(Ncalib1 > Ncalib2, 'Sanity check failed: require Ncalib1 > Ncalib2.');
assert(Ncalib1 <= N(ax.n2) && Ncalib1 <= N(ax.n3), 'Ncalib1 exceeds one of the PE dimensions.');
assert(Ncalib2 <= N(ax.n2) && Ncalib2 <= N(ax.n3), 'Ncalib2 exceeds one of the PE dimensions.');
assert(Nacs    <= N(ax.n2) && Nacs    <= N(ax.n3), 'Nacs exceeds one of the PE dimensions.');
assert(Ndummy >= 0 && Ndummy == round(Ndummy), 'Ndummy must be a nonnegative integer.');
assert(NsettlePerPart >= 0 && NsettlePerPart == round(NsettlePerPart), 'NsettlePerPart must be a nonnegative integer.');

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
else % Siemens
    rfDeadTime =  100e-6;
    rfRingdownTime = 100e-6;
    adcDeadTime = 20e-6;
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

% Create a new sequence object
seq = mr.Sequence(sys);

%% Setup: RF, ADC, readout, prewinders

rf = mr.makeBlockPulse(alpha*pi/180,sys,'Duration',rfLen, 'SliceThickness', fov(ax.n2), 'use', 'excitation');

% Define gradients and ADC events
deltak=1./fov;

% readout sanity check: N(ax.n1)=256 and ro_os=4 -> 1024 ADC points.
dwell = round((ro_dur / N(ax.n1) / ro_os) / sys.adcRasterTime) * sys.adcRasterTime;
Tread = dwell * N(ax.n1) * ro_os;
Nx_os = N(ax.n1) * ro_os;
fprintf('Readout dimension N(ax.n1) = %d, ro_os = %d, ADC samples Nx_os = %d\n', N(ax.n1), ro_os, Nx_os);
fprintf('RO duration input = %.6f ms, rasterized Tread = %.6f ms, dwell = %.6f us\n', ro_dur*1e3, Tread*1e3, dwell*1e6);

gro = mr.makeTrapezoid(ax.d1,'Amplitude',N(ax.n1)*deltak(ax.n1)/ro_dur,'FlatTime',ceil((ro_dur+sys.adcDeadTime)/sys.gradRasterTime)*sys.gradRasterTime,'system',sys);
adc = mr.makeAdc(Nx_os,'Duration',ro_dur,'Delay',gro.riseTime,'system',sys);
assert(adc.numSamples == Nx_os, 'ADC sample count mismatch.');

groPre = mr.makeTrapezoid(ax.d1,'Area',-gro.amplitude*(adc.dwell*(adc.numSamples/2+0.5)+0.5*gro.riseTime),'system',sys_lowPNS); % Siemens sample-center convention
gpe1 = mr.makeTrapezoid(ax.d2,'Area',-deltak(ax.n2)*(N(ax.n2)/2),'system',sys_lowPNS); % maximum PE1/PAR gradient
gpe2 = mr.makeTrapezoid(ax.d3,'Area',-deltak(ax.n3)*(N(ax.n3)/2),'system',sys_lowPNS); % maximum PE2/LIN gradient

[gro1,groSp]=mr.splitGradientAt(gro,gro.riseTime+gro.flatTime);
if ro_spoil>0
    groSp=mr.makeExtendedTrapezoidArea(gro.channel,gro.amplitude,0,deltak(ax.n1)/2*N(ax.n1)*ro_spoil,sys_lowPNS);
end

% Prewinder duration matching
rf.delay = mr.calcDuration(groSp,gpe1,gpe2);
gPre_dur = max([mr.calcDuration(groPre), mr.calcDuration(gpe1), mr.calcDuration(gpe2)]);
gPre_dur = ceil(gPre_dur/sys.gradRasterTime)*sys.gradRasterTime;
groPre   = mr.makeTrapezoid(ax.d1, 'Area', groPre.area, 'Duration', gPre_dur, 'system', sys_lowPNS);
gpe1Pre  = mr.makeTrapezoid(ax.d2, 'Area', gpe1.area, 'Duration', gPre_dur, 'system', sys_lowPNS);
gpe2Pre  = mr.makeTrapezoid(ax.d3, 'Area', gpe2.area, 'Duration', gPre_dur, 'system', sys_lowPNS);

gro1.delay=mr.calcDuration(groPre);
adc.delay=gro1.delay+gro.riseTime;
gro1=mr.addGradients({gro1,groPre},'system',sys);

% PE steps -- physical k-space encoding order
pe1Steps=((0:N(ax.n2)-1)-N(ax.n2)/2)/N(ax.n2)*2;
pe2Steps=((0:N(ax.n3)-1)-N(ax.n3)/2)/N(ax.n3)*2;

%% Precompute no-wave, sin-wave, and cos-wave PE gradients

% Mode IDs used in acquisition table
MODE_NOWAVE = 1;
MODE_SIN    = 2;
MODE_COS    = 3;
modeNames = {'nowave', 'sin', 'cos'};

gpe1Pre_nowave  = cell(1, N(ax.n2));
gpe1Post_nowave = cell(1, N(ax.n2));
gpe1Pre_sin     = cell(1, N(ax.n2));
gpe1Post_sin    = cell(1, N(ax.n2));
gpe1Pre_cos     = cell(1, N(ax.n2));
gpe1Post_cos    = cell(1, N(ax.n2));

gpe2Pre_nowave  = cell(1, N(ax.n3));
gpe2Post_nowave = cell(1, N(ax.n3));
gpe2Pre_sin     = cell(1, N(ax.n3));
gpe2Post_sin    = cell(1, N(ax.n3));
gpe2Pre_cos     = cell(1, N(ax.n3));
gpe2Post_cos    = cell(1, N(ax.n3));

allPostDur = [];

% PE1/PAR axis: no-wave always; cos wave in SAG, sin wave in COR following the original code convention.
for i = 1:N(ax.n2)
    gpe1Pre_i = mr.scaleGrad(gpe1Pre, pe1Steps(i));

    gpe1Pre_nowave{i}  = gpe1Pre_i;
    gpe1Post_nowave{i} = mr.scaleGrad(gpe1, -pe1Steps(i));
    gpe1Pre_nowave{i}.id  = seq.registerGradEvent(gpe1Pre_nowave{i});
    gpe1Post_nowave{i}.id = seq.registerGradEvent(gpe1Post_nowave{i});

    % Default sin/cos copies are no-wave unless this axis carries that wave mode.
    gpe1Pre_sin{i}  = gpe1Pre_nowave{i};
    gpe1Post_sin{i} = gpe1Post_nowave{i};
    gpe1Pre_cos{i}  = gpe1Pre_nowave{i};
    gpe1Post_cos{i} = gpe1Post_nowave{i};

    if strcmp(slOrientation, 'SAG')
        debugFlag = (i == 1);
        [gpe1Pre_cos{i}, gpe1Post_cos{i}] = defineCosineWaveGradient(Tread, sys, sys_lowPNS, Ncycles, gwave_max, swave_max, gpe1Pre_i, gro, adc, physical_slew_max, debugFlag, debugFlag);
        gpe1Pre_cos{i}.id  = seq.registerGradEvent(gpe1Pre_cos{i});
        gpe1Post_cos{i}.id = seq.registerGradEvent(gpe1Post_cos{i});
    elseif strcmp(slOrientation, 'TRA')
        debugFlag = (i == 1);
        [gpe1Pre_cos{i}, gpe1Post_cos{i}] = defineCosineWaveGradient(Tread, sys, sys_lowPNS, Ncycles, gwave_max, swave_max, gpe1Pre_i, gro, adc, physical_slew_max, debugFlag, debugFlag);
        gpe1Pre_cos{i}.id  = seq.registerGradEvent(gpe1Pre_cos{i});
        gpe1Post_cos{i}.id = seq.registerGradEvent(gpe1Post_cos{i});
    end

    allPostDur = [allPostDur, mr.calcDuration(gpe1Post_nowave{i}), mr.calcDuration(gpe1Post_sin{i}), mr.calcDuration(gpe1Post_cos{i})]; %#ok<AGROW>
end

% PE2/LIN axis: no-wave always; sin wave in SAG, cos-like helper in COR following the original code convention.
for j = 1:N(ax.n3)
    gpe2Pre_j = mr.scaleGrad(gpe2Pre, pe2Steps(j));

    gpe2Pre_nowave{j}  = gpe2Pre_j;
    gpe2Post_nowave{j} = mr.scaleGrad(gpe2, -pe2Steps(j));
    gpe2Pre_nowave{j}.id  = seq.registerGradEvent(gpe2Pre_nowave{j});
    gpe2Post_nowave{j}.id = seq.registerGradEvent(gpe2Post_nowave{j});

    % Default sin/cos copies are no-wave unless this axis carries that wave mode.
    gpe2Pre_sin{j}  = gpe2Pre_nowave{j};
    gpe2Post_sin{j} = gpe2Post_nowave{j};
    gpe2Pre_cos{j}  = gpe2Pre_nowave{j};
    gpe2Post_cos{j} = gpe2Post_nowave{j};

    if strcmp(slOrientation, 'SAG')
        debugFlag = (j == 1);
        [gpe2Pre_sin{j}, gpe2Post_sin{j}] = defineSineWaveGradient(Tread, sys, sys_lowPNS, Ncycles, gwave_max, swave_max, gpe2Pre_j, gro, adc, physical_slew_max, debugFlag, debugFlag);
        gpe2Pre_sin{j}.id  = seq.registerGradEvent(gpe2Pre_sin{j});
        gpe2Post_sin{j}.id = seq.registerGradEvent(gpe2Post_sin{j});
    elseif strcmp(slOrientation, 'TRA')
        debugFlag = (j == 1);
        [gpe2Pre_sin{j}, gpe2Post_sin{j}] = defineSineWaveGradient(Tread, sys, sys_lowPNS, Ncycles, gwave_max, swave_max, gpe2Pre_j, gro, adc, physical_slew_max, debugFlag, debugFlag);
        gpe2Pre_sin{j}.id  = seq.registerGradEvent(gpe2Pre_sin{j});
        gpe2Post_sin{j}.id = seq.registerGradEvent(gpe2Post_sin{j});
    end

    allPostDur = [allPostDur, mr.calcDuration(gpe2Post_nowave{j}), mr.calcDuration(gpe2Post_sin{j}), mr.calcDuration(gpe2Post_cos{j})]; %#ok<AGROW>
end

gpe1PreByMode  = {gpe1Pre_nowave,  gpe1Pre_sin,  gpe1Pre_cos};
gpe1PostByMode = {gpe1Post_nowave, gpe1Post_sin, gpe1Post_cos};
gpe2PreByMode  = {gpe2Pre_nowave,  gpe2Pre_sin,  gpe2Pre_cos};
gpe2PostByMode = {gpe2Post_nowave, gpe2Post_sin, gpe2Post_cos};

% Use one global RF delay and one global TR for all modes and part transitions.
rf.delay = max([mr.calcDuration(groSp), allPostDur]);
TRinner = mr.calcDuration(rf)+mr.calcDuration(gro1);
TE = mr.calcDuration(rf) - (rf.delay + mr.calcRfCenter(rf)) + adc.delay + adc.dwell*(adc.numSamples/2+0.5);
ESP = TRinner;

fprintf('Merged GRE timing: TRinner = %.6f ms, TE_nominal = %.6f ms, ESP = %.6f ms\n', TRinner*1e3, TE*1e3, ESP*1e3);
fprintf('Ndummy = %d, NsettlePerPart = %d\n', Ndummy, NsettlePerPart);

% pre-register objects that do not change while looping
groSp.id=seq.registerGradEvent(groSp);
gro1.id=seq.registerGradEvent(gro1);
[~, rf.shapeIDs]=seq.registerRfEvent(rf); % RF phase changes dynamically; only shapes are registered

%% Build compact multi-part acquisition table

ky_calib1 = centerBlockIndices(N(ax.n3), Ncalib1);
ky_calib2 = centerBlockIndices(N(ax.n3), Ncalib2);
ky_acs    = centerBlockIndices(N(ax.n3), Nacs);

kz_calib1 = centerBlockIndices(N(ax.n2), Ncalib1);
kz_calib2 = centerBlockIndices(N(ax.n2), Ncalib2);
kz_acs    = centerBlockIndices(N(ax.n2), Nacs);

parts = struct('id', {}, 'name', {}, 'mode', {}, 'kyList', {}, 'kzList', {}, 'isACS', {});
parts(1).id = 0; parts(1).name = 'nowave_kywide_kznarrow'; parts(1).mode = MODE_NOWAVE; parts(1).kyList = ky_calib1; parts(1).kzList = kz_calib2; parts(1).isACS = false;
parts(2).id = 1; parts(2).name = 'sin_kywide_kznarrow';    parts(2).mode = MODE_SIN;    parts(2).kyList = ky_calib1; parts(2).kzList = kz_calib2; parts(2).isACS = false;
parts(3).id = 2; parts(3).name = 'nowave_kzwide_kynarrow'; parts(3).mode = MODE_NOWAVE; parts(3).kyList = ky_calib2; parts(3).kzList = kz_calib1; parts(3).isACS = false;
parts(4).id = 3; parts(4).name = 'cos_kzwide_kynarrow';    parts(4).mode = MODE_COS;    parts(4).kyList = ky_calib2; parts(4).kzList = kz_calib1; parts(4).isACS = false;
parts(5).id = 4; parts(5).name = 'acs_nowave_center';      parts(5).mode = MODE_NOWAVE; parts(5).kyList = ky_acs;    parts(5).kzList = kz_acs;    parts(5).isACS = true;

% Build table as scalar struct array. Gradients use physical indices iPhys/jPhys.
% Labels use compact local indices iLocal/jLocal inside each SET.
acqTable = struct('partArrayIdx', {}, 'partID', {}, 'mode', {}, 'isACS', {}, ...
                  'iPhys', {}, 'jPhys', {}, 'iLocal', {}, 'jLocal', {});
partStart = zeros(1, numel(parts));
partStop  = zeros(1, numel(parts));
for p = 1:numel(parts)
    partStart(p) = numel(acqTable) + 1;
    kyList = parts(p).kyList;
    kzList = parts(p).kzList;
    for jLocal = 1:numel(kyList)
        jPhys = kyList(jLocal);
        for iLocal = 1:numel(kzList)
            iPhys = kzList(iLocal);
            row.partArrayIdx = p;
            row.partID = parts(p).id;
            row.mode = parts(p).mode;
            row.isACS = parts(p).isACS;
            row.iPhys = iPhys;
            row.jPhys = jPhys;
            row.iLocal = iLocal;
            row.jLocal = jLocal;
            acqTable(end+1) = row; %#ok<SAGROW>
        end
    end
    partStop(p) = numel(acqTable);
end

nAcqExpected = 4*Ncalib1*Ncalib2 + Nacs*Nacs;
assert(numel(acqTable) == nAcqExpected, 'Acquisition table length mismatch.');
fprintf('Merged calibration acquired ADC count: %d\n', numel(acqTable));
for p = 1:numel(parts)
    fprintf('  SET %d: %-24s mode=%s, LIN=%d, PAR=%d, ADCs=%d\n', ...
        parts(p).id, parts(p).name, modeNames{parts(p).mode}, numel(parts(p).kyList), numel(parts(p).kzList), numel(parts(p).kyList)*numel(parts(p).kzList));
end

%% Labels: compact local LIN/PAR + SET part ID

maxLocalLin = max(arrayfun(@(p) numel(p.kyList), parts));
maxLocalPar = max(arrayfun(@(p) numel(p.kzList), parts));

lblLIN = cell(1, maxLocalLin);
for iY = 1:maxLocalLin
    lblLIN{iY} = mr.makeLabel('SET', 'LIN', iY - 1);
end

lblPAR = cell(1, maxLocalPar);
for iZ = 1:maxLocalPar
    lblPAR{iZ} = mr.makeLabel('SET', 'PAR', iZ - 1);
end

lblSET = cell(1, numel(parts));
for p = 1:numel(parts)
    lblSET{p} = mr.makeLabel('SET', 'SET', parts(p).id);
end

lblECO = mr.makeLabel('SET', 'ECO', 0);
lblRefOn  = mr.makeLabel('SET', 'REF', true);
lblRefOff = mr.makeLabel('SET', 'REF', false);

%% Sequence loop: continuous RF-spoiled GRE train

rf_phase = 0;
rf_inc = 0;
prevMode = [];
prevI = [];
prevJ = [];

% Dummy scans use the same acq table cyclically immediately preceding the
% first acquired line. No ADC, no labels.
dummyTableIdx = mod((-Ndummy:-1), numel(acqTable)) + 1;

tic;

for kk = 1:numel(dummyTableIdx)
    row = acqTable(dummyTableIdx(kk));
    mode = row.mode;
    iPhys = row.iPhys;
    jPhys = row.jPhys;

    rf.phaseOffset  = rf_phase/180*pi;
    adc.phaseOffset = rf_phase/180*pi;
    rf_inc   = mod(rf_inc + rfSpoilingInc, 360.0);
    rf_phase = mod(rf_phase + rf_inc, 360.0);

    if isempty(prevMode)
        seq.addBlock(rf);
    else
        seq.addBlock(rf, groSp, gpe1PostByMode{prevMode}{prevI}, gpe2PostByMode{prevMode}{prevJ});
    end
    seq.addBlock(gro1, gpe1PreByMode{mode}{iPhys}, gpe2PreByMode{mode}{jPhys});

    prevMode = mode;
    prevI = iPhys;
    prevJ = jPhys;
end

for p = 1:numel(parts)
    % Optional settling TRs using this part's waveform mode and PE trajectory.
    if NsettlePerPart > 0
        partRows = acqTable(partStart(p):partStop(p));
        settleIdx = mod((-NsettlePerPart:-1), numel(partRows)) + 1;
        for kk = 1:numel(settleIdx)
            row = partRows(settleIdx(kk));
            mode = row.mode;
            iPhys = row.iPhys;
            jPhys = row.jPhys;

            rf.phaseOffset  = rf_phase/180*pi;
            adc.phaseOffset = rf_phase/180*pi;
            rf_inc   = mod(rf_inc + rfSpoilingInc, 360.0);
            rf_phase = mod(rf_phase + rf_inc, 360.0);

            if isempty(prevMode)
                seq.addBlock(rf);
            else
                seq.addBlock(rf, groSp, gpe1PostByMode{prevMode}{prevI}, gpe2PostByMode{prevMode}{prevJ});
            end
            seq.addBlock(gro1, gpe1PreByMode{mode}{iPhys}, gpe2PreByMode{mode}{jPhys});

            prevMode = mode;
            prevI = iPhys;
            prevJ = jPhys;
        end
    end

    % Acquired rows for this part.
    for kk = partStart(p):partStop(p)
        row = acqTable(kk);
        mode = row.mode;
        iPhys = row.iPhys;
        jPhys = row.jPhys;

        rf.phaseOffset  = rf_phase/180*pi;
        adc.phaseOffset = rf_phase/180*pi;
        rf_inc   = mod(rf_inc + rfSpoilingInc, 360.0);
        rf_phase = mod(rf_phase + rf_inc, 360.0);

        if isempty(prevMode)
            seq.addBlock(rf);
        else
            seq.addBlock(rf, groSp, gpe1PostByMode{prevMode}{prevI}, gpe2PostByMode{prevMode}{prevJ});
        end

        if row.isACS
            refLabel = lblRefOn;
        else
            refLabel = lblRefOff;
        end

        seq.addBlock(adc, gro1, gpe1PreByMode{mode}{iPhys}, gpe2PreByMode{mode}{jPhys}, ...
            lblPAR{row.iLocal}, lblLIN{row.jLocal}, lblSET{p}, lblECO, refLabel);

        prevMode = mode;
        prevI = iPhys;
        prevJ = jPhys;
    end
end

% Complete the final readout's spoiler/rewinder. No extra delay is added.
seq.addBlock(groSp, gpe1PostByMode{prevMode}{prevI}, gpe2PostByMode{prevMode}{prevJ});

fprintf('Sequence ready (blocks generation took %g seconds)\n', toc);
fprintf('Total RF excitations: %d dummy + %d settling + %d acquired = %d\n', ...
    Ndummy, NsettlePerPart*numel(parts), numel(acqTable), Ndummy + NsettlePerPart*numel(parts) + numel(acqTable));

%% Label validation

adc_lbl = seq.evalLabels('evolution','adc');

assert(numel(adc_lbl.LIN) == numel(acqTable), 'Unexpected number of acquired ADC events.');
assert(isfield(adc_lbl, 'SET'), 'SET label was not found in evalLabels output.');

expectedSET = [acqTable.partID]';
expectedPAR = [acqTable.iLocal]' - 1;
expectedLIN = [acqTable.jLocal]' - 1;
expectedREF = [acqTable.isACS]';

assert(all(adc_lbl.SET(:) == expectedSET), 'SET order mismatch.');
assert(all(adc_lbl.PAR(:) == expectedPAR), 'Compact PAR label order mismatch.');
assert(all(adc_lbl.LIN(:) == expectedLIN), 'Compact LIN label order mismatch.');
assert(all(logical(adc_lbl.REF(:)) == expectedREF), 'REF label mismatch.');

fprintf('SET range: %d ... %d\n', min(adc_lbl.SET), max(adc_lbl.SET));
fprintf('Compact LIN range: %d ... %d\n', min(adc_lbl.LIN), max(adc_lbl.LIN));
fprintf('Compact PAR range: %d ... %d\n', min(adc_lbl.PAR), max(adc_lbl.PAR));

uniqueTriples = unique([adc_lbl.SET(:), adc_lbl.PAR(:), adc_lbl.LIN(:)], 'rows');
assert(size(uniqueTriples, 1) == numel(acqTable), ...
    'Duplicate compact [SET, PAR, LIN] labels were found.');

for p = 1:numel(parts)
    nThis = sum(adc_lbl.SET(:) == parts(p).id);
    nExpected = numel(parts(p).kyList) * numel(parts(p).kzList);
    assert(nThis == nExpected, 'Unexpected number of ADCs for SET %d.', parts(p).id);
end

%% check whether the timing of the sequence is correct
[ok, error_report]=seq.checkTiming;

if (ok)
    fprintf('Timing check passed successfully\n');
else
    fprintf('Timing check failed! Error listing follows:\n');
    fprintf([error_report{:}]);
    fprintf('\n');
end

%% Define sequence metadata and output filename
% These definitions are written before safety/frequency checks so the saved
% .seq file is available even when optional checks are skipped or fail.
seq.setDefinition('FOV', fov);
seq.setDefinition('SliceThickness', fov(ax.n2) / N(ax.n2));
seq.setDefinition('TR', TRinner);
seq.setDefinition('TE', TE);
seq.setDefinition('FlipAngle', alpha);
seq.setDefinition('Nx', N(1));
seq.setDefinition('Ny', N(2));
seq.setDefinition('Nz', N(3));
seq.setDefinition('ro_os', ro_os);
seq.setDefinition('Nx_os', Nx_os);
seq.setDefinition('Ndummy', Ndummy);
seq.setDefinition('NsettlePerPart', NsettlePerPart);
seq.setDefinition('Ncalib1', Ncalib1);
seq.setDefinition('Ncalib2', Ncalib2);
seq.setDefinition('Nacs', Nacs);
seq.setDefinition('NParts', numel(parts));
seq.setDefinition('OrientationMapping', slOrientation);
seq.setDefinition('ReceiverGainHigh',1);
seq.setDefinition('ReadoutOversamplingFactor', ro_os);

% Definitions for compact local-label -> physical-k-space mapping.
% Physical indices are stored as zero-based start/stop values.
for p = 1:numel(parts)
    prefix = ['Part' num2str(parts(p).id) '_'];
    seq.setDefinition([prefix 'Name'], parts(p).name);
    seq.setDefinition([prefix 'Mode'], modeNames{parts(p).mode});
    seq.setDefinition([prefix 'SetID'], parts(p).id);
    seq.setDefinition([prefix 'IsACS'], double(parts(p).isACS));
    seq.setDefinition([prefix 'NLinLocal'], numel(parts(p).kyList));
    seq.setDefinition([prefix 'NParLocal'], numel(parts(p).kzList));
    seq.setDefinition([prefix 'KyPhysStart0'], parts(p).kyList(1)-1);
    seq.setDefinition([prefix 'KyPhysStop0'],  parts(p).kyList(end)-1);
    seq.setDefinition([prefix 'KzPhysStart0'], parts(p).kzList(1)-1);
    seq.setDefinition([prefix 'KzPhysStop0'],  parts(p).kzList(end)-1);
end

seqFilename = ['gre_witness_merged_calib_', num2str(N(1)), 'x', num2str(N(2)), 'x', num2str(N(3)), ...
    '_Nxos', num2str(Nx_os), '_Ncalib', num2str(Ncalib1), 'x', num2str(Ncalib2), ...
    '_Nacs', num2str(Nacs), '_Ndummy', num2str(Ndummy), '_Nsettle', num2str(NsettlePerPart), ...
    '_os', num2str(ro_os), tag_wave_details, '_', sys_type];
seq.setDefinition('Name', seqFilename);

%% Write sequence
% Save the sequence before optional PNS/CNS and forbidden-frequency checks.
if ~exist(fullfile(out_path, 'v141'), 'dir'), mkdir(fullfile(out_path, 'v141')); end
if ~exist(fullfile(out_path, 'v151'), 'dir'), mkdir(fullfile(out_path, 'v151')); end

if write_v141_format
    seqFile_v141 = fullfile(out_path, 'v141', [seqFilename '_v141.seq']);
    seq.write_v141(seqFile_v141);    % Write to pulseq file (legacy v1.4.1 format)
    fprintf('Write to file (v141): %s\n', seqFile_v141);

    seqFile_v151 = fullfile(out_path, 'v151', [seqFilename '.seq']);
    seq.write(seqFile_v151);         % Also write current format
    fprintf('Write to file (v151): %s\n', seqFile_v151);
else
    seqFile_v151 = fullfile(out_path, 'v151', [seqFilename '.seq']);
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

%% plot, etc
% seq.plot('TimeRange',[0 TRout*2], 'label', 'par,lin');
% seq.plot('TimeRange',[0 TRout*2], 'stacked',1);

return;

%% Local helper for center index blocks

function idx = centerBlockIndices(Ndim, nCenter)
    %CENTERBLOCKINDICES Return 1-based contiguous center indices.
    % For even nCenter, this matches the common ACS convention:
    % centerIdx-n/2 : centerIdx+n/2-1 with centerIdx=floor(N/2)+1.
    centerIdx = floor(Ndim/2) + 1;
    if mod(nCenter, 2) == 0
        idx = (centerIdx - nCenter/2) : (centerIdx + nCenter/2 - 1);
    else
        idx = (centerIdx - floor(nCenter/2)) : (centerIdx + floor(nCenter/2));
    end
    assert(idx(1) >= 1 && idx(end) <= Ndim, 'Requested center block exceeds dimension.');
end

%% Define helper function for make extended trapezoid gradient with fixed area, endpoints and duration

function [gPreRamp, preRampWave] = makeFixedDurationPreRamp(channel, A_target, G_end, T_total, sys)
    %MAKEFIXEDDURATIONPRERAMP Fixed-duration PE prephaser/rampup gradient.
    %
    % Creates an extended trapezoid that:
    %   starts at 0
    %   ends at G_end
    %   has total area A_target
    %   has duration T_total
    %
    % Units:
    %   A_target : Hz/m * s
    %   G_end    : Hz/m
    %   T_total  : s
    %
    % Uses:
    %   sys.maxGrad : Hz/m
    %   sys.maxSlew : Hz/m/s

    dt = sys.gradRasterTime;

    T_total = round(T_total/dt) * dt;

    G_limit = sys.maxGrad;
    S_limit = sys.maxSlew;

    if T_total <= 2*dt
        error('T_total is too short for a fixed-duration pre-ramp gradient.');
    end

    % Try a simple 3-point waveform:
    %
    %   t: 0, T/2, T
    %   G: 0, G_mid, G_end
    %
    % Area:
    %   A = 0.5*T_mid*G_mid ...
    %     + 0.5*(T_total - T_mid)*(G_mid + G_end)

    T_mid = round((T_total/2)/dt) * dt;

    if T_mid > 0 && T_mid < T_total

        G_mid = ( ...
            A_target ...
            - 0.5*(T_total - T_mid)*G_end ...
            ) / (0.5*T_mid + 0.5*(T_total - T_mid));

        slew_1 = abs(G_mid) / T_mid;
        slew_2 = abs(G_end - G_mid) / (T_total - T_mid);

        grad_ok = max(abs([0, G_mid, G_end])) <= G_limit;
        slew_ok = max([slew_1, slew_2]) <= S_limit;

        if grad_ok && slew_ok
            times = [0, T_mid, T_total];
            amps  = [0, G_mid, G_end];

            [gPreRamp, preRampWave] = makeExtendedTrapezoidAndWaveform( ...
                channel, times, amps, T_total, sys);
            return;
        end
    end

    % If 3-point fails, try a 4-point waveform:
    %
    %   t: 0, r, T-r, T
    %   G: 0, G_flat, G_flat, G_end
    %
    % Area:
    %   A = 0.5*r*G_flat ...
    %     + (T_total - 2*r)*G_flat ...
    %     + 0.5*r*(G_flat + G_end)
    %
    % Simplified:
    %   A = (T_total - r)*G_flat + 0.5*r*G_end
    %
    % Therefore:
    %   G_flat = (A - 0.5*r*G_end)/(T_total - r)

    best = [];
    bestScore = inf;

    max_r_index = floor((T_total/dt)/2) - 1;

    for ir = 1:max_r_index

        r = ir * dt;

        if 2*r >= T_total
            continue;
        end

        G_flat = (A_target - 0.5*r*G_end) / (T_total - r);

        slew_1 = abs(G_flat) / r;
        slew_2 = abs(G_end - G_flat) / r;

        grad_peak = max(abs([0, G_flat, G_end]));
        slew_peak = max([slew_1, slew_2]);

        grad_ok = grad_peak <= G_limit;
        slew_ok = slew_peak <= S_limit;

        if grad_ok && slew_ok

            % Prefer the mildest slew solution
            score = slew_peak;

            if score < bestScore
                bestScore = score;
                best.r = r;
                best.G_flat = G_flat;
                best.slew_peak = slew_peak;
                best.grad_peak = grad_peak;
            end
        end
    end

    if isempty(best)
        error(['Could not design fixed-duration pre-ramp gradient. ', ...
               'Try reducing gwave_max or increasing prephaser duration.']);
    end

    r = best.r;
    G_flat = best.G_flat;

    times = [0, r, T_total-r, T_total];
    amps  = [0, G_flat, G_flat, G_end];

    [gPreRamp, preRampWave] = makeExtendedTrapezoidAndWaveform( ...
        channel, times, amps, T_total, sys);

end

function [gExt, wave] = makeExtendedTrapezoidAndWaveform( ...
    channel, times, amps, T_total, sys)

    dt = sys.gradRasterTime;

    % Make the actual extended trapezoid object
    gExt = mr.makeExtendedTrapezoid(channel, ...
        'times', times, ...
        'amplitudes', amps, ...
        'system', sys);

    % Convert the same edge-defined waveform to Pulseq center-sampled waveform
    n = round(T_total / dt);

    tCenters = ((0:n-1) + 0.5) * dt;

    wave = interp1(times, amps, tCenters, 'linear');

    % Force row vector
    wave = wave(:).';

end


%% Define cosine wave

function [gpe_wave_cos, gpe_post] = defineCosineWaveGradient(Tread, sys, sys_lowPNS, Ncycles, gwave_max, swave_max, gpePre, gro, adc, physical_slew_max, waveInfoFlag, debugFlag)
    %DEFINECOSINEWAVEGRADIENT Create PE cosine wave gradient with pre/post compensation.
    %
    % Outputs
    %   gpe_wave_cos : PE prephaser + rampup + cosine wave gradient
    %   gpe_post     : PE rampdown/post-rewinder gradient
    %
    % Inputs
    %   Tread      : cosine wave duration, usually ADC/readout duration [s]
    %   sys        : Pulseq system struct
    %   Ncycles    : number of cosine cycles during Tread
    %   gwave_max  : max wave gradient amplitude [mT/m]
    %   swave_max  : max wave slew rate [T/m/s]
    %   gpePre     : nominal PE prephaser gradient
    %   gro        : readout gradient; gro.riseTime is used for rampup
    if nargin < 10
        debugFlag = false;
        waveInfoFlag = false;
    end

    % Basic checks
    if Ncycles <= 0
        error('Ncycles must be positive.');
    end

    if Tread <= 0
        error('Tread must be positive.');
    end

    % Get PE channel and prephaser duration from gpePre
    gpe_channel = gpePre.channel;
    gpePre_dur  = mr.calcDuration(gpePre);

    % Design cosine wave
    wavepoints_cos = round(Tread / sys.gradRasterTime);
    tWaveUnit_cos  = sys.gradRasterTime;

    TreadRaster = wavepoints_cos * tWaveUnit_cos;

    tWavePeriod_cos = TreadRaster / Ncycles;
    w_cos = 2*pi / tWavePeriod_cos;
    if waveInfoFlag
        fprintf('w_cos: %.6f rad/s, tWavePeriod_cos: %.6f ms\n', ...
        w_cos, tWavePeriod_cos*1e3);
    end

    % Determine cosine amplitude from gradient and slew limits
    % Pulseq internal gradient units are Hz/m.
    % Reference design uses G/cm, then converts to Hz/m.
    swave_max = min(physical_slew_max, swave_max);

    swave_max_gauss = swave_max * 100;   % T/m/s -> G/cm/s
    gwave_max_gauss = gwave_max / 10;    % mT/m -> G/cm

    if swave_max_gauss >= w_cos * gwave_max_gauss
        G0_cos = gwave_max_gauss;
        if waveInfoFlag
            disp(['wave amplitude is not slew limited, using g0_cos = ', ...
                num2str(G0_cos*10), ' mT/m']);
        end
    else
        G0_cos = swave_max_gauss / w_cos;
        if waveInfoFlag
            disp(['wave amplitude is slew limited, using g0_cos = ', ...
                num2str(G0_cos*10), ' mT/m']);
        end
    end

    % Convert amplitude from G/cm to Pulseq Hz/m
    scaling_factor = sys.gamma * 1e-2;
    G0_cos_pulseq  = G0_cos * scaling_factor;

    % Build cosine waveform
    tWavepoints_cos = ((0:wavepoints_cos)) * tWaveUnit_cos;  %TODO: plus 0.5 or not?
    gWave_cos = G0_cos_pulseq * cos(w_cos * tWavepoints_cos);
    
    % cover the extra dead-time region too.
    targetCosDur = adc.numSamples * adc.dwell + sys.adcDeadTime;
    % Current waveform duration
    currentCosDur = numel(gWave_cos) * sys.gradRasterTime;
    % Add constant samples at G0_cos_pulseq if needed
    nPad = round((targetCosDur - currentCosDur) / sys.gradRasterTime);
    if nPad > 0
        gWave_cos = [gWave_cos, G0_cos_pulseq * ones(1, nPad)];
    elseif nPad < 0
        error('nPad smaller than 0')
    end
    gWave_cos = gWave_cos(:).';

    gWave_cos_helper = mr.makeArbitraryGrad(gpe_channel, gWave_cos, 'system', sys, 'first', G0_cos_pulseq, 'last',  G0_cos_pulseq);
    % Design rampdown/post compensation
    gWave_cos_area_max_unit = gWave_cos_helper.area / (nPad + 1);
    if waveInfoFlag
        disp(['gWave_cos_area_max_unit: ' num2str(gWave_cos_area_max_unit)])
    end

    % Design merged PE prephaser + wave rampup
    % The merged event starts at 0, ends at G0_cos_pulseq, has total area gpePre.area, and lasts gpePre_dur + gro.riseTime.
    T_preRamp = gpePre_dur + gro.riseTime;
    T_preRamp = round(T_preRamp / sys.gradRasterTime) * sys.gradRasterTime;
    A_preRamp_target = gpePre.area - gWave_cos_area_max_unit / 2 / (sys.gradRasterTime / adc.dwell);
    G_rampup_end = G0_cos_pulseq;
    [gpePre_rampup, preRampWave] = makeFixedDurationPreRamp(gpe_channel, A_preRamp_target, G_rampup_end, T_preRamp, sys);

    % Concatenate pre-ramp and cosine manually.
    % This avoids mr.addGradients row/column zero-fill issues.
    gpeWaveFull = [preRampWave, gWave_cos];

    % gpe_wave_cos = mr.makeArbitraryGrad(gpe_channel, gpeWaveFull, 'system', sys, 'first', 0, 'last', G0_cos_pulseq, 'oversampling', true);  % has timing errors
    gpe_wave_cos = mr.makeArbitraryGrad(gpe_channel, gpeWaveFull, 'system', sys, 'first', 0, 'last', G0_cos_pulseq);

    % Nominal post-rewinder area is just -gpePre.area.
    % Then compensate the residual area of the cosine waveform.
    gpePost_area_new = -gpe_wave_cos.area;

    gpe_post = mr.makeExtendedTrapezoidArea(gpe_channel, G0_cos_pulseq, 0, gpePost_area_new, sys_lowPNS);

    % Sanity check of time
    tol = sys.gradRasterTime/10;
    adcEndObject = mr.calcDuration(adc);
    fullDur_obj    = mr.calcDuration(gpe_wave_cos);
    if abs(adcEndObject - fullDur_obj) > tol
        error(['Timing mismatch: ADC duration (including delay) = %.6f ms, ', ...
            'Wave object (including prephase) = %.6f ms, diff = %.6f us'], ...
            adcEndObject*1e3, ...
            fullDur_obj*1e3, ...
            (adcEndObject - fullDur_obj)*1e6);
    end

    % Debug print based on actual constructed objects
    if debugFlag

        dt = sys.gradRasterTime;

        % Actual durations from generated waveform arrays
        preRampDur_wave = numel(preRampWave) * dt;
        cosDur_wave     = numel(gWave_cos) * dt;
        fullDur_wave    = numel(gpeWaveFull) * dt;

        % Actual durations from Pulseq objects
        preRampDur_obj = mr.calcDuration(gpePre_rampup);
        cosDur_obj     = mr.calcDuration(gWave_cos_helper);
        fullDur_obj    = mr.calcDuration(gpe_wave_cos);
        postDur_obj    = mr.calcDuration(gpe_post);

        % ADC timing from actual ADC object
        adcStart     = adc.delay;
        adcAcqDur    = adc.numSamples * adc.dwell;
        adcEndAcq    = adcStart + adcAcqDur;
        adcEndObject = mr.calcDuration(adc);

        % Cosine timing inside the final concatenated PE waveform
        cosStart_actual = preRampDur_wave;
        cosEnd_actual   = fullDur_wave;

        fprintf('\n');
        fprintf('================ Cosine PE wave debug ================\n');

        fprintf('gpePre.area                         = %.9g 1/m\n', gpePre.area);
        fprintf('gpePre duration                     = %.6f ms\n', gpePre_dur*1e3);
        fprintf('gro.riseTime                        = %.6f ms\n', gro.riseTime*1e3);

        fprintf('\n--- Pre-ramp timing ---\n');
        fprintf('preRampWave samples                 = %d\n', numel(preRampWave));
        fprintf('preRamp duration from waveform       = %.6f ms\n', preRampDur_wave*1e3);
        fprintf('preRamp duration from object         = %.6f ms\n', preRampDur_obj*1e3);
        fprintf('gpePre duration + gro.riseTime       = %.6f ms\n', ...
            (gpePre_dur + gro.riseTime)*1e3);
        fprintf('preRampWave duration - adc.delay     = %.6f us\n', ...
            (preRampDur_wave - adcStart)*1e6);

        fprintf('\n--- Cosine wave timing ---\n');
        fprintf('gWave_cos samples                   = %d\n', numel(gWave_cos));
        fprintf('cos duration from waveform           = %.6f ms\n', cosDur_wave*1e3);
        fprintf('cos duration from helper object      = %.6f ms\n', cosDur_obj*1e3);
        fprintf('Tread input                          = %.6f ms\n', Tread*1e3);
        fprintf('Tread rasterized                     = %.6f ms\n', TreadRaster*1e3);

        fprintf('\n--- ADC timing ---\n');
        fprintf('adc.delay                            = %.6f ms\n', adcStart*1e3);
        fprintf('adc.numSamples * adc.dwell           = %.6f ms\n', adcAcqDur*1e3);
        fprintf('adc acquisition end                  = %.6f ms\n', adcEndAcq*1e3);
        fprintf('mr.calcDuration(adc)                 = %.6f ms\n', adcEndObject*1e3);

        fprintf('\n--- Final PE wave timing ---\n');
        fprintf('cosine starts at                     = %.6f ms\n', cosStart_actual*1e3);
        fprintf('cosine ends at                       = %.6f ms\n', cosEnd_actual*1e3);
        fprintf('gpe_wave_cos duration from waveform  = %.6f ms\n', fullDur_wave*1e3);
        fprintf('gpe_wave_cos duration from object    = %.6f ms\n', fullDur_obj*1e3);
        fprintf('gpe_post duration                    = %.6f ms\n', postDur_obj*1e3);

        fprintf('\n--- Timing differences ---\n');
        fprintf('cos start - adc start                = %.6f us\n', ...
            (cosStart_actual - adcStart)*1e6);
        fprintf('cos end - adc acquisition end        = %.6f us\n', ...
            (cosEnd_actual - adcEndAcq)*1e6);
        fprintf('cos duration - adc acquisition dur   = %.6f us\n', ...
            (cosDur_wave - adcAcqDur)*1e6);
        fprintf('full PE wave - mr.calcDuration(obj)  = %.6f us\n', ...
            (fullDur_wave - fullDur_obj)*1e6);

        fprintf('\n--- Areas ---\n');
        fprintf('preRamp object area                  = %.9g 1/m\n', gpePre_rampup.area);
        fprintf('cos helper area                      = %.9g 1/m\n', gWave_cos_helper.area);
        fprintf('full gpe_wave_cos area               = %.9g 1/m\n', gpe_wave_cos.area);
        fprintf('gpe_post target area                 = %.9g 1/m\n', gpePost_area_new);
        fprintf('gpe_post actual area                 = %.9g 1/m\n', gpe_post.area);

        fprintf('======================================================\n\n');

    end
end

%% Define sine wave

function [gpe_wave_sin, gpe_post] = defineSineWaveGradient(Tread, sys, sys_lowPNS, Ncycles, gwave_max, swave_max, gpePre, gro, adc, physical_slew_max, waveInfoFlag, debugFlag)
    %DEFINECOSINEWAVEGRADIENT Create PE sine wave gradient with pre/post compensation.
    %
    % Outputs
    %   gpe_wave_sin : PE prephaser + rampup + sine wave gradient
    %   gpe_post     : PE rampdown/post-rewinder gradient
    %
    % Inputs
    %   Tread      : sine wave duration, usually ADC/readout duration [s]
    %   sys        : Pulseq system struct
    %   Ncycles    : number of sine cycles during Tread
    %   gwave_max  : max wave gradient amplitude [mT/m]
    %   swave_max  : max wave slew rate [T/m/s]
    %   gpePre     : nominal PE prephaser gradient
    %   gro        : readout gradient; gro.riseTime is used for rampup
    if nargin < 10
        debugFlag = false;
        waveInfoFlag = false;
    end

    % Basic checks
    if Ncycles <= 0
        error('Ncycles must be positive.');
    end

    if Tread <= 0
        error('Tread must be positive.');
    end

    % Get PE channel and prephaser duration from gpePre
    gpe_channel = gpePre.channel;
    gpePre_dur  = mr.calcDuration(gpePre);

    % Design sine wave
    wavepoints_sin = round(Tread / sys.gradRasterTime);
    tWaveUnit_sin = sys.gradRasterTime;

    TreadRaster = wavepoints_sin * tWaveUnit_sin;

    tWavePeriod_sin = TreadRaster / Ncycles;
    w_sin = 2*pi / tWavePeriod_sin;
    if waveInfoFlag
        fprintf('w_sin: %.6f rad/s, tWavePeriod_sin: %.6f ms\n', ...
        w_sin, tWavePeriod_sin*1e3);
    end

    % Determine sine amplitude from gradient and slew limits
    % Pulseq internal gradient units are Hz/m.
    % Reference design uses G/cm, then converts to Hz/m.
    swave_max = min(physical_slew_max, swave_max);

    swave_max_gauss = swave_max * 100;   % T/m/s -> G/cm/s
    gwave_max_gauss = gwave_max / 10;    % mT/m -> G/cm

    if swave_max_gauss >= w_sin * gwave_max_gauss
        G0_sin = gwave_max_gauss;
        if waveInfoFlag
            disp(['wave amplitude is not slew limited, using g0_sin = ', ...
                num2str(G0_sin*10), ' mT/m']);
        end
    else
        G0_sin = swave_max_gauss / w_sin;
        if waveInfoFlag
            disp(['wave amplitude is slew limited, using g0_sin = ', ...
                num2str(G0_sin*10), ' mT/m']);
        end
    end

    % Convert amplitude from G/cm to Pulseq Hz/m
    scaling_factor = sys.gamma * 1e-2;
    G0_sin_pulseq  = G0_sin * scaling_factor;

    % Build cosine waveform
    tWavepoints_sin = (0:wavepoints_sin) * tWaveUnit_sin;  %TODO: minus 1 or not?
    gWave_sin = G0_sin_pulseq * sin(w_sin * tWavepoints_sin);
    
    % cover the extra dead-time region too.
    targetCosDur = adc.numSamples * adc.dwell + sys.adcDeadTime;
    % Current waveform duration
    currentCosDur = numel(gWave_sin) * sys.gradRasterTime;
    % Add constant samples at G0_sin_pulseq if needed
    nPad = round((targetCosDur - currentCosDur) / sys.gradRasterTime);
    if nPad > 0
        gWave_sin = [gWave_sin, zeros(1, nPad)];
    elseif nPad < 0
        error('nPad smaller than 0')
    end
    % Pad the front
    nPadPre = round(gro.riseTime / sys.gradRasterTime);
    gWave_sin = [zeros(1, nPadPre) gWave_sin];
    gWave_sin = gWave_sin(:).';

    gWave_sin_helper = mr.makeArbitraryGrad(gpe_channel, gWave_sin, 'system', sys, 'first', 0, 'last',  0);

    % Design merged PE prephaser + wave
    % Extract the waveform from gpePre
    tCorners = [0, gpePre.riseTime, gpePre.riseTime + gpePre.flatTime, gpePre.riseTime + gpePre.flatTime + gpePre.fallTime] + gpePre.delay;
    aCorners = [0, gpePre.amplitude, gpePre.amplitude, 0];
    % Filter out duplicates (crucial if the blip is perfectly triangular)
    [tCorners_unq, idx_unq] = unique(tCorners, 'stable');
    aCorners_unq = aCorners(idx_unq);
    % Sample the blip onto the raster grid
    dt = sys.gradRasterTime;
    n = round(gpePre_dur / dt);
    tCenters = ((0:n-1) + 0.5) * dt;
    preWave = interp1(tCorners_unq, aCorners_unq, tCenters, 'linear', 0);
    % Force row vector
    preWave = preWave(:).';

    % Concatenate prephaser and sine manually.
    % This avoids mr.addGradients row/column zero-fill issues.
    gpeWaveFull = [preWave, gWave_sin];
    gpe_wave_sin = mr.makeArbitraryGrad(gpe_channel, gpeWaveFull, 'system', sys, 'first', 0, 'last', 0);

    % Nominal post-rewinder area is just -gpePre.area.
    % Then compensate the residual area of the sine waveform.
    gpePost_area_new = -gpe_wave_sin.area;

    gpe_post = mr.makeTrapezoid(gpe_channel, 'Area', gpePost_area_new, 'system', sys_lowPNS);

    % Sanity check of time
    tol = sys.gradRasterTime/10;
    adcEndObject = mr.calcDuration(adc);
    fullDur_obj    = mr.calcDuration(gpe_wave_sin);
    if abs(adcEndObject - fullDur_obj) > tol
        error(['Timing mismatch: ADC duration (including delay) = %.6f ms, ', ...
            'Wave object (including prephase) = %.6f ms, diff = %.6f us'], ...
            adcEndObject*1e3, ...
            fullDur_obj*1e3, ...
            (adcEndObject - fullDur_wobj)*1e6);
    end

    % Debug print based on actual constructed objects
    if debugFlag

        dt = sys.gradRasterTime;

        % Actual durations from generated waveform arrays
        preRampDur_wave = numel(preWave) * dt;
        sinDur_wave     = numel(gWave_sin) * dt;
        fullDur_wave    = numel(gpeWaveFull) * dt;

        % Actual durations from Pulseq objects
        sinDur_obj     = mr.calcDuration(gWave_sin_helper);
        fullDur_obj    = mr.calcDuration(gpe_wave_sin);
        postDur_obj    = mr.calcDuration(gpe_post);

        % ADC timing from actual ADC object
        adcStart     = adc.delay;
        adcAcqDur    = adc.numSamples * adc.dwell;
        adcEndAcq    = adcStart + adcAcqDur;
        adcEndObject = mr.calcDuration(adc);

        % Sine timing inside the final concatenated PE waveform
        sinStart_actual = preRampDur_wave;
        sinEnd_actual   = fullDur_wave;

        fprintf('\n');
        fprintf('================ Sine PE wave debug ================\n');

        fprintf('gpePre.area                          = %.9g 1/m\n', gpePre.area);
        fprintf('gpePre duration                      = %.6f ms\n', gpePre_dur*1e3);
        fprintf('gro.riseTime                         = %.6f ms\n', gro.riseTime*1e3);

        fprintf('\n--- Pre-ramp timing ---\n');
        fprintf('preRampWave samples                  = %d\n', numel(preWave));
        fprintf('preRamp duration from waveform       = %.6f ms\n', preRampDur_wave*1e3);
        fprintf('gpePre duration + gro.riseTime       = %.6f ms\n', ...
            (gpePre_dur + gro.riseTime)*1e3);
        fprintf('preRampWave duration - adc.delay     = %.6f us\n', ...
            (preRampDur_wave - adcStart)*1e6);

        fprintf('\n--- Sine wave timing ---\n');
        fprintf('gWave_sin samples                    = %d\n', numel(gWave_sin));
        fprintf('sin duration from waveform           = %.6f ms\n', sinDur_wave*1e3);
        fprintf('sin duration from helper object      = %.6f ms\n', sinDur_obj*1e3);
        fprintf('Tread input                          = %.6f ms\n', Tread*1e3);
        fprintf('Tread rasterized                     = %.6f ms\n', TreadRaster*1e3);

        fprintf('\n--- ADC timing ---\n');
        fprintf('adc.delay                            = %.6f ms\n', adcStart*1e3);
        fprintf('adc.numSamples * adc.dwell           = %.6f ms\n', adcAcqDur*1e3);
        fprintf('adc acquisition end                  = %.6f ms\n', adcEndAcq*1e3);
        fprintf('mr.calcDuration(adc)                 = %.6f ms\n', adcEndObject*1e3);

        fprintf('\n--- Final PE wave timing ---\n');
        fprintf('gpe_wave_sin duration from waveform  = %.6f ms\n', fullDur_wave*1e3);
        fprintf('gpe_wave_sin duration from object    = %.6f ms\n', fullDur_obj*1e3);
        fprintf('gpe_post duration                    = %.6f ms\n', postDur_obj*1e3);

        fprintf('\n--- Timing differences ---\n');
        fprintf('full PE wave - mr.calcDuration(obj)  = %.6f us\n', ...
            (fullDur_wave - fullDur_obj)*1e6);

        fprintf('\n--- Areas ---\n');
        fprintf('preRamp object area                  = %.9g 1/m\n', gpePre.area);
        fprintf('sin helper area                      = %.9g 1/m\n', gWave_sin_helper.area);
        fprintf('full gpe_wave_sin area               = %.9g 1/m\n', gpe_wave_sin.area);
        fprintf('gpe_post target area                 = %.9g 1/m\n', gpePost_area_new);
        fprintf('gpe_post actual area                 = %.9g 1/m\n', gpe_post.area);

        fprintf('======================================================\n\n');

    end
end

%% Open-source setup helper functions

function pathValue = getDirectoryFromWorkspaceOrPrompt(varName, promptText, allowEmpty, createIfMissing)
    pathValue = '';
    if evalin('base', sprintf('exist(''%s'', ''var'')', varName))
        pathValue = evalin('base', varName);
        if isstring(pathValue), pathValue = char(pathValue); end
        pathValue = normalizeUserPath(pathValue);
    end

    while isempty(pathValue) || ~exist(pathValue, 'dir')
        if ~isempty(pathValue) && ~exist(pathValue, 'dir')
            if createIfMissing
                reply = input(sprintf('%s does not exist: %s. Create it? [Y/n]: ', varName, pathValue), 's');
                if isempty(reply) || strcmpi(reply, 'y') || strcmpi(reply, 'yes')
                    mkdir(pathValue);
                    break;
                end
            else
                fprintf('%s does not exist: %s\n', varName, pathValue);
            end
        end

        if allowEmpty
            userText = input(sprintf('%s: ', promptText), 's');
            if isempty(strtrim(userText))
                pathValue = '';
                assignin('base', varName, pathValue);
                return;
            end
        else
            userText = input(sprintf('%s: ', promptText), 's');
            if isempty(strtrim(userText))
                fprintf('A valid path is required.\n');
                continue;
            end
        end
        pathValue = normalizeUserPath(strtrim(userText));
    end

    assignin('base', varName, pathValue);
end

function fileValue = getFileFromWorkspaceOrPrompt(varName, promptText, allowEmpty)
    fileValue = '';
    if evalin('base', sprintf('exist(''%s'', ''var'')', varName))
        fileValue = evalin('base', varName);
        if isstring(fileValue), fileValue = char(fileValue); end
        fileValue = normalizeUserPath(fileValue);
    end

    while isempty(fileValue) || ~exist(fileValue, 'file')
        if ~isempty(fileValue) && ~exist(fileValue, 'file')
            fprintf('%s does not exist or is not a file: %s\n', varName, fileValue);
        end
        userText = input(sprintf('%s: ', promptText), 's');
        if isempty(strtrim(userText)) && allowEmpty
            fileValue = '';
            assignin('base', varName, fileValue);
            return;
        elseif isempty(strtrim(userText))
            fprintf('A valid file path is required.\n');
            continue;
        end
        fileValue = normalizeUserPath(strtrim(userText));
    end

    assignin('base', varName, fileValue);
end

function value = selectStringOption(varName, promptText, options, defaultValue)
    if nargin < 4 || isempty(defaultValue)
        defaultValue = options{1};
    end

    value = defaultValue;
    if evalin('base', sprintf('exist(''%s'', ''var'')', varName))
        workspaceValue = evalin('base', varName);
        if isstring(workspaceValue), workspaceValue = char(workspaceValue); end
        if ischar(workspaceValue) && any(strcmp(workspaceValue, options))
            value = workspaceValue;
        else
            warning('Ignoring invalid workspace value for %s.', varName);
        end
    end

    defaultIdx = find(strcmp(value, options), 1);
    if isempty(defaultIdx), defaultIdx = 1; end

    fprintf('\n%s:\n', promptText);
    for ii = 1:numel(options)
        fprintf('  %2d) %s\n', ii, options{ii});
    end
    reply = input(sprintf('Select 1-%d [default %d: %s]: ', numel(options), defaultIdx, options{defaultIdx}), 's');

    if ~isempty(strtrim(reply))
        numericChoice = str2double(reply);
        if ~isnan(numericChoice) && numericChoice == round(numericChoice) && numericChoice >= 1 && numericChoice <= numel(options)
            value = options{numericChoice};
        else
            trimmedReply = strtrim(reply);
            idx = find(strcmp(trimmedReply, options), 1);
            if isempty(idx)
                error('Invalid %s selection: %s', varName, reply);
            end
            value = options{idx};
        end
    else
        value = options{defaultIdx};
    end

    assignin('base', varName, value);
end

function tf = promptYesNoFromWorkspace(varName, promptText, defaultValue)
    if evalin('base', sprintf('exist(''%s'', ''var'')', varName))
        workspaceValue = evalin('base', varName);
        if islogical(workspaceValue) || isnumeric(workspaceValue)
            tf = logical(workspaceValue);
            return;
        elseif ischar(workspaceValue) || isstring(workspaceValue)
            txt = lower(strtrim(char(workspaceValue)));
            if any(strcmp(txt, {'y', 'yes', 'true', '1'}))
                tf = true;
                return;
            elseif any(strcmp(txt, {'n', 'no', 'false', '0'}))
                tf = false;
                return;
            end
        end
        warning('Ignoring invalid workspace value for %s.', varName);
    end

    if defaultValue
        suffix = '[Y/n]';
    else
        suffix = '[y/N]';
    end

    reply = input(sprintf('%s %s: ', promptText, suffix), 's');
    reply = lower(strtrim(reply));
    if isempty(reply)
        tf = logical(defaultValue);
    elseif any(strcmp(reply, {'y', 'yes', 'true', '1'}))
        tf = true;
    elseif any(strcmp(reply, {'n', 'no', 'false', '0'}))
        tf = false;
    else
        error('Please answer yes or no for %s.', promptText);
    end
    assignin('base', varName, tf);
end

function pathValue = normalizeUserPath(pathValue)
    if isempty(pathValue)
        return;
    end
    if isstring(pathValue), pathValue = char(pathValue); end
    pathValue = strtrim(pathValue);
    if startsWith(pathValue, ['~' filesep]) || strcmp(pathValue, '~')
        homeDir = getenv('HOME');
        if isempty(homeDir) && ispc
            homeDir = getenv('USERPROFILE');
        end
        if strcmp(pathValue, '~')
            pathValue = homeDir;
        else
            pathValue = fullfile(homeDir, pathValue(3:end));
        end
    end
end

function pathValue = ensureTrailingFilesep(pathValue)
    if isempty(pathValue)
        return;
    end
    if pathValue(end) ~= filesep
        pathValue = [pathValue filesep];
    end
end

