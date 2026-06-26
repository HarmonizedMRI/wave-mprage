% mprage_3d_wave.m
% Author: Yiyun Dong
% Affiliation: Athinoula A. Martinos Center for Biomedical Imaging
% Date: 2026-06-26
%
% Build based on Maxim's MPRAGE Pulseq demo:
% https://github.com/pulseq-admin/pulseq/blob/master/matlab/demoSeq/writeMPRAGE.m

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

% sequence (for real scan)
alpha    = 7;
ro_dur   = 5120e-6;  % also Tread for wave
ro_os    = 4;        % also os_factor for wave
ro_spoil = 3;        % additional k-max excursion for RO spoiling
TI       = 1.1;
TRout    = 2.5;
% TE & TR in the inner loop are as short as possible derived from the above parameters and the system specs
% more in-depth parameters
rfSpoilingInc = 50;              % RF spoiling increment
rfLen         = 100e-6;
ax            = struct; % encoding axes

% geometry
% sagittal fov options % remember to enable OrientationMapping SAG in setDefinition section below
slOrientation = 'SAG';
fov = [192 256 256]*1e-3;        % FOV [x y z], in meters

% Requested voxel size [x y z], in mm. The matrix N is derived from fov and
% res, with each dimension rounded to the nearest even integer so that the
% k-space center index remains well defined for the PE ordering/TI logic.
% Examples:
%   res = [1.0  1.0  1.0 ];  % N ~= [192 256 256]
%   res = [1.5  1.0  1.0 ];  % N ~= [128 256 256]
%   res = [1.0  1.5  1.0 ];  % N ~= [192 170 256]
%   res = [1.25 1.25 1.0 ];  % N ~= [154 204 256]
res = [1.0 1.0 1.0];          % requested resolution [x y z], in mm
N = 2 * round((fov(:).' * 1e3 ./ res) / 2);
actualRes = fov(:).' ./ N * 1e3; % actual achieved resolution [x y z], in mm
fprintf('Requested resolution [x y z] = [%.4g %.4g %.4g] mm. Derived N = [%d %d %d]. Actual resolution = [%.4g %.4g %.4g] mm.\n', ...
    res(1), res(2), res(3), N(1), N(2), N(3), actualRes(1), actualRes(2), actualRes(3));

ax.d1 = 'z';
ax.d2 = 'x';

% Parallel imaging / undersampling options and fixed-ETL scheduling.
% R1/ACS1num apply to PE_x = ax.d2 = PAR.
% R2/ACS2num apply to PE_y = ax.d3 = LIN.
% R = 1 means fully sampled. ACSnum = 0 means no separate ACS/reference region.
%
% Imaging phase:
%   The sampled PE_x/PE_y table after R1/R2 is packed into fixed ETLtarget
%   inversion blocks using the segmented fractional-PE_y rule when useful.
%   Otherwise one sampled PE_y line is acquired per inversion and the rest
%   of the ETL is filled with no-ADC dummy RF/readout slots.
%
% ACS phase:
%   The existing rectangular ACS squeezing is preserved. Set ACS1num so that
%   ACS1num * squeeze_acs = ETLtarget when possible, e.g. 24*8 or 32*6.
%   The final residual ACS block is padded with no-ADC dummy slots.
R1 = 2;
R2 = 3;
ACS1num = 32;
ACS2num = 32;

ETLtarget = 192;
etlSeg = struct;
etlSeg.sMin      = 16;
etlSeg.KMax      = 12;
etlSeg.PMax      = 16;
etlSeg.fillerMax = 0.10;
etlSeg.savedMin  = 16;

ax.d3=setdiff('xyz',[ax.d1 ax.d2]); % automatically set the slowest dimension
ax.n1=strfind('xyz',ax.d1);
ax.n2=strfind('xyz',ax.d2);
ax.n3=strfind('xyz',ax.d3);

if R1 < 1 || R1 ~= round(R1) || R2 < 1 || R2 ~= round(R2)
    error('R1 and R2 must be positive integers.');
end
if ACS1num < 0 || ACS1num ~= round(ACS1num) || ACS2num < 0 || ACS2num ~= round(ACS2num)
    error('ACS1num and ACS2num must be non-negative integers.');
end

% wave
gwave_max     = 8;  % mT/m
swave_max     = 200; % T/m/s
Ncycles       = 10;
% isUseWave_cos = true;
% isUseWave_sin = true;
isUseWave_cos = false;
isUseWave_sin = false;
tag_wave_details = ['_amp' num2str(gwave_max) '_cycles' num2str(Ncycles) '_' slOrientation];


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

% Create a new sequence object
seq = mr.Sequence(sys);

%% Setup

% Create alpha-degree hard pulse and gradient
rf = mr.makeBlockPulse(alpha*pi/180,sys,'Duration',rfLen, 'SliceThickness', fov(ax.n2), 'use', 'excitation');
% rf180 = mr.makeAdiabaticPulse('hypsec',sys,'Duration',10.24e-3,'dwell',1e-5,...
%     'use', 'inversion');
rf180 = mr.makeAdiabaticPulse('hypsec', sys, ...
    'Duration', 10.24e-3, ...
    'dwell', 1e-5, ...
    'use', 'inversion', ...
    'pythonCmd', '/opt/homebrew/Caskroom/miniforge/base/envs/ptx314/bin/python');

% Define other gradients and ADC events
deltak=1./fov;

% readout sanity check: Make sure Tread is divisible
dwell = round((ro_dur / N(ax.n1) / ro_os) / sys.adcRasterTime) * sys.adcRasterTime;
Tread = dwell * N(ax.n1) * ro_os;
disp(['RO duration (sampling unit):' num2str(ro_dur) ', Tread: ' num2str(Tread)])

gro = mr.makeTrapezoid(ax.d1,'Amplitude',N(ax.n1)*deltak(ax.n1)/ro_dur,'FlatTime',ceil((ro_dur+sys.adcDeadTime)/sys.gradRasterTime)*sys.gradRasterTime,'system',sys);
adc = mr.makeAdc(N(ax.n1)*ro_os,'Duration',ro_dur,'Delay',gro.riseTime,'system',sys);
groPre = mr.makeTrapezoid(ax.d1,'Area',-gro.amplitude*(adc.dwell*(adc.numSamples/2+0.5)+0.5*gro.riseTime),'system',sys_lowPNS); % the first 0.5 is necessary to acount for the Siemens sampling in the center of the dwell periods
gpe1 = mr.makeTrapezoid(ax.d2,'Area',-deltak(ax.n2)*(N(ax.n2)/2),'system',sys_lowPNS); % maximum PE1 gradient
gpe2 = mr.makeTrapezoid(ax.d3,'Area',-deltak(ax.n3)*(N(ax.n3)/2),'system',sys_lowPNS); % maximum PE2 gradient
gslSp = mr.makeTrapezoid(ax.d3,'Area',max(deltak.*N)*4,'Duration',10e-3,'system',sys_lowPNS); % spoil with 4x cycles per voxel
% we cut the RO gradient into two parts for the optimal spoiler timing
[gro1,groSp]=mr.splitGradientAt(gro,gro.riseTime+gro.flatTime);
% gradient spoiling
if ro_spoil>0
    groSp=mr.makeExtendedTrapezoidArea(gro.channel,gro.amplitude,0,deltak(ax.n1)/2*N(ax.n1)*ro_spoil,sys_lowPNS);
end

% Adjust timing of the fast loop 
% we will have two blocks in the inner loop:
% 1: spoilers/rewinders + RF 
% 2: prewinder,phase neconding + readout 
rf.delay = mr.calcDuration(groSp,gpe1,gpe2);
% Prolong the prewinders
gPre_dur = max([mr.calcDuration(groPre), mr.calcDuration(gpe1), mr.calcDuration(gpe2)]);
gPre_dur = ceil(gPre_dur/sys.gradRasterTime)*sys.gradRasterTime;
groPre   = mr.makeTrapezoid(ax.d1, 'Area', groPre.area, 'Duration', gPre_dur, 'system', sys_lowPNS);
gpe1Pre  = mr.makeTrapezoid(ax.d2, 'Area', gpe1.area, 'Duration', gPre_dur, 'system', sys_lowPNS);
gpe2Pre  = mr.makeTrapezoid(ax.d3, 'Area', gpe2.area, 'Duration', gPre_dur, 'system', sys_lowPNS);
% Merge the readout partially with its prewinder
gro1.delay=mr.calcDuration(groPre);
adc.delay=gro1.delay+gro.riseTime;
gro1=mr.addGradients({gro1,groPre},'system',sys);

% Precompute PE1 cosine-wave prewinders and rewinders and sanity check
gpe1PreWave  = cell(1, N(ax.n2));
gpe1PostWave = cell(1, N(ax.n2));
gpe1PostDur = zeros(1, N(ax.n2));
tol = sys.gradRasterTime/10;
% peSteps -- control reordering
pe1Steps=((0:N(ax.n2)-1)-N(ax.n2)/2)/N(ax.n2)*2;
pe2Steps=((0:N(ax.n3)-1)-N(ax.n3)/2)/N(ax.n3)*2;
for i = 1:N(ax.n2)
    % Nominal PE1 prewinder for this partition step
    gpe1Pre_i = mr.scaleGrad(gpe1Pre, pe1Steps(i));
    if isUseWave_cos
        % Print debug info only for the first PE1 step
        debugFlag = (i == 1);
        if slOrientation == "SAG"
            [gpe1PreWave{i}, gpe1PostWave{i}] = defineCosineWaveGradient(Tread, sys, sys_lowPNS, Ncycles, gwave_max, swave_max, gpe1Pre_i, gro, adc, physical_slew_max, debugFlag, debugFlag);
        elseif slOrientation == "COR"
            [gpe1PreWave{i}, gpe1PostWave{i}] = defineSineWaveGradient(Tread, sys, sys_lowPNS, Ncycles, gwave_max, swave_max, gpe1Pre_i, gro, adc, physical_slew_max, debugFlag, debugFlag);
        end
    else
        % No wave: use normal scaled prewinder and rewinder
        gpe1PreWave{i} = gpe1Pre_i;
        % Use gpe1 for the rewinder because it is the shorter rewinder object
        gpe1PostWave{i} = mr.scaleGrad(gpe1, -pe1Steps(i));
    end
    % Register the prepared gradients
    gpe1PreWave{i}.id  = seq.registerGradEvent(gpe1PreWave{i});
    gpe1PostWave{i}.id = seq.registerGradEvent(gpe1PostWave{i});
    % Store post duration for rf.delay
    gpe1PostDur(i) = mr.calcDuration(gpe1PostWave{i});
end

% Precompute PE2 sine-wave prewinders and rewinders and sanity check
gpe2PreWave  = cell(1, N(ax.n3));
gpe2PostWave = cell(1, N(ax.n3));
gpe2PostDur = zeros(1, N(ax.n3));
tol = sys.gradRasterTime/10;
for i = 1:N(ax.n3)
    % Nominal PE2 prewinder for this partition step
    gpe2Pre_i = mr.scaleGrad(gpe2Pre, pe2Steps(i));
    if isUseWave_sin
        % Print debug info only for the first PE1 step
        debugFlag = (i == 1);
        if slOrientation == "SAG"
            [gpe2PreWave{i}, gpe2PostWave{i}] = defineSineWaveGradient(Tread, sys, sys_lowPNS, Ncycles, gwave_max, swave_max, gpe2Pre_i, gro, adc, physical_slew_max, debugFlag, debugFlag);
        elseif slOrientation == "COR"
            [gpe2PreWave{i}, gpe2PostWave{i}] = defineCosineWaveGradient(Tread, sys, sys_lowPNS, Ncycles, gwave_max, swave_max, gpe2Pre_i, gro, adc, physical_slew_max, debugFlag, debugFlag);
        end
    else
        % No wave: use normal scaled prewinder and rewinder
        gpe2PreWave{i} = gpe2Pre_i;
        % Use gpe2 for the rewinder because it is the shorter rewinder object
        gpe2PostWave{i} = mr.scaleGrad(gpe2, -pe2Steps(i));
    end
    % Register the prepared gradients
    gpe2PreWave{i}.id  = seq.registerGradEvent(gpe2PreWave{i});
    gpe2PostWave{i}.id = seq.registerGradEvent(gpe2PostWave{i});
    % Store post duration for rf.delay
    gpe2PostDur(i) = mr.calcDuration(gpe2PostWave{i});
end

% Recalculate the RF delay
rf.delay = max([mr.calcDuration(groSp), max(gpe1PostDur), max(gpe2PostDur)]);

% Calculate timing
TRinner=mr.calcDuration(rf)+mr.calcDuration(gro1); % we'll need it for the TI delay

% Undersampling setup for both PE directions, acquired in two separate phases:
%   1) undersampled imaging k-space
%   2) full ACS/reference k-space, acquired separately even if some lines
%      duplicate image k-space locations.
%
% Image PAR/LIN labels use full 0-based matrix indices. ACS PAR/LIN labels
% use ACS-local 0-based indices, while gradients still use the global
% physical k-space indices. This avoids refscan index shifts in TWIX.
[PE1_img, ~, ~, centerPE1LineIdx] = makePESamplingPattern(N(ax.n2), R1, 0);
[PE2_img, ~, ~, centerPE2LineIdx] = makePESamplingPattern(N(ax.n3), R2, 0);
[~, ~, PE1_acs_target, ~] = makePESamplingPattern(N(ax.n2), 1, ACS1num);
[~, ~, PE2_acs_target, ~] = makePESamplingPattern(N(ax.n3), 1, ACS2num);

nPE1Img = numel(PE1_img);
nPE2Img = numel(PE2_img);

acsEnabled = (ACS1num > 0) && (ACS2num > 0);
if xor(ACS1num > 0, ACS2num > 0)
    warning(['Only one ACS dimension is nonzero. The separate ACS loop will be skipped; ', ...
             'set both ACS1num and ACS2num > 0 for a 2D ACS/reference region.']);
end
if R1 > 1 && ~acsEnabled
    warning('R1 > 1 but no 2D ACS region is enabled. PE_x is undersampled without dedicated PE_x ACS data.');
end
if R2 > 1 && ~acsEnabled
    warning('R2 > 1 but no 2D ACS region is enabled. PE_y is undersampled without dedicated PE_y ACS data.');
end

% Fixed-ETL scheduler setup. Acceleration is already represented by the
% sampled PE lists; do not multiply by R1/R2 again here.
if nPE1Img > ETLtarget
    error('The sampled imaging PE_x count (%d) exceeds ETLtarget (%d).', nPE1Img, ETLtarget);
end

etlPlan_img = chooseFixedETLPlan(nPE1Img, ETLtarget, etlSeg);

if acsEnabled
    if ACS1num > ETLtarget
        error('ACS1num (%d) exceeds ETLtarget (%d).', ACS1num, ETLtarget);
    end
    squeeze_acs = max(1, floor(ETLtarget / ACS1num));
    if ACS1num * squeeze_acs ~= ETLtarget
        warning(['ACS1num*squeeze_acs = %d, not ETLtarget = %d. ', ...
                 'All ACS blocks will be padded to ETLtarget with no-ADC dummy slots.'], ...
                 ACS1num*squeeze_acs, ETLtarget);
    end
else
    squeeze_acs = [];
end

% Expected pair sets for high-level reporting only.
imgPairsGlobal = makePEPairList(PE1_img, PE2_img, []);
if acsEnabled
    acsPairsGlobal = makePEPairList(PE1_acs_target, PE2_acs_target, []);
else
    acsPairsGlobal = zeros(0,2);
end

inv180TailToEnd = mr.calcDuration(rf180) - mr.calcRfCenter(rf180) - rf180.delay;
rfStartToCenter = rf.delay + mr.calcRfCenter(rf);

fprintf(['PE imaging undersampling: R1=%d -> %d/%d sampled PE_x lines; ', ...
         'R2=%d -> %d/%d sampled PE_y lines. ETLtarget=%d.\n'], ...
    R1, nPE1Img, N(ax.n2), R2, nPE2Img, N(ax.n3), ETLtarget);
fprintf(['IMG fixed-ETL plan: mode=%s, segment length=%d, ', ...
         'segments/PE_y=%d, segments/block=%d, filler/PE_y=%d, efficiency=%.4f.\n'], ...
    etlPlan_img.mode, etlPlan_img.s, etlPlan_img.K, etlPlan_img.P, ...
    etlPlan_img.F, etlPlan_img.efficiency);
if acsEnabled
    fprintf(['ACS separate loop: ACS1=%d PE_x lines, ACS2=%d PE_y lines. ', ...
             'ACS rectangular squeeze=%d, ACS blocks padded to ETLtarget=%d. ', ...
             'ACS lines are duplicated if also present in image k-space.\n'], ...
        ACS1num, ACS2num, squeeze_acs, ETLtarget);
else
    fprintf('ACS target disabled; no separate ACS/reference loop will be added.\n');
end

% all LABELS / counters and flags are automatically initialized to 0 in the beginning.
% Use explicit SET labels for PAR/LIN rather than INC counters, because
% undersampling skips k-space lines.

% Image labels: full 0-based matrix coordinates.
lblLIN_img = [];
for iY = 1:N(ax.n3)
    lblLIN_img{iY} = mr.makeLabel('SET', 'LIN', iY - 1); 
end

lblPAR_img = [];
for iZ = 1:N(ax.n2)
    lblPAR_img{iZ} = mr.makeLabel('SET', 'PAR', iZ - 1); 
end

% ACS labels: local 0-based ACS coordinates. These are deliberately not
% full-matrix indices, so twixObj.refscan is indexed from 0 within the ACS
% block rather than shifted/cropped by the global ACS position.
lblLIN_acs = [];
for iY = 1:max(ACS2num, 1)
    lblLIN_acs{iY} = mr.makeLabel('SET', 'LIN', iY - 1);
end

lblPAR_acs = [];
for iZ = 1:max(ACS1num, 1)
    lblPAR_acs{iZ} = mr.makeLabel('SET', 'PAR', iZ - 1);
end

lblECO = mr.makeLabel('SET', 'ECO', 0);

% REF/IMA flags for Siemens PAT/GRAPPA:
%   ordinary image k-space: REF=false, IMA=false
%   separate ACS k-space:  REF=true,  IMA=false
% IMA here is the PATRefAndImaScan flag, not the ordinary image flag.
lblSetRefScan = mr.makeLabel('SET', 'REF', true);
lblResetRefScan = mr.makeLabel('SET', 'REF', false);
lblResetImaScan = mr.makeLabel('SET', 'IMA', false);

% pre-register objects that do not change while looping
gslSp.id=seq.registerGradEvent(gslSp);
groSp.id=seq.registerGradEvent(groSp);
gro1.id=seq.registerGradEvent(gro1);
[~, rf.shapeIDs]=seq.registerRfEvent(rf); % the phase of the RF object will change, therefore we only pre-register the shapes 
[rf180.id, rf180.shapeIDs]=seq.registerRfEvent(rf180); % 
lblSetRefScan.id = seq.registerLabelEvent(lblSetRefScan);
lblResetRefScan.id = seq.registerLabelEvent(lblResetRefScan);
lblResetImaScan.id = seq.registerLabelEvent(lblResetImaScan);

% Expected label evolution recorded as we build the sequence. This is more
% robust than set-based checks because image and ACS phases may intentionally
% duplicate the same global k-space location with different local labels.
expectedPhase = {};          % 'IMG' or 'ACS'
expectedPAR = [];            % label value seen by ICE/TWIX
expectedLIN = [];            % label value seen by ICE/TWIX
expectedREF = [];
expectedIMA = [];
expectedGlobalPAR = [];      % true physical PE_x position, 0-based
expectedGlobalLIN = [];      % true physical PE_y position, 0-based

% Build fixed-ETL block tables before adding sequence blocks. Each block is a
% 192-slot RF/readout train. Only slots with isAcquire=true receive ADC/labels.
centerSlotTarget = floor(ETLtarget/2) + 1;

imgBlocks = buildSegmentedFixedETLBlocks(PE1_img, PE2_img, ETLtarget, etlPlan_img, ...
    centerPE1LineIdx, centerPE2LineIdx, centerSlotTarget);
[nImgRealSlots, nImgDummySlots] = countFixedETLBlocks(imgBlocks);
assertGlobalCenterAtTarget(imgBlocks, centerPE1LineIdx, centerPE2LineIdx, centerSlotTarget, 'IMG');
fprintf('IMG phase: %d fixed-ETL inversion block(s), real ADCs=%d, dummy slots=%d.\n', ...
    numel(imgBlocks), nImgRealSlots, nImgDummySlots);

if acsEnabled
    acsBlocks = buildRectangularFixedETLBlocks(PE1_acs_target, PE2_acs_target, squeeze_acs, ...
        ETLtarget, centerPE1LineIdx, centerPE2LineIdx, centerSlotTarget);
    [nAcsRealSlots, nAcsDummySlots] = countFixedETLBlocks(acsBlocks);
    assertGlobalCenterAtTarget(acsBlocks, centerPE1LineIdx, centerPE2LineIdx, centerSlotTarget, 'ACS');
    fprintf('ACS phase: %d rectangular fixed-ETL inversion block(s), real ADCs=%d, dummy slots=%d.\n', ...
        numel(acsBlocks), nAcsRealSlots, nAcsDummySlots);
else
    acsBlocks = makeEmptyFixedETLBlockStruct();
    nAcsRealSlots = 0;
    nAcsDummySlots = 0;
end

% start the sequence
tic;
for phaseIdx = 1:2

    if phaseIdx == 1
        phaseName = 'IMG';
        blocks_phase = imgBlocks;
    else
        phaseName = 'ACS';
        if ~acsEnabled
            continue;
        end
        blocks_phase = acsBlocks;
    end

    for jBlock = 1:numel(blocks_phase)
        block = blocks_phase(jBlock);
        nSlotsThisBlock = numel(block.isAcquire);
        if nSlotsThisBlock ~= ETLtarget
            error('%s block %d has %d slots, expected ETLtarget=%d.', ...
                phaseName, jBlock, nSlotsThisBlock, ETLtarget);
        end

        centerSlot = block.centerSlot;
        if centerSlot ~= centerSlotTarget
            error('%s block %d center slot is %d, expected %d.', ...
                phaseName, jBlock, centerSlot, centerSlotTarget);
        end

        % TI target: the block builder explicitly places a kx=0 slot at the
        % center of the fixed ETL. For the block containing the true global
        % k-space center, that real (kx=0, ky=0) ADC is therefore at TI.
        nAcqBeforeCenter = centerSlot - 1;

        TIdelay = round((TI - nAcqBeforeCenter*TRinner - inv180TailToEnd - rfStartToCenter) ...
            / sys.blockDurationRaster) * sys.blockDurationRaster;
        TRoutDelay = TRout - TRinner*nSlotsThisBlock - TIdelay - mr.calcDuration(rf180);

        if TIdelay < 0
            error(['TIdelay is negative for %s fixed-ETL block %d ', ...
                   '(centerSlot=%d, slots=%d). ', ...
                   'Reduce ETLtarget, reduce TI, or lengthen TRout.'], ...
                   phaseName, jBlock, centerSlot, nSlotsThisBlock);
        end
        if TIdelay < mr.calcDuration(gslSp)
            warning(['TIdelay (%.3f ms) is shorter than gslSp duration (%.3f ms) ', ...
                     'for %s fixed-ETL block %d. The spoiler will set the effective delay block duration.'], ...
                     TIdelay*1e3, mr.calcDuration(gslSp)*1e3, phaseName, jBlock);
        end
        if TRoutDelay < 0
            error(['TRoutDelay is negative for %s fixed-ETL block %d ', ...
                   '(slots=%d). Lengthen TRout or reduce ETLtarget.'], ...
                   phaseName, jBlock, nSlotsThisBlock);
        end

        seq.addBlock(rf180);
        seq.addBlock(TIdelay,gslSp);
        rf_phase=0;
        rf_inc=0;
        isFirstSlotInBlock = true;
        prevI = [];
        prevJ = [];

        for slotIdx = 1:nSlotsThisBlock
            iGlobal = block.iGlobal(slotIdx);
            jGlobal = block.jGlobal(slotIdx);
            iPos    = block.iPos(slotIdx);
            jPos    = block.jPos(slotIdx);

            rf.phaseOffset=rf_phase/180*pi;
            adc.phaseOffset=rf_phase/180*pi;
            rf_inc=mod(rf_inc+rfSpoilingInc, 360.0);
            rf_phase=mod(rf_phase+rf_inc, 360.0);

            if isFirstSlotInBlock
                seq.addBlock(rf);
                isFirstSlotInBlock = false;
            else
                % Rewind previous PE_x/PE_y step using precomputed wave rewinders.
                seq.addBlock(rf, groSp, gpe1PostWave{prevI}, gpe2PostWave{prevJ});
            end

            if block.isAcquire(slotIdx)
                if strcmp(phaseName, 'IMG')
                    lblPAR_use = lblPAR_img{iGlobal};
                    lblLIN_use = lblLIN_img{jGlobal};
                    lblREF = lblResetRefScan;
                    lblIMA = lblResetImaScan;
                    parLabelVal = iGlobal - 1;
                    linLabelVal = jGlobal - 1;
                    refVal = 0;
                    imaVal = 0;
                else
                    % ACS labels are local to the ACS target region. Gradients
                    % still use global physical PE indices.
                    lblPAR_use = lblPAR_acs{iPos};
                    lblLIN_use = lblLIN_acs{jPos};
                    lblREF = lblSetRefScan;
                    lblIMA = lblResetImaScan;
                    parLabelVal = iPos - 1;
                    linLabelVal = jPos - 1;
                    refVal = 1;
                    imaVal = 0;
                end

                % Real acquired slot: add ADC and phase-specific labels.
                seq.addBlock(adc, gro1, gpe1PreWave{iGlobal}, gpe2PreWave{jGlobal}, ...
                    lblPAR_use, lblLIN_use, lblECO, lblREF, lblIMA);

                expectedPhase{end+1,1} = phaseName; %#ok<SAGROW>
                expectedPAR(end+1,1) = parLabelVal; %#ok<SAGROW>
                expectedLIN(end+1,1) = linLabelVal; %#ok<SAGROW>
                expectedREF(end+1,1) = refVal; %#ok<SAGROW>
                expectedIMA(end+1,1) = imaVal; %#ok<SAGROW>
                expectedGlobalPAR(end+1,1) = iGlobal - 1; %#ok<SAGROW>
                expectedGlobalLIN(end+1,1) = jGlobal - 1; %#ok<SAGROW>
            else
                % Dummy slot: play the same RF/readout timing and PE gradients,
                % but do not acquire ADC and do not emit PAR/LIN/REF/IMA labels.
                seq.addBlock(gro1, gpe1PreWave{iGlobal}, gpe2PreWave{jGlobal});
            end

            prevI = iGlobal;
            prevJ = jGlobal;
        end

        % Final rewinder after the last RF/readout slot in this fixed-ETL block.
        seq.addBlock(groSp, gpe1PostWave{prevI}, gpe2PostWave{prevJ}, mr.makeDelay(TRoutDelay));
    end
end
fprintf('Sequence ready (blocks generation took %g seconds)\n', toc);

adc_lbl = seq.evalLabels('evolution','adc');

fprintf('LIN label range: %d ... %d\n', min(adc_lbl.LIN), max(adc_lbl.LIN));
fprintf('PAR label range: %d ... %d\n', min(adc_lbl.PAR), max(adc_lbl.PAR));

assert(numel(adc_lbl.LIN) == numel(expectedLIN), 'Unexpected number of ADC acquisitions.');
assert(all(adc_lbl.LIN(:) == expectedLIN(:)), 'LIN label evolution does not match the requested image/ACS labels.');
assert(all(adc_lbl.PAR(:) == expectedPAR(:)), 'PAR label evolution does not match the requested image/ACS labels.');
if isfield(adc_lbl, 'REF')
    assert(all((adc_lbl.REF(:) ~= 0) == logical(expectedREF(:))), 'REF label evolution does not match the requested image/ACS labels.');
    assert(all((adc_lbl.IMA(:) ~= 0) == logical(expectedIMA(:))), 'IMA/PATRefAndIma label evolution does not match the requested image/ACS labels.');
end

phaseIsImg = strcmp(expectedPhase, 'IMG');
phaseIsAcs = strcmp(expectedPhase, 'ACS');
imgLabelPairs = [expectedPAR(phaseIsImg), expectedLIN(phaseIsImg)];
imgExpectedPairs0 = imgPairsGlobal - 1;
assert(size(unique(imgLabelPairs, 'rows'),1) == size(imgExpectedPairs0,1), 'Image loop has duplicate image PAR/LIN labels.');
assert(isempty(setdiff(imgExpectedPairs0, imgLabelPairs, 'rows')) && isempty(setdiff(imgLabelPairs, imgExpectedPairs0, 'rows')), ...
    'Image loop PAR/LIN labels do not match the requested undersampled image mask.');
assert(all(expectedREF(phaseIsImg) == 0), 'Image loop should have REF=false for all ADCs.');
assert(all(expectedIMA(phaseIsImg) == 0), 'Image loop should have IMA/PATRefAndIma=false for all ADCs.');

if acsEnabled
    acsLabelPairs = [expectedPAR(phaseIsAcs), expectedLIN(phaseIsAcs)];
    [acsLocalI, acsLocalJ] = ndgrid(0:ACS1num-1, 0:ACS2num-1);
    acsExpectedLocalPairs = [acsLocalI(:), acsLocalJ(:)];
    assert(size(acsLabelPairs,1) == ACS1num*ACS2num, 'ACS loop has the wrong number of ADC acquisitions.');
    assert(size(unique(acsLabelPairs, 'rows'),1) == ACS1num*ACS2num, 'ACS loop has duplicate ACS-local PAR/LIN labels.');
    assert(isempty(setdiff(acsExpectedLocalPairs, acsLabelPairs, 'rows')) && isempty(setdiff(acsLabelPairs, acsExpectedLocalPairs, 'rows')), ...
        'ACS loop PAR/LIN labels do not match local ACS coordinates.');
    assert(all(expectedREF(phaseIsAcs) == 1), 'ACS loop should have REF=true for all ADCs.');
    assert(all(expectedIMA(phaseIsAcs) == 0), 'ACS loop should have IMA/PATRefAndIma=false for all ADCs.');

    % The ACS gradients should still cover the desired global ACS target.
    acsGlobalPairs0 = [expectedGlobalPAR(phaseIsAcs), expectedGlobalLIN(phaseIsAcs)];
    acsExpectedGlobalPairs0 = acsPairsGlobal - 1;
    assert(isempty(setdiff(acsExpectedGlobalPairs0, acsGlobalPairs0, 'rows')) && isempty(setdiff(acsGlobalPairs0, acsExpectedGlobalPairs0, 'rows')), ...
        'ACS loop gradients do not cover the intended global ACS target region.');
else
    assert(~any(phaseIsAcs), 'ACS phase should be skipped when ACS is disabled.');
end

if isfield(adc_lbl, 'REF')
    fprintf('REF count: %d, PATRefAndIma count: %d\n', ...
        sum(adc_lbl.REF ~= 0), sum(adc_lbl.IMA ~= 0));
end

if R1 == 1 && R2 == 1 && ACS1num == 0 && ACS2num == 0
    assert(numel(expectedLIN) == N(ax.n2)*N(ax.n3), ...
        'Full-kspace sanity check failed: unexpected acquisition count.');
    assert(size(unique(imgLabelPairs, 'rows'),1) == N(ax.n2)*N(ax.n3), ...
        'Full-kspace sanity check failed: not every PE pair was acquired exactly once.');
    assert(all(expectedREF == 0), 'Full-kspace sanity check failed: REF should be false.');
    assert(all(expectedIMA == 0), 'Full-kspace sanity check failed: IMA/PATRefAndIma should be false.');
    fprintf('Full-kspace no-squeeze sanity check passed.\n');
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
seq.setDefinition('SliceThickness', fov(ax.n2) / N(ax.n2));        % [m] slab thickness — Siemens-recognized
seq.setDefinition('TR', TRout);                         % [s] — Siemens-recognized
seq.setDefinition('FlipAngle', alpha);                  % [deg] — Siemens-recognized
seq.setDefinition('Nx', N(1));                          % readout matrix — Siemens-recognized
seq.setDefinition('Ny', N(2));                          % full PE resolution — Siemens-recognized
seq.setDefinition('Nz', N(3));                          % full partition resolution — Siemens-recognized
res_mm = fov(:).' ./ N(:).' * 1e3;
seq.setDefinition('RequestedResolutionX_mm', res(1));
seq.setDefinition('RequestedResolutionY_mm', res(2));
seq.setDefinition('RequestedResolutionZ_mm', res(3));
seq.setDefinition('ResolutionX_mm', res_mm(1));
seq.setDefinition('ResolutionY_mm', res_mm(2));
seq.setDefinition('ResolutionZ_mm', res_mm(3));
seq.setDefinition('ro_os', ro_os);
seq.setDefinition('OrientationMapping', slOrientation);         % only when programming in saggital orientation
seq.setDefinition('ReceiverGainHigh',1);

% Add those to ensure image labels/ICE recon could run
phaseResolution = fov(ax.n1)/N(ax.n1) / (fov(ax.n3)/N(ax.n3));
seq.setDefinition('kSpaceCenterLine', centerPE2LineIdx-1);
seq.setDefinition('kSpaceCenterPartition', centerPE1LineIdx-1);
seq.setDefinition('PhaseResolution', phaseResolution);
seq.setDefinition('ReadoutOversamplingFactor', ro_os);
seq.setDefinition('PE1_R', R1);
seq.setDefinition('PE2_R', R2);
seq.setDefinition('PE1_ACS', ACS1num);
seq.setDefinition('PE2_ACS', ACS2num);
seq.setDefinition('PE1_ImgLines', nPE1Img);
seq.setDefinition('PE2_ImgLines', nPE2Img);
seq.setDefinition('PE1_ACSTargetLines', numel(PE1_acs_target));
seq.setDefinition('PE2_ACSTargetLines', numel(PE2_acs_target));
seq.setDefinition('ACS_Pairs', size(acsPairsGlobal,1));
seq.setDefinition('ETL_Target', ETLtarget);
seq.setDefinition('ETL_CenterSlot', centerSlotTarget - 1);  % 0-based slot index for diagnostics
seq.setDefinition('ETL_Mode_IMG', etlPlan_img.mode);
seq.setDefinition('ETL_SegLen_IMG', etlPlan_img.s);
seq.setDefinition('ETL_SegmentsPerKy_IMG', etlPlan_img.K);
seq.setDefinition('ETL_SegmentsPerBlock_IMG', etlPlan_img.P);
seq.setDefinition('ETL_FillerPerKy_IMG', etlPlan_img.F);
seq.setDefinition('ETL_Efficiency_IMG', etlPlan_img.efficiency);
seq.setDefinition('ETL_Blocks_IMG', numel(imgBlocks));
seq.setDefinition('ETL_RealSlots_IMG', nImgRealSlots);
seq.setDefinition('ETL_DummySlots_IMG', nImgDummySlots);
if acsEnabled
    seq.setDefinition('PE2_Squeeze_ACS', squeeze_acs);
    seq.setDefinition('ETL_Blocks_ACS', numel(acsBlocks));
    seq.setDefinition('ETL_RealSlots_ACS', nAcsRealSlots);
    seq.setDefinition('ETL_DummySlots_ACS', nAcsDummySlots);
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
tag_etl = ['_ETL', num2str(ETLtarget), '_', etlPlan_img.mode];

seqFilename = ['mprage_3d', tag_wave, '_', num2str(N(1)), 'x', num2str(N(2)), 'x', num2str(N(3)), ...
    tag_res, tag_etl, '_R1_', num2str(R1), '_R2_', num2str(R2), ...
    '_ACS1_', num2str(ACS1num), '_ACS2_', num2str(ACS2num), ...
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


%% very optional slow step, but useful for testing during development e.g. for the real TE, TR or for staying within slew rate limits  

rep = seq.testReport; 
fprintf([rep{:}]); 

%%
calculateKspacePP()
tic;
[kfa,ta,kf]=seq.calculateKspacePP();
toc
figure;plot(kf(1,:),kf(2,:));
hold on;plot(kfa(1,:),kfa(2,:),'r.');

%% Optional timing diagnostics from actual event times
% This block uses ADC sample times to estimate echo-center timing per echo.
% Run by commenting out the 'return' above.
warning('off', 'mr:restoreShape');
[ktraj_adc, t_adc, ktraj, t_ktraj, t_excitation, t_refocusing] = seq.calculateKspacePP();

nAdcSamples = numel(t_adc);
adcSamplesPerReadout = adc.numSamples;
assert(mod(nAdcSamples, adcSamplesPerReadout) == 0, ...
    'ADC sample count (%d) is not divisible by ADC samples/readout (%d).', nAdcSamples, adcSamplesPerReadout);
nReadouts = nAdcSamples / adcSamplesPerReadout;
Nechoes = 1;
assert(mod(nReadouts, Nechoes) == 0, 'Readout count (%d) is not divisible by Nechoes (%d).', nReadouts, Nechoes);
nExcWithAdc = nReadouts / Nechoes;

% Dummy excitations do not have ADC in this script, so use the tail excitations.
assert(numel(t_excitation) >= nExcWithAdc, 'Not enough excitation timestamps to match ADC readouts.');
t_exc_use = t_excitation(end - nExcWithAdc + 1:end);

% ADC center per readout: midpoint between first and last ADC sample times.
t_adc_2d = reshape(t_adc, adcSamplesPerReadout, nReadouts);
t_ro_center = 0.5 * (t_adc_2d(1,:) + t_adc_2d(end,:));
t_ro_center_2d = reshape(t_ro_center, Nechoes, nExcWithAdc).';  % [excitation x echo]

TE_meas = t_ro_center_2d - t_exc_use(:);  % [s], per excitation and per echo
TE_meas_mean = mean(TE_meas, 1);
TE_meas_std = std(TE_meas, 0, 1);

fprintf('\nMeasured TE from ADC-center relative to RF center (seq.calculateKspacePP):\n');
for c = 1:Nechoes
    fprintf('  Echo %d: mean = %.6f s, std = %.6g s\n', ...
        c, TE_meas_mean(c), TE_meas_std(c));
end

TR_meas = diff(t_exc_use);
fprintf('Measured TR from excitation centers: mean = %.6f s, std = %.6g s\n\n', ...
    mean(TR_meas), std(TR_meas));

warning('on', 'mr:restoreShape');

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

    % Build sine waveform
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


%% Fixed-ETL scheduling helper functions

function plan = chooseFixedETLPlan(M, E, opts)
    %CHOOSEFIXEDETLPLAN Choose segmented fractional-PE_y or dummy-fill mode.
    % M is the number of sampled PE_x positions after R1/ACS selection.
    % E is the desired fixed ETL / number of RF-readout slots per inversion.

    if M < 1 || M ~= round(M)
        error('M must be a positive integer.');
    end
    if E < 1 || E ~= round(E)
        error('E must be a positive integer.');
    end
    if M > E
        error('Sampled PE_x count M=%d exceeds ETL E=%d.', M, E);
    end

    divE = find(mod(E, 1:E) == 0);
    candidates = struct('s', {}, 'K', {}, 'P', {}, 'F', {}, 'saved', {}, 'efficiency', {});

    for ii = 1:numel(divE)
        s = divE(ii);
        K = ceil(M / s);      % number of kx segments per sampled PE_y line
        P = E / s;            % number of segment slots per inversion
        F = K*s - M;          % no-ADC filler slots per sampled PE_y line
        saved = E - K*s;      % extra slots available for the next PE_y line

        isValid = ...
            (P > K) && ...
            (s >= opts.sMin) && ...
            (K <= opts.KMax) && ...
            (P <= opts.PMax) && ...
            (F / (K*s) <= opts.fillerMax) && ...
            (saved >= opts.savedMin);

        if isValid
            c = numel(candidates) + 1;
            candidates(c).s = s;
            candidates(c).K = K;
            candidates(c).P = P;
            candidates(c).F = F;
            candidates(c).saved = saved;
            candidates(c).efficiency = M / (K*s);
        end
    end

    if isempty(candidates)
        plan.mode = 'dummy';
        plan.s = E;
        plan.K = 1;
        plan.P = 1;
        plan.F = E - M;
        plan.saved = 0;
        plan.efficiency = M / E;
    else
        % Prefer the largest segment length. This minimizes fragmentation and
        % PE_y switching. Tie-breaker: higher efficiency, then fewer segments.
        score = zeros(1, numel(candidates));
        for c = 1:numel(candidates)
            score(c) = candidates(c).s*1e6 + candidates(c).efficiency*1e3 - candidates(c).P;
        end
        [~, bestIdx] = max(score);
        best = candidates(bestIdx);

        plan.mode = 'segmented';
        plan.s = best.s;
        plan.K = best.K;
        plan.P = best.P;
        plan.F = best.F;
        plan.saved = best.saved;
        plan.efficiency = best.efficiency;
    end
end

function blocks = buildSegmentedFixedETLBlocks(PE1list, PE2list, E, plan, centerPE1Idx, centerPE2Idx, centerSlotTarget)
    %BUILDSEGMENTEDFIXEDETLBLOCKS Pack an undersampled image PE table into fixed-ETL blocks.
    %
    % For segmented mode, each PE_y line is split into K residue-class kx
    % segments. P segments are placed in each inversion block and expanded in
    % kx-major order. All blocks are then circularly shifted so that a kx=0
    % slot is at centerSlotTarget. If the block contains the true global
    % center (kx=0, ky=0), that real ADC slot is chosen as the center slot.

    if strcmp(plan.mode, 'dummy')
        blocks = buildRectangularFixedETLBlocks(PE1list, PE2list, 1, E, ...
            centerPE1Idx, centerPE2Idx, centerSlotTarget);
        return;
    end

    M = numel(PE1list);
    L = numel(PE2list);
    s = plan.s;
    K = plan.K;
    P = plan.P;

    centerIPos = findClosestListPos(PE1list, centerPE1Idx);
    centerJPos = findClosestListPos(PE2list, centerPE2Idx);

    % Segment stream: y0-seg1, y0-seg2, ..., y1-seg1, ...
    entryJPos = zeros(1, L*K);
    entrySeg  = zeros(1, L*K);
    cc = 0;
    for jPos = 1:L
        for segIdx = 1:K
            cc = cc + 1;
            entryJPos(cc) = jPos;
            entrySeg(cc)  = segIdx;
        end
    end

    nBlocks = ceil(numel(entryJPos) / P);
    blocks = makeEmptyFixedETLBlockStruct();

    for b = 1:nBlocks
        segSlots = cell(1, P);
        for e = 1:P
            entryIdx = (b-1)*P + e;
            if entryIdx <= numel(entryJPos)
                jPos = entryJPos(entryIdx);
                segIdx = entrySeg(entryIdx);
                kxPosList = segIdx:K:M;  % residue-class segment: odd/even for K=2
                segSlots{e} = makeSegmentSlotArrays(PE1list, PE2list, kxPosList, jPos, s, ...
                    centerIPos, centerJPos, true);
            else
                % Final block padding: full dummy segment at center kx / center ky.
                segSlots{e} = makeSegmentSlotArrays(PE1list, PE2list, [], centerJPos, s, ...
                    centerIPos, centerJPos, false);
            end
        end

        iGlobal = zeros(1, E);
        jGlobal = zeros(1, E);
        iPos    = zeros(1, E);
        jPos    = zeros(1, E);
        isAcquire = false(1, E);
        outSlot = 0;

        % kx-major expansion: local kx offset first, then segment entry.
        for t = 1:s
            for e = 1:P
                outSlot = outSlot + 1;
                iGlobal(outSlot) = segSlots{e}.iGlobal(t);
                jGlobal(outSlot) = segSlots{e}.jGlobal(t);
                iPos(outSlot)    = segSlots{e}.iPos(t);
                jPos(outSlot)    = segSlots{e}.jPos(t);
                isAcquire(outSlot) = segSlots{e}.isAcquire(t);
            end
        end

        block.iGlobal = iGlobal;
        block.jGlobal = jGlobal;
        block.iPos = iPos;
        block.jPos = jPos;
        block.isAcquire = isAcquire;
        block.centerSlot = [];
        block = forceCenterSlot(block, centerSlotTarget, centerPE1Idx, centerPE2Idx, ...
            centerIPos, centerJPos, PE1list, PE2list);
        blocks(end+1) = block; %#ok<AGROW>
    end
end

function blocks = buildRectangularFixedETLBlocks(PE1list, PE2list, pe2BlockSize, E, centerPE1Idx, centerPE2Idx, centerSlotTarget)
    %BUILDRECTANGULARFIXEDETLBLOCKS Existing rectangular PE_y squeezing + dummy padding.
    % The real acquisition order inside each block matches the original code:
    % PE_x position outer, local PE_y block inner.

    if pe2BlockSize < 1 || pe2BlockSize ~= round(pe2BlockSize)
        error('pe2BlockSize must be a positive integer.');
    end

    M = numel(PE1list);
    L = numel(PE2list);
    if M > E
        error('PE_x count M=%d exceeds fixed ETL E=%d.', M, E);
    end

    centerIPos = findClosestListPos(PE1list, centerPE1Idx);
    centerJPos = findClosestListPos(PE2list, centerPE2Idx);

    pe2BlockStartPos = 1:pe2BlockSize:L;
    blocks = makeEmptyFixedETLBlockStruct();

    for b = 1:numel(pe2BlockStartPos)
        pe2PosBlock = pe2BlockStartPos(b) : min(pe2BlockStartPos(b) + pe2BlockSize - 1, L);
        nReal = M * numel(pe2PosBlock);
        if nReal > E
            error('Rectangular block has %d real slots, exceeding ETLtarget=%d.', nReal, E);
        end

        iGlobal = zeros(1, E);
        jGlobal = zeros(1, E);
        iPos    = zeros(1, E);
        jPos    = zeros(1, E);
        isAcquire = false(1, E);

        outSlot = 0;
        for ii = 1:M
            for jj = pe2PosBlock
                outSlot = outSlot + 1;
                iPos(outSlot) = ii;
                jPos(outSlot) = jj;
                iGlobal(outSlot) = PE1list(ii);
                jGlobal(outSlot) = PE2list(jj);
                isAcquire(outSlot) = true;
            end
        end

        % Pad residual slots with no-ADC dummy readouts. Use the previous real
        % coordinate for smoothness when possible; forceCenterSlot can replace
        % one dummy by center kx/ky if the block has no real kx=0 slot.
        for ss = (outSlot+1):E
            if outSlot >= 1
                iPos(ss) = iPos(outSlot);
                jPos(ss) = jPos(outSlot);
                iGlobal(ss) = iGlobal(outSlot);
                jGlobal(ss) = jGlobal(outSlot);
            else
                iPos(ss) = centerIPos;
                jPos(ss) = centerJPos;
                iGlobal(ss) = PE1list(centerIPos);
                jGlobal(ss) = PE2list(centerJPos);
            end
            isAcquire(ss) = false;
        end

        block.iGlobal = iGlobal;
        block.jGlobal = jGlobal;
        block.iPos = iPos;
        block.jPos = jPos;
        block.isAcquire = isAcquire;
        block.centerSlot = [];
        block = forceCenterSlot(block, centerSlotTarget, centerPE1Idx, centerPE2Idx, ...
            centerIPos, centerJPos, PE1list, PE2list);
        blocks(end+1) = block; %#ok<AGROW>
    end
end

function seg = makeSegmentSlotArrays(PE1list, PE2list, kxPosList, jPos, s, centerIPos, centerJPos, useNearestDummy)
    %MAKESEGMENTSLOTARRAYS Build one fixed-length kx segment slot array.

    seg.iGlobal = zeros(1, s);
    seg.jGlobal = zeros(1, s);
    seg.iPos = zeros(1, s);
    seg.jPos = zeros(1, s);
    seg.isAcquire = false(1, s);

    if isempty(kxPosList)
        for t = 1:s
            seg.iPos(t) = centerIPos;
            seg.jPos(t) = centerJPos;
            seg.iGlobal(t) = PE1list(centerIPos);
            seg.jGlobal(t) = PE2list(centerJPos);
        end
        return;
    end

    nReal = numel(kxPosList);
    for t = 1:s
        if t <= nReal
            ii = kxPosList(t);
            seg.iPos(t) = ii;
            seg.jPos(t) = jPos;
            seg.iGlobal(t) = PE1list(ii);
            seg.jGlobal(t) = PE2list(jPos);
            seg.isAcquire(t) = true;
        else
            if useNearestDummy
                ii = kxPosList(end);
                jj = jPos;
            else
                ii = centerIPos;
                jj = centerJPos;
            end
            seg.iPos(t) = ii;
            seg.jPos(t) = jj;
            seg.iGlobal(t) = PE1list(ii);
            seg.jGlobal(t) = PE2list(jj);
            seg.isAcquire(t) = false;
        end
    end
end

function block = forceCenterSlot(block, targetSlot, centerPE1Idx, centerPE2Idx, centerIPos, centerJPos, PE1list, PE2list)
    %FORCECENTERSLOT Circularly shift the block so kx=0 is at targetSlot.
    % Priority: true global center real ADC, then any real kx=0 ADC, then a
    % dummy slot converted to kx=0/ky=0.

    realGlobalCenter = find(block.isAcquire & block.iGlobal == centerPE1Idx & block.jGlobal == centerPE2Idx);
    if ~isempty(realGlobalCenter)
        candidates = realGlobalCenter;
    else
        realKxCenter = find(block.isAcquire & block.iGlobal == centerPE1Idx);
        if ~isempty(realKxCenter)
            candidates = realKxCenter;
        else
            dummySlots = find(~block.isAcquire);
            if isempty(dummySlots)
                error(['No kx=0 slot and no dummy slot available to place at the ETL center. ', ...
                       'Relax the segmentation constraints or use dummy-fill mode.']);
            end
            [~, dd] = min(abs(dummySlots - targetSlot));
            dummyIdx = dummySlots(dd);
            block.iPos(dummyIdx) = centerIPos;
            block.jPos(dummyIdx) = centerJPos;
            block.iGlobal(dummyIdx) = PE1list(centerIPos);
            block.jGlobal(dummyIdx) = PE2list(centerJPos);
            candidates = dummyIdx;
        end
    end

    [~, cc] = min(abs(candidates - targetSlot));
    centerIdx = candidates(cc);
    shift = targetSlot - centerIdx;

    block.iGlobal = circshift(block.iGlobal, [0 shift]);
    block.jGlobal = circshift(block.jGlobal, [0 shift]);
    block.iPos = circshift(block.iPos, [0 shift]);
    block.jPos = circshift(block.jPos, [0 shift]);
    block.isAcquire = circshift(block.isAcquire, [0 shift]);
    block.centerSlot = targetSlot;

    if block.iGlobal(targetSlot) ~= centerPE1Idx
        error('Internal fixed-ETL scheduler error: target slot is not kx=0 after centering.');
    end
end

function pos = findClosestListPos(listVals, targetVal)
    if isempty(listVals)
        error('Cannot find a position in an empty PE list.');
    end
    [~, pos] = min(abs(listVals - targetVal));
end

function [nReal, nDummy] = countFixedETLBlocks(blocks)
    nReal = 0;
    nDummy = 0;
    for b = 1:numel(blocks)
        nReal = nReal + sum(blocks(b).isAcquire);
        nDummy = nDummy + sum(~blocks(b).isAcquire);
    end
end

function assertGlobalCenterAtTarget(blocks, centerPE1Idx, centerPE2Idx, centerSlotTarget, phaseName)
    found = false;
    for b = 1:numel(blocks)
        idx = find(blocks(b).isAcquire & ...
                   blocks(b).iGlobal == centerPE1Idx & ...
                   blocks(b).jGlobal == centerPE2Idx);
        if ~isempty(idx)
            found = true;
            assert(numel(idx) == 1, '%s global k-space center appears more than once in block %d.', phaseName, b);
            assert(idx == centerSlotTarget, ...
                '%s global k-space center is at slot %d, expected ETL center slot %d.', ...
                phaseName, idx, centerSlotTarget);
        end
    end
    if ~found
        warning('%s global k-space center was not found among real ADC slots.', phaseName);
    end
end

function blocks = makeEmptyFixedETLBlockStruct()
    blocks = struct('iGlobal', {}, 'jGlobal', {}, 'iPos', {}, 'jPos', {}, 'isAcquire', {}, 'centerSlot', {});
end

%% Define sequence block

function pairs = makePEPairList(PE1list, PE2list, includePairMask)
    %MAKEPEPAIRLIST Ordered 1-based PE pair list with PE_x outer, PE_y inner.
    %
    % includePairMask is optional. If nonempty, only pairs whose mask entry
    % is true are returned. This is useful for ACS-only blocks where common
    % image/ACS lines are intentionally not reacquired.

    if nargin < 3
        includePairMask = [];
    end

    pairs = zeros(numel(PE1list)*numel(PE2list), 2);
    count = 0;
    for ii = 1:numel(PE1list)
        i = PE1list(ii);
        for jj = 1:numel(PE2list)
            j = PE2list(jj);
            if ~isempty(includePairMask) && ~includePairMask(i,j)
                continue;
            end
            count = count + 1;
            pairs(count,:) = [i, j];
        end
    end
    pairs = pairs(1:count,:);
end

function [PEsamp, PEsamp_u, PEsamp_ACS, centerLineIdx] = makePESamplingPattern(nPE, R, ACSnum)
    %MAKEPESAMPLINGPATTERN Centered undersampling pattern with optional ACS.
    %
    % Inputs
    %   nPE    : full number of phase-encode lines in this dimension
    %   R      : acceleration factor. R=1 gives all lines.
    %   ACSnum : number of fully sampled central ACS lines. 0 disables ACS.
    %
    % Outputs are 1-based full-matrix k-space indices. These are used to
    % index gradient tables directly; labels are written as index-1.

    if R < 1 || R ~= round(R)
        error('Acceleration factor R must be a positive integer.');
    end
    if ACSnum < 0 || ACSnum ~= round(ACSnum)
        error('ACSnum must be a non-negative integer.');
    end
    if ACSnum > nPE
        error('ACSnum cannot exceed the number of PE lines.');
    end

    centerLineIdx = floor(nPE/2) + 1;

    PEsamp_u = [];
    count = 1;
    for idx = 1:nPE
        if mod(idx - centerLineIdx, R) == 0
            PEsamp_u(count) = idx; %#ok<AGROW>
            count = count + 1;
        end
    end

    if ACSnum > 0
        acsStart = centerLineIdx - floor(ACSnum/2);
        acsStart = max(1, min(acsStart, nPE - ACSnum + 1));
        PEsamp_ACS = acsStart : (acsStart + ACSnum - 1);
    else
        PEsamp_ACS = [];
    end

    PEsamp = union(PEsamp_u, PEsamp_ACS);
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

