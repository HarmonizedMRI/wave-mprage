function [PEsamp, centerLineIdx] = makeAcceleratedPESamplingPattern(nPE, R)
%MAKEACCELERATEDPESAMPLINGPATTERN Centered accelerated PE index list.
%
% Inputs
%   nPE : full number of phase-encode lines
%   R   : positive integer acceleration factor; R=1 returns all lines
%
% Outputs are 1-based full-matrix indices. The global k-space center is
% always included, and labels should be written as index-1.

    if nPE < 1 || nPE ~= round(nPE)
        error('nPE must be a positive integer.');
    end
    if R < 1 || R ~= round(R)
        error('Acceleration factor R must be a positive integer.');
    end

    centerLineIdx = floor(nPE/2) + 1;
    PEsamp = find(mod((1:nPE) - centerLineIdx, R) == 0);
end
