function [gpe_wave_sin, gpe_post, offsetInfo] = defineHighSlewSineWaveGradient( ...
        Tread, sys, sys_sequence, sys_wave, sys_lowPNS, ...
        Ncycles, gwave_max, swave_max, ...
        gpePre, gro, adc, physical_slew_max, centerWaveAroundNowave, ...
        cosRampUpDuration, cosRampDownDuration, waveInfoFlag, debugFlag)
    %DEFINESINEWAVEGRADIENT Create PE sine wave and optional centering blips.
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
    %   centerWaveAroundNowave : select zero-crossing or centered-corkscrew
    %   cosRampUp/DownDuration : preferred durations for those trapezoids
    if nargin < 17
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
    % Include the endpoint sample so the integer-cycle sine returns to zero.
    tWavepoints_sin = (0:wavepoints_sin) * tWaveUnit_sin;
    gWave_sin_active = G0_sin_pulseq * sin(w_sin * tWavepoints_sin);

    idealKRadius = G0_sin_pulseq/w_sin;
    if mod(Ncycles, 2) == 0
        % Preserve the accepted even-cycle behavior: the rasterized
        % half-readout sine moment closes, while the centered option uses
        % the established continuous radius G0/omega. This covers the
        % high-slew C10/A12.732 and C20/A6.3662 cases.
        rawCenterMoment = 0;
        centeringKRadius = idealKRadius;
    else
        % For the supported 25-cycle case, the first half-readout contains
        % an unmatched half sine-cycle. Use the exact Pulseq cell-centered
        % raster sum, whose closed form is G0*dt*cot(pi*N/M).
        assert(Ncycles == 25, 'Only the Ncycles=25 odd case is supported.');
        assert(mod(wavepoints_sin, 2) == 0, ...
            'Odd-cycle center correction requires an even raster length.');
        nHalfWaveCells = wavepoints_sin/2;
        rawCenterMoment = sum(gWave_sin_active(1:nHalfWaveCells)) ...
            * tWaveUnit_sin;
        expectedRawCenterMoment = G0_sin_pulseq*tWaveUnit_sin ...
            * cot(pi*Ncycles/wavepoints_sin);
        rasterAreaTol = max(1e-9, 1e-10*abs(expectedRawCenterMoment));
        assert(abs(rawCenterMoment-expectedRawCenterMoment) <= rasterAreaTol, ...
            'Odd-cycle sine center moment disagrees with the finite-sum formula.');
        centeringKRadius = rawCenterMoment/2;
    end

    if centerWaveAroundNowave
        sinePreOffsetArea = -centeringKRadius;
    else
        sinePreOffsetArea = -rawCenterMoment;
    end
    sinePostOffsetArea = -sinePreOffsetArea;

    gWave_sin = gWave_sin_active;

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
    gWave_sin_without_prepad = gWave_sin;
    % Pad the front
    nPadPre = round(gro.riseTime / sys.gradRasterTime);
    gWave_sin = [zeros(1, nPadPre) gWave_sin];
    gWave_sin = gWave_sin(:).';

    gWave_sin_helper = mr.makeArbitraryGrad(gpe_channel, gWave_sin, ...
        'system', sys_wave, 'first', 0, 'last', 0);

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
    assertHighSlewArbitraryWaveformWithinSystem( ...
        preWave, 0, 0, sys_lowPNS, 'sine PE prephaser');
    assertHighSlewArbitraryWaveformWithinSystem( ...
        gWave_sin, 0, 0, sys_wave, 'active sine wave');
    gpe_wave_sin = mr.makeArbitraryGrad(gpe_channel, gpeWaveFull, ...
        'system', sys_sequence, 'first', 0, 'last', 0);

    % Nominal post-rewinder area is just -gpePre.area.
    % Then compensate the residual area of the sine waveform.
    gpePost_area_new = -gpe_wave_sin.area;

    gpe_post = mr.makeTrapezoid(gpe_channel, 'Area', gpePost_area_new, 'system', sys_lowPNS);

    % Preserve the complete baseline waveform above. Add only the selected
    % sine-axis pre/post pair. For 25 cycles, option-off cancels the exact
    % half-cycle raster moment; option-on uses half that area to center the
    % sampled corkscrew. Cosine and existing PE/ramp compensation are intact.
    offsetInfo = struct( ...
        'enabled', false, ...
        'centerWaveAroundNowave', centerWaveAroundNowave, ...
        'idealKRadius', idealKRadius, ...
        'kRadius', centeringKRadius, ...
        'rawCenterMoment', rawCenterMoment, ...
        'preArea', sinePreOffsetArea, ...
        'postArea', sinePostOffsetArea, ...
        'preDuration', 0, ...
        'postDuration', 0, ...
        'preMatchedCosineDuration', false, ...
        'postMatchedCosineDuration', false);

    if abs(sinePreOffsetArea) > max(1e-12, eps(max(1, abs(sinePreOffsetArea))))
        gpe_wave_sin_baseline = gpe_wave_sin;
        gpe_post_baseline = gpe_post;

        % Design PE+offset as one low-PNS trapezoid. Superposing two
        % individually legal trapezoids can exceed the assigned slew limit.
        mergedPreArea = gpePre.area + sinePreOffsetArea;
        [gSinPreMerged, preMatched] = makeHighSlewAreaTrapezoidNearDuration( ...
            gpe_channel, mergedPreArea, cosRampUpDuration, sys_lowPNS);
        mergedPreDuration = mr.calcDuration(gSinPreMerged);
        if abs(mergedPreDuration-adc.delay) > sys.gradRasterTime/10
            error(['Merged sine PE/offset duration %.6f ms does not match ', ...
                'the active-wave start at %.6f ms.'], ...
                mergedPreDuration*1e3, adc.delay*1e3);
        end
        mergedPreWave = sampleHighSlewTrapezoidOnRaster( ...
            gSinPreMerged, mergedPreDuration, sys.gradRasterTime);
        gpeWaveFullMerged = [mergedPreWave, gWave_sin_without_prepad];
        gpe_wave_sin = mr.makeArbitraryGrad( ...
            gpe_channel, gpeWaveFullMerged, 'system', sys_sequence, ...
            'first', 0, 'last', 0);

        mergedPostArea = -gpe_wave_sin.area;
        [gpe_post, postMatched] = makeHighSlewAreaTrapezoidNearDuration( ...
            gpe_channel, mergedPostArea, cosRampDownDuration, sys_lowPNS);

        areaTol = max(1e-9, 1e-10*max(abs( ...
            [sinePreOffsetArea sinePostOffsetArea])));
        assert(abs((gpe_wave_sin.area-gpe_wave_sin_baseline.area) ...
            - sinePreOffsetArea) <= areaTol, ...
            'Sine pre-offset area changed in the merged prephaser.');
        assert(abs((gpe_post.area-gpe_post_baseline.area) ...
            - sinePostOffsetArea) <= areaTol, ...
            'Sine post-offset area changed in the merged rewinder.');

        offsetInfo.enabled = true;
        offsetInfo.preArea = sinePreOffsetArea;
        offsetInfo.postArea = sinePostOffsetArea;
        offsetInfo.preDuration = mr.calcDuration(gSinPreMerged);
        offsetInfo.postDuration = mr.calcDuration(gpe_post);
        offsetInfo.preMatchedCosineDuration = preMatched;
        offsetInfo.postMatchedCosineDuration = postMatched;
    end

    % Everything before the first active sine cell, including the optional
    % offset trapezoid, must still satisfy the low-PNS envelope.
    nLowPNSPreCells = round(adc.delay/sys.gradRasterTime);
    assertHighSlewArbitraryWaveformWithinSystem( ...
        gpe_wave_sin.waveform(1:nLowPNSPreCells), 0, 0, ...
        sys_lowPNS, 'sine PE/offset/ramp-up');
    assertHighSlewGradientEventWithinSystem( ...
        gpe_post, sys_lowPNS, 'sine offset/rewinder/ramp-down');

    % Sanity check of time
    tol = sys.gradRasterTime/10;
    adcEndObject = mr.calcDuration(adc);
    fullDur_obj    = mr.calcDuration(gpe_wave_sin);
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

        if offsetInfo.enabled
            fprintf('\n--- Sine parity/centering offset ---\n');
            fprintf('ideal radius G0/omega                = %.9g 1/m\n', ...
                offsetInfo.idealKRadius);
            fprintf('effective raster radius              = %.9g 1/m\n', ...
                offsetInfo.kRadius);
            fprintf('raw ADC-center sine moment           = %+.9g 1/m\n', ...
                offsetInfo.rawCenterMoment);
            fprintf('pre/post offset areas                = %+.9g / %+.9g 1/m\n', ...
                offsetInfo.preArea, offsetInfo.postArea);
            fprintf('pre/post offset durations            = %.6f / %.6f ms\n', ...
                offsetInfo.preDuration*1e3, offsetInfo.postDuration*1e3);
            fprintf('matched cosine pre/post durations    = %d / %d\n', ...
                offsetInfo.preMatchedCosineDuration, ...
                offsetInfo.postMatchedCosineDuration);
        end

        fprintf('======================================================\n\n');

    end
end
