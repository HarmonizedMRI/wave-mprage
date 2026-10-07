function waveform = sampleHighSlewTrapezoidOnRaster(gradient, duration, dt)
    assert(strcmp(gradient.type, 'trap'), ...
        'Expected a trapezoid for raster sampling.');
    times = [0, gradient.riseTime, ...
        gradient.riseTime+gradient.flatTime, ...
        gradient.riseTime+gradient.flatTime+gradient.fallTime] ...
        + gradient.delay;
    amplitudes = [0, gradient.amplitude, gradient.amplitude, 0];
    [times, uniqueIndices] = unique(times, 'stable');
    amplitudes = amplitudes(uniqueIndices);
    nCells = round(duration/dt);
    cellCenters = ((0:nCells-1)+0.5)*dt;
    waveform = interp1(times, amplitudes, cellCenters, 'linear', 0);
    waveform = waveform(:).';
end
