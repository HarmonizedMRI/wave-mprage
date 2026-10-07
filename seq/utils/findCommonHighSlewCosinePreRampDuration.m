function duration = findCommonHighSlewCosinePreRampDuration( ...
        targetAreas, endAmplitude, minDuration, system)
    dt = system.gradRasterTime;
    duration = ceil(minDuration/dt)*dt;
    maxDuration = 20e-3;
    while duration <= maxDuration + dt/10
        feasible = true;
        for ii = 1:numel(targetAreas)
            try
                makeHighSlewFixedDurationPreRamp( ...
                    'x', targetAreas(ii), endAmplitude, duration, system);
            catch
                feasible = false;
                break;
            end
        end
        if feasible
            return;
        end
        duration = duration + dt;
    end
    error(['Could not find a common low-PNS cosine pre-ramp duration ', ...
        'within %.3f ms.'], maxDuration*1e3);
end
