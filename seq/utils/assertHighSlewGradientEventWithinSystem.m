function assertHighSlewGradientEventWithinSystem(event, system, segmentName)
    if strcmp(event.type, 'grad')
        if abs(event.tt(1)) <= system.gradRasterTime/10
            gradPeak = max(abs(event.waveform));
            slewPeak = max(abs(diff(event.waveform)./diff(event.tt)));
            tol = 1 + 1e-10;
            assert(gradPeak <= system.maxGrad*tol, ...
                '%s exceeds its assigned gradient limit.', segmentName);
            assert(slewPeak <= system.maxSlew*tol, ...
                '%s exceeds its assigned slew limit.', segmentName);
        else
            assertHighSlewArbitraryWaveformWithinSystem( ...
                event.waveform, event.first, event.last, system, segmentName);
        end
    elseif strcmp(event.type, 'trap')
        gradPeak = abs(event.amplitude);
        slewPeak = max(abs([event.amplitude/event.riseTime, ...
            event.amplitude/event.fallTime]));
        tol = 1 + 1e-10;
        assert(gradPeak <= system.maxGrad*tol, ...
            '%s exceeds its assigned gradient limit.', segmentName);
        assert(slewPeak <= system.maxSlew*tol, ...
            '%s exceeds its assigned slew limit.', segmentName);
    else
        error('Unsupported gradient event type for %s.', segmentName);
    end
end
