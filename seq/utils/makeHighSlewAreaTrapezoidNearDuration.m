function [gTrap, matchedPreferredDuration] = makeHighSlewAreaTrapezoidNearDuration( ...
        channel, area, preferredDuration, system)
    %MAKEAREATRAPEZOIDNEARDURATION Match a reference ramp when feasible.
    % The shortest feasible trapezoid is used only when the requested area
    % cannot fit inside the preferred cosine-ramp duration.
    dt = system.gradRasterTime;
    preferredDuration = round(preferredDuration/dt)*dt;
    gShortest = mr.makeTrapezoid(channel, 'Area', area, 'system', system);
    shortestDuration = mr.calcDuration(gShortest);

    if preferredDuration + dt/10 >= shortestDuration
        gTrap = mr.makeTrapezoid(channel, 'Area', area, ...
            'Duration', preferredDuration, 'system', system);
        matchedPreferredDuration = ...
            abs(mr.calcDuration(gTrap)-preferredDuration) <= dt/10;
    else
        gTrap = gShortest;
        matchedPreferredDuration = false;
    end
end
