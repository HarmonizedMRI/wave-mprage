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
