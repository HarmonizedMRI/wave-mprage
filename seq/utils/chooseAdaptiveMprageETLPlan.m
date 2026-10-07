function [selectedETL, selectedPlan, info] = chooseAdaptiveMprageETLPlan( ...
        M, L, searchOpts, planOpts, timing)
    %CHOOSEADAPTIVEMPRAGEETLPLAN Keep nominal ETL unless it is unsuitable.
    %
    % M and L are the actual sampled PAR/inner and LIN/outer line counts.
    % The existing fixed-ETL planner is evaluated first at nominalETL. An
    % outer search over [minETL,maxETL] is triggered only when the nominal
    % plan has a small common divisor, falls back to dummy mode, exceeds the
    % preferred total dummy fraction, or violates exact TI/TRout timing. If
    % sampled PAR exceeds a candidate ETL, PAR remains the inner direction
    % and is segmented across multiple inversion blocks.

    validatePositiveInteger(M, 'M');
    validatePositiveInteger(L, 'L');

    requiredSearchFields = {'nominalETL', 'minETL', 'maxETL', ...
        'minCommonDivisor', 'preferredDummyFraction', ...
        'hardDummyFraction'};
    for iField = 1:numel(requiredSearchFields)
        fieldName = requiredSearchFields{iField};
        if ~isfield(searchOpts, fieldName)
            error('searchOpts.%s is required.', fieldName);
        end
    end

    validatePositiveInteger(searchOpts.nominalETL, ...
        'searchOpts.nominalETL');
    validatePositiveInteger(searchOpts.minETL, 'searchOpts.minETL');
    validatePositiveInteger(searchOpts.maxETL, 'searchOpts.maxETL');
    validatePositiveInteger(searchOpts.minCommonDivisor, ...
        'searchOpts.minCommonDivisor');
    if searchOpts.minETL > searchOpts.maxETL
        error('searchOpts.minETL must not exceed searchOpts.maxETL.');
    end
    if searchOpts.nominalETL > searchOpts.maxETL
        error('searchOpts.nominalETL must not exceed searchOpts.maxETL.');
    end
    if searchOpts.preferredDummyFraction < 0 || ...
            searchOpts.preferredDummyFraction > 1
        error('searchOpts.preferredDummyFraction must be in [0,1].');
    end
    if searchOpts.hardDummyFraction < ...
            searchOpts.preferredDummyFraction || ...
            searchOpts.hardDummyFraction > 1
        error(['searchOpts.hardDummyFraction must be in ', ...
            '[preferredDummyFraction,1].']);
    end

    nominal = evaluateCandidate(M, L, searchOpts.nominalETL, ...
        planOpts, timing);
    triggerReasons = {};
    if ~nominal.exists
        triggerReasons{end+1} = 'nominal_no_feasible_segmentation';
    else
        if M > searchOpts.nominalETL
            triggerReasons{end+1} = 'sampled_PAR_exceeds_nominal';
        end
        if nominal.commonDivisor < searchOpts.minCommonDivisor
            triggerReasons{end+1} = 'small_common_divisor';
        end
        if strcmp(nominal.plan.mode, 'dummy')
            triggerReasons{end+1} = 'dummy_mode';
        end
        if nominal.dummyFraction > searchOpts.preferredDummyFraction
            triggerReasons{end+1} = 'excess_dummy_fraction';
        end
        if ~nominal.timingOK
            triggerReasons{end+1} = 'timing_infeasible';
        end
    end

    if isempty(triggerReasons)
        selected = nominal;
        searchTriggered = false;
        candidatesEvaluated = 1;
    else
        searchTriggered = true;
        selected = emptyCandidate();
        candidatesEvaluated = 0;
        for E = searchOpts.minETL:searchOpts.maxETL
            candidate = evaluateCandidate(M, L, E, planOpts, timing);
            if ~candidate.exists
                continue;
            end
            candidatesEvaluated = candidatesEvaluated + 1;
            if ~candidate.timingOK || ...
                    candidate.dummyFraction > searchOpts.hardDummyFraction
                continue;
            end
            candidate.rankClass = candidateClass(candidate, searchOpts);
            if ~selected.exists || isBetterCandidate(candidate, selected, ...
                    searchOpts.nominalETL)
                selected = candidate;
            end
        end
        if ~selected.exists
            error(['No feasible MPRAGE ETL in [%d,%d] for sampled ', ...
                'PAR=%d, LIN=%d. Nominal trigger: %s.'], ...
                searchOpts.minETL, searchOpts.maxETL, M, L, ...
                strjoin(triggerReasons, ','));
        end
    end

    selectedETL = selected.E;
    selectedPlan = selected.plan;
    info = struct;
    info.searchTriggered = searchTriggered;
    if searchTriggered
        info.triggerReason = strjoin(triggerReasons, ',');
    else
        info.triggerReason = 'none';
    end
    info.candidatesEvaluated = candidatesEvaluated;
    info.nominalETL = searchOpts.nominalETL;
    info.minETL = searchOpts.minETL;
    info.maxETL = searchOpts.maxETL;
    info.centerSlot = selected.centerSlot;
    info.commonDivisor = selected.commonDivisor;
    info.nBlocks = selected.nBlocks;
    info.realSlots = selected.realSlots;
    info.dummySlots = selected.dummySlots;
    info.dummyFraction = selected.dummyFraction;
    info.totalEfficiency = selected.realSlots / selected.totalSlots;
    info.tiDelay = selected.tiDelay;
    info.troutDelay = selected.troutDelay;
end

function candidate = evaluateCandidate(M, L, E, planOpts, timing)
    candidate = emptyCandidate();
    if E < M
        plan = chooseLongParFixedETLPlan(M, E, planOpts);
        if isempty(plan)
            return;
        end
    else
        plan = chooseFixedETLPlan(M, E, planOpts);
    end

    candidate.exists = true;
    candidate.E = E;
    candidate.plan = plan;
    candidate.centerSlot = floor(E/2) + 1;
    candidate.commonDivisor = gcd(M, E);
    candidate.realSlots = M * L;

    if strcmp(candidate.plan.mode, 'dummy')
        candidate.nBlocks = L;
    else
        candidate.nBlocks = ceil(L * candidate.plan.K / candidate.plan.P);
    end
    candidate.totalSlots = candidate.nBlocks * E;
    candidate.dummySlots = candidate.totalSlots - candidate.realSlots;
    candidate.dummyFraction = candidate.dummySlots / candidate.totalSlots;

    rawTIDelay = timing.TI ...
        - (candidate.centerSlot-1) * timing.TRinner ...
        - timing.invTailToEnd - timing.rfStartToCenter;
    candidate.tiDelay = round(rawTIDelay / timing.blockRaster) ...
        * timing.blockRaster;
    candidate.troutDelay = timing.TRout - E * timing.TRinner ...
        - candidate.tiDelay - timing.invDuration;
    tol = timing.blockRaster / 10;
    candidate.timingOK = ...
        candidate.tiDelay >= timing.spoilerDuration - tol && ...
        candidate.troutDelay >= -tol;
    candidate.rankClass = inf;
end

function plan = chooseLongParFixedETLPlan(M, E, opts)
    % Split one sampled PAR line over multiple inversion blocks when M>E.
    % buildSegmentedFixedETLBlocks already supports this K>P layout. Every
    % residue-class segment must contain a filler slot so a block without a
    % real PAR-center sample can still place a dummy at the prescribed TI.

    divE = find(mod(E, 1:E) == 0);
    candidates = struct('s', {}, 'K', {}, 'P', {}, 'F', {}, ...
        'saved', {}, 'efficiency', {});

    for ii = 1:numel(divE)
        s = divE(ii);
        K = ceil(M / s);
        P = E / s;
        F = K*s - M;

        isValid = ...
            (K > P) && ...
            (s >= opts.sMin) && ...
            (K <= opts.KMax) && ...
            (P <= opts.PMax) && ...
            (F / (K*s) <= opts.fillerMax) && ...
            (s > ceil(M / K));

        if isValid
            c = numel(candidates) + 1;
            candidates(c).s = s;
            candidates(c).K = K;
            candidates(c).P = P;
            candidates(c).F = F;
            candidates(c).saved = E - K*s;
            candidates(c).efficiency = M / (K*s);
        end
    end

    if isempty(candidates)
        plan = [];
        return;
    end

    score = zeros(1, numel(candidates));
    for c = 1:numel(candidates)
        score(c) = candidates(c).s*1e6 ...
            + candidates(c).efficiency*1e3 - candidates(c).P;
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

function rankClass = candidateClass(candidate, searchOpts)
    isSegmented = strcmp(candidate.plan.mode, 'segmented');
    hasGoodDivisor = ...
        candidate.commonDivisor >= searchOpts.minCommonDivisor;
    hasPreferredDummy = ...
        candidate.dummyFraction <= searchOpts.preferredDummyFraction;

    if isSegmented && hasGoodDivisor && hasPreferredDummy
        rankClass = 0;
    elseif isSegmented && hasPreferredDummy
        rankClass = 1;
    elseif hasGoodDivisor
        rankClass = 2;
    else
        rankClass = 3;
    end
end

function tf = isBetterCandidate(a, b, nominalETL)
    keysA = [a.rankClass, a.nBlocks, ...
        abs(a.centerSlot-(floor(nominalETL/2)+1)), ...
        a.dummySlots, -a.plan.s, -a.E];
    keysB = [b.rankClass, b.nBlocks, ...
        abs(b.centerSlot-(floor(nominalETL/2)+1)), ...
        b.dummySlots, -b.plan.s, -b.E];
    tf = false;
    for iKey = 1:numel(keysA)
        if keysA(iKey) < keysB(iKey)
            tf = true;
            return;
        elseif keysA(iKey) > keysB(iKey)
            return;
        end
    end
end

function candidate = emptyCandidate()
    candidate = struct( ...
        'exists', false, ...
        'E', [], ...
        'plan', [], ...
        'centerSlot', [], ...
        'commonDivisor', [], ...
        'nBlocks', [], ...
        'realSlots', [], ...
        'totalSlots', [], ...
        'dummySlots', [], ...
        'dummyFraction', [], ...
        'tiDelay', [], ...
        'troutDelay', [], ...
        'timingOK', false, ...
        'rankClass', inf);
end

function validatePositiveInteger(value, name)
    if ~isscalar(value) || value < 1 || value ~= round(value)
        error('%s must be a positive integer.', name);
    end
end
