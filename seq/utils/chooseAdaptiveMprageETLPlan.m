function [selectedETL, selectedPlan, info] = chooseAdaptiveMprageETLPlan( ...
        M, L, searchOpts, planOpts, timing)
    %CHOOSEADAPTIVEMPRAGEETLPLAN Keep nominal ETL unless it is unsuitable.
    %
    % M and L are the actual sampled PAR/inner and LIN/outer line counts.
    % The existing fixed-ETL planner is evaluated first at nominalETL. An
    % outer search over [minETL,maxETL] is triggered only when the nominal
    % plan has a small common divisor, falls back to dummy mode, exceeds the
    % preferred total dummy fraction, or violates exact TI/TRout timing.

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
        triggerReasons{end+1} = 'nominal_below_sampled_PAR';
    else
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
        return;
    end

    candidate.exists = true;
    candidate.E = E;
    candidate.plan = chooseFixedETLPlan(M, E, planOpts);
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
