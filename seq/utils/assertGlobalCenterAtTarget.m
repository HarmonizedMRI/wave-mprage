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
