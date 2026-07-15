function block = forceCenterSlot(block, targetSlot, centerPE1Idx, centerPE2Idx, centerIPos, centerJPos, PE1list, PE2list)
    %FORCECENTERSLOT Circularly shift the block so kx=0 is at targetSlot.
    % Priority: true global center real ADC, then any real kx=0 ADC, then a
    % dummy slot converted to kx=0/ky=0.

    realGlobalCenter = find(block.isAcquire & block.iGlobal == centerPE1Idx & block.jGlobal == centerPE2Idx);
    if ~isempty(realGlobalCenter)
        candidates = realGlobalCenter;
    else
        realKxCenter = find(block.isAcquire & block.iGlobal == centerPE1Idx);
        if ~isempty(realKxCenter)
            candidates = realKxCenter;
        else
            dummySlots = find(~block.isAcquire);
            if isempty(dummySlots)
                error(['No kx=0 slot and no dummy slot available to place at the ETL center. ', ...
                       'Relax the segmentation constraints or use dummy-fill mode.']);
            end
            [~, dd] = min(abs(dummySlots - targetSlot));
            dummyIdx = dummySlots(dd);
            block.iPos(dummyIdx) = centerIPos;
            block.jPos(dummyIdx) = centerJPos;
            block.iGlobal(dummyIdx) = PE1list(centerIPos);
            block.jGlobal(dummyIdx) = PE2list(centerJPos);
            candidates = dummyIdx;
        end
    end

    [~, cc] = min(abs(candidates - targetSlot));
    centerIdx = candidates(cc);
    shift = targetSlot - centerIdx;

    block.iGlobal = circshift(block.iGlobal, [0 shift]);
    block.jGlobal = circshift(block.jGlobal, [0 shift]);
    block.iPos = circshift(block.iPos, [0 shift]);
    block.jPos = circshift(block.jPos, [0 shift]);
    block.isAcquire = circshift(block.isAcquire, [0 shift]);
    block.centerSlot = targetSlot;

    if block.iGlobal(targetSlot) ~= centerPE1Idx
        error('Internal fixed-ETL scheduler error: target slot is not kx=0 after centering.');
    end
end
