function blocks = buildRectangularFixedETLBlocks(PE1list, PE2list, pe2BlockSize, E, centerPE1Idx, centerPE2Idx, centerSlotTarget)
    %BUILDRECTANGULARFIXEDETLBLOCKS Existing rectangular PE_y squeezing + dummy padding.
    % The real acquisition order inside each block matches the original code:
    % PE_x position outer, local PE_y block inner.

    if pe2BlockSize < 1 || pe2BlockSize ~= round(pe2BlockSize)
        error('pe2BlockSize must be a positive integer.');
    end

    M = numel(PE1list);
    L = numel(PE2list);
    if M > E
        error('PE_x count M=%d exceeds fixed ETL E=%d.', M, E);
    end

    centerIPos = findClosestListPos(PE1list, centerPE1Idx);
    centerJPos = findClosestListPos(PE2list, centerPE2Idx);

    pe2BlockStartPos = 1:pe2BlockSize:L;
    blocks = makeEmptyFixedETLBlockStruct();

    for b = 1:numel(pe2BlockStartPos)
        pe2PosBlock = pe2BlockStartPos(b) : min(pe2BlockStartPos(b) + pe2BlockSize - 1, L);
        nReal = M * numel(pe2PosBlock);
        if nReal > E
            error('Rectangular block has %d real slots, exceeding ETLtarget=%d.', nReal, E);
        end

        iGlobal = zeros(1, E);
        jGlobal = zeros(1, E);
        iPos    = zeros(1, E);
        jPos    = zeros(1, E);
        isAcquire = false(1, E);

        outSlot = 0;
        for ii = 1:M
            for jj = pe2PosBlock
                outSlot = outSlot + 1;
                iPos(outSlot) = ii;
                jPos(outSlot) = jj;
                iGlobal(outSlot) = PE1list(ii);
                jGlobal(outSlot) = PE2list(jj);
                isAcquire(outSlot) = true;
            end
        end

        % Pad residual slots with no-ADC dummy readouts. Use the previous real
        % coordinate for smoothness when possible; forceCenterSlot can replace
        % one dummy by center kx/ky if the block has no real kx=0 slot.
        for ss = (outSlot+1):E
            if outSlot >= 1
                iPos(ss) = iPos(outSlot);
                jPos(ss) = jPos(outSlot);
                iGlobal(ss) = iGlobal(outSlot);
                jGlobal(ss) = jGlobal(outSlot);
            else
                iPos(ss) = centerIPos;
                jPos(ss) = centerJPos;
                iGlobal(ss) = PE1list(centerIPos);
                jGlobal(ss) = PE2list(centerJPos);
            end
            isAcquire(ss) = false;
        end

        block.iGlobal = iGlobal;
        block.jGlobal = jGlobal;
        block.iPos = iPos;
        block.jPos = jPos;
        block.isAcquire = isAcquire;
        block.centerSlot = [];
        block = forceCenterSlot(block, centerSlotTarget, centerPE1Idx, centerPE2Idx, ...
            centerIPos, centerJPos, PE1list, PE2list);
        blocks(end+1) = block; %#ok<AGROW>
    end
end
