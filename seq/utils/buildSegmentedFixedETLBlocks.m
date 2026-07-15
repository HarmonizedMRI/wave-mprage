function blocks = buildSegmentedFixedETLBlocks(PE1list, PE2list, E, plan, centerPE1Idx, centerPE2Idx, centerSlotTarget)
    %BUILDSEGMENTEDFIXEDETLBLOCKS Pack an undersampled image PE table into fixed-ETL blocks.
    %
    % For segmented mode, each PE_y line is split into K residue-class kx
    % segments. P segments are placed in each inversion block and expanded in
    % kx-major order. All blocks are then circularly shifted so that a kx=0
    % slot is at centerSlotTarget. If the block contains the true global
    % center (kx=0, ky=0), that real ADC slot is chosen as the center slot.

    if strcmp(plan.mode, 'dummy')
        blocks = buildRectangularFixedETLBlocks(PE1list, PE2list, 1, E, ...
            centerPE1Idx, centerPE2Idx, centerSlotTarget);
        return;
    end

    M = numel(PE1list);
    L = numel(PE2list);
    s = plan.s;
    K = plan.K;
    P = plan.P;

    centerIPos = findClosestListPos(PE1list, centerPE1Idx);
    centerJPos = findClosestListPos(PE2list, centerPE2Idx);

    % Segment stream: y0-seg1, y0-seg2, ..., y1-seg1, ...
    entryJPos = zeros(1, L*K);
    entrySeg  = zeros(1, L*K);
    cc = 0;
    for jPos = 1:L
        for segIdx = 1:K
            cc = cc + 1;
            entryJPos(cc) = jPos;
            entrySeg(cc)  = segIdx;
        end
    end

    nBlocks = ceil(numel(entryJPos) / P);
    blocks = makeEmptyFixedETLBlockStruct();

    for b = 1:nBlocks
        segSlots = cell(1, P);
        for e = 1:P
            entryIdx = (b-1)*P + e;
            if entryIdx <= numel(entryJPos)
                jPos = entryJPos(entryIdx);
                segIdx = entrySeg(entryIdx);
                kxPosList = segIdx:K:M;  % residue-class segment: odd/even for K=2
                segSlots{e} = makeSegmentSlotArrays(PE1list, PE2list, kxPosList, jPos, s, ...
                    centerIPos, centerJPos, true);
            else
                % Final block padding: full dummy segment at center kx / center ky.
                segSlots{e} = makeSegmentSlotArrays(PE1list, PE2list, [], centerJPos, s, ...
                    centerIPos, centerJPos, false);
            end
        end

        iGlobal = zeros(1, E);
        jGlobal = zeros(1, E);
        iPos    = zeros(1, E);
        jPos    = zeros(1, E);
        isAcquire = false(1, E);
        outSlot = 0;

        % kx-major expansion: local kx offset first, then segment entry.
        for t = 1:s
            for e = 1:P
                outSlot = outSlot + 1;
                iGlobal(outSlot) = segSlots{e}.iGlobal(t);
                jGlobal(outSlot) = segSlots{e}.jGlobal(t);
                iPos(outSlot)    = segSlots{e}.iPos(t);
                jPos(outSlot)    = segSlots{e}.jPos(t);
                isAcquire(outSlot) = segSlots{e}.isAcquire(t);
            end
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
