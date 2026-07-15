function seg = makeSegmentSlotArrays(PE1list, PE2list, kxPosList, jPos, s, centerIPos, centerJPos, useNearestDummy)
    %MAKESEGMENTSLOTARRAYS Build one fixed-length kx segment slot array.

    seg.iGlobal = zeros(1, s);
    seg.jGlobal = zeros(1, s);
    seg.iPos = zeros(1, s);
    seg.jPos = zeros(1, s);
    seg.isAcquire = false(1, s);

    if isempty(kxPosList)
        for t = 1:s
            seg.iPos(t) = centerIPos;
            seg.jPos(t) = centerJPos;
            seg.iGlobal(t) = PE1list(centerIPos);
            seg.jGlobal(t) = PE2list(centerJPos);
        end
        return;
    end

    nReal = numel(kxPosList);
    for t = 1:s
        if t <= nReal
            ii = kxPosList(t);
            seg.iPos(t) = ii;
            seg.jPos(t) = jPos;
            seg.iGlobal(t) = PE1list(ii);
            seg.jGlobal(t) = PE2list(jPos);
            seg.isAcquire(t) = true;
        else
            if useNearestDummy
                ii = kxPosList(end);
                jj = jPos;
            else
                ii = centerIPos;
                jj = centerJPos;
            end
            seg.iPos(t) = ii;
            seg.jPos(t) = jj;
            seg.iGlobal(t) = PE1list(ii);
            seg.jGlobal(t) = PE2list(jj);
            seg.isAcquire(t) = false;
        end
    end
end
