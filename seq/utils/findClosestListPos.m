function pos = findClosestListPos(listVals, targetVal)
    if isempty(listVals)
        error('Cannot find a position in an empty PE list.');
    end
    [~, pos] = min(abs(listVals - targetVal));
end
