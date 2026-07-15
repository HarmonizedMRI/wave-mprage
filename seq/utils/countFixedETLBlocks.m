function [nReal, nDummy] = countFixedETLBlocks(blocks)
    nReal = 0;
    nDummy = 0;
    for b = 1:numel(blocks)
        nReal = nReal + sum(blocks(b).isAcquire);
        nDummy = nDummy + sum(~blocks(b).isAcquire);
    end
end
