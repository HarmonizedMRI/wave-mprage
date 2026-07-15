function pairs = makePEPairList(PE1list, PE2list, includePairMask)
    %MAKEPEPAIRLIST Ordered 1-based PE pair list with PE_x outer, PE_y inner.
    %
    % includePairMask is optional. If nonempty, only pairs whose mask entry
    % is true are returned. This is useful for ACS-only blocks where common
    % image/ACS lines are intentionally not reacquired.

    if nargin < 3
        includePairMask = [];
    end

    pairs = zeros(numel(PE1list)*numel(PE2list), 2);
    count = 0;
    for ii = 1:numel(PE1list)
        i = PE1list(ii);
        for jj = 1:numel(PE2list)
            j = PE2list(jj);
            if ~isempty(includePairMask) && ~includePairMask(i,j)
                continue;
            end
            count = count + 1;
            pairs(count,:) = [i, j];
        end
    end
    pairs = pairs(1:count,:);
end
