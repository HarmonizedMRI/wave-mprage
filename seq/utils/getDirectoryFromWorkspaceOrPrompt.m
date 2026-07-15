function pathValue = getDirectoryFromWorkspaceOrPrompt(varName, promptText, allowEmpty, createIfMissing)
    pathValue = '';
    if evalin('base', sprintf('exist(''%s'', ''var'')', varName))
        pathValue = evalin('base', varName);
        if isstring(pathValue), pathValue = char(pathValue); end
        pathValue = normalizeUserPath(pathValue);
    end

    while isempty(pathValue) || ~exist(pathValue, 'dir')
        if ~isempty(pathValue) && ~exist(pathValue, 'dir')
            if createIfMissing
                reply = input(sprintf('%s does not exist: %s. Create it? [Y/n]: ', varName, pathValue), 's');
                if isempty(reply) || strcmpi(reply, 'y') || strcmpi(reply, 'yes')
                    mkdir(pathValue);
                    break;
                end
            else
                fprintf('%s does not exist: %s\n', varName, pathValue);
            end
        end

        if allowEmpty
            userText = input(sprintf('%s: ', promptText), 's');
            if isempty(strtrim(userText))
                pathValue = '';
                assignin('base', varName, pathValue);
                return;
            end
        else
            userText = input(sprintf('%s: ', promptText), 's');
            if isempty(strtrim(userText))
                fprintf('A valid path is required.\n');
                continue;
            end
        end
        pathValue = normalizeUserPath(strtrim(userText));
    end

    assignin('base', varName, pathValue);
end
