function fileValue = getFileFromWorkspaceOrPrompt(varName, promptText, allowEmpty)
    fileValue = '';
    if evalin('base', sprintf('exist(''%s'', ''var'')', varName))
        fileValue = evalin('base', varName);
        if isstring(fileValue), fileValue = char(fileValue); end
        fileValue = normalizeUserPath(fileValue);
    end

    while isempty(fileValue) || ~exist(fileValue, 'file')
        if ~isempty(fileValue) && ~exist(fileValue, 'file')
            fprintf('%s does not exist or is not a file: %s\n', varName, fileValue);
        end
        userText = input(sprintf('%s: ', promptText), 's');
        if isempty(strtrim(userText)) && allowEmpty
            fileValue = '';
            assignin('base', varName, fileValue);
            return;
        elseif isempty(strtrim(userText))
            fprintf('A valid file path is required.\n');
            continue;
        end
        fileValue = normalizeUserPath(strtrim(userText));
    end

    assignin('base', varName, fileValue);
end
