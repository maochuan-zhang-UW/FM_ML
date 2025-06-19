clc;
clear;
close all;

% Define path and groups
path = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/';
groups = {'E1', 'E2', 'E3', 'E4', 'S1', 'W1', 'W2'};

% Define fields to keep
fieldsToKeep = {
    'ID', 'on', 'lon', 'lat', 'depth', ...
    'Po_AS1', 'Po_SVD_AS1', ...
    'Po_AS2', 'Po_SVD_AS2', ...
    'Po_CC1', 'Po_SVD_CC1', ...
    'Po_EC1', 'Po_SVD_EC1', ...
    'Po_EC2', 'Po_SVD_EC2', ...
    'Po_EC3', 'Po_SVD_EC3', ...
    'Po_ID1', 'Po_SVD_ID1', ...
    'PoALL'
};

% Initialize empty array
combinedFelix = [];

% Helper function to create a clean entry
makeCleanEntry = @(f) cell2struct(cellfun(@(fn) getFieldSafe(f, fn), fieldsToKeep, 'UniformOutput', false), fieldsToKeep, 2);

% Process all files
for gp = 1:length(groups)
    filePath = fullfile(path, ['E_', groups{gp}, '.mat']);
    if ~exist(filePath, 'file')
        fprintf('Warning: %s not found\n', filePath);
        continue;
    end

    load(filePath, 'Felix');
    for i = 1:numel(Felix)
        entry = makeCleanEntry(Felix(i));
        combinedFelix = [combinedFelix; entry];  % Safe concatenation
    end
end

% Save result
Felix = combinedFelix;
save(fullfile(path, 'E_Combined_SelectedFields.mat'), 'Felix');
fprintf('Saved %d entries with selected fields.\n', length(Felix));

% Safe field getter
function val = getFieldSafe(s, field)
    if isfield(s, field)
        val = s.(field);
        if numel(val) > 1
            val = val(1); % force scalar
        end
    else
        val = NaN;
    end
end
