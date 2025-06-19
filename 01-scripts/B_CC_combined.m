% Script to combine all B_<group>_<field>.mat files into a single .mat file
clc; clear; close all;

% Configuration
path2 = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/B_CC/'; % Path to B_CC directory
output_file = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/B_CC/B_CC_combined.mat'; % Path for combined output file
fields = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
groups = {'W12', 'W2', 'S1', 'E1', 'E2', 'E3', 'E4', 'W1', 'E12', 'E23', 'E34'};

% Initialize storage for combined data
combined_filteredResultsMatrix = [];
combined_resultsMatrix_cb = [];

% Loop through groups and fields to load and combine data
for gp = 1:length(groups)
    for kz = 3%1:length(fields)
        % Construct file name
        file_name = fullfile(path2, sprintf('B_%s_%s.mat', groups{gp}, fields{kz}));
        
        % Check if file exists
        if exist(file_name, 'file')
            % Load the .mat file
            data = load(file_name);
            
            % Append filteredResultsMatrix
            if isfield(data, 'filteredResultsMatrix') && ~isempty(data.filteredResultsMatrix)
                % Add group and field identifiers as additional columns if needed
                group_id = repmat(gp, size(data.filteredResultsMatrix, 1), 1); % Group index
                field_id = repmat(kz, size(data.filteredResultsMatrix, 1), 1); % Field index
                combined_filteredResultsMatrix = [combined_filteredResultsMatrix; ...
                    [data.filteredResultsMatrix, group_id, field_id]];
            end
            
            % Append resultsMatrix_cb
            if isfield(data, 'resultsMatrix_cb') && ~isempty(data.resultsMatrix_cb)
                group_id = repmat(gp, size(data.resultsMatrix_cb, 1), 1); % Group index
                field_id = repmat(kz, size(data.resultsMatrix_cb, 1), 1); % Field index
                combined_resultsMatrix_cb = [combined_resultsMatrix_cb; ...
                    [data.resultsMatrix_cb, group_id, field_id]];
            end
            
            fprintf('Processed file: %s\n', file_name);
        else
            fprintf('File not found: %s\n', file_name);
        end
    end
end

% Load Felix.ID (adjust path and format as needed)

load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/D_man/D_CC1_128.mat')
% Filter combined_filteredResultsMatrix
Felix_ID=[Felix.ID];
% Keep rows where first column matches Felix.ID

[~, IA_col1, ~] = intersect(combined_filteredResultsMatrix(:,1), Felix_ID);
[~, IA_col2, ~] = intersect(combined_filteredResultsMatrix(:,2), Felix_ID);
% Combine unique row indices from both columns
matching_rows_frm = unique([IA_col1; IA_col2]);
Felix_CC=Felix(matching_rows_frm);
filtered_matrix_frm = combined_filteredResultsMatrix(matching_rows_frm, :);
[~, IA_col1_cb, ~] = intersect(combined_resultsMatrix_cb(:,1), Felix_ID);
[~, IA_col2_cb, ~] = intersect(combined_resultsMatrix_cb(:,2), Felix_ID);
% Combine unique row indices from both columns
matching_rows_cb = unique([IA_col1_cb; IA_col2_cb]);
filtered_matrix_cb = combined_resultsMatrix_cb(matching_rows_cb, :);
% Save combined data to a single .mat file
save(output_file, 'filtered_matrix_frm', 'filtered_matrix_cb');
fprintf('Combined data saved to: %s\n', output_file);
% Save combined data to a single .mat file

