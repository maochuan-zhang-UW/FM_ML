% Script to calculate the percentage of matching non-zero values and the count of non-zero pairs
% between Po_* (ground truth) and PoML_W*, CFM_W*, EQP_* (ML predictions) for each station in the Felix dataset.
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/F_ML/A_wave_dB15_DT_CFM_EQP_modified.mat')
% Define the stations
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};

% Initialize arrays to store percentages and counts
percent_PoML = zeros(1, length(stations));
percent_CFM = zeros(1, length(stations));
percent_EQP = zeros(1, length(stations));
count_PoML = zeros(1, length(stations));
count_CFM = zeros(1, length(stations));
count_EQP = zeros(1, length(stations));

% Loop through each station
for i = 1:length(stations)
    station = stations{i};
    
    % Field names for ground truth and predictions
    po_field = sprintf('Po_%s', station);
    poml_field = sprintf('PoML_W_%s', station);
    cfm_field = sprintf('CFM_W_%s', station);
    eqp_field = sprintf('EQP_%s', station);
    
    % Extract data for the current station
    po_data = [Felix.(po_field)];
    poml_data = [Felix.(poml_field)];
    cfm_data = [Felix.(cfm_field)];
    eqp_data = [Felix.(eqp_field)];
    
    % Filter for non-zero values in both ground truth and predictions
    % PoML_W*
    non_zero_idx_poml = po_data ~= 0 & poml_data ~= 0;
    count_PoML(i) = sum(non_zero_idx_poml); % Count of non-zero pairs
    if count_PoML(i) > 0
        matches_poml = sum(po_data(non_zero_idx_poml) == poml_data(non_zero_idx_poml));
        percent_PoML(i) = (matches_poml / count_PoML(i)) * 100;
    else
        percent_PoML(i) = NaN; % No non-zero pairs
    end
    
    % CFM_W*
    non_zero_idx_cfm = po_data ~= 0 & cfm_data ~= 0;
    count_CFM(i) = sum(non_zero_idx_cfm); % Count of non-zero pairs
    if count_CFM(i) > 0
        matches_cfm = sum(po_data(non_zero_idx_cfm) == cfm_data(non_zero_idx_cfm));
        percent_CFM(i) = (matches_cfm / count_CFM(i)) * 100;
    else
        percent_CFM(i) = NaN; % No non-zero pairs
    end
    
    % EQP_*
    non_zero_idx_eqp = po_data ~= 0 & eqp_data ~= 0;
    count_EQP(i) = sum(non_zero_idx_eqp); % Count of non-zero pairs
    if count_EQP(i) > 0
        matches_eqp = sum(po_data(non_zero_idx_eqp) == eqp_data(non_zero_idx_eqp));
        percent_EQP(i) = (matches_eqp / count_EQP(i)) * 100;
    else
        percent_EQP(i) = NaN; % No non-zero pairs
    end
end

% Display results
fprintf('Percentage of matching non-zero values and count of non-zero pairs for each station:\n');
fprintf('Station\tPoML_W*\tCount\tCFM_W*\tCount\tEQP_*\tCount\n');
for i = 1:length(stations)
    fprintf('%s\t%.2f%%\t%d\t%.2f%%\t%d\t%.2f%%\t%d\n', ...
        stations{i}, percent_PoML(i), count_PoML(i), ...
        percent_CFM(i), count_CFM(i), percent_EQP(i), count_EQP(i));
end