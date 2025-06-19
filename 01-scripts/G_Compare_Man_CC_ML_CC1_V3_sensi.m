% -------------------------------------------------------------------------
% Analyzing Polarity Agreement between ML, CC, and Manual with Sensitivity
% -------------------------------------------------------------------------
% Load the data
clc;clear;
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/F_ML/F_CC1_MLPo_DiTing128.mat');

% Define the stations
stations = {'CC1'};
results = struct();

% Loop through each station
for s = 1:length(stations)
    station = stations{s};
    po_field   = ['Po_' station];         % CC field
    poml_field = ['PoML1_W1_' station];   % ML field
    poCC_field = [station '_Po'];         % Manual field

    % Initialize counters
    match_CC_ML = 0; total_CC_ML = 0;
    match_CC_Man = 0; total_CC_Man = 0;
    match_Man_ML = 0; total_Man_ML = 0;
    
    % Two views for ML agreement when both CC and Manual exist
    match_ML_agree_CC = 0;
    match_ML_agree_Man = 0;
    total_both_exist = 0;

    % New counters for CC & ML agreement with Manual
    match_CC_ML_Agree_Manual = 0;
    total_CC_ML_Agree = 0;

    % Sensitivity counters
    total_CC_determined = 0;
    total_ML_determined = 0;
    total_Manual_determined = 0;
    total_events = length(Felix);

    % Loop through each event in Felix
    for i = 1:total_events
        % Check if all three fields are present in the structure
        if isfield(Felix(i), po_field) && isfield(Felix(i), poml_field) && isfield(Felix(i), poCC_field)
            % Extract polarity values
            val_CC = sign(Felix(i).(po_field));
            val_Man = sign(Felix(i).(poCC_field));
            val_ML_raw = Felix(i).(poml_field);

            % Convert ML raw value to polarity representation
            if ischar(val_ML_raw)
                if strcmp(val_ML_raw, 'U')
                    val_ML = 1;
                elseif strcmp(val_ML_raw, 'D')
                    val_ML = -1;
                else
                    val_ML = 0;
                end
            else
                val_ML = val_ML_raw;
            end

            % -----------------------------------------------------------------
            % Sensitivity Count (if value is non-zero, it is determined)
            % -----------------------------------------------------------------
            if val_CC ~= 0
                total_CC_determined = total_CC_determined + 1;
            end
            if val_ML ~= 0
                total_ML_determined = total_ML_determined + 1;
            end
            if val_Man ~= 0
                total_Manual_determined = total_Manual_determined + 1;
            end

            % -----------------------------------------------------------------
            % 1. CC vs ML Agreement
            % -----------------------------------------------------------------
            if val_CC ~= 0 && val_ML ~= 0
                total_CC_ML = total_CC_ML + 1;
                
                if val_CC == val_ML
                    match_CC_ML = match_CC_ML + 1;
                    
                    % -----------------------------------------------------------------
                    % 2. If CC and ML agree, check if it also matches Manual
                    % -----------------------------------------------------------------
                    if val_Man ~= 0
                        total_CC_ML_Agree = total_CC_ML_Agree + 1;
                        if val_Man == val_CC
                            match_CC_ML_Agree_Manual = match_CC_ML_Agree_Manual + 1;
                        end
                    end
                end
            end

            % -----------------------------------------------------------------
            % 3. CC vs Manual Agreement
            % -----------------------------------------------------------------
            if val_CC ~= 0 && val_Man ~= 0
                total_CC_Man = total_CC_Man + 1;
                if val_CC == val_Man
                    match_CC_Man = match_CC_Man + 1;
                end
            end

            % -----------------------------------------------------------------
            % 4. Manual vs ML Agreement
            % -----------------------------------------------------------------
            if val_Man ~= 0 && val_ML ~= 0
                total_Man_ML = total_Man_ML + 1;
                if val_Man == val_ML
                    match_Man_ML = match_Man_ML + 1;
                end
            end

            % -----------------------------------------------------------------
            % 5. ML agreement when both CC and Manual exist
            % -----------------------------------------------------------------
            if val_CC ~= 0 && val_Man ~= 0 && val_ML ~= 0
                total_both_exist = total_both_exist + 1;
                if val_ML == val_CC
                    match_ML_agree_CC = match_ML_agree_CC + 1;
                end
                if val_ML == val_Man
                    match_ML_agree_Man = match_ML_agree_Man + 1;
                end
            end
        end
    end

    % ---------------------------------------------------------------------
    % Store results
    % ---------------------------------------------------------------------
    results.(station).CC_vs_ML = [match_CC_ML, total_CC_ML, 100 * match_CC_ML / max(1, total_CC_ML)];
    results.(station).CC_vs_Manual = [match_CC_Man, total_CC_Man, 100 * match_CC_Man / max(1, total_CC_Man)];
    results.(station).Manual_vs_ML = [match_Man_ML, total_Man_ML, 100 * match_Man_ML / max(1, total_Man_ML)];
    results.(station).Sensitivity = [total_CC_determined, total_ML_determined, total_Manual_determined, total_events];
end

% -------------------------------------------------------------------------
% Display Results
% -------------------------------------------------------------------------
fprintf('\n%-6s | %-40s | %-10s | %-10s | %-10s\n', 'Station', 'Comparison', 'Match', 'Total', 'Agreement');
fprintf('-----------------------------------------------------------------------------------------------------\n');
for s = 1:length(stations)
    station = stations{s};
    fprintf('%-6s | %-40s | %-10d | %-10d | %-9.2f%%\n', station, 'CC vs ML', results.(station).CC_vs_ML);
    fprintf('%-6s | %-40s | %-10d | %-10d | %-9.2f%%\n', station, 'CC vs Manual', results.(station).CC_vs_Manual);
    fprintf('%-6s | %-40s | %-10d | %-10d | %-9.2f%%\n', station, 'Manual vs ML', results.(station).Manual_vs_ML);

    % Sensitivity Output
    CC_sens = results.(station).Sensitivity(1) / results.(station).Sensitivity(4);
    ML_sens = results.(station).Sensitivity(2) / results.(station).Sensitivity(4);
    Man_sens = results.(station).Sensitivity(3) / results.(station).Sensitivity(4);

    fprintf('%-6s | %-40s | %-10.2f | %-10.2f | %-10.2f\n', station, 'Sensitivity (CC, ML, Manual)', CC_sens, ML_sens, Man_sens);
end
