% -------------------------------------------------------------------------
% Analyzing Polarity Agreement between ML, CC, and Manual
% -------------------------------------------------------------------------
% Load the data
clear;clc;
%load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/F_ML/F_CC1_MLPo_DiTing128_Stone.mat');

%load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/D_man/D_CC1_CCPo.mat')
load('/Users/mczhang/Documents/GitHub/FM5_ML/01-scripts/CFM/Felix_with_predictions_timeshift.mat');
Felix = [Felix{:}]; % ML_stone
% Define the stations
stations = {'CC1'};
results = struct();

% Loop through each station
for s = 1:length(stations)
    station = stations{s};
    po_field   = ['Po_' station];         % CC field
    %poml_field = ['PoML1_W_' station];    % ML field of DiTing
    pomlqu_field =['PoCon_ML_Ian_' station]; % ML quality control

    poml_field = [station '_CFM_Po'];  % ML CFM
    poCC_field = [station '_CCPo'];         % Manual field
    wave_field = ['W1_' station];  % waveform

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
    match_Man_Agree_CC_ML_exist = 0;  % Initialize the counter

    % New counters for CC vs ML without Manual
    match_CC_ML_No_Manual = 0;
    total_CC_ML_No_Manual = 0;

    % Loop through each event in Felix
    for i = 1:length(Felix)
        % Check if all three fields are present in the structure
        if isfield(Felix(i), po_field) && isfield(Felix(i), poml_field)
            % Extract polarity values
            val_CC = sign(Felix(i).(po_field));
            val_Man = sign(Felix(i).(poCC_field));
            val_ML_raw = Felix(i).(poml_field);
            quality_ML = Felix(i).(pomlqu_field); % Extract ML quality control value

            % Only process ML if quality is > 0.9
            if val_ML_raw==1
                val_ML = 1;
            elseif val_ML_raw==-1
                val_ML = -1;
            else
                val_ML = 0;
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

            % -----------------------------------------------------------------
            % 6. Manual agrees with both CC & ML exist
            % -----------------------------------------------------------------
            if val_CC ~= 0 && val_Man ~= 0 && val_ML ~= 0
                % Check if Manual agrees with both CC and ML when all three exist
                if val_Man == val_CC && val_Man == val_ML
                    match_Man_Agree_CC_ML_exist = match_Man_Agree_CC_ML_exist + 1;
                end
            end

            % -------------- CC vs ML when Manual is Missing ---------------
            if val_Man == 0 && val_CC ~= 0 && val_ML ~= 0
                total_CC_ML_No_Manual = total_CC_ML_No_Manual + 1;
                if val_CC == val_ML
                    match_CC_ML_No_Manual = match_CC_ML_No_Manual + 1;
                end
            end
        end
    end

    % ---------------------------------------------------------------------
    % Store results
    % ---------------------------------------------------------------------
    results.(station).CC_vs_ML = [match_CC_ML, total_CC_ML, 100 * match_CC_ML / max(1, total_CC_ML)];
    results.(station).CC_vs_ML_No_Manual = [match_CC_ML_No_Manual, total_CC_ML_No_Manual, 100 * match_CC_ML_No_Manual / max(1, total_CC_ML_No_Manual)];
    results.(station).CC_vs_Manual = [match_CC_Man, total_CC_Man, 100 * match_CC_Man / max(1, total_CC_Man)];
    results.(station).Manual_vs_ML = [match_Man_ML, total_Man_ML, 100 * match_Man_ML / max(1, total_Man_ML)];
    results.(station).ML_agree_CCbasis = [match_ML_agree_CC, total_both_exist, 100 * match_ML_agree_CC / max(1, total_both_exist)];
    results.(station).ML_agree_Manbasis = [match_ML_agree_Man, total_both_exist, 100 * match_ML_agree_Man / max(1, total_both_exist)];
    results.(station).CC_ML_Agree_Manual = [match_CC_ML_Agree_Manual, total_CC_ML_Agree, 100 * match_CC_ML_Agree_Manual / max(1, total_CC_ML_Agree)];
    results.(station).Sensitivity = [total_CC_determined, total_ML_determined, total_Manual_determined, total_events];
    results.(station).Man_Agree_Both_CC_ML_exist = [match_Man_Agree_CC_ML_exist, total_both_exist, 100 * match_Man_Agree_CC_ML_exist / max(1, total_both_exist)];

end

% -------------------------------------------------------------------------
% Display Results
% -------------------------------------------------------------------------
fprintf('\n%-6s | %-40s | %-10s | %-10s | %-10s\n', 'Station', 'Comparison', 'Match', 'Total', 'Agreement');
fprintf('-----------------------------------------------------------------------------------------------------\n');
for s = 1:length(stations)
    station = stations{s};
    fprintf('%-6s | %-40s | %-10d | %-10d | %-9.2f%%\n', station, 'CC vs ML', results.(station).CC_vs_ML);
    fprintf('%-6s | %-40s | %-10d | %-10d | %-9.2f%%\n', station, 'CC vs ML (No Manual)', results.(station).CC_vs_ML_No_Manual);
    fprintf('%-6s | %-40s | %-10d | %-10d | %-9.2f%%\n', station, 'CC vs Man', results.(station).CC_vs_Manual);
    fprintf('%-6s | %-40s | %-10d | %-10d | %-9.2f%%\n', station, 'ML vs Man', results.(station).Manual_vs_ML);
    fprintf('%-6s | %-40s | %-10d | %-10d | %-9.2f%%\n', station, 'ML agrees with CC (both CC & Man exist)', results.(station).ML_agree_CCbasis);
    fprintf('%-6s | %-40s | %-10d | %-10d | %-9.2f%%\n', station, 'ML agrees with Man (both CC & Man exist)', results.(station).ML_agree_Manbasis);
    %fprintf('%-6s | %-40s | %-10d | %-10d | %-9.2f%%\n', station, 'Man agrees with both CC & ML exist', results.(station).Man_Agree_Both_CC_ML_exist);

    fprintf('%-6s | %-40s | %-10d | %-10d | %-9.2f%%\n', station, 'Man agrees with both CC & ML agreed ', results.(station).CC_ML_Agree_Manual);
end
fprintf('-----------------------------------------------------------------------------------------------------\n');
% Sensitivity Output
for s = 1:length(stations)
    station = stations{s};
    CC_sens = results.(station).Sensitivity(1) / results.(station).Sensitivity(4);
    ML_sens = results.(station).Sensitivity(2) / results.(station).Sensitivity(4);
    Man_sens = results.(station).Sensitivity(3) / results.(station).Sensitivity(4);
    fprintf('%-6s | %-40s | %-10.2f | %-10.2f | %-10.2f\n', station, 'Sensitivity (CC, ML, Manual)', CC_sens, ML_sens, Man_sens);
end