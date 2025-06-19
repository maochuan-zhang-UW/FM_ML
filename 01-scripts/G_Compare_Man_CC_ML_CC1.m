%clc; 
clear;

load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/F_ML/F_CC1_MLPo_DiTing250.mat');

stations = {'CC1'};
results = struct();

for s = 1:length(stations)
    station = stations{s};
    po_field   = ['Po_' station];         % CC
    poml_field = ['PoML1_W_' station];    % ML
    poCC_field = [station '_Po'];         % Manual
    wave_field = ['W_' station];

    % Initialize counters
    match_CC_ML = 0; total_CC_ML = 0;
    match_CC_Man = 0; total_CC_Man = 0;
    match_Man_ML = 0; total_Man_ML = 0;

    % Two views for ML agreement when both CC and Manual exist
    match_ML_agree_CC = 0;
    match_ML_agree_Man = 0;
    total_both_exist = 0;

    for i = 1:length(Felix)
        if isfield(Felix(i), po_field) && isfield(Felix(i), poml_field) && isfield(Felix(i), poCC_field)
            % Get values
            val_CC = sign(Felix(i).(po_field));
            val_Man = sign(Felix(i).(poCC_field));
            val_ML_raw = Felix(i).(poml_field);
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

            % CC vs ML
            if val_CC ~= 0 && val_ML ~= 0
                total_CC_ML = total_CC_ML + 1;
                if val_CC == val_ML
                    match_CC_ML = match_CC_ML + 1;
                end
            end

            % CC vs Manual
            if val_CC ~= 0 && val_Man ~= 0
                total_CC_Man = total_CC_Man + 1;
                if val_CC == val_Man
                    match_CC_Man = match_CC_Man + 1;
                end
            end

            % Manual vs ML
            if val_Man ~= 0 && val_ML ~= 0
                total_Man_ML = total_Man_ML + 1;
                if val_Man == val_ML
                    match_Man_ML = match_Man_ML + 1;
                end
            end

            % ML agreement check when both CC and Manual exist
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

    % Store results
    results.(station).CC_vs_ML = [match_CC_ML, total_CC_ML, 100 * match_CC_ML / max(1, total_CC_ML)];
    results.(station).CC_vs_Manual = [match_CC_Man, total_CC_Man, 100 * match_CC_Man / max(1, total_CC_Man)];
    results.(station).Manual_vs_ML = [match_Man_ML, total_Man_ML, 100 * match_Man_ML / max(1, total_Man_ML)];
    results.(station).ML_agree_CCbasis = [match_ML_agree_CC, total_both_exist, 100 * match_ML_agree_CC / max(1, total_both_exist)];
    results.(station).ML_agree_Manbasis = [match_ML_agree_Man, total_both_exist, 100 * match_ML_agree_Man / max(1, total_both_exist)];
end

% Print Results
fprintf('\n%-6s | %-25s | %-10s | %-10s | %-10s\n', 'Station', 'Comparison', 'Match', 'Total', 'Agreement');
fprintf('-------------------------------------------------------------------------\n');
for s = 1:length(stations)
    station = stations{s};
    fprintf('%-6s | %-25s | %-10d | %-10d | %-9.2f%%\n', station, 'CC vs ML', results.(station).CC_vs_ML);
    fprintf('%-6s | %-25s | %-10d | %-10d | %-9.2f%%\n', station, 'CC vs Manual', results.(station).CC_vs_Manual);
    fprintf('%-6s | %-25s | %-10d | %-10d | %-9.2f%%\n', station, 'Manual vs ML', results.(station).Manual_vs_ML);
    fprintf('%-6s | %-25s | %-10d | %-10d | %-9.2f%%\n', station, 'ML agrees with CC (both exist)', results.(station).ML_agree_CCbasis);
    fprintf('%-6s | %-25s | %-10d | %-10d | %-9.2f%%\n', station, 'ML agrees with Manual (both exist)', results.(station).ML_agree_Manbasis);
end
