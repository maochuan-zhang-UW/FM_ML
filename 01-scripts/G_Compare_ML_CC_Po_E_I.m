% Clear workspace and load the .mat file
clc; clear;
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/E_MLPo_AS1_MLPo.mat');

% Define station and quality flags to analyze
stations = {'AS1'};
qual_flags = {'E', 'I'};
results = struct();

for s = 1:length(stations)
    station = stations{s};
    po_field = ['Po_' station];
    poml_field = ['PoML1_W_' station];
    poml_qual_field = ['Po_W_' station '_Sp'];
    
    % Initialize results for each quality flag
    for q = 1:length(qual_flags)
        qual = qual_flags{q};
        results.(station).(qual).total_comparable = 0;
        results.(station).(qual).matching = 0;
        results.(station).(qual).ground_truth = [];
        results.(station).(qual).predictions = [];
    end
    
    % Loop through all events
    for i = 1:length(Felix)
        if isempty(Felix(i).(po_field)) || isempty(Felix(i).(poml_qual_field))
            continue;
        end
        
        % Get quality flag, ground truth, and prediction
        qual_value = Felix(i).(poml_qual_field);
        po_value = Felix(i).(po_field);
        poml_value = Felix(i).(poml_field);
        
        % Skip if quality flag is not 'E' or 'I'
        if ~ismember(qual_value, qual_flags)
            continue;
        end
        
        % Convert PoML to numeric (assuming format: 'U', 'D', '-')
        if ischar(poml_value)
            if strcmp(poml_value, 'U')
                poml_numeric = 1;
            elseif strcmp(poml_value, 'D')
                poml_numeric = -1;
            else
                poml_numeric = 0;
            end
        else
            poml_numeric = poml_value; % In case it's already numeric
        end
        
        % Only consider cases where both are non-zero
        if po_value ~= 0 && poml_numeric ~= 0
            results.(station).(qual_value).total_comparable = ...
                results.(station).(qual_value).total_comparable + 1;
            results.(station).(qual_value).ground_truth(end+1) = po_value;
            results.(station).(qual_value).predictions(end+1) = poml_numeric;
            
            if po_value == poml_numeric
                results.(station).(qual_value).matching = ...
                    results.(station).(qual_value).matching + 1;
            end
        end
    end
    
    % Calculate statistics for each quality flag
    for q = 1:length(qual_flags)
        qual = qual_flags{q};
        if results.(station).(qual).total_comparable > 0
            results.(station).(qual).agreement = 100 * ...
                results.(station).(qual).matching / ...
                results.(station).(qual).total_comparable;
            results.(station).(qual).disagreements = ...
                results.(station).(qual).total_comparable - ...
                results.(station).(qual).matching;
        else
            results.(station).(qual).agreement = NaN;
            results.(station).(qual).disagreements = 0;
        end
    end
end

% Display results for each station and quality flag
fprintf('%-6s %-8s %-15s %-15s %-15s\n', 'Station', 'Quality', 'Comparable', 'Matching', 'Agreement');
fprintf('------------------------------------------------------------\n');
for s = 1:length(stations)
    station = stations{s};
    for q = 1:length(qual_flags)
        qual = qual_flags{q};
        fprintf('%-6s %-8s %-15d %-15d %-15.2f%%\n', ...
                station, qual, ...
                results.(station).(qual).total_comparable, ...
                results.(station).(qual).matching, ...
                results.(station).(qual).agreement);
    end
end

% Create confusion matrices for each quality flag (optional)
figure;
for s = 1:length(stations)
    station = stations{s};
    for q = 1:length(qual_flags)
        qual = qual_flags{q};
        if results.(station).(qual).total_comparable > 0
            subplot(length(stations), length(qual_flags), (s-1)*length(qual_flags) + q);
            cm = confusionmat(results.(station).(qual).ground_truth, ...
                              results.(station).(qual).predictions);
            confusionchart(cm, {'-1', '1'});
            title([station ' (' qual ') Confusion Matrix']);
            xlabel('Predicted');
            ylabel('Ground Truth');
        end
    end
end
