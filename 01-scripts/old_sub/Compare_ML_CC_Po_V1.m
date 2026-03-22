% Load the .mat file
%load('your_file.mat'); % Replace with your actual filename
clc;clear;
load('/Users/mczhang/Documents/GitHub/FM3/02-data/E_Po/E_All_polished_updated_V2.mat');
% Define station names and corresponding fields
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
%stations = {'AS1'};
results = struct();

for s = 1:length(stations)
    station = stations{s};
    po_field = ['Po_' station];
    poml_field = ['PoML_W_' station];
    poml_qual_field=['Po_W_' station '_Sp'];
    
    % Initialize counters for this station
    results.(station).total_comparable = 0;
    results.(station).matching = 0;
    results.(station).ground_truth = [];
    results.(station).predictions = [];
    
    % Loop through all events
    for i = 1:length(Felix)
        if isempty(Felix(i).(po_field))
            continue;
        end
        % Get ground truth and prediction
        po_value = Felix(i).(po_field);
        poml_value = Felix(i).(poml_field);
        
        % Convert PoML to numeric (assuming format: '1', '-', '0')
        if ischar(poml_value)
            if strcmp(poml_value, 'U')
                poml_numeric = 1;
            elseif strcmp(poml_value, 'D')
                poml_numeric = -1;
            else
                poml_numeric = 0;
            end
        else
            poml_numeric = poml_value; % in case it's already numeric
        end
        
        % Only consider cases where both are non-zero
        if po_value ~= 0 && poml_numeric ~= 0
            results.(station).total_comparable = results.(station).total_comparable + 1;
            results.(station).ground_truth(end+1) = po_value;
            results.(station).predictions(end+1) = poml_numeric;
            
            if po_value == poml_numeric
                results.(station).matching = results.(station).matching + 1;
            end
        end
    end
    
    % Calculate statistics for this station
    if results.(station).total_comparable > 0
        results.(station).agreement = 100 * results.(station).matching / results.(station).total_comparable;
        results.(station).disagreements = results.(station).total_comparable - results.(station).matching;
    else
        results.(station).agreement = NaN;
        results.(station).disagreements = 0;
    end
end

% Display results for each station
fprintf('%-6s %-15s %-15s %-15s\n', 'Station', 'Comparable', 'Matching', 'Agreement');
fprintf('------------------------------------------------\n');
for s = 1:length(stations)
    station = stations{s};
    fprintf('%-6s %-15d %-15d %-15.2f%%\n', ...
            station, ...
            results.(station).total_comparable, ...
            results.(station).matching, ...
            results.(station).agreement);
end

% Create confusion matrices (optional)
figure;
for s = 1:length(stations)
    station = stations{s};
    if results.(station).total_comparable > 0
        subplot(3, 3, s);
        cm = confusionmat(results.(station).ground_truth, results.(station).predictions);
        confusionchart(cm, {'-1', '1'});
        title([station ' Confusion Matrix']);
        xlabel('Predicted');
        ylabel('Ground Truth');
    end
end