% Load the .mat file
clc;clear;
load('/Users/mczhang/Documents/GitHub/FM3/02-data/E_Po/E_All_polished_updated_V2.mat')

% Define station names and corresponding fields
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
%stations = {'AS1'};
results = struct();

for s = 1:length(stations)
    station = stations{s};
    po_field = ['Po_' station];
    poml_field = ['PoML_W_' station];
    poml_qual_field = ['Po_W_' station '_Sp'];
    
    % Initialize counters for this station
    results.(station).total_comparable = 0;
    results.(station).matching = 0;
    results.(station).ground_truth = [];
    results.(station).predictions = [];
    results.(station).quality = [];
    
    % Initialize counters for E and I quality
    results.(station).total_comparable_E = 0;
    results.(station).matching_E = 0;
    results.(station).ground_truth_E = [];
    results.(station).predictions_E = [];
    
    results.(station).total_comparable_I = 0;
    results.(station).matching_I = 0;
    results.(station).ground_truth_I = [];
    results.(station).predictions_I = [];
    
    % Loop through all events
    for i = 1:length(Felix)
        % Get ground truth, prediction, and quality
        po_value = Felix(i).(po_field);
        poml_value = Felix(i).(poml_field);
        poml_qual_value = Felix(i).(poml_qual_field);
        
        % Convert PoML to numeric
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
            results.(station).quality{end+1} = poml_qual_value;
            
            if po_value == poml_numeric
                results.(station).matching = results.(station).matching + 1;
            end
            
            % Check quality and add to respective groups
            if strcmp(poml_qual_value, 'E')
                results.(station).total_comparable_E = results.(station).total_comparable_E + 1;
                results.(station).ground_truth_E(end+1) = po_value;
                results.(station).predictions_E(end+1) = poml_numeric;
                if po_value == poml_numeric
                    results.(station).matching_E = results.(station).matching_E + 1;
                end
            elseif strcmp(poml_qual_value, 'I')
                results.(station).total_comparable_I = results.(station).total_comparable_I + 1;
                results.(station).ground_truth_I(end+1) = po_value;
                results.(station).predictions_I(end+1) = poml_numeric;
                if po_value == poml_numeric
                    results.(station).matching_I = results.(station).matching_I + 1;
                end
            end
        end
    end
    
    % Calculate statistics for all quality data
    if results.(station).total_comparable > 0
        % Accuracy
        results.(station).accuracy = results.(station).matching / results.(station).total_comparable;
        
        % Confusion matrix
        cm = confusionmat(results.(station).ground_truth, results.(station).predictions);
        
        % Precision and Sensitivity (Recall) for U (1) and D (-1)
        % Precision = TP / (TP + FP)
        % Sensitivity = TP / (TP + FN)
        
        % For U polarity (1)
        TP_U = cm(2,2); % True positives for U
        FP_U = cm(1,2); % False positives for U
        FN_U = cm(2,1); % False negatives for U
        
        results.(station).precision_U = TP_U / (TP_U + FP_U);
        results.(station).sensitivity_U = TP_U / (TP_U + FN_U);
        
        % For D polarity (-1)
        TP_D = cm(1,1); % True positives for D
        FP_D = cm(2,1); % False positives for D
        FN_D = cm(1,2); % False negatives for D
        
        results.(station).precision_D = TP_D / (TP_D + FP_D);
        results.(station).sensitivity_D = TP_D / (TP_D + FN_D);
    else
        results.(station).accuracy = NaN;
        results.(station).precision_U = NaN;
        results.(station).sensitivity_U = NaN;
        results.(station).precision_D = NaN;
        results.(station).sensitivity_D = NaN;
    end
    
    % Calculate statistics for E quality data
    if results.(station).total_comparable_E > 0
        cm_E = confusionmat(results.(station).ground_truth_E, results.(station).predictions_E);
        
        % For U polarity (1)
        TP_U_E = cm_E(2,2);
        FP_U_E = cm_E(1,2);
        FN_U_E = cm_E(2,1);
        
        results.(station).precision_U_E = TP_U_E / (TP_U_E + FP_U_E);
        results.(station).sensitivity_U_E = TP_U_E / (TP_U_E + FN_U_E);
        
        % For D polarity (-1)
        TP_D_E = cm_E(1,1);
        FP_D_E = cm_E(2,1);
        FN_D_E = cm_E(1,2);
        
        results.(station).precision_D_E = TP_D_E / (TP_D_E + FP_D_E);
        results.(station).sensitivity_D_E = TP_D_E / (TP_D_E + FN_D_E);
    else
        results.(station).precision_U_E = NaN;
        results.(station).sensitivity_U_E = NaN;
        results.(station).precision_D_E = NaN;
        results.(station).sensitivity_D_E = NaN;
    end
    
    % Calculate statistics for I quality data
    if results.(station).total_comparable_I > 0
        cm_I = confusionmat(results.(station).ground_truth_I, results.(station).predictions_I);
        
        % For U polarity (1)
        TP_U_I = cm_I(2,2);
        FP_U_I = cm_I(1,2);
        FN_U_I = cm_I(2,1);
        
        results.(station).precision_U_I = TP_U_I / (TP_U_I + FP_U_I);
        results.(station).sensitivity_U_I = TP_U_I / (TP_U_I + FN_U_I);
        
        % For D polarity (-1)
        TP_D_I = cm_I(1,1);
        FP_D_I = cm_I(2,1);
        FN_D_I = cm_I(1,2);
        
        results.(station).precision_D_I = TP_D_I / (TP_D_I + FP_D_I);
        results.(station).sensitivity_D_I = TP_D_I / (TP_D_I + FN_D_I);
    else
        results.(station).precision_U_I = NaN;
        results.(station).sensitivity_U_I = NaN;
        results.(station).precision_D_I = NaN;
        results.(station).sensitivity_D_I = NaN;
    end
end

% Display results for each station
fprintf('%-6s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n', ...
    'Station', 'Acc', 'Prec_U', 'Prec_D', 'Sens_U', 'Sens_D', ...
    'Prec_U_E', 'Prec_D_E', 'Prec_U_I', 'Prec_D_I');
fprintf('----------------------------------------------------------------------------------------------------\n');
for s = 1:length(stations)
    station = stations{s};
    fprintf('%-6s %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f\n', ...
            station, ...
            results.(station).accuracy, ...
            results.(station).precision_U, ...
            results.(station).precision_D, ...
            results.(station).sensitivity_U, ...
            results.(station).sensitivity_D, ...
            results.(station).precision_U_E, ...
            results.(station).precision_D_E, ...
            results.(station).precision_U_I, ...
            results.(station).precision_D_I);
end

% Display sensitivity for E and I quality
fprintf('\n%-6s %-10s %-10s %-10s %-10s\n', ...
    'Station', 'Sens_U_E', 'Sens_D_E', 'Sens_U_I', 'Sens_D_I');
fprintf('--------------------------------------------------\n');
for s = 1:length(stations)
    station = stations{s};
    fprintf('%-6s %-10.2f %-10.2f %-10.2f %-10.2f\n', ...
            station, ...
            results.(station).sensitivity_U_E, ...
            results.(station).sensitivity_D_E, ...
            results.(station).sensitivity_U_I, ...
            results.(station).sensitivity_D_I);
end

% Create confusion matrices (optional)
figure;
for s = 1:length(stations)
    station = stations{s};
    if results.(station).total_comparable > 0
        subplot(3, 3, s);
        cm = confusionmat(results.(station).ground_truth, results.(station).predictions);
        confusionchart(cm, {'D (-1)', 'U (1)'});
        title([station ' Confusion Matrix']);
        xlabel('Predicted');
        ylabel('Ground Truth');
    end
end