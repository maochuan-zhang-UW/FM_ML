% -------------------------------------------------------------------------
% Plot Confusion Matrices for CC vs Manual and ML vs Manual
% -------------------------------------------------------------------------
% Load the data
clear;
clear;close all
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/D_man/D_CC1_CCPo.mat')

% Define the stations
stations = {'CC1'};
conf_data.CC_vs_Man = struct();
conf_data.ML_vs_Man = struct();

% Loop through each station
for s = 1:length(stations)
    station = stations{s};
    po_field   = ['Po_' station];         % CC field
    poml_field = ['PoML1_W1_' station];   % ML field of DiTing
    poCC_field = [station '_CCPo'];       % Manual field

    % Initialize arrays to store true and predicted labels
    true_labels_CC = [];
    pred_labels_CC = [];
    true_labels_ML = [];
    pred_labels_ML = [];

    % Loop through each event in Felix
    for i = 1:length(Felix)
        % Check if all three fields are present
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

            % Collect data for CC vs Manual (Manual as ground truth)
            if val_Man ~= 0 && val_CC ~= 0
                true_labels_CC = [true_labels_CC; val_Man];
                pred_labels_CC = [pred_labels_CC; val_CC];
            end

            % Collect data for ML vs Manual (Manual as ground truth)
            if val_Man ~= 0 && val_ML ~= 0
                true_labels_ML = [true_labels_ML; val_Man];
                pred_labels_ML = [pred_labels_ML; val_ML];
            end
        end
    end

    % Store confusion matrix data
    conf_data.CC_vs_Man.(station).true = true_labels_CC;
    conf_data.CC_vs_Man.(station).pred = pred_labels_CC;
    conf_data.ML_vs_Man.(station).true = true_labels_ML;
    conf_data.ML_vs_Man.(station).pred = pred_labels_ML;
end

% -------------------------------------------------------------------------
% Plot Confusion Matrices
% -------------------------------------------------------------------------
for s = 1:length(stations)
    station = stations{s};
    
    % Define labels for confusion matrix
    labels = {'Down (-1)', 'Up (+1)'};
    
    % CC vs Manual Confusion Matrix
    figure;
    true_cc = conf_data.CC_vs_Man.(station).true;
    pred_cc = conf_data.CC_vs_Man.(station).pred;
    
    % Convert to categorical for confusion matrix
    true_cc_cat = categorical(true_cc, [-1, 1], labels);
    pred_cc_cat = categorical(pred_cc, [-1, 1], labels);
    
    % Compute confusion matrix
    cm = confusionmat(true_cc_cat, pred_cc_cat, 'Order', labels);
    
    % Calculate row-normalized percentages
    row_sums = sum(cm, 2);
    percentages = (cm ./ row_sums) * 100;
    percentages(isnan(percentages)) = 0; % Handle case where row sum is 0
    
    % Create display values (count and percentage)
    display_values = strings(size(cm));
    for i = 1:numel(cm)
        row = ceil(i/2);
        col = mod(i-1, 2) + 1;
        display_values(row, col) = sprintf('%d (%.1f%%)', cm(row, col), percentages(row, col));
    end
    
    % Plot heatmap using imagesc with percentages
    imagesc(percentages);
    colormap('summer');
    colorbar;
    caxis([0 100]); % Set color axis to percentage range (0 to 100%)
    
    % Set axes properties
    set(gca, 'XTick', 1:2, 'XTickLabel', labels, 'YTick', 1:2, 'YTickLabel', labels);
    xlabel('Predicted (CC)');
    ylabel('True (Manual)');
    title(sprintf('Confusion Matrix: CC vs Manual (%s)', station));
    
    % Add text labels for counts and percentages
    for i = 1:2
        for j = 1:2
            text(j, i, display_values(i,j), 'HorizontalAlignment', 'center', 'Color', 'black', 'FontWeight', 'bold');
        end
    end
    
    % Adjust axis to make square
    axis equal tight;
    
    % ML vs Manual Confusion Matrix
    figure;
    true_ml = conf_data.ML_vs_Man.(station).true;
    pred_ml = conf_data.ML_vs_Man.(station).pred;
    
    % Convert to categorical for confusion matrix
    true_ml_cat = categorical(true_ml, [-1, 1], labels);
    pred_ml_cat = categorical(pred_ml, [-1, 1], labels);
    
    % Compute confusion matrix
    cm = confusionmat(true_ml_cat, pred_ml_cat, 'Order', labels);
    
    % Calculate row-normalized percentages
    row_sums = sum(cm, 2);
    percentages = (cm ./ row_sums) * 100;
    percentages(isnan(percentages)) = 0; % Handle case where row sum is 0
    
    % Create display values (count and percentage)
    display_values = strings(size(cm));
    for i = 1:numel(cm)
        row = ceil(i/2);
        col = mod(i-1, 2) + 1;
        display_values(row, col) = sprintf('%d (%.1f%%)', cm(row, col), percentages(row, col));
    end
    
    % Plot heatmap using imagesc with percentages
    imagesc(percentages);
    colormap('summer');
    colorbar;
    caxis([0 100]); % Set color axis to percentage range (0 to 100%)
    
    % Set axes properties
    set(gca, 'XTick', 1:2, 'XTickLabel', labels, 'YTick', 1:2, 'YTickLabel', labels);
    xlabel('Predicted (ML)');
    ylabel('True (Manual)');
    title(sprintf('Confusion Matrix: ML vs Manual (%s)', station));
    
    % Add text labels for counts and percentages
    for i = 1:2
        for j = 1:2
            text(j, i, display_values(i,j), 'HorizontalAlignment', 'center', 'Color', 'black', 'FontWeight', 'bold');
        end
    end
    
    % Adjust axis to make square
    axis equal tight;
end
