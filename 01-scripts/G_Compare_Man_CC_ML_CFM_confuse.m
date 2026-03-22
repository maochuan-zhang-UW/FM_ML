% MATLAB code to plot confusion matrix for ML vs Manual polarity
clear; close all;

% Load the data
load('/Users/mczhang/Documents/GitHub/FM5_ML/01-scripts/CFM/Felix_with_predictions_timeshift.mat');
Felix = [Felix{:}]; % Convert to array if needed

% Define the station and fields
station = 'CC1';
poml_field = [station '_CFM_Po']; % ML predictions
poCC_field = [station '_CCPo']; % Manual labels
pomlqu_field = ['PoCon_ML_Ian_' station]; % ML quality control

% Initialize arrays to store true and predicted labels
true_labels_ML = [];
pred_labels_ML = [];

% Loop through each event in Felix
for i = 1:length(Felix)
    % Check if required fields exist
    if isfield(Felix(i), poml_field) && isfield(Felix(i), poCC_field) && isfield(Felix(i), pomlqu_field)
        % Extract values
        val_Man = sign(Felix(i).(poCC_field)); % Manual polarity (+1, -1, 0)
        val_ML_raw = Felix(i).(poml_field); % ML prediction
        quality_ML = Felix(i).(pomlqu_field); % ML quality control

        % Process ML polarity based on quality threshold
        if val_ML_raw == 1 && quality_ML > 0.9
            val_ML = 1;
        elseif val_ML_raw == -1 && quality_ML > 0.9
            val_ML = -1;
        else
            val_ML = 0;
        end

        % Collect data for ML vs Manual (Manual as ground truth)
        if val_Man ~= 0 && val_ML ~= 0
            true_labels_ML = [true_labels_ML; val_Man];
            pred_labels_ML = [pred_labels_ML; val_ML];
        end
    end
end

% Define labels for confusion matrix
labels = {'Down (-1)', 'Up (+1)'};

% Define custom colormap (new light blue to light red)
custom_colormap = [0.702, 0.804, 0.890; 0.984, 0.706, 0.682]; % Light blue (#B3CDE3), Light red (#FBB4AE)

% ML vs Manual Confusion Matrix
figure;
true_ml = true_labels_ML;
pred_ml = pred_labels_ML;

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
colormap(custom_colormap);
colorbar;
caxis([0 100]); % Set color axis to percentage range (0 to 100%)

% Set axes properties with larger font
set(gca, 'XTick', 1:2, 'XTickLabel', labels, 'YTick', 1:2, 'YTickLabel', labels, ...
    'FontSize', 14); % Axes font size
xlabel('Predicted (ML)', 'FontSize', 16); % X-label font size
ylabel('True (Manual)', 'FontSize', 16); % Y-label font size
title(sprintf('Confusion Matrix: ML vs Manual (%s)', station), 'FontSize', 18); % Title font size

% Add text labels for counts and percentages with larger font
for i = 1:2
    for j = 1:2
        text(j, i, display_values(i,j), 'HorizontalAlignment', 'center', ...
            'Color', 'black', 'FontWeight', 'bold', 'FontSize', 20); % Text font size
    end
end

% Adjust axis to make square
axis equal tight;

% Display confusion matrix counts for verification
disp('Confusion Matrix Counts:');
disp(['True Negatives (ML = -1, Manual = -1): ', num2str(cm(1,1))]);
disp(['False Positives (ML = +1, Manual = -1): ', num2str(cm(1,2))]);
disp(['False Negatives (ML = -1, Manual = +1): ', num2str(cm(2,1))]);
disp(['True Positives (ML = +1, Manual = +1): ', num2str(cm(2,2))]);