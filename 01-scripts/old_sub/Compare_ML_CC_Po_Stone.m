clc; clear;
load('/Users/mczhang/Documents/GitHub/FM5_ML/Stone_polarPicker/data/E_All_polished_updated_with_MLpredictions_0710.mat');

Felix = [Felix{:}];
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
results = struct();

for s = 1:length(stations)
    station = stations{s};
    po_field = ['Po_' station];
    poml_field = ['Po_ML_Ian_' station];

    results.(station).total_comparable = 0;
    results.(station).matching = 0;
    results.(station).ground_truth = [];
    results.(station).predictions = [];

    for i = 1:length(Felix)
        if isempty(Felix(i).(po_field))
            continue;
        end

        po_value = Felix(i).(po_field);
        poml_value = Felix(i).(poml_field);

        if ischar(poml_value)
            if strcmp(poml_value, 'Positive')
                poml_numeric = 1;
            elseif strcmp(poml_value, 'Negative')
                poml_numeric = -1;
            else
                poml_numeric = 0;
            end
        else
            poml_numeric = poml_value;
        end

        if po_value ~= 0 && poml_numeric ~= 0
            results.(station).total_comparable = results.(station).total_comparable + 1;
            results.(station).ground_truth(end+1) = po_value;
            results.(station).predictions(end+1) = poml_numeric;

            if po_value == poml_numeric
                results.(station).matching = results.(station).matching + 1;
            end
        end
    end

    if results.(station).total_comparable > 0
        results.(station).agreement = 100 * results.(station).matching / results.(station).total_comparable;
        results.(station).disagreements = results.(station).total_comparable - results.(station).matching;
    else
        results.(station).agreement = NaN;
        results.(station).disagreements = 0;
    end
end

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

% Confusion matrices (optional)
figure;
for s = 1:length(stations)
    station = stations{s};
    if results.(station).total_comparable > 0
        subplot(3, 3, s);
        cm = confusionmat(results.(station).ground_truth, results.(station).predictions, 'Order', [-1 1]);
        confusionchart(cm, {'D (-1)', 'U (+1)'});
        title([station ' Confusion Matrix']);
        xlabel('Predicted');
        ylabel('Ground Truth');
    end
end

%% ── Summary table ───────────────────────────────────────────────────────
fprintf('\n%-6s %8s %8s %8s %8s %12s %12s %10s\n', ...
        'Station','True↑','True↓','Pred↑','Pred↓','Prec↑(1)','Prec↓(-1)','Accuracy');
fprintf('-------------------------------------------------------------------------------\n');

for s = 1:numel(stations)
    station = stations{s};
    gt   = results.(station).ground_truth;     % ground‑truth vector (‑1 / +1)
    pred = results.(station).predictions;      % prediction vector (‑1 / +1)

    if isempty(gt)           % skip stations that had no comparable readings
        continue;
    end

    % Basic counts
    true_up    = sum(gt  ==  1);
    true_down  = sum(gt  == -1);
    pred_up    = sum(pred == 1);
    pred_down  = sum(pred == -1);

    % Confusion‑matrix elements
    tp = sum(gt ==  1 & pred ==  1);   % true positives  (+1 correctly predicted)
    tn = sum(gt == -1 & pred == -1);   % true negatives  (‑1 correctly predicted)
    fp = sum(gt == -1 & pred ==  1);   % false positives (‑1 mis‑predicted as +1)
    fn = sum(gt ==  1 & pred == -1);   % false negatives (+1 mis‑predicted as ‑1)

    % Metrics (guard against divide‑by‑zero)
    prec_up   = tp / max(tp+fp,1) * 100;      % precision for +1
    prec_down = tn / max(tn+fn,1) * 100;      % precision for ‑1
    accuracy  = (tp + tn) / numel(gt) * 100;  % overall accuracy

    fprintf('%-6s %8d %8d %8d %8d %12.2f %12.2f %10.2f\n', ...
            station, true_up, true_down, pred_up, pred_down, prec_up, prec_down, accuracy);
end

