clc; clear;
%load('/Users/mczhang/Documents/GitHub/FM3/02-data/E_Po/E_All_polished_updated_V2.mat');

load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/D_AS1_MLPo.mat');

stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
results = struct();

for s = 1:length(stations)
    station = stations{s};
    po_field = ['Po_' station];
    poml_field = ['PoML_W_' station];

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
            if strcmp(poml_value, 'U')
                poml_numeric = 1;
            elseif strcmp(poml_value, 'D')
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


