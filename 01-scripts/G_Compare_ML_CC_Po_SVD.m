clc; clear;
% Load SVD fields (small Felix)
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/E_Combined_SelectedFields.mat');  % loads smaller Felix
FelixSVD = Felix; clear Felix;  % Rename to avoid overwriting
% Load main Felix struct
load('/Users/mczhang/Documents/GitHub/FM3/02-data/E_Po/E_All_polished_updated_V2.mat');  % loads Felix
% Match by ID and add Po_SVD_* fields
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
id_main = [Felix.ID];
id_svd = [FelixSVD.ID];

[common_ids, ia, ib] = intersect(id_main, id_svd);
for s = 1:length(stations)
    field_svd = ['Po_SVD_' stations{s}];
    values = arrayfun(@(x) getfield_safe(x, field_svd), FelixSVD(ib));  % get values from SVD file
    for i = 1:length(ia)
        Felix(ia(i)).(field_svd) = values(i);  % assign to main Felix
    end
end

% Prepare log bins
log_bins = -7:0.5:-1;
bin_centers = log_bins(1:end-1) + 0.25;

figure; hold on;
accuracy_all = [];  % to store all station curves

% Loop over stations
for s = 1:length(stations)
    station = stations{s};
    po_field = ['Po_' station];
    poml_field = ['PoML_W_' station];
    svd_field = ['Po_SVD_' station];

    svds = [];
    matches = [];

    for i = 1:length(Felix)
        if ~isfield(Felix(i), po_field) || ~isfield(Felix(i), poml_field) || ~isfield(Felix(i), svd_field)
            continue;
        end

        po = Felix(i).(po_field);
        poml = Felix(i).(poml_field);
        svd = abs(Felix(i).(svd_field));

        if isempty(po) || isempty(poml) || isempty(svd) || po == 0 || isnan(svd)
            continue;
        end

        if ischar(poml)
            if strcmp(poml, 'U')
                poml = 1;
            elseif strcmp(poml, 'D')
                poml = -1;
            else
                poml = 0;
            end
        end

        if poml == 0
            continue;
        end

        svds(end+1) = svd;
        matches(end+1) = double(po == poml);
    end

    % Bin by log10(SVD)
    accuracy = zeros(size(bin_centers));
    for b = 1:length(log_bins)-1
        idx = log10(svds) >= log_bins(b) & log10(svds) < log_bins(b+1);
        if sum(idx) > 0
            accuracy(b) = mean(matches(idx));
        else
            accuracy(b) = NaN;
        end
    end

    accuracy_all = [accuracy_all; accuracy];  % collect for averaging
    semilogx(10.^bin_centers, accuracy * 100, '-o', 'DisplayName', station, 'LineWidth', 1.5);
end

% Add average accuracy line
avg_accuracy = nanmean(accuracy_all, 1);
semilogx(10.^bin_centers, avg_accuracy * 100, 'k--', 'LineWidth', 2.5, 'DisplayName', 'Average');


xlabel('SVD Strength (|Po\_SVD|)');
ylabel('Accuracy (%)');
title('Polarity Accuracy vs. SVD Strength');
legend('Location', 'best');
grid on;
%xlim([10^-8, 10^-1]);
set(gca, 'XScale', 'log');  % Force x-axis to log scale
xticks([1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1]);  % Customize tick locations
xticklabels({'10^{-7}','10^{-6}','10^{-5}','10^{-4}','10^{-3}','10^{-2}','10^{-1}'});  % Optional: formatted tick labels
xlim([1e-7 1e-1]);  % Set axis range
xlabel('SVD Strength |Po\_SVD| (log scale)');

function val = getfield_safe(s, field)
    if isfield(s, field) && ~isempty(s.(field))
        val = s.(field);
        if numel(val) > 1
            val = val(1); % ensure scalar
        end
    else
        val = NaN;
    end
end
