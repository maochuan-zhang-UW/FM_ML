clc; clear;

% Load Felix struct
load('/Users/mczhang/Documents/GitHub/FM3/02-data/E_Po/E_All_polished_updated_V2.mat');

% Now analyze accuracy vs depth per station
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
depth_bins = 0:0.2:2.0;
bin_centers = depth_bins(1:end-1) + 0.1;

figure;
hold on;
% Store accuracy for each station to compute overall average
all_accuracies = zeros(length(stations), length(bin_centers));
for s = 1:length(stations)
    station = stations{s};
    po_field = ['Po_' station];
    poml_field = ['PoML_W_' station];

    depths = [];
    matches = [];

    for i = 1:length(Felix)
        if  isempty(Felix(i).(po_field)) || isempty(Felix(i).(poml_field)) || isempty(Felix(i).depth)
            continue;
        end

        po = Felix(i).(po_field);
        poml = Felix(i).(poml_field);
        depth = Felix(i).depth;

        if po == 0
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

        depths(end+1) = depth;
        matches(end+1) = double(po == poml);
    end

    % Bin by depth
    accuracy = zeros(size(bin_centers));
    for b = 1:length(depth_bins)-1
        idx = depths >= depth_bins(b) & depths < depth_bins(b+1);
        if sum(idx) > 0
            accuracy(b) = mean(matches(idx));
        else
            accuracy(b) = NaN;
        end
    end
    all_accuracies(s, :) = accuracy;
    plot(bin_centers, accuracy * 100, '-o', 'DisplayName', station);
end
% Compute and plot overall average accuracy
overall_accuracy = nanmean(all_accuracies, 1);
plot(bin_centers, overall_accuracy * 100, '-*', 'LineWidth', 2, 'Color', 'k', 'DisplayName', 'Overall Average');

xlabel('Depth (km)');
ylabel('Accuracy (%)');
title('Polarity Accuracy vs. Depth');
legend('show');
grid on;