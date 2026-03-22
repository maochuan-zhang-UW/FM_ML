% Script to compute the percentage of times PoML_W_*, CFM_W_*, and EQP_* match the ground truth Po_* (non-zero) for each station
% Input: Felix struct array with fields for 7 stations (AS1, AS2, CC1, EC1, EC2, EC3, ID1)

% Load the data
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/F_ML/A_wave_dB15_DT_CFM_EQP.mat')

% Define stations and methods
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
methods = {'PoML_W_', 'CFM_W_', 'EQP_'};

% Initialize counters
counts = zeros(length(stations), length(methods)); % Matches for each method and station
nonzero_Po = zeros(length(stations), 1); % Non-zero Po_* counts

% Loop through all entries in Felix
for i = 1:length(Felix)
    for s = 1:length(stations)
        station = stations{s};
        po_field = ['Po_' station];
        if Felix(i).(po_field) ~= 0
            nonzero_Po(s) = nonzero_Po(s) + 1;
            for m = 1:length(methods)
                method_field = [methods{m} station];
                if Felix(i).(method_field) == Felix(i).(po_field)
                    counts(s, m) = counts(s, m) + 1;
                end
            end
        end
    end
end

% Calculate percentages
percentages = zeros(length(stations), length(methods));
for s = 1:length(stations)
    if nonzero_Po(s) > 0
        percentages(s, :) = (counts(s, :) / nonzero_Po(s)) * 100;
    end
end

% Display results
fprintf('Percentage of times each field matches the ground truth (Po_*) for each station (when Po_* is non-zero):\n');
for s = 1:length(stations)
    fprintf('\nStation %s:\n', stations{s});
    for m = 1:length(methods)
        fprintf('  %s%s: %.2f%%\n', methods{m}, stations{s}, percentages(s, m));
    end
    fprintf('  Non-zero Po_%s count: %d\n', stations{s}, nonzero_Po(s));
end