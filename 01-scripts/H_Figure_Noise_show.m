% Script to plot 20 normalized waveforms per station in a single subplot with y-axis offset
clc; clear;
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/H_Noi/H_Noise_200.mat')

% List of stations and their W_* and Po_* fields
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
wFields = {'W_AS1', 'W_AS2', 'W_CC1', 'W_EC1', 'W_EC2', 'W_EC3', 'W_ID1'};
poFields = {'Po_AS1', 'Po_AS2', 'Po_CC1', 'Po_EC1', 'Po_EC2', 'Po_EC3', 'Po_ID1'};

% Number of events to select per station
numEventsToSelect = 20;

% Create a single figure with 1x7 subplots
figure('Name', 'Waveforms for All Stations', 'NumberTitle', 'off', 'Position', [100, 100, 1400, 400]);

% Loop through each station
for s = 1:length(stations)
    % Create subplot for this station
    subplot(1, 7, s);
    hold on;
    
    % Find indices of non-empty W_* events for this station
    % More robust check: ensure data exists and is not empty
    nonEmptyIndices = [];
    for idx = 1:length(Felix)
        if isfield(Felix(idx), wFields{s}) && ...
           ~isempty(Felix(idx).(wFields{s})) && ...
           all(~isnan(Felix(idx).(wFields{s}))) && ...
           any(Felix(idx).(wFields{s}) ~= 0) % Additional check for non-zero data
            nonEmptyIndices = [nonEmptyIndices, idx];
        end
    end
    
    % Check if there are enough non-empty events
    numAvailable = length(nonEmptyIndices);
    numToSelect = min(numEventsToSelect, numAvailable);
    
    if numToSelect == 0
        fprintf('No non-empty events for station %s\n', stations{s});
        continue;
    elseif numAvailable < numEventsToSelect
        fprintf('Warning: Only %d non-empty events available for station %s (requested %d)\n', ...
                numAvailable, stations{s}, numEventsToSelect);
    end
    
    % Randomly select indices from non-empty events
    selectedIndices = nonEmptyIndices(randperm(numAvailable, numToSelect));
    
    % Plot each selected event with offset
    for i = 1:numToSelect
        waveData = Felix(selectedIndices(i)).(wFields{s});
        
        % Normalize waveform to range [-1, 1]
        if ~isempty(waveData) && max(abs(waveData)) > 0
            waveData = waveData / max(abs(waveData));
        else
            % Skip plotting if data is still empty or zero
            continue;
        end
        
        % Apply y-axis offset
        offset = (i-1) * 2; % Offset by +2 for each waveform
        plot(waveData + offset, 'b-', 'LineWidth', 0.5);
        
        % Add polarity label at the end of the waveform
        % if isfield(Felix(selectedIndices(i)), poFields{s}) && ...
        %    ~isempty(Felix(selectedIndices(i)).(poFields{s}))
        %     polarity = Felix(selectedIndices(i)).(poFields{s});
        %     text(length(waveData), offset, sprintf('Po: %s', num2str(polarity)), ...
        %         'FontSize', 8, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');
        % end
    end
    
    % Customize subplot
    title(sprintf('Station %s\n(%d events)', stations{s}, numToSelect));
    xlabel('Sample Index');
    ylabel('Normalized Amplitude + Offset');
    grid on;
    
    % Adjust y-axis limits to fit all waveforms
    ylim([-2, (numToSelect * 2) + 1]);
    box on;
end

% Adjust figure layout
sgtitle('Example Noise Waveforms for All Stations (20 Events Each)');
set(gcf, 'Position', [100, 100, 1400, 400]);
hold off;