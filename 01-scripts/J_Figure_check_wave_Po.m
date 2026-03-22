% Script to plot 20 normalized waveforms per station in a single subplot with y-axis offset
clc; clear;
%load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/H_Noi/H_Noise_200.mat')
%load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_wave/A_wave_forML1sec200Hz.mat')
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_wave/A_wave_dB20_polish.mat')
% List of stations and their W_* and Po_* fields
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
wFields = {'W_AS1', 'W_AS2', 'W_CC1', 'W_EC1', 'W_EC2', 'W_EC3', 'W_ID1'};
poFields = {'Po_AS1', 'Po_AS2', 'Po_CC1', 'Po_EC1', 'Po_EC2', 'Po_EC3', 'Po_ID1'};
nonEmptyCounts = [288, 799, 353, 565, 656, 800, 419]; % Provided counts

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
    nonEmptyIndices = find(arrayfun(@(x) ~isempty(x.(wFields{s})), Felix));
    
    % Check if there are enough non-empty events
    numAvailable = length(nonEmptyIndices);
    numToSelect = min(numEventsToSelect, numAvailable);
    
    if numToSelect == 0
        fprintf('No non-empty events for station %s\n', stations{s});
        continue;
    end
    
    % Randomly select indices
    selectedIndices = randsample(nonEmptyIndices, numToSelect);
    
    % Plot each selected event with offset
    for i = 1:numToSelect
        waveData = Felix(selectedIndices(i)).(wFields{s});
        
        % Normalize waveform to range [-1, 1]
        if ~isempty(waveData)
            waveData = waveData / max(abs(waveData));
        else
            waveData = zeros(size(waveData)); % Handle empty data
        end
        
        % Apply y-axis offset
        offset = (i-1) * 2; % Offset by +2 for each waveform
        plot(waveData + offset, 'b-', 'LineWidth', 0.5);
        
        % Store polarity for legend (optional)
        polarity = Felix(selectedIndices(i)).(poFields{s});
         % Add polarity label at the end of the waveform
        text( 10, offset, sprintf('Po: %s', num2str(polarity)), ...
            'FontSize', 12, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
        %legendLabels{i} = sprintf('Event %d, Po: %s', Felix(selectedIndices(i)).ID, num2str(polarity));
    end
    
    % Customize subplot
    title(sprintf('Station %s', stations{s}));
    xlabel('Sample Index');
    ylabel('Normalized Amplitude');
    grid on;
    
    % Adjust y-axis limits to fit all waveforms
    ylim([-2, (numToSelect * 2)]);
    box on;
    
    % Optional: Add legend (commented out to avoid clutter)
    % legend(legendLabels, 'Location', 'best', 'FontSize', 6);
end

% Adjust figure layout
sgtitle('Example Normalized Waveforms for All Stations (20 Events Each)');
set(gcf, 'Position', [100, 100, 1400, 400]); % Adjust figure size
hold off;