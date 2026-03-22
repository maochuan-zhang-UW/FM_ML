clc; clear;

% Base directory for your DT.mat files
baseDir = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/N_Po/';

% Station list and field names
stations    = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
wFields     = {'W_AS1', 'W_AS2', 'W_CC1', 'W_EC1', 'W_EC2', 'W_EC3', 'W_ID1'};
manPoFields = {'Man_AS1', 'Man_AS2', 'Man_CC1', 'Man_EC1', 'Man_EC2', 'Man_EC3', 'Man_ID1'};
poMLFields  = {'PoML_W_AS1','PoML_W_AS2','PoML_W_CC1','PoML_W_EC1','PoML_W_EC2','PoML_W_EC3','PoML_W_ID1'};

% Number of events to select per station
numEventsToSelect = 20;

% Create a single figure with 1x7 subplots
figure('Name', 'Waveforms + Polarity (Man vs PoML)', ...
       'NumberTitle', 'off', 'Position', [100, 100, 1600, 600]);

for s = 1:length(stations)
    % Load each station's Felix data
    matFile = fullfile(baseDir, sprintf('N_Po%s_DT.mat', stations{s}));
    if ~isfile(matFile)
        fprintf('File not found: %s\n', matFile);
        continue;
    end
    load(matFile, 'Felix');

    subplot(1, 7, s);
    hold on;

    % Find non-empty waveforms
    nonEmptyIndices = find(arrayfun(@(x) ~isempty(x.(wFields{s})), Felix));
    numAvailable = length(nonEmptyIndices);
    numToSelect = min(numEventsToSelect, numAvailable);

    if numToSelect == 0
        fprintf('No non-empty events for station %s\n', stations{s});
        continue;
    end

    % Randomly select indices
    selectedIndices = randsample(nonEmptyIndices, numToSelect);

    % Plot waveforms
    for i = 1:numToSelect
        waveData = Felix(selectedIndices(i)).(wFields{s});

        % Normalize
        if ~isempty(waveData)
            waveData = waveData / max(abs(waveData));
        else
            waveData = zeros(size(waveData));
        end

        % Offset for visibility
        offset = (i-1) * 2;

        % Get polarities
        manPol  = Felix(selectedIndices(i)).(manPoFields{s});
        poMLPol = Felix(selectedIndices(i)).(poMLFields{s});

        % Choose color: blue if match, red if mismatch
        if manPol == poMLPol
            plotColor = 'b';
        else
            plotColor = 'r';
        end

        % Plot waveform with color
        plot(waveData + offset, '-', 'Color', plotColor, 'LineWidth', 0.5);

        % Annotate polarity values
        text(5, offset, sprintf('Man:%d | ML:%d', manPol, poMLPol), ...
            'FontSize', 8, 'HorizontalAlignment', 'left', ...
            'VerticalAlignment', 'middle', 'Color', plotColor);
    end

    % Subplot settings
    title(sprintf('Station %s', stations{s}));
    xlabel('Sample Index');
    ylabel('Amp + Offset');
    ylim([-2, (numToSelect * 2) + 2]);
    grid on;
end

sgtitle('20 Random Normalized Waveforms per Station (Blue=Match, Red=Mismatch)');
