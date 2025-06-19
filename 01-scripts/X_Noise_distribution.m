clear; % Clear workspace
% Load data
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_wave/A_wave_forML1sec.mat');
numStructs = length(Felix);
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
numStations = length(stations);
times = zeros(numStructs, 1);
noiseValues = zeros(numStructs, numStations);

% Define start date for filtering (June 19, 2015)
startDate = datetime(2015, 6, 19);

% Extract data from structs
for i = 1:numStructs
    times(i) = Felix(i).on;
    for j = 1:numStations
        fieldName = ['NSP_' stations{j}];
        if isfield(Felix(i), fieldName)
            noise = Felix(i).(fieldName)(1); % Take first value of NSP array
            % Skip if noise value is 1
            if noise == 1
                noiseValues(i, j) = NaN; % Use NaN to skip in plot
            else
                noiseValues(i, j) = noise;
            end
        else
            noiseValues(i, j) = NaN; % Handle missing fields
        end
    end
end

% Convert times to datetime for filtering and plotting
timesDT = datetime(times, 'ConvertFrom', 'datenum');

% Filter data for dates >= June 19, 2015
validTimeIdx = timesDT >= startDate;
timesDT = timesDT(validTimeIdx);
noiseValues = noiseValues(validTimeIdx, :);

% Create figure with 7 subplots
figure('Position', [100, 100, 1200, 800]); % Adjust figure size
for j = 1:numStations
    subplot(4, 2, j); % 4 rows, 2 columns, j-th subplot (leaving 8th slot empty)
    validIdx = ~isnan(noiseValues(:, j));
    if any(validIdx)
        scatter(timesDT(validIdx), noiseValues(validIdx, j), 4, 'b', 'filled', ...
            'DisplayName', stations{j});
        % Customize subplot
        xlabel('Date');
        ylabel('Noise Value');
        title(['Station ' stations{j}]);
        grid on;
        % Format x-axis as dates
        % Set x-axis ticks to every 3 months
        minDate = max(startDate, min(timesDT(validIdx))); % Earliest date (after filtering)
        maxDate = max(timesDT(validIdx)); % Latest date
        tickDates = minDate:calmonths(3):maxDate; % 3-month intervals
        set(gca, 'XTick', tickDates); % Set ticks to datetime values
        set(gca, 'XTickLabel', datestr(tickDates, 'mm/yyyy')); % Format as MM/YYYY
        set(gca, 'XTickLabelRotation', 45); % Rotate labels for readability
        %datetick('x', 'yyyy-mm-dd', 'keeplimits');
        %set(gca, 'XTickLabelRotation', 45); % Rotate labels for readability
        
        % Set consistent y-axis limits if needed
        if any(noiseValues(validIdx, j) > 0)
            set(gca, 'YScale', 'log'); % Log scale for large range
        end
    else
        % Empty subplot if no valid data
        text(0.5, 0.5, ['No Data for ' stations{j}], ...
            'HorizontalAlignment', 'center', 'FontSize', 10);
        axis off;
    end
end

% Add overall title
sgtitle('Noise Values Across Stations (Starting June 19, 2015)');
% Adjust layout
set(gcf, 'Color', 'w'); % White background
