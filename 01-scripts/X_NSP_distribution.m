clear; % Clear workspace
% Load data
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_wave/A_wave_forML1sec100hz.mat');
numStructs = length(Felix);
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
numStations = length(stations);
times = zeros(numStructs, 1);
noiseValues = zeros(numStructs, numStations); % NSP(1)
sWaveValues = zeros(numStructs, numStations); % NSP(2)
pWaveValues = zeros(numStructs, numStations); % NSP(3)

% Define start date for filtering (June 19, 2015)
startDate = datetime(2015, 6, 19);

% Extract data from structs
for i = 1:numStructs
    times(i) = Felix(i).on;
    for j = 1:numStations
        fieldName = ['NSP_' stations{j}];
        if isfield(Felix(i), fieldName)
            nsp = Felix(i).(fieldName); % Get NSP array
            % Check if NSP has at least 3 elements
            if length(nsp) >= 3
                % Noise (NSP(1))
                if nsp(1) == 1
                    noiseValues(i, j) = NaN;
                else
                    noiseValues(i, j) = nsp(1);
                end
                % S-wave (NSP(2))
                if nsp(2) == 1
                    sWaveValues(i, j) = NaN;
                else
                    sWaveValues(i, j) = nsp(2);
                end
                % P-wave (NSP(3))
                if nsp(3) == 1
                    pWaveValues(i, j) = NaN;
                else
                    pWaveValues(i, j) = nsp(3);
                end
            else
                noiseValues(i, j) = NaN;
                sWaveValues(i, j) = NaN;
                pWaveValues(i, j) = NaN;
            end
        else
            noiseValues(i, j) = NaN;
            sWaveValues(i, j) = NaN;
            pWaveValues(i, j) = NaN;
        end
    end
end

% Convert times to datetime for filtering and plotting
timesDT = datetime(times, 'ConvertFrom', 'datenum');

% Filter data for dates >= June 19, 2015
validTimeIdx = timesDT >= startDate;
timesDT = timesDT(validTimeIdx);
noiseValues = noiseValues(validTimeIdx, :);
sWaveValues = sWaveValues(validTimeIdx, :);
pWaveValues = pWaveValues(validTimeIdx, :);

% Create figure with 7 subplots
figure('Position', [100, 100, 1200, 800]); % Adjust figure size
for j = 1:numStations
    subplot(4, 2, j); % 4 rows, 2 columns, j-th subplot (leaving 8th slot empty)
    validIdx = ~isnan(noiseValues(:, j)) | ~isnan(sWaveValues(:, j)) | ~isnan(pWaveValues(:, j));
    if any(validIdx)
        % Create primary axes for scatter plots
        ax1 = gca;
        hold on;
        % Scatter plots for Noise, S-wave, P-wave
        if any(~isnan(noiseValues(:, j)))
            scatter(ax1, timesDT(~isnan(noiseValues(:, j))), noiseValues(~isnan(noiseValues(:, j)), j), ...
                4, 'b', 'filled', 'DisplayName', 'Noise');
        end
        if any(~isnan(sWaveValues(:, j)))
            scatter(ax1, timesDT(~isnan(sWaveValues(:, j))), sWaveValues(~isnan(sWaveValues(:, j)), j), ...
                4, 'r', '^', 'DisplayName', 'S-wave');
        end
        if any(~isnan(pWaveValues(:, j)))
            scatter(ax1, timesDT(~isnan(pWaveValues(:, j))), pWaveValues(~isnan(pWaveValues(:, j)), j), ...
                4, 'g', 's', 'DisplayName', 'P-wave');
        end
        % Customize primary axes
        xlabel(ax1, 'Date');
        ylabel(ax1, 'Value');
        title(ax1, ['Station ' stations{j}]);
        grid(ax1, 'on');
        % Set x-axis ticks to every 3 months
        minDate = max(startDate, min(timesDT(validIdx))); % Earliest date
        maxDate = max(timesDT(validIdx)); % Latest date
        tickDates = dateshift(minDate:calmonths(3):maxDate, 'start', 'month'); % Align to start of month
        set(ax1, 'XTick', tickDates); % Set ticks to datetime values
        set(ax1, 'XTickLabel', datestr(tickDates, 'mm/yyyy')); % Format as MM/YYYY
        set(ax1, 'XTickLabelRotation', 45); % Rotate labels
        % Set log scale for primary y-axis if all valid data is positive
        allValues = [noiseValues(validIdx, j); sWaveValues(validIdx, j); pWaveValues(validIdx, j)];
        if any(allValues > 0) && all(allValues(~isnan(allValues)) > 0)
            set(ax1, 'YScale', 'log');
            yLimits = [min(allValues(allValues > 0)), max(allValues)];
            set(ax1, 'YLim', yLimits);
        end
        legend(ax1, 'show', 'Location', 'best');

        % % Create inset axes for box plot (top-right corner)
        % ax2 = axes('Position', [ax1.Position(1) + ax1.Position(3)*0.65, ...
        %                         ax1.Position(2) + ax1.Position(4)*0.55, ...
        %                         ax1.Position(3)*0.3, ax1.Position(4)*0.35], ...
        %            'Box', 'on');
        % hold(ax2, 'on');
        % % Prepare data for box plot
        % boxData = [noiseValues(validIdx, j), sWaveValues(validIdx, j), pWaveValues(validIdx, j)];
        % % Box plot with matching colors
        % boxplot(ax2, boxData, 'Labels', {'N', 'S', 'P'}, ...
        %         'Widths', 0.15, 'Symbol', '', ...
        %         'Colors', ['b', 'r', 'g']);
        % ylabel(ax2, 'Values');
        % % Set log scale for box plot if all data is positive
        % if any(allValues > 0) && all(allValues(~isnan(allValues)) > 0)
        %     set(ax2, 'YScale', 'log');
        %     set(ax2, 'YLim', yLimits); % Match y-limits with scatter plot
        % end
        % set(ax2, 'FontSize', 8); % Smaller font for inset
        % hold(ax2, 'off');
        % hold(ax1, 'off');
    else
        % Empty subplot if no valid data
        text(0.5, 0.5, ['No Data for ' stations{j}], ...
            'HorizontalAlignment', 'center', 'FontSize', 10);
        axis off;
    end
end

% Add overall title
sgtitle('Noise, S-wave, and P-wave Values Across Stations (Starting June 19, 2015)');
% Adjust layout
set(gcf, 'Color', 'w'); % White background