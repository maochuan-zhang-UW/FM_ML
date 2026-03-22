clear; 
% Load data
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_wave/A_wave_forML1sec100hz.mat');
numStructs = length(Felix);
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
numStations = length(stations);

% --- Step 1: Collect SNR values for each station ---
snrValues = cell(numStations, 1); % store SNR per station
allSnr = []; % for total

for i = 1:numStructs
    for j = 1:numStations
        nspField = ['NSP_' stations{j}];
        wField   = ['W_' stations{j}];
        
        if isfield(Felix(i), nspField) && isfield(Felix(i), wField)
            nsp = Felix(i).(nspField);
            if length(nsp) >= 3 && nsp(1) > 0 && nsp(3) > 0
                SNR_dB = 10 * log10(nsp(3) / nsp(1));
                snrValues{j}(end+1) = SNR_dB; %#ok<AGROW>
                allSnr(end+1) = SNR_dB; %#ok<AGROW>
            end
        end
    end
end

% --- Step 2: Define SNR range for x-axis ---
xVals = 0:1:50; % dB values (adjust if needed)
markDB = [5 10 15 20]; % annotate at these dB values

% --- Step 3: Plot 8 subplots ---
figure('Position', [100, 100, 1200, 900]);

for j = 1:numStations+1
    subplot(4, 2, j);
    if j <= numStations
        snrData = snrValues{j};
        titleStr = ['Station ' stations{j}];
    else
        snrData = allSnr;
        titleStr = 'All Stations Total';
    end
    
    % compute survival counts (# with SNR > threshold)
    yVals = arrayfun(@(x) sum(snrData > x), xVals);
    
    plot(xVals, yVals, 'LineWidth', 1.5);
    xlabel('SNR (dB)');
    ylabel('Count');
    title(titleStr);
    grid on;
    hold on;
    
    % annotate specific dB values
    for k = 1:length(markDB)
        dbVal = markDB(k);
        yVal = sum(snrData > dbVal);
        plot(dbVal, yVal, 'ro', 'MarkerSize', 6, 'HandleVisibility','off'); % mark point
        text(dbVal, yVal, sprintf('%d', yVal), ...
            'VerticalAlignment','bottom','HorizontalAlignment','left', 'FontSize', 8);
    end
end

sgtitle('SNR Distribution: Count of W_{station} > Threshold (dB)');
set(gcf, 'Color', 'w');
