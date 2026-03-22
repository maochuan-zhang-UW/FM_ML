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
                if SNR_dB >= 0.5   % keep only non-negative SNR
                    snrValues{j}(end+1) = SNR_dB; %#ok<AGROW>
                    allSnr(end+1) = SNR_dB; %#ok<AGROW>
                end
            end
        end
    end
end

% --- Step 2: Plot histograms ---
figure('Position', [100, 100, 1600, 600]); % wider for 2x4 layout

edges = 0:1:50;  % bin edges (adjust as needed)

for j = 1:numStations+1
    subplot(2, 4, j);   % 2 rows, 4 columns
    if j <= numStations
        snrData = snrValues{j};
        titleStr = ['Station ' stations{j}];
    else
        snrData = allSnr;
        titleStr = 'All Stations Total';
    end
    
    histogram(snrData, edges, 'FaceColor', [0.2 0.5 0.8], 'EdgeColor', 'k');
    xlabel('SNR (dB)');
    ylabel('Count');
    title(titleStr);
    grid on;
end

sgtitle('SNR Distribution Histograms (SNR ≥ 0)');
set(gcf, 'Color', 'w');
