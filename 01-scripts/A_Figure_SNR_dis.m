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
                SNR_dB = 20 * log10(nsp(3) / nsp(1));
                if SNR_dB >= 0.1   % keep only non-negative SNR
                    snrValues{j}(end+1) = SNR_dB; %#ok<AGROW>
                    allSnr(end+1) = SNR_dB; %#ok<AGROW>
                end
            end
        end
    end
end

% --- Step 2: Plot histograms ---
binEdges = 0:1:50;  
binWidth = binEdges(2) - binEdges(1);

% --- Step 3: Plot 8 subplots: SNR distribution (histograms) ---
figure('Position', [100, 100, 800, 900]);

for j = 1:numStations+1
    subplot(4, 2, j);

    if j <= numStations
        snrData = snrValues{j};
        nameStr = ['Station ' stations{j}];
    else
        snrData = allSnr;
        nameStr = 'All Stations Total';
    end

    % Remove NaN and SNR < 0
    snrData = snrData(~isnan(snrData) & snrData >= 0.1);

    % Compute median
    medVal = median(snrData);

    % Plot histogram (distribution of SNR)
    h = histogram(snrData, 'BinEdges', binEdges, 'Normalization', 'count');
    xlabel('SNR (dB)');
    ylabel('Count');
    xlim([0 60]);
    title(sprintf('%s — Median = %.2f dB', nameStr, medVal));
    grid on;
    hold on;

    % ---------- Fit lognormal ONLY for last subplot ----------
    if j == numStations + 1
        % Fit lognormal to all-station SNR
        pd = fitdist(snrData', 'Lognormal');   % uses log(SNR)

        % Use same bins as histogram
        [counts, edges] = histcounts(snrData, binEdges);
        binCenters = edges(1:end-1) + diff(edges)/2;

        % Lognormal PDF evaluated at bin centers
        pdfVals = pdf(pd, binCenters);

        % Scale PDF to match histogram COUNTS: pdf * N * binWidth
        N = numel(snrData);
        yFit = pdfVals * N * binWidth;

        % Plot fitted curve
        plot(binCenters, yFit, 'r-', 'LineWidth', 2);

        legend('Counts', sprintf('Lognormal fit (\\mu=%.2f, \\sigma=%.2f)', ...
                                 pd.mu, pd.sigma), ...
               'Location', 'northeast');
    end
end

sgtitle('SNR Distribution');
set(gcf, 'Color', 'w');
