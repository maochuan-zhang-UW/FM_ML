clear;close all;

% Load data
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_wave/A_wave_forML1sec100hz.mat');

stations = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};
numStations = numel(stations);
numStructs = numel(Felix);

allSnr = [];

% --- Collect SNR for all stations ---
for i = 1:numStructs
    for j = 1:numStations
        nspField = ['NSP_' stations{j}];
        wField   = ['W_' stations{j}];

        if isfield(Felix(i), nspField) && isfield(Felix(i), wField)
            nsp = Felix(i).(nspField);
            if numel(nsp) >= 3 && nsp(1) > 0 && nsp(3) > 0
                SNR_dB = 20 * log10(nsp(3) / nsp(1));
                if SNR_dB >= 0.1
                    allSnr(end+1) = SNR_dB; %#ok<AGROW>
                end
            end
        end
    end
end

% --- Histogram parameters ---
binEdges = 0:1:60;
binWidth = binEdges(2) - binEdges(1);

% --- Plot ONLY the last histogram ---
figure('Position',[300 300 700 450],'Color','w');

% Clean data
snrData = allSnr(~isnan(allSnr) & allSnr >= 0.1);
medVal = median(snrData);

% Histogram
histogram(snrData,'BinEdges',binEdges,'Normalization','count');
hold on;
grid on;

xlabel('SNR (dB)');
ylabel('Count');
xlim([0 60]);
title(sprintf('All Stations — Median = %.2f dB', medVal));

% --- Lognormal fit ---
pd = fitdist(snrData','Lognormal');

[counts, edges] = histcounts(snrData, binEdges);
binCenters = edges(1:end-1) + diff(edges)/2;

pdfVals = pdf(pd, binCenters);
N = numel(snrData);
yFit = pdfVals * N * binWidth;

set(gca, 'FontSize',22)
%plot(binCenters, yFit, 'r-', 'LineWidth', 2);

%legend('Counts', ...
%       sprintf('Lognormal fit (\\mu=%.2f, \\sigma=%.2f)', pd.mu, pd.sigma), ...
%       'Location','northeast');
