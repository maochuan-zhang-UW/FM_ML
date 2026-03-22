clear; close all;

outDir = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/K_aug';
stations = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};

figure('Position',[100 100 1400 800]);

allSNR = [];

for s = 1:numel(stations)
    station = stations{s};
    varName = [station '_add'];
    matFile = fullfile(outDir, [varName '.mat']);
    load(matFile, varName);   % load station data
    data = eval(varName);

    % SNR field for this station
    snrField = ['SNR_' station];
    snrVals = [data.(snrField)];

    % Collect for global histogram
    allSNR = [allSNR snrVals];

    % Plot histogram for this station
    subplot(2,4,s);
    histogram(snrVals, 50, 'FaceColor', 'k', 'EdgeColor', 'none');
    xlabel('SNR (dB)');
    ylabel('Count');
    title([station ' SNR']);
    grid on; box on;
end

% --- Last subplot: all stations ---
subplot(2,4,8);
histogram(allSNR, 50, 'FaceColor', 'b', 'EdgeColor', 'none');
xlabel('SNR (dB)');
ylabel('Count');
title('All Stations Combined');
grid on; box on;

sgtitle('SNR Distributions per Station and Combined');
