clear; close all;

outDir = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/K_aug';
stations = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};

figure('Position',[100 100 1400 800]); 

for s = 1:numel(stations)
    station = stations{s};
    varName = [station '_add'];
    matFile = fullfile(outDir, [varName '.mat']);
    load(matFile, varName);   % load station data
    data = eval(varName);

    subplot(2,4,s); % 7 subplots in a 2x4 grid (last one empty)
    scatter([data.lon], [data.lat], 8, 'k', 'filled');
    xlabel('Longitude'); ylabel('Latitude');
    title([station ' events']);
    grid on; box on;
end

sgtitle('Event Locations by Station (Augmented Datasets)');
