% Precompute the magnitude and SNR data for Figure 1 panels (c) and (d).
%
% Reading the 176 MB prediction catalog takes ~1 min, so the result is cached
% to 02-data/fig01_panelCD_data.mat and the figure script just loads that.
%
% Run headless from the repository root:
%   /Applications/MATLAB_R2024a.app/bin/matlab -batch "run('04-manuscripts/Figure01_panelCD_prep.m')"

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(scriptDir);

catalogFile = fullfile(repoRoot, '02-data', 'A_wave_2015_2021_h5_conf80_predictions.mat');
templateFile = fullfile(repoRoot, '02-data', 'A_wave_dB20_cleaned.mat');
magFile = fullfile(repoRoot, '07-files', 'evMw.txt');
outFile = fullfile(repoRoot, '02-data', 'fig01_panelCD_data.mat');

stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};

% Signal-to-noise convention for the 64-sample model input windows:
%   noise  = peak absolute amplitude over samples 1:NOISE_END  (pre-arrival)
%   signal = peak absolute amplitude over samples NOISE_END+1:end
% This differs from the 200-sample RMS convention in
% 01-scripts/data_preparation/01_build_training_dataset.py because the
% prediction catalog stores waveforms already cropped to 64 samples.
NOISE_END = 20;

fprintf('Loading catalog: %s\n', catalogFile);
S = load(catalogFile, 'Felix');
C = S.Felix;
clear S;
nEvent = numel(C);
fprintf('  %d events\n', nEvent);

fprintf('Loading templates: %s\n', templateFile);
T = load(templateFile, 'Felix');
templateID = double([T.Felix.ID]);
clear T;
fprintf('  %d template events\n', numel(templateID));

% ---- Per-event quantities -------------------------------------------------
evID = double(cellfun(@(s) double(s.ID), C));
evLon = cellfun(@(s) double(s.lon), C);
evLat = cellfun(@(s) double(s.lat), C);
evDepth = cellfun(@(s) double(s.depth), C);
evTime = cellfun(@(s) double(s.on), C);
isTemplate = ismember(evID, templateID);

mw = load(magFile);
[hasMw, loc] = ismember(evID, mw(:, 1));
evMw = nan(size(evID));
evMw(hasMw) = mw(loc(hasMw), 2);

fprintf('Events: %d template, %d normal; %d (%.1f%%) have a magnitude\n', ...
    sum(isTemplate), sum(~isTemplate), sum(hasMw), 100 * sum(hasMw) / nEvent);

% ---- Per station-waveform SNR --------------------------------------------
% One value per station-event pair, pooled over all seven stations.
maxSlots = nEvent * numel(stations);
snrDb = nan(maxSlots, 1);
snrIsTemplate = false(maxSlots, 1);
snrStation = zeros(maxSlots, 1);
snrEventID = zeros(maxSlots, 1);
k = 0;

for si = 1:numel(stations)
    field = ['W_' stations{si}];
    fprintf('  SNR for %s ...\n', stations{si});
    for i = 1:nEvent
        w = C{i}.(field);
        if numel(w) <= NOISE_END
            continue;
        end
        aw = abs(double(w(:)));
        noiseAmp = max(aw(1:NOISE_END));
        signalAmp = max(aw(NOISE_END + 1:end));
        if ~(noiseAmp > 0) || ~isfinite(signalAmp)
            continue;
        end
        k = k + 1;
        snrDb(k) = 20 * log10(signalAmp / noiseAmp);
        snrIsTemplate(k) = isTemplate(i);
        snrStation(k) = si;
        snrEventID(k) = evID(i);
    end
end

snrDb = snrDb(1:k);
snrIsTemplate = snrIsTemplate(1:k);
snrStation = snrStation(1:k);
snrEventID = snrEventID(1:k);

fprintf('SNR waveforms: %d total (%d template, %d normal)\n', ...
    k, sum(snrIsTemplate), sum(~snrIsTemplate));
fprintf('  template SNR  median %.1f dB, 5-95%%: %.1f to %.1f dB\n', ...
    median(snrDb(snrIsTemplate)), prctile(snrDb(snrIsTemplate), 5), prctile(snrDb(snrIsTemplate), 95));
fprintf('  normal   SNR  median %.1f dB, 5-95%%: %.1f to %.1f dB\n', ...
    median(snrDb(~snrIsTemplate)), prctile(snrDb(~snrIsTemplate), 5), prctile(snrDb(~snrIsTemplate), 95));
fprintf('  template Mw   median %.2f\n', median(evMw(isTemplate), 'omitnan'));
fprintf('  normal   Mw   median %.2f\n', median(evMw(~isTemplate), 'omitnan'));

noiseEnd = NOISE_END; %#ok<NASGU>
save(outFile, 'evID', 'evLon', 'evLat', 'evDepth', 'evTime', 'evMw', ...
    'isTemplate', 'snrDb', 'snrIsTemplate', 'snrStation', 'snrEventID', ...
    'stations', 'noiseEnd', '-v7');
fprintf('Saved: %s\n', outFile);
