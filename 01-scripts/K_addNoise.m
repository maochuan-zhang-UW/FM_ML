% Load the data files
clear;close all;
signal_file = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_wave/A_wave_dB20_polish.mat';
noise_file = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/H_Noi/H_Noise_200.mat';

signal_data = load(signal_file);
noise_data = load(noise_file);

% Access Felix struct from both files
signal_felix = signal_data.Felix;
noise_felix = noise_data.Felix;

% Define stations (based on fields provided, extended to 7 for subplot)
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};

% Function to find first non-empty W_* field index with exactly 200 points
find_valid_index = @(felix, station) find(arrayfun(@(x) isfield(x, ['W_' station]) && ...
    ~isempty(x.(['W_' station])) && length(x.(['W_' station])) == 200, felix), 1, 'first');

% Display valid stations for debugging
fprintf('Checking for valid stations with 200-point signal and noise data:\n');
valid_stations = {};
valid_indices = zeros(1, length(stations));
for s = 1:length(stations)
    station = stations{s};
    idx_signal = find_valid_index(signal_felix, station);
    idx_noise = find_valid_index(noise_felix, station);
    if ~isempty(idx_signal) && ~isempty(idx_noise) && idx_signal == idx_noise
        fprintf('Station %s: Valid data at Felix index %d (signal length: %d, noise length: %d)\n', ...
            station, idx_signal, length(signal_felix(idx_signal).(['W_' station])), ...
            length(noise_felix(idx_noise).(['W_' station])));
        valid_stations{end+1} = station;
        valid_indices(s) = idx_signal;
    else
        fprintf('Station %s: Invalid or missing data (signal idx: %s, noise idx: %s)\n', ...
            station, mat2str(idx_signal), mat2str(idx_noise));
    end
end

if isempty(valid_stations)
    error('No stations have valid 200-point signal and noise data.');
end

% Select first valid station for the 27-wave stacked plot
selected_station = valid_stations{1};
selected_idx = valid_indices(find(strcmp(stations, selected_station), 1));

% Extract signal and noise
signal = signal_felix(selected_idx).(['W_' selected_station]);
noise = noise_felix(selected_idx).(['W_' selected_station]);

% Ensure column vectors
signal = signal(:);
noise = noise(:);

% Time axis (200 points)
t = 1:length(signal);

% Precompute RMS for signal and noise
rms_signal = rms(signal);
rms_noise = rms(noise);

% Prepare the 27 waves: signal, noise, and synthetics at 1-25 dB
waves = cell(27, 1);
labels = cell(27, 1);

% Normalize and store signal
waves{1} = signal / max(abs(signal));
labels{1} = 'Signal';

% Normalize and store noise
waves{2} = noise / max(abs(noise));
labels{2} = 'Noise';

% Generate synthetics for dB 1 to 25
for db = 1:25
    % Desired RMS for noise at this SNR
    desired_rms_noise = rms_signal / (10^(db / 20));
    
    % Scale factor for noise
    scale = desired_rms_noise / rms_noise;
    
    % Scaled noise
    scaled_noise = scale * noise;
    
    % Synthetic signal
    synthetic = signal + scaled_noise;
    
    % Normalize synthetic
    waves{db + 2} = synthetic / max(abs(synthetic));
    labels{db + 2} = sprintf('%d dB', db);
end

% Figure 1: Stacked plot of 27 waves in one column with offset 2
figure(1);
hold on;
offset = 2;
y_base = (27:-1:1) * offset;  % Starting y for each wave (top to bottom)
for i = 1:27
    plot(t, waves{i} + y_base(i), 'k');
    text(t(end) + 5, y_base(i), labels{i}, 'VerticalAlignment', 'middle');
end
hold off;
xlabel('Time (samples)');
ylabel('Normalized Amplitude');
title(sprintf('Waveforms of Station %s', selected_station));
xlim([1, length(t) + 20]);  % Extra space for labels
grid on;
set(gca, 'YTick', []);  % Optional: remove y-ticks since offsets are arbitrary