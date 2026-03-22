clear; close all;

% --- Load data ---
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/K_aug/AS1.mat')   % AS1
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/H_Noi/H_noise_dB20_snrValue.mat'); % snrValues
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/H_Noi/H_Noise_200.mat');           % Felix noise

% --- Step 1: Fit lognormal distribution to SNR data ---
snrData = snrValues{1};       % Example: AS1 station
snrData = snrData(snrData > 0); % remove non-positive values
pd = fitdist(snrData(:),'Lognormal');

% --- Step 2: Generate synthetic SNR values (dB) ---
numSamples = length(AS1) * 10;  % 10 noisy copies per waveform
synthetic_SNR = random(pd, numSamples, 1);

% --- Step 3: Augment AS1 with noisy variants ---
AS1_add = [];
j = 1;        % index into synthetic_SNR
n = 1;        % running ID2 counter

for i = 1:length(AS1)
    % --- Keep original ---
    tmp.ID2       = n; n = n + 1;
    tmp.ID        = AS1(i).ID;
    tmp.lon       = AS1(i).lon;
    tmp.lat       = AS1(i).lat;
    tmp.depth     = AS1(i).depth;
    tmp.W_AS1     = AS1(i).W_AS1;
    tmp.Man_AS1   = AS1(i).Man_AS1;
    tmp.Po_AS1    = AS1(i).Po_AS1;

    % Compute SNR for original waveform
    noise_seg   = tmp.W_AS1(1:80);
    signal_seg  = tmp.W_AS1(81:160);
    tmp.SNR_AS1 = 20*log10(rms(signal_seg)/rms(noise_seg));

    AS1_add = [AS1_add tmp]; %#ok<AGROW>
    
    % --- Generate 10 noisy copies ---
    for k = 1:10
        signal = AS1(i).W_AS1;
        rms_signal = rms(signal);

        % Target SNR in dB
        target_snr_dB = synthetic_SNR(j); 
        j = j + 1;

        % Convert target SNR dB → desired RMS(noise)
        rms_noise_target = rms_signal / (10^(target_snr_dB/20));

        % --- Select valid random noise ---
        valid = false;
        while ~valid
            noisenum = randi([1, length(Felix)]);
            if isfield(Felix(noisenum),'W_AS1') && ~isempty(Felix(noisenum).W_AS1)
                noise = Felix(noisenum).W_AS1;
                if length(noise) == 200 && any(noise)   % length 200 and not all zeros
                    valid = true;
                end
            end
        end

        % --- Scale noise to target RMS ---
        rms_noise_actual = rms(noise);
        scale = rms_noise_target / rms_noise_actual;
        scaled_noise = scale * noise;

        % --- Synthetic waveform ---
        synthetic = signal + scaled_noise;

        % --- Store as new struct ---
        tmp2.ID2     = n; n = n + 1;
        tmp2.ID      = AS1(i).ID;
        tmp2.lon     = AS1(i).lon;
        tmp2.lat     = AS1(i).lat;
        tmp2.depth   = AS1(i).depth;
        tmp2.W_AS1   = synthetic;
        tmp2.Man_AS1 = AS1(i).Man_AS1;
        tmp2.Po_AS1  = AS1(i).Po_AS1;

        % Compute SNR for synthetic waveform
        noise_seg   = synthetic(1:80);
        signal_seg  = synthetic(81:160);
        tmp2.SNR_AS1 = 20*log10(rms(signal_seg)/rms(noise_seg));

        AS1_add = [AS1_add tmp2]; %#ok<AGROW>
    end
end

fprintf('Final augmented dataset size: %d entries\n', numel(AS1_add));

% --- Quick check plot ---
% --- Quick check stacked plot ---
figure; hold on;
start=3;
numToPlot = 10;   % first 10 waveforms
for i = start*numToPlot+start+1:start*numToPlot+start+11
    % Get waveform and normalize
    waveData = AS1_add(i).W_AS1;
    if ~isempty(waveData)
        waveData = waveData / max(abs(waveData));
    else
        waveData = zeros(size(waveData));
    end
    
    % Offset by +2 for stacking
    offset = (i-1) * 2;
    
    % Plot waveform
    plot(waveData + offset, 'k-', 'LineWidth', 1);
    
    % Build label: Man, Po, SNR
    manVal = AS1_add(i).Man_AS1;
    poVal  = AS1_add(i).Po_AS1;
    snrVal = AS1_add(i).SNR_AS1;
    if i==start*numToPlot+start+1

    text(5, offset+0.6, sprintf('Man: %d | %.2f dB (signal)', ...
        manVal, snrVal), ...
        'FontSize', 10, 'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'middle');
    end
    
    text(5, offset+0.6, sprintf('Man: %d | %.2f dB', ...
        manVal, snrVal), ...
        'FontSize', 10, 'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'middle');
end
ylim([65 88]);
xlabel('Sample');
ylabel('Waveforms');
title('Augmented Waveforms (Station AXAS1)');
set(gca,'YTick',[]); % Hide y-ticks since stacked
grid on;
box on;

