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
% --- Quick check stacked plot ---
figure; hold on;set(gcf,'Color','w');
start = 3;
numToPlot = 10;   % number of waveforms
idx0 = start*numToPlot + start + 1;   % first index

for i = idx0 : idx0 + numToPlot
    % --- Get waveform and normalize ---
    waveData = AS1_add(i).W_AS1;
    if isempty(waveData)
        continue
    end
    
    % Normalize for display
    waveData = waveData / max(abs(waveData));
    
    % --- Vertical offset ---
    offset = (i - idx0) * 2;
    
    % --- Plotting ---
    if i == idx0
        plot(waveData + offset, 'r', 'LineWidth', 1.8); % Original in Red
    else
        plot(waveData + offset, 'k', 'LineWidth', 1.0); % Augmented in Black
    end
    
    % --- Label each waveform with its SNR ---
    snrVal = AS1_add(i).SNR_AS1;
    
    % Determine label color: Red for the original, Blue for augmented
    txtColor = 'blue';
    if i == idx0, txtColor = 'red'; end
    
    % Place text at the end of the waveform (sample 205) for clarity
    text(205, offset, sprintf('%.1f dB', snrVal), ...
        'FontSize', 9, ...
        'Color', txtColor, ...
        'VerticalAlignment', 'middle');
end

% --- Axes & formatting ---
xlim([0 240]); % Increased x-limit to make room for text labels

ylim([-1 2*(numToPlot+1) + 1]);
xlabel('Sample index','FontSize',12)
ylabel('Normalized waveforms (Stacked)','FontSize',12)
title('Augmented Waveforms with SNR Labels','FontSize',14,'FontWeight','bold')
set(gca, 'YTick', [], 'FontSize', 11, 'LineWidth', 1.1, 'Box', 'on')
grid on
set(gca,'GridAlpha',0.15)


ylim([-1 21])

