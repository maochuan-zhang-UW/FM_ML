% Extract SNR in dB for CC1
clc;clear;close all
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/D_man/D_CC1_128.mat')
% Extract SNR in dB for CC1
numStructs = length(Felix);
snr_all = [];
snr_picked = [];

for i = 1:numStructs
    if isfield(Felix(i), 'NSP_CC1') && isfield(Felix(i), 'CC1_Po')
        nsp = Felix(i).NSP_CC1;
        if ~isempty(nsp) && length(nsp) >= 3
            noise_val  = nsp(1);
            signal_val = nsp(3);

            % Only take valid cases (positive values)
            if noise_val > 0 && signal_val > 0
                snr_val_dB = 20*log10(signal_val / noise_val);

                % Skip if SNR is exactly 0 dB
                if snr_val_dB == 0
                    continue;
                end

                % Collect all SNRs
                snr_all(end+1) = snr_val_dB; %#ok<*AGROW>

                % Collect only when polarity picked
                if Felix(i).CC1_Po ~= 0
                    snr_picked(end+1) = snr_val_dB;
                end
            end
        end
    end
end

% Plot histogram
figure;
hold on;

% Histogram of all SNR values (gray)
h1 = histogram(snr_all, 'BinWidth', 2, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none');

% Histogram of picked SNR values (blue)
h2 = histogram(snr_picked, 'BinWidth', 2, 'FaceColor', [0.2 0.4 0.8], 'EdgeColor', 'none');

% Formatting
xlabel('SNR (dB, CC1)');
ylabel('Count');
legend([h1 h2], {'All SNR', 'Picked (CC1\_Po ≠ 0)'});
title('SNR Distribution at CC1 (in dB, excluding 0 dB)');
grid on;
hold off;

