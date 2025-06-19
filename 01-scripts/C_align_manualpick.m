clc; clear;

% Load the data
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/F_ML/F_CC1_MLPo_DiTing128_Stone.mat');
Felix = [Felix{:}]; % Expand the Felix structure

% Extract the main waveform and its parameters
main_waveform = Felix(1).W1_CC1(:); % Ensure it's a column vector
L = length(main_waveform); % Should be 128 samples
Fs = 100; % Sampling rate: 100 Hz
t = linspace(-0.64, 0.64, L); % Time vector

% --- Define the cross-correlation window [-0.1, 0.4] ---
start_idx = find(t >= -0.1, 1, 'first');
end_idx = find(t <= 0.4, 1, 'last');

% Subset the main waveform to the desired time window
main_window = main_waveform(start_idx:end_idx);

% --- Normalization of the main waveform in the window ---
main_window = (main_window - min(main_window)) / (max(main_window) - min(main_window));
main_window = 2 * (main_window - 0.5); % Normalize to [-1, 1]

% --- Normalization of the full main waveform for plotting ---
main_waveform = (main_waveform - min(main_waveform)) / (max(main_waveform) - min(main_waveform));
main_waveform = 2 * (main_waveform - 0.5); % Normalize to [-1, 1]

% Initialize variables
num_waveforms = length(Felix);
cc_values = zeros(num_waveforms-1, 1);
indices = (2:num_waveforms)';

% Preallocate space for cross-correlations
window_length = length(main_window);
cross_corrs = zeros(2 * window_length - 1, num_waveforms-1);

% --- Vectorized cross-correlation calculation in window ---
parfor i = 2:num_waveforms
    other_waveform = Felix(i).W1_CC1(:); % Ensure it's a column vector
    
    % --- Window subset for each waveform ---
    other_window = other_waveform(start_idx:end_idx);
    
    % --- Normalization of each waveform in the window ---
    other_window = (other_window - min(other_window)) / (max(other_window) - min(other_window));
    other_window = 2 * (other_window - 0.5); % Normalize to [-1, 1]
    
    % Cross-correlation within the window
    cross_corrs(:, i-1) = xcorr(main_window, other_window, 'coeff');
end

% Find maximum cross-correlation for each waveform
[max_cc, max_idx] = max(cross_corrs);
cc_values = max_cc(:);

% Sort and select the top 20 waveforms with the highest correlation
[~, sort_idx] = sort(cc_values, 'descend');
top_20_indices = indices(sort_idx(1:20));
top_20_cc = cc_values(sort_idx(1:20));
top_20_lags = max_idx(sort_idx(1:20)) - window_length;

% Display the CC coefficients
disp('Top 20 Cross-Correlation Coefficients:');
table(top_20_indices, top_20_cc)

% --- Plotting (Vertical Alignment) ---
figure;
hold on;
plot(t, main_waveform, 'k', 'LineWidth', 1.5); % Plot the main waveform in black

% Vertical offset for alignment
vertical_offset = 1; % Separation between waveforms

% Container for stacking
aligned_waveforms = zeros(L, 20);

% Plot the aligned waveforms
for j = 1:length(top_20_indices)
    idx = top_20_indices(j);
    shift = top_20_lags(j);
    other_waveform = Felix(idx).W1_CC1(:);
    
    % --- Normalization of the full waveform ---
    other_waveform = (other_waveform - min(other_waveform)) / (max(other_waveform) - min(other_waveform));
    other_waveform = 2 * (other_waveform - 0.5);

    % Alignment logic
    if shift > 0
        aligned_waveform = [zeros(shift, 1); other_waveform];
        aligned_waveform = aligned_waveform(1:L);
    else
        aligned_waveform = other_waveform(abs(shift) + 1:end);
        if length(aligned_waveform) < L
            aligned_waveform = [aligned_waveform; zeros(L - length(aligned_waveform), 1)];
        end
    end
    
    % Store for stacking
    aligned_waveforms(:, j) = aligned_waveform;

    % Plot each aligned waveform with vertical offset
    plot(t, aligned_waveform + j * vertical_offset);
end

% Plot each waveform
for j = 1:20
    plot(t, aligned_waveforms(:, j) + 21 * vertical_offset, 'Color', [0.7, 0.7, 0.7]); % Gray for individual
end

% --- Plot all 20 waveforms and the average stacking at Position 21 ---
average_stack = mean(aligned_waveforms, 2);
hold on;
% Plot in the same figure, at vertical position 21
plot(t, average_stack + 21 * vertical_offset, 'r', 'LineWidth', 1.5);

% Add label
text(t(1), 21 * vertical_offset, 'Stacked Average', 'Color', 'r', 'FontSize', 10);

% --- Formatting ---
title('Vertically Aligned Top 20 Waveforms with Main Waveform');
legend('Main Waveform', 'Top 20 Waveforms', 'Stacked Average');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
%xlim([-0.2 0.4]);
hold off;
