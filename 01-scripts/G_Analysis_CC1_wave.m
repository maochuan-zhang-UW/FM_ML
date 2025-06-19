clc;clear;close all;
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/F_ML/F_CC1_MLPo_DiTing128.mat');
% Time parameters
t_start = -0.25;  % seconds
t_end = 1.0;      % seconds
fs = 200;         % Hz
n_points = 250;
t = linspace(t_start, t_end, n_points);

% Index range for plotting -0.25 to 0.25 s
t_plot_end = 0.25;
plot_indices = t >= t_start & t <= t_plot_end;

% Station and fields
station = 'CC1';
po_field = ['Po_' station];       % CC polarity
poCC_field = [station '_Po'];     % Manual polarity
wave_field = ['W_' station];      % Wave data

% Find events where CC and Manual polarities differ
diff_polarity_events = [];
for i = 1:length(Felix)
    if isfield(Felix(i), po_field) && isfield(Felix(i), poCC_field)
        val_CC = sign(Felix(i).(po_field));
        val_Man = sign(Felix(i).(poCC_field));
        if val_CC ~= 0 && val_Man ~= 0 && val_CC ~= val_Man
            diff_polarity_events = [diff_polarity_events, i];
        end
    end
end

% Plot waves for events with different polarities
figure('Position', [100, 100, 310, 600]);
hold on;
for idx = 1:min(10, length(diff_polarity_events))
    i = diff_polarity_events(idx);
    wave = Felix(i).(wave_field);
    if length(wave) == n_points
        % Get polarities
        val_CC = sign(Felix(i).(po_field));
        val_Man = sign(Felix(i).(poCC_field));
        
        % Normalize wave for better visualization
        if max(abs(wave)) ~= 0
            wave = wave / max(abs(wave));
        end
        
        % Plot with black color and line width 2
        plot(t(plot_indices), wave(plot_indices) + idx * 2, 'k-', 'LineWidth', 2);
        hold on;
        plot([0,0], [wave(plot_indices) + idx * 2-0.5,wave(plot_indices) + idx * 2+0.5], 'r', 'LineWidth', 2);
        % Label polarities
        polarity_text = sprintf('CC: %s, Man: %s', ...
            ternary(val_CC > 0, 'Up', 'Down'), ...
            ternary(val_Man > 0, 'Up', 'Down'));
        text(t_start + 0.02, idx * 2+0.2, polarity_text, 'VerticalAlignment', 'middle');
    end
end
xlim([-0.25 0.25]);
ylim([0 idx * 2+1]);
% Plot formatting
xlabel('Time (s)');
ylabel('Normalized Amplitude (Offset by Event)');
title(sprintf('Waveforms with Different CC and Manual Polarities for %s', station));
grid on;
%legend('show');
hold off;

% Save the plot
print('-dpng', 'waveforms_different_polarity.png');

% Helper function for ternary

 operation
function y = ternary(cond, a, b)
    if cond
        y = a;
    else
        y = b;
    end
end