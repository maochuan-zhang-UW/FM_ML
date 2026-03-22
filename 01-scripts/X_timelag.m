%% Reproduce ONLY panel (a): waveform comparison
clear; close all;

% Load saved data for panel (a)
load('plot_data_cc_fig02.mat');

figure;
set(gcf, 'position', [500, 400, 600, 450]);

% ---- Subplot (a) only ----
ax_a = axes;  % single axes instead of subplot grid
set(ax_a, 'Color', 'w'); 
h1 = plot(x, normalize(waveform1), 'k-', 'LineWidth', 4); hold on;
h2 = plot(x - shift2, normalize(waveform2), '-.b', 'LineWidth', 2); hold on;
h3 = plot(x - shift3, normalize(waveform3), '-.r', 'LineWidth', 2); hold on;

% Reference vertical lines
plot([0, 0], [1, -1], 'k-', 'LineWidth', 4); hold on;
plot([0.02, 0.02], [1, -1], 'r-.', 'LineWidth', 2); hold on;
plot([0.01, 0.01], [1, -1], 'b-.', 'LineWidth', 2); hold on;

% Horizontal annotation line
plot([0.0, 0.02], [-0.8, -0.8], 'k', 'LineWidth', 2);

legend([h1, h2, h3], ...
    {'Eq 1 (+ ve polarity)', 'Eq 2 (- ve polarity)', 'Eq 3 (+ ve polarity)'}, ...
    'Location', 'best', 'FontSize', 12);

xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
xlim([-0.2, 0.2]);

%text(-0.195, 3.5, '(a)', 'FontSize', 12, 'FontWeight', 'bold');
text(-0.03, -1.5, 'Pick Time Shift', 'FontSize', 12);

grid on;
set(gca, 'FontSize', 12);
