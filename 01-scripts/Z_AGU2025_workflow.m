%% ============================================
% WORKFLOW DIAGRAM — FINAL CLEAN VERSION
% Functions are at the end (MATLAB compliant)
%% ============================================

clear; close all; clc;

figure('Color','w','Position',[50 50 1400 600]);

% Full-figure axes for block drawing
ax = axes('Position',[0 0 1 1]);
axis(ax, 'off');
hold(ax,'on');

%% ===========================
% Draw Blocks
%% ===========================

% Seismic Arrivals
drawRect(ax, [0.05 0.70 0.22 0.07], 'Input', [0.83 0.93 0.83], true);
drawRect(ax, [0.05 0.64 0.22 0.05], '3-channel waveform', [1 1 1]);
%drawRect(ax, [0.05 0.59 0.22 0.05], 'Machine learning', [1 1 1]);

% Sliced waveforms
%drawRect(ax, [0.33 0.59 0.18 0.08], 'Sliced waveforms', [1 1 1]);

% S/P ratios
drawRect(ax, [0.33 0.48 0.18 0.07], 'S/P amplitude ratios', [1 1 1]);

% DL ellipse
drawEllipse(ax, [0.58 0.57 0.18 0.10], 'Trained + Fine-tuned', [1 1 0.85]);

% FMP block
drawRect(ax, [0.82 0.58 0.15 0.08], 'First-Motion Polarity', [1 1 1]);

% Basic Info
%drawRect(ax, [0.05 0.33 0.22 0.07], 'Input', [0.83 0.93 0.83], true);
drawRect(ax, [0.05 0.27 0.22 0.05], 'Earthquake location', [1 1 1]);
drawRect(ax, [0.05 0.22 0.22 0.05], 'Station location', [1 1 1]);
drawRect(ax, [0.05 0.17 0.22 0.05], 'Velocity model', [1 1 1]);

% HASH hexagon
drawHex(ax, [0.62 0.32], 0.07, 'SKHASH', [0.80 0.87 1]);

% Focal Mechanism
drawRect(ax, [0.82 0.40 0.15 0.07], 'Focal mechanism', [0.83 0.93 0.83], true);
drawRect(ax, [0.82 0.35 0.15 0.05], 'Strike', [1 1 1]);
drawRect(ax, [0.82 0.30 0.15 0.05], 'Dip', [1 1 1]);
drawRect(ax, [0.82 0.25 0.15 0.05], 'Rake', [1 1 1]);

%% ===========================
% Stress Inversion (new block)
%% ===========================
drawRect(ax, [0.82 0.15 0.15 0.07], 'Stress inversion', [0.83 0.93 0.83], true);
drawRect(ax, [0.82 0.10 0.15 0.05], 'P axis', [1 1 1]);
drawRect(ax, [0.82 0.05 0.15 0.05], 'T axis', [1 1 1]);
drawRect(ax, [0.82 0.00 0.15 0.05], 'Shape ratio', [1 1 1]);


%% ===========================
% Draw Arrows (annotation coords)
%% ===========================

annotation('arrow',[0.27 0.33],[0.615 0.615]);    % Arrivals → Sliced
annotation('arrow',[0.51 0.58],[0.615 0.615]);    % Sliced → DL
annotation('arrow',[0.76 0.82],[0.615 0.615]);    % DL → FMP

annotation('arrow',[0.42 0.58],[0.48 0.36]);      % S/P → HASH
annotation('arrow',[0.27 0.58],[0.215 0.31]);     % Basic info → HASH
annotation('arrow',[0.88 0.66],[0.57 0.35]);      % FMP → HASH

annotation('arrow',[0.69 0.82],[0.32 0.32]);      % HASH → FM

title('Workflow Diagram (DL Polarity + HASH)', 'FontSize', 16);
print(gcf, 'workflow_editable.pdf', '-dpdf', '-painters');
c
%% ============================================
%            FUNCTION DEFINITIONS
% (must appear at the END of the file)
%% ============================================

function drawRect(ax, pos, txt, fc, bold)
    if nargin < 5, bold=false; end
    rectangle(ax,'Position',pos,'FaceColor',fc,'Curvature',0.1,'LineWidth',1.5);
    text(ax, pos(1)+pos(3)/2, pos(2)+pos(4)/2, txt, ...
         'HorizontalAlignment','center','VerticalAlignment','middle', ...
         'FontSize',14,'FontWeight', ternary(bold,'bold','normal'));
end

function drawEllipse(ax, pos, txt, fc)
    rectangle(ax,'Position',pos,'Curvature',[1 1], ...
              'FaceColor',fc,'LineWidth',1.5);
    text(ax, pos(1)+pos(3)/2, pos(2)+pos(4)/2, txt, ...
         'HorizontalAlignment','center','VerticalAlignment','middle', ...
         'FontSize',15);
end

function drawHex(ax, center, radius, txt, fc)
    theta = linspace(0,2*pi,7);
    x = center(1) + radius*cos(theta);
    y = center(2) + radius*sin(theta);
    patch(ax, x, y, fc, 'EdgeColor','black','LineWidth',1.6);
    text(ax, center(1), center(2), txt, ...
         'HorizontalAlignment','center','VerticalAlignment','middle', ...
         'FontSize',15);
end

function out = ternary(cond, a, b)
    if cond
        out = a;
    else
        out = b;
    end
end

