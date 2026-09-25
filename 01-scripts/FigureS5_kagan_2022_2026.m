% FigureS5_kagan_2022_2026.m
%
% Distribution of the Kagan angle between the composite focal mechanisms
% obtained from cross-correlation polarities and those obtained from
% AxialPolCap polarities, for the 2022-01 to 2026-03 catalog.
%
% Both mechanism sets come from SKHASH runs that share an event set, a
% composite grouping, the S/P data and the station geometry, and differ only
% in the polarity used, so the Kagan angle measures the effect of the
% polarity method alone.
%
% Reads the CSV written by 01-scripts/compare_skhash_cc_ml.py.  Set the
% environment variable KAGAN_TAG to 'pol' (polarity-only inversion) or 'sp'
% (polarity plus S/P amplitude ratio); default 'pol'.
%
% Run headless from the repository root (no window is opened):
%   /Applications/MATLAB_R2024a.app/bin/matlab -batch "run('01-scripts/FigureS5_kagan_2022_2026.m')"

clear; clc;

scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(scriptDir);

tag = getenv('KAGAN_TAG');
if isempty(tag), tag = 'pol'; end

csvFile = fullfile(repoRoot, '02-data', sprintf('skhash_kagan_%s_2022_2026.csv', tag));
outPng  = fullfile(repoRoot, '03-figs', sprintf('FigureS5_kagan_2022_2026_%s.png', tag));
outPdf  = fullfile(repoRoot, '03-figs', sprintf('FigureS5_kagan_2022_2026_%s.pdf', tag));

if ~isfile(csvFile)
    error('Missing %s -- run compare_skhash_cc_ml.py --tag %s first.', csvFile, tag);
end

T  = readtable(csvFile);
kg = T.kagan_deg;
kg = kg(isfinite(kg));

fprintf('Paired composites: %d\nKagan angle: median %.1f deg, mean %.1f deg\n', ...
        numel(kg), median(kg), mean(kg));
for th = [15 30 45]
    fprintf('  within %2d deg: %5.1f%%\n', th, 100*sum(kg <= th)/numel(kg));
end

% ===================== Figure ============================================
CMAX = 120;                       % Kagan angle is bounded by 120 degrees

figW = 3.6; figH = 2.9;
fig = figure('Visible','off', 'Units','inches', 'Position',[0 0 figW figH], ...
             'PaperUnits','inches', 'PaperPosition',[0 0 figW figH], ...
             'PaperSize',[figW figH], 'Color','w');
set(fig, 'DefaultAxesFontName','Helvetica', 'DefaultTextFontName','Helvetica');

ax1 = axes('Parent',fig,'Position',[0.165 0.175 0.800 0.775]);
hold(ax1,'on'); box(ax1,'on');
edges = 0:5:CMAX;
histogram(ax1, kg, edges, 'FaceColor',[0.25 0.45 0.80], 'EdgeColor','k', ...
          'LineWidth',0.4, 'FaceAlpha',0.9);
md = median(kg);
yl = ylim(ax1);
plot(ax1, [md md], yl, 'r--', 'LineWidth',1.2);
text(ax1, md+4, yl(2)*0.92, sprintf('median %.0f%c', md, char(176)), ...
     'Color','r', 'FontSize',8);
ylim(ax1, yl);
xlim(ax1, [0 CMAX]);
set(ax1,'FontSize',8,'LineWidth',0.7,'Layer','top','XTick',0:20:CMAX);
xlabel(ax1, sprintf('Kagan angle (%c)', char(176)), 'FontSize',8.5);
ylabel(ax1, 'Number of composites', 'FontSize',8.5);
grid(ax1,'on'); set(ax1,'GridAlpha',0.15);

exportgraphics(fig, outPng, 'Resolution', 300);
exportgraphics(fig, outPdf, 'ContentType','vector');
close(fig);
fprintf('Saved:\n  %s\n  %s\n', outPng, outPdf);
