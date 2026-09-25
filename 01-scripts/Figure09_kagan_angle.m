% FigureS5_kagan_histogram.m
%
% Figure 9 for the AxialPolCap manuscript: agreement between the composite
% focal mechanisms obtained from cross-correlation polarities and those
% obtained from AxialPolCap polarities, measured by the Kagan angle.
%
% Distribution of the Kagan angle only; its spatial pattern is shown in the
% bottom row of Figure 8.
%
% Uses the same catalog pairing as Figure 8, so both figures describe the
% same 3,145 clusters.
%
% Run headless from the repository root (no window is opened, the figure is
% never displayed):
%   /Applications/MATLAB_R2024a.app/bin/matlab -batch "run('01-scripts/Figure09_kagan_angle.m')"

clear; clc;

scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(scriptDir);
fm4Root   = '/Users/mcZhang/Documents/GitHub/FM4';

addpath(fullfile(repoRoot, '07-files'));
addpath(fullfile(fm4Root, '01-scripts'));        % kagan.m

mlFile = fullfile(repoRoot, '02-data', 'G_HASH_All_ML_sameClusterasbeforev_confidence.mat');
ccFile = fullfile(fm4Root, '02-data', 'FMs_Composite_Single_SK_HASH_Final.mat');
outPng = fullfile(repoRoot, '03-figs', 'FigureS5_kagan_hist_matlab.png');
outPdf = fullfile(repoRoot, '03-figs', 'FigureS5_kagan_hist_matlab.pdf');

% ===================== Load and pair the two catalogs ====================
M = load(mlFile, 'event1');        ml = M.event1;
C = load(ccFile, 'eventHASH_org'); cc = C.eventHASH_org;

mlIds = arrayfun(@(e) double(e.id), ml);
ccIds = arrayfun(@(e) double(e.id), cc);
[common, iML, iCC] = intersect(mlIds, ccIds);
ml = ml(iML);  cc = cc(iCC);

% ===================== Kagan angle, cluster by cluster ===================
n  = numel(common);
kg = nan(n,1);  lon = nan(n,1);  lat = nan(n,1);
for i = 1:n
    a = double(reshape(ml(i).avmech, 1, []));
    b = double(reshape(cc(i).avmech, 1, []));
    if numel(a) < 3 || numel(b) < 3 || any(~isfinite(a(1:3))) || any(~isfinite(b(1:3)))
        continue
    end
    kg(i)  = kagan(a(1:3), b(1:3));
    lon(i) = double(ml(i).lon);
    lat(i) = double(ml(i).lat);
end
ok = isfinite(kg) & isfinite(lon) & isfinite(lat);
kg = kg(ok); lon = lon(ok); lat = lat(ok);

fprintf('Paired clusters: %d\nKagan angle: median %.1f deg, mean %.1f deg\n', ...
        numel(kg), median(kg), mean(kg));
for th = [15 30 45]
    fprintf('  within %2d deg: %5.1f%%\n', th, 100*sum(kg <= th)/numel(kg));
end

% ===================== Figure ============================================
lonLim = [-130.030 -129.970];
latLim = [  45.920   45.972];
axial_calderaRim;                 % -> calderaRim (:,1)=lon (:,2)=lat
staLon = [-129.9992 -130.0141 -130.0089 -129.9797 -129.9770 -129.9800 -129.9740];
staLat = [  45.9336   45.9338   45.9547   45.9496   45.9396   45.9358   45.9257];

CMAX = 120;                       % colour / histogram upper limit

figW = 3.6; figH = 2.9;
fig = figure('Visible','off', 'Units','inches', 'Position',[0 0 figW figH], ...
             'PaperUnits','inches', 'PaperPosition',[0 0 figW figH], ...
             'PaperSize',[figW figH], 'Color','w');
set(fig, 'DefaultAxesFontName','Helvetica', 'DefaultTextFontName','Helvetica');

% ---- (a) distribution --------------------------------------------------
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
ylabel(ax1, 'Number of clusters', 'FontSize',8.5);
grid(ax1,'on'); set(ax1,'GridAlpha',0.15);

exportgraphics(fig, outPng, 'Resolution', 300);
exportgraphics(fig, outPdf, 'ContentType','vector');
close(fig);
fprintf('Saved:\n  %s\n  %s\n', outPng, outPdf);
