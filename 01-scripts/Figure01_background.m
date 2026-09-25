clear; clc; close all;

% Figure 1 for the AxialPolCap manuscript.
%
%   (a) Bathymetry, eruptive features, caldera rim and OBS stations.
%   (b) Catalog earthquakes in x/y km with the k-means regions R1-R7,
%       template earthquakes highlighted in blue.
%   (c) Magnitude distribution of the earthquakes in (b), templates vs rest.
%   (d) SNR distribution of the waveforms, templates vs rest.
%
% Panels (a) and (b) follow the layout of FM4/04-final-paper/Figure01_final.m.
% Panels (c) and (d) read the cache written by Figure01_panelCD_prep.m.
%
% Run headless from the repository root:
%   /Applications/MATLAB_R2024a.app/bin/matlab -batch "run('04-manuscripts/Figure01_background.m')"

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(scriptDir);
fm4Root = '/Users/mcZhang/Documents/GitHub/FM4';
addpath(fullfile(repoRoot, '07-files'));

topoFile = fullfile(repoRoot, '07-files', 'Axial-em300-gmt-25m.grd');
coastFile = fullfile(repoRoot, '07-files', 'coastlines_ne110m.txt');
fiss2015File = fullfile(repoRoot, '07-files', 'Fissures2015', 'JdF:Axial_Clague', 'Axial-2015-fissures-points-geo-v2.txt');
fiss2011File = fullfile(repoRoot, '07-files', 'Fissures2011', 'JdF:Axial_Clague', 'Axial-2011-fissures-points-geo-v2.txt');
fiss1998File = fullfile(repoRoot, '07-files', 'Axial-1998-Fissures.txt');
cmapFile = fullfile(fm4Root, 'ColormapZMC.mat');
panelBFile = fullfile(fm4Root, '02-data', 'A_All', 'Felix_kmean_morethan5.mat');
templateFile = fullfile(repoRoot, '02-data', 'A_wave_dB20_cleaned.mat');
magFile = fullfile(repoRoot, '07-files', 'evMw.txt');
panelCDFile = fullfile(repoRoot, '02-data', 'fig01_panelCD_data.mat');

outPng = fullfile(repoRoot, '03-figs', 'Figure01_background_matlab.png');
outPdf = fullfile(repoRoot, '03-figs', 'Figure01_background_matlab.pdf');

mustExist(topoFile, 'Topography grid');
mustExist(panelBFile, 'Panel (b) catalog');
mustExist(panelCDFile, 'Panel (c)/(d) cache (run Figure01_panelCD_prep.m first)');

% Population colours, shared by panels (b), (c) and (d).
templateColor = [0.00 0.20 0.90];
normalColor = [0.55 0.55 0.55];

% Drawn at the final printed width so Word does not scale it down (review
% comment 18: "Font is small and distorted. Make larger size and use a
% San-serif font").  Fonts below are therefore true printed point sizes.
FIG_W_IN = 7.2;
fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 FIG_W_IN FIG_W_IN]);
set(fig, 'DefaultAxesFontName', 'Arial', 'DefaultTextFontName', 'Arial', ...
         'DefaultLegendFontName', 'Arial', 'DefaultColorbarFontName', 'Arial');
set(fig, 'PaperUnits', 'inches', 'PaperPosition', [0 0 FIG_W_IN FIG_W_IN], ...
         'PaperSize', [FIG_W_IN FIG_W_IN]);

% ===================== Panel (a): bathymetry map =========================
ax1 = axes('Parent', fig, 'Position', [0.085 0.40 0.32 0.55]);
set(ax1, 'FontSize', 9, 'LineWidth', 0.8);

x = ncread(topoFile, 'x');
y = ncread(topoFile, 'y');
z = ncread(topoFile, 'z');
if isequal(size(z), [numel(y), numel(x)])
    Zfull = double(z);
elseif isequal(size(z), [numel(x), numel(y)])
    Zfull = double(z');
else
    error('Unexpected bathymetry grid size for z: [%d %d]', size(z, 1), size(z, 2));
end

lonLim1 = [-130.10, -129.90];
latLim1 = [45.85, 46.10];
xMask = x >= lonLim1(1) & x <= lonLim1(2);
yMask = y >= latLim1(1) & y <= latLim1(2);
[X, Y] = meshgrid(double(x(xMask)), double(y(yMask)));
Z = Zfull(yMask, xMask);
Z = fillmissing(Z, 'nearest', 1);
Z = fillmissing(Z, 'nearest', 2);
Z = Z / 1000;   % metres -> km, to match the depth units in panel (b)

contourf(ax1, X, Y, Z, 20, 'LineColor', 'none');
if exist(cmapFile, 'file') == 2
    cm = load(cmapFile);
    colormap(ax1, cm.ColormapZMC * 0.7 + 0.3);   % pastel blend, as in FM4
else
    colormap(ax1, parula(256));
end
hold(ax1, 'on');

% Blue box marking the area shown in panel (b)
boxLat = [45.9142 45.9952 45.9952 45.9142 45.9142];
boxLon = [-130.0438 -130.0438 -129.9649 -129.9649 -130.0438];
plot(ax1, boxLon, boxLat, 'b-', 'LineWidth', 2);

fissureHandle = [];
fissureHandle = plotGroupedTxt(ax1, fiss2015File, 'ORIG_FID', 'LONGITUDE', 'LATITUDE', 'k-', 1, fissureHandle);
fissureHandle = plotGroupedTxt(ax1, fiss2011File, 'ORIG_FID', 'LONGITUDE', 'LATITUDE', 'k-', 1, fissureHandle);
fissureHandle = plotGroupedNumeric(ax1, fiss1998File, 'k-', 1, fissureHandle);

axial_calderaRim;
calderaRimLL = calderaRim;
plot(ax1, calderaRimLL(:, 1), calderaRimLL(:, 2), 'k-', 'LineWidth', 2);

alphaVal = 0.5;
lava1998Handle = [];
lava2011Handle = [];
lava2015Handle = [];

if exist('axial_lava1998.m', 'file') == 2
    axial_lava1998;
    lava1998Handle = fill(ax1, lava(1).xy(:, 1), lava(1).xy(:, 2), [0.5 0 0], ...
        'EdgeColor', 'none', 'FaceAlpha', alphaVal);
    for i = 2:numel(lava)
        fill(ax1, lava(i).xy(:, 1), lava(i).xy(:, 2), [0.5 0 0], ...
            'EdgeColor', 'none', 'FaceAlpha', alphaVal);
    end
end

axial_lava2011;
lava2011Handle = fill(ax1, lava(1).xy(:, 1), lava(1).xy(:, 2), [0 0 0.9], ...
    'EdgeColor', 'none', 'FaceAlpha', alphaVal);
for i = 2:numel(lava)
    fill(ax1, lava(i).xy(:, 1), lava(i).xy(:, 2), [0 0 0.9], ...
        'EdgeColor', 'none', 'FaceAlpha', alphaVal);
end

axial_lava2015;
lava2015Handle = fill(ax1, flow2015(1).lon, flow2015(1).lat, [0 0.5 0], ...
    'EdgeColor', 'none', 'FaceAlpha', alphaVal);
for i = 2:numel(flow2015)
    fill(ax1, flow2015(i).lon, flow2015(i).lat, [0 0.5 0], ...
        'EdgeColor', 'none', 'FaceAlpha', alphaVal);
end

stations = axialStationTable();
stationHandle = plot(ax1, [stations.lon], [stations.lat], 'sk', ...
    'MarkerFaceColor', 'k', 'MarkerSize', 8);

legendHandles = stationHandle;
legendLabels = {'OBS'};
if ~isempty(fissureHandle)
    legendHandles(end + 1) = fissureHandle;
    legendLabels{end + 1} = 'Fissures';
end
if ~isempty(lava1998Handle)
    legendHandles(end + 1) = lava1998Handle;
    legendLabels{end + 1} = 'Lava Flows 1998';
end
if ~isempty(lava2011Handle)
    legendHandles(end + 1) = lava2011Handle;
    legendLabels{end + 1} = 'Lava Flows 2011';
end
if ~isempty(lava2015Handle)
    legendHandles(end + 1) = lava2015Handle;
    legendLabels{end + 1} = 'Lava Flows 2015';
end
lg1 = legend(ax1, legendHandles, legendLabels, 'Location', 'southwest', 'FontSize', 6);
set(lg1, 'ItemTokenSize', [10 8]);

text(ax1, -130.037, 45.965, 'AXIAL CALDERA', 'FontSize', 8, 'Color', 'w', 'FontWeight', 'bold');
text(ax1, -130.038, 46.012, 'NORTH RIFT', 'FontSize', 8, 'Color', 'k', 'FontWeight', 'bold', 'Rotation', 75);
text(ax1, -129.985, 45.858, 'SOUTH RIFT', 'FontSize', 8, 'Color', 'k', 'FontWeight', 'bold', 'Rotation', 75);
text(ax1, -130.035, 45.932, 'ASHES', 'FontSize', 7, 'Color', 'w', 'FontWeight', 'bold');
text(ax1, -130.033, 45.990, 'CASM', 'FontSize', 7, 'Color', 'w', 'FontWeight', 'bold');
text(ax1, -130.012, 45.922, 'INTERNATIONAL', 'FontSize', 7, 'Color', 'w', 'FontWeight', 'bold');
text(ax1, -130.006, 45.917, 'DISTRICT', 'FontSize', 7, 'Color', 'w', 'FontWeight', 'bold');
text(ax1, -130.09, 46.09, '(a)', 'FontSize', 10, 'Color', 'w', 'FontWeight', 'bold');

xlabel(ax1, 'Longitude');
ylabel(ax1, 'Latitude');
xlim(ax1, lonLim1);
ylim(ax1, latLim1);
grid(ax1, 'on');
pbaspect(ax1, [diff(lonLim1) * cosd(mean(latLim1)), diff(latLim1), 1]);

cb1 = colorbar(ax1, 'eastoutside');
set(cb1, 'Position', [0.300 0.435 0.013 0.085]);
ylabel(cb1, 'Depth (km)');

% ===================== Panel (b): catalog in x/y km ======================
ax2 = axes('Parent', fig, 'Position', [0.50 0.40 0.35 0.55]);
set(ax2, 'FontSize', 9, 'LineWidth', 0.8);
hold(ax2, 'on');

B = load(panelBFile, 'Felix');
FelixB = B.Felix;
FelixB([FelixB.depth] > 2) = [];
bLat = [FelixB.lat];
bLon = [FelixB.lon];
bDep = [FelixB.depth];
bID = double([FelixB.ID]);

T = load(templateFile, 'Felix');
templateID = double([T.Felix.ID]);
isTemplateB = ismember(bID, templateID);

[bx, by] = latlon2xy_axial(bLat, bLon);
scatter(ax2, bx, by, 2, bDep, 'filled');
cmap2 = summer(256);
colormap(ax2, flipud(cmap2));

% Template earthquakes highlighted. Marker area is matched to the catalog
% dots (scatter sizes are areas in points^2) so the templates read as a
% subset rather than as a separate, heavier layer.
templateHandle = scatter(ax2, bx(isTemplateB), by(isTemplateB), 4, ...
    templateColor, 'filled');
axis(ax2, 'equal');
xlim(ax2, [-2.7, 3.4]);
ylim(ax2, [-4.5, 4.5]);

[rimX, rimY] = latlon2xy_axial(calderaRimLL(:, 2), calderaRimLL(:, 1));
plot(ax2, rimX, rimY, 'k-', 'LineWidth', 3);

for i = 1:numel(stations)
    [sx, sy] = latlon2xy_axial(stations(i).lat, stations(i).lon);
    plot(ax2, sx, sy, 's', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k', 'MarkerSize', 9);
    text(ax2, sx + 0.15, sy, ['AX' stations(i).name], 'FontSize', 7);
end

text(ax2, -2.5, 4.1, '(b)', 'FontSize', 10, 'Color', 'k', 'FontWeight', 'bold');
xlabel(ax2, 'x (km)');
ylabel(ax2, 'y (km)');
grid(ax2, 'on');
box(ax2, 'on');

cb2 = colorbar(ax2, 'eastoutside');
set(cb2, 'Position', [0.762 0.80 0.018 0.13]);
ylabel(cb2, 'Depth (km)');

% ===================== Panel (c): magnitude ==============================
D = load(panelCDFile);

mw = load(magFile);
[hasMw, loc] = ismember(bID, mw(:, 1));
bMw = nan(size(bID));
bMw(hasMw) = mw(loc(hasMw), 2);

mwT = bMw(isTemplateB);
mwN = bMw(~isTemplateB);
mwT = mwT(isfinite(mwT))';
mwN = mwN(isfinite(mwN))';

ax3 = axes('Parent', fig, 'Position', [0.08 0.07 0.33 0.24]);
set(ax3, 'FontSize', 9, 'LineWidth', 0.8);
hold(ax3, 'on');

mwEdges = floor(min([mwN; mwT]) * 10) / 10 : 0.1 : ceil(max([mwN; mwT]) * 10) / 10;
histogram(ax3, mwN, mwEdges, ...
    'FaceColor', normalColor, 'EdgeColor', 'none', 'FaceAlpha', 0.75);
histogram(ax3, mwT, mwEdges, ...
    'FaceColor', templateColor, 'EdgeColor', 'none', 'FaceAlpha', 0.75);

xlabel(ax3, 'Magnitude (M_w)');
ylabel(ax3, 'Earthquake count');
xlim(ax3, [prctile([mwN; mwT], 0.2), prctile([mwN; mwT], 99.9)]);
grid(ax3, 'on');
box(ax3, 'on');
legend(ax3, {sprintf('Non-template (n = %d)', numel(mwN)), ...
             sprintf('Templates (n = %d)', numel(mwT))}, ...
    'Location', 'northeast', 'FontSize', 7.5);
addPanelLabel(ax3, '(c)');

% ===================== Panel (d): SNR ====================================
% Same event population as panels (b) and (c): waveforms are kept only for
% events that appear in the panel (b) catalog after the depth cut.
inPanelB = ismember(D.snrEventID, bID);
wIsTemplate = ismember(D.snrEventID, bID(isTemplateB));
snrT = D.snrDb(inPanelB & wIsTemplate);
snrN = D.snrDb(inPanelB & ~wIsTemplate);
snrN = snrN(isfinite(snrN));
snrT = snrT(isfinite(snrT));

ax4 = axes('Parent', fig, 'Position', [0.54 0.07 0.33 0.24]);
set(ax4, 'FontSize', 9, 'LineWidth', 0.8);
hold(ax4, 'on');

snrEdges = -10:2:80;
hSnrN = histogram(ax4, snrN, snrEdges, ...
    'FaceColor', normalColor, 'EdgeColor', 'none', 'FaceAlpha', 0.75);
hSnrT = histogram(ax4, snrT, snrEdges, ...
    'FaceColor', templateColor, 'EdgeColor', 'none', 'FaceAlpha', 0.75);

xlabel(ax4, 'SNR (dB)');
ylabel(ax4, 'Waveform count');
xlim(ax4, [snrEdges(1), snrEdges(end)]);

% Headroom above the tallest bar so the legend sits clear of the histogram
% instead of covering the peak.
snrPeak = max([hSnrN.Values, hSnrT.Values]);
ylim(ax4, [0, snrPeak * 1.42]);

grid(ax4, 'on');
box(ax4, 'on');

% SNR = 10 dB reference line: the boundary between the low- and high-SNR
% training categories used in Section 3.2.
snrRefLine = xline(ax4, 10, '--', 'SNR = 10 dB', ...
    'Color', [0.15 0.15 0.15], 'LineWidth', 1.1, 'Alpha', 1, ...
    'FontSize', 7.5, 'LabelOrientation', 'aligned', ...
    'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'left');
snrRefLine.Annotation.LegendInformation.IconDisplayStyle = 'off';

legend(ax4, [hSnrN, hSnrT], ...
    {sprintf('Non-template (n = %d)', numel(snrN)), ...
     sprintf('Templates (n = %d)', numel(snrT))}, ...
    'Location', 'northeast', 'FontSize', 7.5, 'Box', 'on', 'Color', 'w');
addPanelLabel(ax4, '(d)');

% ===================== Inset locator globe ===============================
drawInsetGlobe(fig, coastFile);

exportgraphics(fig, outPng, 'Resolution', 300);
exportgraphics(fig, outPdf, 'ContentType', 'image', 'Resolution', 300);

fprintf('Panel (b): %d earthquakes, %d templates highlighted\n', numel(bID), sum(isTemplateB));
fprintf('Panel (c): Mw median %.2f (templates) vs %.2f (rest)\n', median(mwT), median(mwN));
fprintf('Panel (d): SNR median %.1f dB (templates) vs %.1f dB (rest)\n', median(snrT), median(snrN));
fprintf('Saved:\n  %s\n  %s\n', outPng, outPdf);

% ========================= helper functions ==============================

function addPanelLabel(ax, labelText)
text(ax, 0.02, 0.96, labelText, 'Units', 'normalized', ...
    'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
end

function mustExist(pathStr, label)
if exist(pathStr, 'file') ~= 2
    error('%s not found: %s', label, pathStr);
end
end

function firstHandle = plotGroupedTxt(ax, pathStr, idName, xName, yName, style, width, firstHandle)
if exist(pathStr, 'file') ~= 2
    warning('Figure01:MissingFile', 'Skipping missing file: %s', pathStr);
    return;
end

opts = detectImportOptions(pathStr, 'FileType', 'text');
opts = setvartype(opts, {idName, xName, yName}, 'double');
T = readtable(pathStr, opts);
ids = unique(T.(idName));
for i = 1:numel(ids)
    rows = T.(idName) == ids(i);
    h = plot(ax, T.(xName)(rows), T.(yName)(rows), style, 'LineWidth', width);
    if isempty(firstHandle)
        firstHandle = h;
    end
end
end

function firstHandle = plotGroupedNumeric(ax, pathStr, style, width, firstHandle)
if exist(pathStr, 'file') ~= 2
    warning('Figure01:MissingFile', 'Skipping missing file: %s', pathStr);
    return;
end

fid = fopen(pathStr, 'r');
if fid < 0
    warning('Figure01:MissingFile', 'Unable to open file: %s', pathStr);
    return;
end

C = textscan(fid, '%f %f %f', 'Delimiter', {' ', ','}, ...
    'MultipleDelimsAsOne', true, 'CollectOutput', true);
fclose(fid);

if isempty(C) || isempty(C{1}) || size(C{1}, 2) < 3
    warning('Figure01:BadFormat', 'Unexpected fissure file format: %s', pathStr);
    return;
end

A = C{1};
ids = unique(A(:, 1));
for i = 1:numel(ids)
    rows = A(:, 1) == ids(i);
    h = plot(ax, A(rows, 2), A(rows, 3), style, 'LineWidth', width);
    if isempty(firstHandle)
        firstHandle = h;
    end
end
end

function stations = axialStationTable()
stations = struct( ...
    'name', {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'}, ...
    'lon',  {-129.9992, -130.0141, -130.0089, -129.9797, -129.9738, -129.9785, -129.9780}, ...
    'lat',  {45.9336, 45.9338, 45.9547, 45.9496, 45.9397, 45.9361, 45.9257});
end

function [x, y] = latlon2xy_axial(lat, lon)
% Origin at AXCC1, no rotation (matches FM4/01-scripts/latlon2xy_no_rotate.m).
originLat = 45.9547;
originLon = -130.0089;
xltkm = 111.19;
xlnkm = xltkm * cosd(originLat);
y = (double(lat) - originLat) * xltkm;
x = (double(lon) - originLon) * xlnkm;
end

function drawInsetGlobe(fig, coastFile)
% Orthographic locator globe. The Mapping Toolbox is licensed but not
% installed on this machine, so axesm/geoshow are unavailable and the globe
% is drawn directly from Natural Earth 110 m coastlines exported to text.
pos = [0.250 0.800 0.145 0.145];
ax = axes('Parent', fig, 'Position', pos);
hold(ax, 'on');

lat0 = 35;
lon0 = -145;

theta = linspace(0, 2 * pi, 400);
fill(ax, cos(theta), sin(theta), [1 1 1], ...
    'EdgeColor', [0.25 0.25 0.25], 'LineWidth', 1.0);
drawOrthoGrid(ax, lat0, lon0);

if exist(coastFile, 'file') == 2
    A = load(coastFile);
    [xc, yc] = orthoProject(A(:, 1), A(:, 2), lat0, lon0);
    plotProjectedSegments(ax, xc, yc, [0 0 0], 0.8);
else
    warning('Figure01:MissingCoast', 'Coastline file not found: %s', coastFile);
end

[sx, sy] = orthoProject(46.0, -130.0, lat0, lon0);
plot(ax, sx, sy, 'rp', 'MarkerSize', 11, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');

axis(ax, 'equal');
xlim(ax, [-1.05, 1.05]);
ylim(ax, [-1.05, 1.05]);
axis(ax, 'off');
uistack(ax, 'top');
end

function [x, y] = orthoProject(lat, lon, lat0, lon0)
lat = deg2rad(double(lat));
lon = deg2rad(double(lon));
lat0 = deg2rad(lat0);
lon0 = deg2rad(lon0);

cosc = sin(lat0) .* sin(lat) + cos(lat0) .* cos(lat) .* cos(lon - lon0);
x = cos(lat) .* sin(lon - lon0);
y = cos(lat0) .* sin(lat) - sin(lat0) .* cos(lat) .* cos(lon - lon0);

x(cosc < 0) = NaN;
y(cosc < 0) = NaN;
end

function plotProjectedSegments(ax, x, y, colorVal, width)
mask = isfinite(x) & isfinite(y);
idx = find(mask);
if isempty(idx)
    return;
end

breaks = [0; find(diff(idx) > 1); numel(idx)];
for k = 1:numel(breaks) - 1
    seg = idx(breaks(k) + 1:breaks(k + 1));
    plot(ax, x(seg), y(seg), 'Color', colorVal, 'LineWidth', width);
end
end

function drawOrthoGrid(ax, lat0, lon0)
for lon = -180:30:180
    latVals = -90:2:90;
    lonVals = lon * ones(size(latVals));
    [xg, yg] = orthoProject(latVals, lonVals, lat0, lon0);
    plotProjectedSegments(ax, xg(:), yg(:), [0.75 0.75 0.75], 0.4);
end

for lat = -60:30:60
    lonVals = -180:2:180;
    latVals = lat * ones(size(lonVals));
    [xg, yg] = orthoProject(latVals, lonVals, lat0, lon0);
    plotProjectedSegments(ax, xg(:), yg(:), [0.75 0.75 0.75], 0.4);
end
end
