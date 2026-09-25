% Figure08_mechanism_comparison.m
%
% Figure 8 for the AxialPolCap manuscript: composite focal mechanisms from
% cross-correlation polarities (row 1) and from AxialPolCap polarities
% (row 2), split into the three periods of the 2015 eruption cycle.
%
% Only clusters resolved by both methods are drawn, so the two rows show the
% same events and any difference is due to the polarities alone.
%
% Run headless from the repository root (no MATLAB window is opened and the
% figure is never displayed):
%   /Applications/MATLAB_R2024a.app/bin/matlab -batch "run('01-scripts/Figure08_mechanism_comparison.m')"

clear; clc;

scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(scriptDir);
fm4Root   = '/Users/mcZhang/Documents/GitHub/FM4';

addpath(fullfile(repoRoot, '07-files'));
addpath(fullfile(fm4Root, '01-scripts'));        % plot_balloon.m, kagan.m

mlFile  = fullfile(repoRoot, '02-data', 'G_HASH_All_ML_sameClusterasbeforev_confidence.mat');
ccFile  = fullfile(fm4Root, '02-data', 'FMs_Composite_Single_SK_HASH_Final.mat');
outPng  = fullfile(repoRoot, '03-figs', 'Figure08_mechanisms_matlab.png');
outPdf  = fullfile(repoRoot, '03-figs', 'Figure08_mechanisms_matlab.pdf');

% ===================== Load and pair the two catalogs ====================
M = load(mlFile,  'event1');       ml = M.event1;
C = load(ccFile,  'eventHASH_org'); cc = C.eventHASH_org;

mlIds = arrayfun(@(e) double(e.id), ml);
ccIds = arrayfun(@(e) double(e.id), cc);
[common, iML, iCC] = intersect(mlIds, ccIds);
ml = ml(iML);  cc = cc(iCC);
fprintf('CC mechanisms: %d\nML mechanisms: %d\nCommon (plotted): %d (%.1f%% of CC)\n', ...
        numel(ccIds), numel(mlIds), numel(common), 100*numel(common)/numel(ccIds));

% ===================== Kagan angle between the paired mechanisms =========
nPair = numel(common);   %#ok<NASGU>
kg = nan(nPair,1);
for i = 1:nPair
    a = double(reshape(ml(i).avmech, 1, []));
    b = double(reshape(cc(i).avmech, 1, []));
    if numel(a) >= 3 && numel(b) >= 3 && all(isfinite(a(1:3))) && all(isfinite(b(1:3)))
        kg(i) = kagan(a(1:3), b(1:3));
    end
end
fprintf('Kagan angle: median %.1f deg, %.1f%% within 15 deg\n', ...
        median(kg,'omitnan'), 100*sum(kg<=15)/sum(isfinite(kg)));

% ===================== Common, classifiable subset =======================
% Every panel shows the same clusters, so the three columns carry one n each.
% A cluster is kept only when BOTH methods return a mechanism the P/T/B scheme
% classifies as normal, reverse, or strike-slip (Table S2).
keepKey = {'N','R','S'};
okCC = arrayfun(@(e) any(strcmp(strtrim(char(e.faultType)), keepKey)), cc);
okML = arrayfun(@(e) any(strcmp(strtrim(char(e.faultType)), keepKey)), ml);
keep = okCC(:) & okML(:) & isfinite(kg(:));
fprintf('Classifiable by CC: %d, by AxialPolCap: %d, by both: %d\n', ...
        sum(okCC), sum(okML), sum(keep));
cc = cc(keep);  ml = ml(keep);  kg = kg(keep);

% ===================== Periods of the 2015 eruption ======================
ERUPT = datenum(2015,4,24);
ENDCO = datenum(2015,5,25);        % end of the co-eruptive swarm
t = arrayfun(@(e) double(e.time), ml);
periods = { t <  ERUPT, ...
            t >= ERUPT & t < ENDCO, ...
            t >= ENDCO };
perLabel = {'Before', 'During', 'After'};

% ===================== Fault-type colours (shared) =======================
% Matches the manuscript: normal blue, reverse red, strike-slip green,
% oblique/unclassified black.
% Only the three classifiable types are drawn; mechanisms the P/T/B-plunge
% scheme leaves unclassified ('U', see Table S2) are omitted.
typeKey  = {'N','R','S'};
typeCol  = [0.00 0.20 0.90; 0.85 0.10 0.10; 0.00 0.60 0.20];
typeName = {'Normal','Reverse','Strike-slip'};

% ===================== Map frame =========================================
lonLim = [-130.030 -129.970];
latLim = [  45.920   45.972];
axial_calderaRim;                  % -> calderaRim (:,1)=lon (:,2)=lat


% SRL: 7.0 in text width, 300 dpi, legible sans-serif at printed size.
figW = 7.0;  figH = 8.8;
fig = figure('Visible','off', 'Units','inches', 'Position',[0 0 figW figH], ...
             'PaperUnits','inches', 'PaperPosition',[0 0 figW figH], ...
             'PaperSize',[figW figH], 'Color','w');
set(fig, 'DefaultAxesFontName','Helvetica', 'DefaultTextFontName','Helvetica');

rows   = {cc, ml, []};
rowLab = {'CC polarities', 'AxialPolCap polarities', 'Kagan angle'};
panel  = {'(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)'};
KMAX   = 120;                       % colour limit for the Kagan angle

% geometry: 3 columns x 2 rows
x0 = 0.085; wA = 0.272; dx = 0.293; hA = 0.270;
rowY = [0.700 0.410 0.095];   % top -> bottom
radius = 0.00075;                  % beachball radius in degrees of latitude
aspect = 1/cosd(mean(latLim));

k = 0;
for r = 1:3
    ev = rows{r};
    for c = 1:3
        k = k + 1;
        ax = axes('Parent',fig, 'Position',[x0+(c-1)*dx, rowY(r), wA, hA]);
        hold(ax,'on'); box(ax,'on');
        set(ax,'FontSize',7.5,'LineWidth',0.7,'Layer','top');

        sel = find(periods{c});
        nDrawn = 0;
        if r == 3
            % Kagan angle between the two mechanisms of the same cluster
            kk = kg(sel); xx = arrayfun(@(e) double(e.lon), ml(sel));
            yy = arrayfun(@(e) double(e.lat), ml(sel));
            good = isfinite(kk);
            [~, ord] = sort(kk(good));            % worst agreement on top
            xg = xx(good); yg = yy(good); kgv = kk(good);
            scatter(ax, xg(ord), yg(ord), 5, kgv(ord), 'filled');
            colormap(ax, parula(24)); caxis(ax, [0 KMAX]);
            nDrawn = sum(good);
        end
        for ii = 1:numel(sel)*(r<3)
            e = ev(sel(ii));
            if isempty(e.avfnorm) || isempty(e.avslip), continue; end
            ci = find(strcmp(strtrim(char(e.faultType)), typeKey), 1);
            if isempty(ci), continue; end          % cannot occur after filtering
            hh = plot_balloon(e.avfnorm, e.avslip, e.lon, e.lat, radius, aspect, typeCol(ci,:));
            set(hh(1), 'EdgeColor','none');        % white backdrop, no rim
            set(hh(2), 'LineWidth', 0.25);         % thin circle outline
            set(hh([3 4]), 'EdgeColor','none');    % colour reads without a heavy edge
            nDrawn = nDrawn + 1;
        end

        plot(ax, calderaRim(:,1), calderaRim(:,2), 'k-', 'LineWidth', 1.0);

        xlim(ax,lonLim); ylim(ax,latLim); daspect(ax,[1 cosd(mean(latLim)) 1]);
        set(ax,'XTick',[-130.02 -130.00 -129.98],'YTick',45.92:0.02:45.97);
        xtickformat(ax,'%.2f'); ytickformat(ax,'%.2f');

        if r == 1, title(ax, perLabel{c}, 'FontSize',9, 'FontWeight','bold'); end
        if r == 3, xlabel(ax,'Longitude','FontSize',8); else, set(ax,'XTickLabel',[]); end
        if c == 1
            ylabel(ax,{rowLab{r},'Latitude'},'FontSize',8);
        else
            set(ax,'YTickLabel',[]);
        end
        text(ax, 0.035, 0.955, panel{k}, 'Units','normalized', 'FontSize',9, ...
             'FontWeight','bold', 'VerticalAlignment','top');
        text(ax, 0.965, 0.955, sprintf('n = %d', nDrawn), 'Units','normalized', ...
             'FontSize',7, 'HorizontalAlignment','right', 'VerticalAlignment','top');
        fprintf('  %-24s %-7s n = %4d\n', rowLab{r}, perLabel{c}, nDrawn);
    end
end

% ===================== Shared fault-type legend ==========================
lg = axes('Parent',fig,'Position',[0.085 0.368 0.85 0.026]); axis(lg,'off');
hold(lg,'on'); xlim(lg,[0 1]); ylim(lg,[0 1]);
for i = 1:numel(typeName)
    xx = 0.17 + (i-1)*0.24;
    plot(lg, xx, 0.5, 'o', 'MarkerSize',6, 'MarkerFaceColor',typeCol(i,:), ...
         'MarkerEdgeColor','k', 'LineWidth',0.5);
    text(lg, xx+0.022, 0.5, typeName{i}, 'FontSize',8, 'VerticalAlignment','middle');
end

% Shared colour bar for the Kagan-angle row
cbx = axes('Parent',fig,'Position',[0.300 0.040 0.400 0.016]);
imagesc(cbx, linspace(0,KMAX,256), [0 1], repmat(linspace(0,KMAX,256),2,1));
colormap(cbx, parula(24)); caxis(cbx,[0 KMAX]);
set(cbx,'YTick',[],'XTick',0:20:KMAX,'FontSize',7.5,'TickDir','out','Layer','top');
box(cbx,'on');
xlabel(cbx, sprintf('Kagan angle (%c)', char(176)), 'FontSize',8);

exportgraphics(fig, outPng, 'Resolution', 300);
exportgraphics(fig, outPdf, 'ContentType','image', 'Resolution', 300);
close(fig);
fprintf('Saved:\n  %s\n  %s\n', outPng, outPdf);
