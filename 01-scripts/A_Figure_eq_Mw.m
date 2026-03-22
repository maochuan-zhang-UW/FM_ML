% Only the second subplot content, simplified
clear; clc; close all;

figure('Position',[1306 203 700 648]);  % single figure
axes('Position',[0.12 0.1 0.76 0.82]);  % fill nicely

% --- Load and filter data ---
load('/Users/mczhang/Documents/GitHub/FM3/02-data/A_All/Felix_kmean_morethan5.mat');
Felix([Felix.depth] > 2) = [];

% Convert to local x-y (km)
[x, y] = latlon2xy_no_rotate([Felix.lat], [Felix.lon]);

% --- Plot events (no colorbar, no tick labels) ---
scatter(x, y, 2, [Felix.depth], 'filled'); hold on;
cmap = colormap('summer'); colormap(flipud(cmap));  % use flipped summer
axis equal; xlim([-2.7 3.4]); ylim([-4.5 4.5]);

% --- Plot caldera rim ---
axial_calderaRim;
[calderaRim(:,2), calderaRim(:,1)] = latlon2xy_no_rotate(calderaRim(:,2), calderaRim(:,1));
plot(calderaRim(:,2), calderaRim(:,1), 'k', 'LineWidth', 2);

% --- Plot stations with names only ---
sta = axial_stationsNewOrder; 
sta = sta(1:7);  % first 7 stations
for i = 1:numel(sta)
    [sx, sy] = latlon2xy_no_rotate(sta(i).lat, sta(i).lon);
    plot(sx, sy, 's', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k', 'MarkerSize', 9);
    text(sx + 0.15, sy, sta(i).name, 'FontWeight','bold');  % name only
end

% --- Clean look: no numbers, no labels, no grid/box ---
%set(gca,'XTick',[],'YTic;k',[],'XColor','none','YColor','none');
grid on; box on;
xlabel('X (km)'); ylabel('Y (km)');
set(gca, 'FontSize', 16);


load('/Users/mczhang/Documents/GitHub/FM3/02-data/F_Cl/F_2015Erp_Final_Mw_Mo.mat')
figure;hist([Po_Clu.Mw],20);
xlabel('Mw'); ylabel('Counts');
set(gca, 'FontSize', 16);