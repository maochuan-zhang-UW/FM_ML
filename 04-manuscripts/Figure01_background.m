% Clear the environment
clear; clc; close all

pdf_file = ['/Users/mczhang/Documents/GitHub/FM3/04-final-paper/Figure01_VV.pdf'];
if exist(pdf_file, 'file') == 2
    delete(pdf_file);
end

% Create a single figure with two subplots
figure;
set(gcf, 'Position', [1306 203 1000 648]); % Adjusted width for better proportion

% --- Subplot (121): First Figure ---
ax1 = subplot(1, 2, 1);
set(ax1, 'Position', [0.05 0.11 0.35 0.8]); % [left bottom width height]

% Load topography data from the NetCDF file
x = ncread('Axial-em300-gmt-25m.grd', 'x'); % X coordinates (longitude)
y = ncread('Axial-em300-gmt-25m.grd', 'y'); % Y coordinates (latitude)
z = ncread('Axial-em300-gmt-25m.grd', 'z'); % Z data (elevation)
[X, Y] = meshgrid(x, y);
Z = reshape(z, [3621, 5187]); % Adjust the dimensions as necessary
Z = Z'; % Transpose to align correctly

% Define the constraints for longitude and latitude
lon_min = -130.1; lon_max = -129.9; % Longitude range
lat_min = 45.85; lat_max = 46.1; % Latitude range

% Filter the data within the specified longitude and latitude ranges
x_idx = x >= lon_min & x <= lon_max; % Index for x within range
y_idx = y >= lat_min & y <= lat_max; % Index for y within range

% Apply the filters to extract the relevant portions of X, Y, Z
X_filtered = X(y_idx, x_idx);
Y_filtered = Y(y_idx, x_idx);
Z_filtered = Z(y_idx, x_idx);

% Plot the topography
contourf(X_filtered, Y_filtered, Z_filtered, 20, 'LineColor', 'none'); % Contour plot with filled contours
load('/Users/mczhang/Documents/GitHub/FM/02-data/Alldata/ColormapZMC.mat');
%colormap(ax1, ColormapZMC); % Set colormap for this axes only
% Make colormap pastel (blend with white)
ColormapZMC_pastel = ColormapZMC*0.7 + 0.3;  
colormap(ax1, ColormapZMC_pastel);

hold on;
xx=[-2.7 3.4];
yy=[-4.5 4.5];

[dlat,dlon] = xy2latlon_no_rotate(xx,yy);

% Plotting the four dots and connecting them to form a closed shape (the box)
% lat = [dlat,dlat];
% lon = [dlon,dlon];
% 
% % Connect the dots to form the box
% plot([lon(1), lon(2)], [lat(1), lat(2)], 'b-', 'LineWidth', 2);
% plot([lon(2), lon(4)], [lat(2), lat(4)], 'b-', 'LineWidth', 2);
% plot([lon(4), lon(3)], [lat(4), lat(3)], 'b-', 'LineWidth', 2);
% plot([lon(3), lon(1)], [lat(3), lat(1)], 'b-', 'LineWidth', 2);
lat = [45.9142 45.9952 45.9952 45.9142];
lon = [-130.0438 -130.0438 -129.9649 -129.9649];

% Now your 4 plot commands work correctly
plot([lon(1), lon(2)], [lat(1), lat(2)], 'b-', 'LineWidth', 2); % left
plot([lon(2), lon(3)], [lat(2), lat(3)], 'b-', 'LineWidth', 2); % top
plot([lon(3), lon(4)], [lat(3), lat(4)], 'b-', 'LineWidth', 2); % right
plot([lon(4), lon(1)], [lat(4), lat(1)], 'b-', 'LineWidth', 2); % bottom
hold on;

alpha = 0.5;
doColor = 1;

% Plotting fissures from 2015
fiss = importdata(['/Users/mczhang/Documents/GitHub/FM/02-data/Alldata/' ...
    'Fissures2015/JdF:Axial_Clague/Axial-2015-fissures-points-geo-v2.txt']);
fiss = [fiss.data];
ind = unique(fiss(:,1));
for i = 1:length(ind)
    ind_p = find(fiss(:,1) == ind(i));
    fiss_p = fiss(ind_p,:);
    fissure_plot = plot(fiss_p(:,2), fiss_p(:,3), 'k-', 'LineWidth', 1);
    hold on;
end

% Plotting fissures from 2011
fiss = importdata(['/Users/mczhang/Documents/GitHub/FM/02-data/Alldata/' ...
    'Fissures2011/JdF:Axial_Clague/Axial-2011-fissures-points-geo-v2.txt']);
fiss = [fiss.data];
ind = unique(fiss(:,1));
for i = 1:length(ind)
    ind_p = find(fiss(:,1) == ind(i));
    fiss_p = fiss(ind_p,:);
    plot(fiss_p(:,2),fiss_p(:,3), 'k-', 'LineWidth', 1);
    hold on;
end

% Plotting fissures from 1998
fiss = load('/Users/mczhang/Documents/GitHub/FM/02-data/Alldata/Axial-1998-Fissures.txt');
ind = unique(fiss(:,1));
for i = 1:length(ind)
    ind_p = find(fiss(:,1) == ind(i));
    fiss_p = fiss(ind_p,:);
    plot(fiss_p(:,2),fiss_p(:,3), 'k-', 'LineWidth', 1);
    hold on;
end

% Plotting caldera rim
axial_calderaRim
plot(calderaRim(:,1), calderaRim(:,2), '-k', 'linewidth', 2);
hold on;

% Plotting lava flows from 2011
axial_lava2011
for i = 1:length(lava)
    if doColor
        lava_flow_2011 = fill(lava(i).xy(:,1), lava(i).xy(:,2), [0 0 0.9], 'edgecolor', 'none', 'facealpha', alpha);
    else
        fill(lava(i).xy(:,1), lava(i).xy(:,2), [.95 .95 .95], 'edgecolor', 'none', 'facealpha', alpha);
    end
    hold on;
end

% Plotting lava flows from 2015
axial_lava2015
for i = 1:length(flow2015)
    if doColor
        lava_flow_2015 = fill(flow2015(i).lon, flow2015(i).lat, [0 0.5 0], 'edgecolor', 'none', 'facealpha', abs(alpha));
    else
        fill(flow2015(i).lon, flow2015(i).lat, [.8 .8 .8], 'edgecolor', 'none', 'facealpha', abs(alpha));
    end
    hold on;
end

% Plotting lava flows from 1998
axial_lava1998
for i = 1:length(lava)
    if doColor
        lava_flow_1998 = fill(lava(i).xy(:,1), lava(i).xy(:,2), [0.5 0 0], 'edgecolor', 'none', 'facealpha', alpha);
    else
        fill(lava(i).xy(:,1), lava(i).xy(:,2), [.95 .95 .95], 'edgecolor', 'none', 'facealpha', alpha);
    end
    hold on;
end

% Plotting stations
scale = 0.5;
station = axial_stationsNewOrder;
station_plot = plot([station.lon], [station.lat], 'sk', 'markerfacecolor', 'k', 'markersize', 8);

% Adding legend
legend([station_plot, fissure_plot, lava_flow_1998, lava_flow_2011, lava_flow_2015], ...
    {'OBS', 'Fissures', 'Lava Flows 1998', 'Lava Flows 2011', 'Lava Flows 2015'}, ...
    'Location', 'southwest');

% Add labels for important regions
text(-130.04, 45.96, 'AXIAL CALDERA', 'FontSize', 10, 'Color', 'w', 'FontWeight', 'bold');
text(-130.01, 46.03, 'NORTH RIFT', 'FontSize', 10, 'Color', 'k', 'Rotation', 70);
text(-129.99, 45.852, 'SOUTH RIFT', 'FontSize', 10, 'Color', 'k', 'Rotation', 75);
text(-130.035, 45.932, 'ASHES', 'FontSize', 8, 'Color', 'w', 'FontWeight', 'bold');
text(-130.033, 45.99, 'CASM', 'FontSize', 8, 'Color', 'w', 'FontWeight', 'bold');
text(-130.012, 45.922, 'INTERNATIONAL', 'FontSize', 8, 'Color', 'w', 'FontWeight', 'bold');
text(-130.006, 45.917, 'DISTRICT', 'FontSize', 8, 'Color', 'w', 'FontWeight', 'bold');
text(-130.09, 46.09, '(a)', 'FontSize', 12, 'Color', 'w', 'FontWeight', 'bold');

% Customize the plot
xlabel('Longitude');
ylabel('Latitude');
grid on;
xlim([-130.10, -129.9]);
ylim([45.85, 46.1]);

% Apply pbaspect to make axis equal
lonLim = [-130.1, -129.9];
latLim = [45.85, 46.1];
pbaspect(ax1, [diff(lonLim)*cosd(mean(latLim)) diff(latLim) 1]);

% Create a colorbar for subplot(121)
colorbar_handle1 = colorbar('Location', 'eastoutside');
colorbar_position1 = get(colorbar_handle1, 'Position');
colorbar_position1(1) = 0.31; % Adjust position for subplot
colorbar_position1(2) = 0.14;
colorbar_position1(4) = 0.2;
set(colorbar_handle1, 'Position', colorbar_position1);
ylabel(colorbar_handle1, 'Depth (m)', 'Rotation', 90, 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'center');

% Store the xlim and ylim of ax1 before creating the inset
xlim1 = get(ax1, 'XLim');
ylim1 = get(ax1, 'YLim');

% Create inset axes for the globe plot in subplot(121)
%inset_ax = axes('Position', [0.14, 0.60, 0.45, 0.45]); % [left bottom width height]
inset_ax = axes('Position', [0.14, 0.60, 0.40, 0.40]); % [left bottom width height]
axes(inset_ax);

% Load coastline data
load coastlines;

% Create a globe map projection for the inset
axesm('globe', 'Frame', 'on', 'Grid', 'on');
setm(gca, 'Origin', [0 0 0]);
setm(gca, 'GLineStyle', '-');

% Make the globe surface white
[lat, lon] = meshgrid(-90:90, -180:180);
white_surface = ones(size(lat));
surfm(lat, lon, white_surface, 'FaceColor', 'w');

hold on;
geoshow(coastlat, coastlon, 'DisplayType', 'polygon', 'FaceColor', [0.1 0.5 0.5]);

% Plot a star at the specified location
star_lon = -128.0;
star_lat = 46.0;
plotm(star_lat, star_lon, 'rp', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

% Customize the appearance of the inset
axis off;
set(gcf, 'Color', 'w');
view(-37.5, 46.8);

% Bring the inset axes to the front to ensure visibility
uistack(inset_ax, 'top');

% --- Subplot (122): Second Figure ---

%ax2 = subplot(1, 2, 2);
%set(ax2, 'Position', [0.5 0.11 0.35 0.8]); % Adjusted for proper size and aspect ratio

ax2 = subplot(1, 2, 2);
set(ax2, 'Position', [0.5 0.11 0.35 0.8]); % Adjusted for proper size and aspect ratio

% Load your data
load('/Users/mczhang/Documents/GitHub/FM3/02-data/A_All/Felix_kmean_morethan5.mat');
Felix([Felix.depth] > 2) = [];
y=[Felix.lat];
x= [Felix.lon];

% Scatter plot with data
scatter(x, y, 2, [Felix.depth], 'filled');
cmap = colormap(ax2, 'summer'); % Set colormap for this axes only
colormap(ax2, flipud(cmap));
axis equal;
% xlim([-2.7 3.4]);
% ylim([-4.5 4.5]);
hold on;

% Plot caldera rim
axial_calderaRim;
%[calderaRim(:, 2), calderaRim(:, 1)] = latlon2xy_no_rotate(calderaRim(:, 2), calderaRim(:, 1));
plot(calderaRim(:, 1), calderaRim(:, 2), 'k', 'LineWidth', 3);

% Plot stations
sta = axial_stationsNewOrder;
sta = sta(1:7);
for i = 1:length(sta)
    %[sta(i).x, sta(i).y] = latlon2xy_no_rotate([sta(i).lat], [sta(i).lon]);
    plot(sta(i).lon, sta(i).lat, 's', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k', 'MarkerSize', 10);
    text(sta(i).lon + 0.00015, sta(i).lat, sta(i).name(1:end));
end

grid on;
box on;

%text(-2.5, 4.1, '(b)', 'FontSize', 12, 'Color', 'k', 'FontWeight', 'bold');

% Add colorbar for subplot(122)
colorbar_handle2 = colorbar('Location', 'eastoutside');
colorbar_position2 = get(colorbar_handle2, 'Position');
colorbar_position2(1) = 0.77;
colorbar_position2(2) = 0.65;
colorbar_position2(4) = 0.2;
set(colorbar_handle2, 'Position', colorbar_position2);
ylabel(colorbar_handle2, 'Depth below seafloor (km)', 'Rotation', 90, 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'center');

set(gca, 'FontSize', 12);
hold on;
clear Felix;
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_wave/A_wave_dB20_cleaned.mat');
y=[Felix.lat];
x= [Felix.lon];
% Scatter plot with data
scatter(x, y, 5, 'red', 'filled');

% lonLim = [-130.1, -129.9];
% latLim = [45.85, 46.1];
latLim = [45.9142 45.9952];
lonLim = [-130.0438 -129.9649];
xlim(lonLim);
ylim(latLim);
pbaspect(ax2, [diff(lonLim)*cosd(mean(latLim)) diff(latLim) 1]);
%ax3 = subplot(2, 2, 3);
%set(ax3, 'Position', [0.5 0.11 0.35 0.8]); % Adjusted for proper size and aspect ratio


% % Save the figure
% exportgraphics(gcf, pdf_file, 'Append', true);
% print('Figure01_Combined.png', '-dpng', '-r300');
% %clear;