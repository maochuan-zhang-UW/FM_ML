%% =========================================
% ONE FIGURE — LEFT (1D) vs RIGHT (ML/3D)
%% =========================================
clear; close all;

lonLim = [-130.031 -129.97];
latLim = [45.92 45.972];

regions = struct(...
    'West', struct('Lat', [45.930, 45.953], 'Lon', [-130.029, -130.008]), ...
    'East', struct('Lat', [45.971, 45.930], 'Lon', [-130.0015, -129.975]), ...
    'ID', struct('Lat', [45.921, 45.929], 'Lon', [-130.004, -129.975]));

%% ================= LOAD BOTH CATALOGS =================
load('/Users/mczhang/Documents/GitHub/FM4/02-data/Before22OBSs/G_FM/G_HASH_All.mat');
event1D = event1;

load('/Users/mczhang/Documents/GitHub/FM4/02-data/Before22OBSs/G_FM/G_HASH_All_ML_sameClusterasbefore.mat');
eventML = event1;

%% ================= COMMON FILTER FUNCTION =================
filterEvents = @(ev) ev( ...
    ismember({ev.mechqual}, {'A','B'}) & ...
    [ev.faultType] ~= 'U' & ...
    [ev.lat] <= 45.97 & [ev.lon] >= -130.03);

event1D = filterEvents(event1D);
eventML = filterEvents(eventML);

%% ================= FIGURE =================
figure('Position',[400 100 1559 782],'Color','w');

datasets = {event1D, eventML};
titles   = {'CC FMs','ML FMs'};

for p = 1:2
    ax = subplot(1,2,p);
    hold(ax,'on');

    %% ===== BASEMAP =====
    basemap_2015v2(lonLim, latLim, 100, [0 0], 1, false, ax);
    pbaspect(ax, [diff(lonLim)*cosd(mean(latLim)) diff(latLim) 1]);

    set(ax,'XTick', -130.03:0.01:-129.97);
    set(ax,'GridLineStyle','-','LineWidth',0.5,'GridColor',[0.5 0.5 0.5]);
    grid(ax,'on');

    %% ===== CALDERA RIM =====
    axial_calderaRim;
    plot(calderaRim(:,1), calderaRim(:,2), '-k','LineWidth',3,'HandleVisibility','off');

    %% ===== EVENTS =====
    events = datasets{p};
    nmec = length(events);

    radius = 0.0005;
    scale_event = 1.3;

    for i = 1:nmec
        if ~isempty(events(i).avfnorm)
            plot_balloon(events(i).avfnorm, events(i).avslip, ...
                events(i).lon, events(i).lat, ...
                radius, scale_event, events(i).color2);
        end
    end

    %% ===== REGIONS =====
    regionNames = fieldnames(regions);
    for i = 1:length(regionNames)
        region = regions.(regionNames{i});
        latitudes = [region.Lat(1), region.Lat(1), region.Lat(2), region.Lat(2), region.Lat(1)];
        longitudes = [region.Lon(1), region.Lon(2), region.Lon(2), region.Lon(1), region.Lon(1)];
        plot(longitudes, latitudes, '--','LineWidth',2);
    end

    %% ===== LABELS =====
    xlabel(ax,'Longitude','FontSize',14);
    ylabel(ax,'Latitude','FontSize',14);
    title(ax, sprintf('%s  (%d FMs)', titles{p}, nmec),'FontSize',16);

    axis(ax,[lonLim latLim]);
    set(ax,'FontSize',16);
end

%% ================= LEGEND (FAULT TYPES) =================
% Draw once on right panel
ax = subplot(1,2,2);
legendX = lonLim(1) + 0.008;
legendYs = linspace(latLim(2)-0.002, latLim(2)-0.01, 3);

faultTypes = {'N','R','S'};
faultLabels = {'Normal','Reverse','Strike-slip'};

for k = 1:3
    idx = find(strcmp({eventML.faultType}, faultTypes{k}),1);
    if ~isempty(idx)
        repEvent = eventML(idx);
        plot_balloon(repEvent.avfnorm, repEvent.avslip, ...
            legendX, legendYs(k), 0.001, 1.3, repEvent.color2);
    end
    text(legendX+0.003, legendYs(k), faultLabels{k}, ...
        'FontSize',16,'HorizontalAlignment','left');
end

rectangle(ax,'Position',[legendX-0.0025, min(legendYs)-0.002, 0.018, range(legendYs)+0.0035], ...
    'EdgeColor','k','LineWidth',0.5);