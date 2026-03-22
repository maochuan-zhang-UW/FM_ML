clc; close all;

%% =========================
% Load BOTH catalogs
%% =========================
load('/Users/mczhang/Documents/GitHub/FM3/02-data/G_FM/G_2015Erp_polished.mat');
eventA = event1;   % Mechanisms from HERE

load('/Users/mczhang/Documents/GitHub/FM3/02-data/G_FM/G_HASH_All_ML_sameClusterasbeforev_confidence.mat');
eventB = event1;   % Locations from HERE

%% =========================
% Keep ONLY common events (by ID)
%% =========================
idA = [eventA.id];
idB = [eventB.id];

commonID = intersect(idA, idB);

eventA = eventA(ismember(idA, commonID));
eventB = eventB(ismember(idB, commonID));

fprintf('Common events BEFORE filter = %d\n', length(eventA));

%% =========================
% FORCE eventA to use eventB LOCATION/TIME
%% =========================
[~, ia, ib] = intersect([eventA.id], [eventB.id]);

for k = 1:length(ia)
    eventA(ia(k)).lat   = eventB(ib(k)).lat;
    eventA(ia(k)).lon   = eventB(ib(k)).lon;
    eventA(ia(k)).depth = eventB(ib(k)).depth;
    eventA(ia(k)).time  = eventB(ib(k)).time;
end

%% =========================
% Basic filters
%% =========================
filterFcn = @(ev) ev([ev.lat] <= 45.969 & ...
                     [ev.lon] >= -130.03 & ...
                     ~ismember([ev.mechqual], ['C','D']));

eventA = filterFcn(eventA);
eventB = filterFcn(eventB);

%% =========================
% Re-match AGAIN after filtering (IMPORTANT)
%% =========================
idA = [eventA.id];
idB = [eventB.id];

commonID = intersect(idA, idB);

eventA = eventA(ismember(idA, commonID));
eventB = eventB(ismember(idB, commonID));

fprintf('Common events AFTER filter = %d\n', length(eventA));

%% =========================
% Color logic
%% =========================
for i = 1:length(eventA)
    if eventA(i).mechqual=='A' || eventA(i).mechqual=='B'
        eventA(i).color3 = eventA(i).color2;
    else
        eventA(i).color3 = eventA(i).color2 + 0.5*(1-eventA(i).color2);
    end
end

for i = 1:length(eventB)
    if eventB(i).mechqual=='A' || eventB(i).mechqual=='B'
        eventB(i).color3 = eventB(i).color2;
    else
        eventB(i).color3 = eventB(i).color2 + 0.5*(1-eventB(i).color2);
    end
end

%% =========================
% Figure setup
%% =========================
projects = {'Before','During','After'};
date_BF = datenum(2015,4,24,8,0,0);
date_DR = datenum(2015,5,19);

lonLim = [-130.031 -129.97];
latLim = [45.92 45.970];

figure('Position',[744,309, 1133 , 942],'Color','white');

positions = [
    0.07 0.56 0.27 0.35;
    0.37 0.56 0.27 0.35;
    0.67 0.56 0.27 0.35;
    0.07 0.12 0.27 0.35;
    0.37 0.12 0.27 0.35;
    0.67 0.12 0.27 0.35];

catalogs = {eventA, eventB};
rowLabels = {'G\_2015Erp\_polished','Confidence Catalog'};
panelLabels = {'(a)','(b)','(c)','(d)','(e)','(f)'};

%% =========================
% LOOP rows × columns
%% =========================
for row = 1:2
    event1D = catalogs{row};

    for kp = 1:3
        ax = subplot('Position', positions((row-1)*3+kp,:));
        hold(ax,'on');

        basemap_2015v2(lonLim,latLim,100,[0 0],1,false,ax);
        pbaspect(ax,[diff(lonLim)*cosd(mean(latLim)) diff(latLim) 1]);
        grid(ax,'on');

        %% --- Time filter ---
        if kp==1
            indEv = [event1D.time] < date_BF;
        elseif kp==2
            indEv = [event1D.time] >= date_BF & [event1D.time] < date_DR;
        else
            indEv = [event1D.time] >= date_DR;
        end

        eventsPeriod = event1D(indEv);

        %% --- Plot FMs ---
        radius = 0.0005;
        scale_event = 1.3;

        for i = 1:length(eventsPeriod)
            if ~isempty(eventsPeriod(i).avfnorm)
                plot_balloon(eventsPeriod(i).avfnorm, ...
                             eventsPeriod(i).avslip, ...
                             eventsPeriod(i).lon, ...
                             eventsPeriod(i).lat, ...
                             radius, scale_event, ...
                             eventsPeriod(i).color3);
            end
        end

        %% --- Panel label (a–f) ---
        panelID = (row-1)*3 + kp;
        text(ax, lonLim(1)+0.002, latLim(2)-0.002, panelLabels{panelID}, ...
            'FontSize',16,'FontWeight','bold','VerticalAlignment','top');

        %% --- Column titles ---
        if row==1
            title(ax,projects{kp},'FontSize',14);
        end

        %% --- Row labels ---
        % if kp==1
        %     ylabel(ax,rowLabels{row},'FontSize',14);
        % else
        %     set(ax,'YTickLabel',[]);
        % end

        %% --- Longitude only bottom row ---
        % if row==2
        %     xlabel(ax,'Longitude (°)','FontSize',14);
        % else
        %     set(ax,'XTickLabel',[]);
        % end
    end
end
