%% Main Script: Only (a) and (b) with Legend Box
clear; close all;

projects = {'Before', 'During'};   % Only 2 subplots
date_BF = datenum(2015, 4, 24, 8, 0, 0);
date_DR = datenum(2015, 5, 19);

lonLim = [-130.031 -129.97];
latLim = [45.92 45.970];

load('/Users/mczhang/Documents/GitHub/FM3/02-data/G_FM/G_2015Erp_polished.mat');
event1D = event1;

labels = {'(a)','(b)'};

% clean event selection
event1D([event1D.lat] > 45.969) = [];
event1D([event1D.lon] < -130.03) = [];
event1D([event1D.mechqual] == 'C' | [event1D.mechqual] == 'D') = [];

% lighten poor-quality colors
for i = 1:length(event1D)
    if event1D(i).mechqual == 'A' || event1D(i).mechqual == 'B'
        event1D(i).color3 = event1D(i).color2;
    else
        event1D(i).color3 = event1D(i).color2 + 0.5*(1-event1D(i).color2);
    end
end

% figure layout
set(gcf,'Position',[400 300 1330         883],'Color','white');

positions = [0.07 0.12 0.40 0.78;   % (a)
             0.52 0.12 0.40 0.78];  % (b)

%% =============================
%       LOOP OVER (a) & (b)
% =============================
for kp = 1:2
    ax = subplot('Position', positions(kp,:));
    basemap_2015v2(lonLim, latLim, 100, [0 0], 1, false, ax);

    hold(ax,'on');
    pbaspect(ax,[diff(lonLim)*cosd(mean(latLim)) diff(latLim) 1]);

    %% Select events
    if kp == 1
        inds = find([event1D.time] < date_BF);
    elseif kp == 2
        inds = find([event1D.time] >= date_BF & [event1D.time] < date_DR);
    end
    events = event1D(inds);

    nmec = length(events);
    radius = 0.0005;
    scale_event = 1.3;

    %% Plot events
    for i = 1:nmec
        if ~isempty(events(i).avfnorm)
            plot_balloon(events(i).avfnorm, events(i).avslip, ...
                events(i).lon, events(i).lat, radius, scale_event, events(i).color3);
        end
    end

    %% Axis formatting
    set(ax,'FontSize',20);
    xlabel(ax,'Longitude (°)','FontSize',20);

    if kp == 1
        ylabel(ax,'Latitude (°)','FontSize',20);
    else
        set(ax,'YTickLabel',[]);
    end

    %% Subplot label (a) (b)
   % text(lonLim(1)+0.001, latLim(2)-0.001, labels{kp}, ...
    %    'FontSize', 20, 'FontWeight','bold');

    %% ================
    %   LEGEND BOX
    %% ================
    if kp == 1
        legendX = lonLim(1) + 0.005;
        legendYs = linspace(latLim(2)-0.004, latLim(2)-0.01, 4);

        faultTypes = {'N','R','S','U'};
        faultLabels = {'Normal','Reverse','Strike-slip','Unclassified'};

        for k = 1:4
            idx = find(strcmp({events.faultType}, faultTypes{k}), 1);
            if ~isempty(idx)
                rep = events(idx);
                plot_balloon(rep.avfnorm, rep.avslip, legendX, legendYs(k), ...
                             radius, scale_event, rep.color2);
            end
            text(legendX + 0.003, legendYs(k), faultLabels{k}, ...
                'FontSize', 20, 'HorizontalAlignment','left');
        end

        %% Draw LEGEND BOX (you requested this)
        legendLeft   = legendX - 0.002;
        legendRight  = legendX + 0.022;
        legendBottom = min(legendYs) - 0.002;
        legendTop    = max(legendYs) + 0.002;

        rectangle(ax, 'Position', ...
            [legendLeft, legendBottom, legendRight-legendLeft, legendTop-legendBottom], ...
            'EdgeColor','k','LineWidth',1.8);
    end

end
