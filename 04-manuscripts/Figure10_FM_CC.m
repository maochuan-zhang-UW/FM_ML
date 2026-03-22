%% Main Script: One Tight Figure with 3 Subplots
clear;close all;
% Define project labels and cutoff dates
projects = {'Before', 'During', 'After'};
date_BF = datenum(2015, 4, 24, 8, 0, 0);
date_DR = datenum(2015, 5, 19);
regions = struct(...
    'West', struct('Lat', [45.930, 45.950], 'Lon', [-130.029, -130.008]), ...
    'East', struct('Lat', [45.970, 45.930], 'Lon', [-130.0015, -129.975]), ...
    'ID', struct('Lat', [45.921, 45.929], 'Lon', [-130.004, -129.975]));
lonLim = [-130.031 -129.97];
latLim = [45.92 45.970];
% Load the event data once (assumes event1D exists in the file)
%load('/Users/mczhang/Documents/GitHub/FM3/04-final-paper/old/Event1D_3D.mat');
 load('/Users/mczhang/Documents/GitHub/FM3/02-data/G_FM/G_2015Erp_polished.mat');
 event1D=event1;
labels = {'(c)', '(d)', '(e)'};
labels_fig = {'Fig.6', 'Fig.7', 'Fig.8'};
x_fig=[-130.028,-130.00,-130.003];%west, east and ID
y_fig=[45.931,45.9315,45.9225];
event1D([event1D.lat]>45.969)=[];
event1D([event1D.lon]<-130.03)=[];
event1D([event1D.mechqual]=='C')=[];
event1D([event1D.mechqual]=='D')=[];
for i = 1:length(event1D)
    % Check if mechqual is 'A' or 'B'
    if event1D(i).mechqual == 'A' || event1D(i).mechqual == 'B'
        event1D(i).color3 = event1D(i).color2;
    else
        % Lighten the color: mix 50% of white ([1 1 1]) into the color
        event1D(i).color3 = event1D(i).color2 + 0.5 * (1 - event1D(i).color2);
    end
end

% for i = 1:length(event3D)
%     % Check if mechqual is 'A' or 'B'
%     if event3D(i).mechqual == 'A' || event3D(i).mechqual == 'B'
%         event3D(i).color3 = event3D(i).color2;
%     else
%         % Lighten the color: mix 50% of white ([1 1 1]) into the color
%         event3D(i).color3 = event3D(i).color2 + 0.5 * (1 - event3D(i).color2);
%     end
% end
% Create one figure for all subplots (adjust size as needed)
%figure('Position', [1000 200 1000 400]);
set(gcf, 'Position', [744, 309, 1465 622], ...
    'InvertHardcopy', 'off', ...   % Preserve background color
    'Color', 'white' ...           % Set figure background to white
);
% Define custom positions for the three subplots to reduce gaps.
% Format: [left, bottom, width, height] in normalized units.
% positions = [0.07 0.12 0.27 0.78;  % first subplot
%     0.37 0.12 0.27 0.78;  % second subplot
%     0.67 0.12 0.27 0.78]; % third subplot
positions = [0.07 0.12 0.27 0.78;  % first subplot
    0.37 0.12 0.27 0.78;  % second subplot
    0.67 0.12 0.27 0.78]; % third subplot
%%
%clear event1D;
%hg                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       event1D=event3D;

for kp = 1:length(projects)
    % Create subplot with tight spacing
    ax = subplot('Position', positions(kp, :));
    set(ax, 'XTick', linspace(-130.03,-129.97,7));

    % Plot the basemap on the current axes by passing ax handle.
    % (Ensure you are using the modified basemap_2015 that accepts an axes handle.)
    %basemap_2015v2([-130.03 -129.97], [45.92 45.97], 100, [0 0], 1, false, ax);
    basemap_2015v2(lonLim, latLim, 100, [0 0], 1, false, ax);
    pbaspect(ax, [diff(lonLim)*cosd(mean(latLim)) diff(latLim) 1]);
    hold(ax, 'on');
    set(ax, 'XTick', -130.03:0.01:-129.97); % Denser ticks
    set(ax, 'GridLineStyle', '-', 'LineWidth', 0.5, 'GridColor', [0.5 0.5 0.5]);
   %set(ax, 'MinorGridLineStyle', ':', 'XMinorTick', 'on', 'XMinorGrid', 'on');
    grid(ax, 'on');

    % Filter events for this period
    if kp == 1
        % "Before": events earlier than date_BF
        indEv = find([event1D.time] < date_BF);
    elseif kp == 2
        % "During": events between date_BF and date_DR
        indEv = find([event1D.time] >= date_BF & [event1D.time] < date_DR);
    else
        % "After": events on/after date_DR
        indEv = find([event1D.time] >= date_DR);
    end
    eventsPeriod = event1D(indEv);
    eventsPeriod([eventsPeriod.lat]>45.97)=[];
    eventsPeriod([eventsPeriod.lon]<-130.03)=[];
    nmec = length(eventsPeriod);

    % Plot each event using your plot_balloon function
    radius = 0.0005;
    scale_event = 1.3;

    for i = 1:nmec
        %scatter3(eventsPeriod(i).lon, eventsPeriod(i).lat, eventsPeriod(i).depth, 1)
        if ~isempty(eventsPeriod(i).avfnorm)
            plot_balloon(eventsPeriod(i).avfnorm, eventsPeriod(i).avslip, ...
                eventsPeriod(i).lon, eventsPeriod(i).lat, radius, scale_event, eventsPeriod(i).color3);
            hold(ax, 'on');
        end
    end

    % Set grid and title
    grid(ax, 'on');
    %title(ax, [num2str(nmec) ' FMs ' projects{kp} ' 2015 Eruption'], 'fontsize', 14);
    %set(ax, 'FontSize', 12);
    text(-130.029, 45.968, labels{kp}, 'FontSize', 18, 'FontWeight', 'bold');
    % text(ax, -0.1, 1.05, labels{kp}, 'Units', 'normalized', ...
    %      'FontSize', 14, 'FontWeight', 'bold');

    % For the first subplot, keep both axes labels;
    xlabel(ax, 'Longitude (°)', 'fontsize', 18);
    if kp == 1
        ylabel(ax, 'Latitude (°)', 'fontsize', 18);
    else
        set(ax, 'YTickLabel', []);
        ylabel(ax, '');
    end

    %axis equal;
    if kp == 1
        %--- Define a "legend box" region in data coordinates on the left side ---
        legendX = lonLim(1) + 0.008;  % near the left boundary of the map
        % Choose 4 vertical positions for the markers (adjust as needed)
        legendYs = linspace(latLim(2)-0.002, latLim(2)-0.01, 4);

        % Define the fault types and corresponding labels.
        faultTypes = {'N','R','S','U'};
        faultLabels = {'N - Normal', 'R - Reverse', 'S - Strike-slip', 'U - Unclassified'};

        % Loop over the four fault types.
        for k = 1:length(faultTypes)
            % Search eventsPeriod for the first event with this fault type.
            idx = find(strcmp({eventsPeriod.faultType}, faultTypes{k}), 3, 'first');
            if ~isempty(idx)
                repEvent = eventsPeriod(idx(2));
                % Plot the representative marker at the legend box location.
                plot_balloon(repEvent.avfnorm, repEvent.avslip, ...
                    legendX, legendYs(k), radius, scale_event, repEvent.color2);
            else
                % If none is found, use dummy parameters and a default color.
                defaultColors = containers.Map(...
                    {'N','R','S','U'}, {[1,0,0], [0,0,1], [0,1,0], [0,0,0]});
                plot_balloon(0, 0, legendX, legendYs(k), radius, scale_event, defaultColors(faultTypes{k}));
            end
            % Add a text label to the right of the marker.
            text(legendX + 0.003, legendYs(k), faultLabels{k}, ...
                'HorizontalAlignment', 'left', 'FontSize', 15);
        end

        %--- Draw a rectangle around the legend entries ---
        % Define the left and right bounds of the legend box.
        legendLeft = legendX-0.002 ;     % a little left of the marker
        legendRight = legendX + 0.020;       % extend enough to include the text labels
        legendBottom = min(legendYs) - 0.002;
        legendTop = max(legendYs)+0.001;

        legendWidth = legendRight - legendLeft;
        legendHeight = legendTop - legendBottom;

        rectangle(ax, 'Position', [legendLeft, legendBottom, legendWidth, legendHeight], ...
            'EdgeColor', 'k', 'LineWidth', 0.5);
        hold on;
        % regionNames = fieldnames(regions);
        % for i = 1:length(regionNames)
        %     region = regions.(regionNames{i});
        %     latitudes = [region.Lat(1), region.Lat(1), region.Lat(2), region.Lat(2), region.Lat(1)];
        %     longitudes = [region.Lon(1), region.Lon(2), region.Lon(2), region.Lon(1), region.Lon(1)];
        %     plot3(longitudes, latitudes, zeros(size(latitudes)), 'k--', 'LineWidth', 2, 'DisplayName', [regionNames{i} ' Region']);
        % end
        % for i=1:length(labels_fig)
        %     text(x_fig(i), y_fig(i), labels_fig{i}, 'FontSize', 12, 'FontWeight', 'bold')
        % end
    end

end


