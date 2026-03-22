clc; clear; close all;

lonLim = [-130.03 -129.97];
latLim = [45.92 45.97];

%% =========================================================
% COMPUTE KAGAN USING YOUR FIRST SCRIPT EVENTS
%% =========================================================
load('/Users/mczhang/Documents/GitHub/FM3/02-data/G_FM/G_HASH_All_ML_sameClusterasbeforev_confidence.mat');

event = event1;
%event([event.mechqual] == 'D' | [event.mechqual] == 'D') = [];
clear event1;

%load('/Users/mczhang/Documents/GitHub/FM3/02-data/G_FM/G_2015Erp_polished.mat');
load('/Users/mczhang/Documents/GitHub/FM3/02-data/G_FM/G_2015Erp_polished.mat');

for i = 1:length(event)
    ind = find([event1.id] == event(i).id);
    if ~isempty(ind)
        event(i).kg = kagan( ...
            [event(i).avmech(1), event(i).avmech(2), event(i).avmech(3)], ...
            [event1(ind(1)).avmech(1), event1(ind(1)).avmech(2), event1(ind(1)).avmech(3)]);
    end
end

event(arrayfun(@(x) isempty(x.kg), event)) = [];
data_real = real([event.kg]);

fprintf('Loaded %d events with Kagan\n', length(event));

%% =========================================================
% BUILD FIGURE (Histogram + Map only)
%% =========================================================
figure('Position',[200 300 900 380],'Color','w');
kg_mean   = mean(data_real );
kg_median = median(data_real );


%% ---------- (a) Kagan Histogram ----------
subplot(1,2,1);

edges = linspace(0,120,25);
histogram(data_real, edges, ...
    'FaceColor',[0.2 0.4 0.9], ...
    'EdgeColor','k');

xlabel('Kagan Angle (°)');
ylabel('Count');
%title(sprintf('Median = %.1f°', median(data_real,'omitnan')));

title(sprintf('Kagan Angle Distribution  |  Mean = %.2f°   Median = %.2f°', ...
    kg_mean, kg_median), 'FontWeight','bold');
grid on;
xlim([0 120]);
set(gca,'FontSize',16);

text(0.02,0.95,'(a)','Units','normalized',...
    'FontWeight','bold','FontSize',18);

%% ---------- (b) Spatial Kagan Map ----------
subplot(1,2,2);

ax = gca;
pbaspect(ax,[diff(lonLim)*cosd(mean(latLim)) diff(latLim) 1]);
hold on;

% Plot caldera rim if available
try
    axial_calderaRim;
    plot(calderaRim(:,1),calderaRim(:,2),'k','LineWidth',1.8);
end

scatter3([event.lon],[event.lat],[event.depth], ...
    6, data_real, 'filled');

set(gca,'ZDir','reverse');

colormap(jet);
cb = colorbar;
ylabel(cb,'Kagan (°)');

xlabel('Longitude');
ylabel('Latitude');
zlabel('Depth (km)');

grid on;
box on;
xlim(lonLim);
ylim(latLim);

set(gca,'FontSize',16);

text(0.02,0.95,'(b)','Units','normalized',...
    'FontWeight','bold','FontSize',18);
