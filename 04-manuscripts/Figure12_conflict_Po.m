clear; clc; close all;

%% =========================
% Load files
%% =========================
load('/Users/mczhang/Documents/GitHub/FM4/02-data/Before22OBSs/A_All/A_wavelarge5.mat')   % Felix
load('/Users/mczhang/Documents/GitHub/FM4/02-data/Before22OBSs/F_Cl/F_Cl_All_MLreplace_samecluster_conf.mat') % Po_Clu

%% =========================
% Fields
%% =========================
po_fields = {'Po_AS1','Po_AS2','Po_CC1','Po_EC1','Po_EC2','Po_EC3','Po_ID1'};
w_fields  = {'W_AS1','W_AS2','W_CC1','W_EC1','W_EC2','W_EC3','W_ID1'};

%% =========================
% Build Felix lookup
%% =========================
felix_ids = [Felix.ID];
felix_map = containers.Map(felix_ids, 1:numel(Felix));

%% =========================
% Storage for conflicts
%% =========================
conflict_waveforms = cell(size(po_fields));
conflict_labels    = cell(size(po_fields));

%% =========================
% Find conflicts
%% =========================
for i = 1:numel(Po_Clu)
    
    id = Po_Clu(i).ID;
    if ~isKey(felix_map, id)
        continue
    end
    
    j = felix_map(id);
    
    for k = 1:numel(po_fields)
        
        po_field = po_fields{k};
        w_field  = w_fields{k};
        
        if ~isfield(Po_Clu, po_field) || ~isfield(Felix, w_field)
            continue
        end
        
        val = Po_Clu(i).(po_field);
        if numel(val) < 2
            continue
        end
        
        p1 = val(1);   % CC polarity
        p2 = val(2);   % ML polarity
        
        % Both non-zero & conflicting
        if p1 ~= 0 && p2 ~= 0 && p1 ~= p2
            
            wave = Felix(j).(w_field);
            if isempty(wave)
                continue
            end
            
            conflict_waveforms{k}{end+1} = wave;
            conflict_labels{k}{end+1} = sprintf('%d → %d', p1, p2);
        end
    end
end

%% =========================
% Plot ONE figure with 7 subplots
%% =========================
figure('Color','w','Position',[100 100 1400 600]);

rng('shuffle')   % use rng(1) if you want reproducible random selection

for k = 1:numel(po_fields)
    
    subplot(1,7,k); hold on
    
    waves  = conflict_waveforms{k};
    labels = conflict_labels{k};
    
    if isempty(waves)
        title([po_fields{k} ' (no conflict)'], 'Interpreter','none')
        axis off
        continue
    end
    
    % ----- Randomly select up to 10 -----
    n_total = numel(waves);
    nplot   = min(n_total, 10);
    rand_idx = randperm(n_total, nplot);
    
    for n = 1:nplot
        
        idx = rand_idx(n);
        w = waves{idx};
        if isempty(w), continue, end
        
        w = w / max(abs(w));    % normalize
        yshift = n;
        
        plot(w + yshift, 'k','LineWidth', 3)
        %text(10, yshift, labels{idx}, 'fontsize', 10)
        text(5, yshift+0.2, labels{idx}, 'fontsize', 10, 'Color', 'b')
    end
    
    % ---- Pick time at sample 50 ----
    x_pick = 50;
    yl = ylim;
    plot([x_pick x_pick], yl, 'r--', 'LineWidth', 1.2);
    
    title(po_fields{k}, 'FontWeight','bold', 'Interpreter','none')
    xlim([0 100]);
    ylim([0 11]);
    box on;

    ylabel('Trace')
    set(gca,'fontsize',9)
end

sgtitle('Random 10 Conflicting Polarities (CC - ML)');


%% =========================
% Plot spatial distribution (3x3 layout, small dots)
%% =========================
% Plot spatial distribution (3x3, station-specific)
%% =========================
figure('Color','w','Position',[100 100 900 800]);

lonLim = [-130.031 -129.97];
latLim = [45.92 45.972];

% Load stations
axial_stationsNewOrder;   % creates variable: station

for k = 1:numel(po_fields)
    
    subplot(3,3,k); hold on
    
    %% ----- SAME polarity (gray) -----
    same_lon = lon_all_valid{k};
    same_lat = lat_all_valid{k};
    
    if ~isempty(lon_conflict{k})
        is_conf = ismember([same_lon(:) same_lat(:)], ...
                           [lon_conflict{k}(:) lat_conflict{k}(:)], ...
                           'rows');
        same_lon = same_lon(~is_conf);
        same_lat = same_lat(~is_conf);
    end
    
    scatter(same_lon, same_lat, 3, [0.7 0.7 0.7], 'filled')
    
    %% ----- CONFLICT (blue) -----
    scatter(lon_conflict{k}, lat_conflict{k}, 6, [0 0.45 0.9], 'filled')
    
    %% ===== CALDERA RIM =====
    axial_calderaRim;
    plot(calderaRim(:,1), calderaRim(:,2), '-k', ...
        'LineWidth',1.5,'HandleVisibility','off');
    
    %% ===== Plot ONLY corresponding station =====
    
    % Convert Po_AS1 -> AXAS1
    st_name = ['AX' po_fields{k}(4:end)];
    
    % Find matching station
    idx = find(strcmp({station.name}, st_name));
    
    if ~isempty(idx)
        plot(station(idx).lon, station(idx).lat, 'sk', ...
            'markerfacecolor','k', 'markersize',7);
        
        % Optional label
        text(station(idx).lon, station(idx).lat, st_name, ...
            'fontsize',7, 'verticalalignment','bottom', ...
            'horizontalalignment','center');
    end
    
    %% ===== Axis limits =====
    xlim(lonLim)
    ylim(latLim)
    box on
    
    title(po_fields{k}, 'Interpreter','none', 'FontWeight','bold')
    
    if k > 6
        xlabel('Lon')
    end
    if mod(k,3)==1
        ylabel('Lat')
    end
    
    set(gca,'fontsize',9)
end

sgtitle('Polarity Agreement vs Conflict (Gray = Same, Blue = Conflict)')