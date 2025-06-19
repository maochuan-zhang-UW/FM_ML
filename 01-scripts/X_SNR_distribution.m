clc; clear;

load('/Users/mczhang/Documents/GitHub/FM3/02-data/E_Po/E_All_polished_updated_V2.mat');

stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
snr_data = table();

% Collect SNR data for all stations
for s = 1:length(stations)
    station = stations{s};
    nsp_field = ['NSP_' station];
    
    % Initialize temporary arrays
    valid_idx = false(length(Felix), 1);
    snr_vals = zeros(length(Felix), 1);
    lons = zeros(length(Felix), 1);
    lats = zeros(length(Felix), 1);
    depths = zeros(length(Felix), 1);
    count = 0;
    
    % Single pass through Felix
    for i = 1:length(Felix)
        if isfield(Felix(i), nsp_field) && ~isempty(Felix(i).(nsp_field)) && length(Felix(i).(nsp_field)) >= 3 && isfield(Felix(i), 'depth')
            nsp = Felix(i).(nsp_field);
            noise_val = nsp(1);
            p_val = nsp(3);
            if noise_val > 0 && p_val > 0 && noise_val ~= 1
                count = count + 1;
                valid_idx(i) = true;
                snr_vals(count) = p_val;
                lons(count) = Felix(i).lon;
                lats(count) = Felix(i).lat;
                depths(count) = Felix(i).depth;
            end
        end
    end
    
    % Trim arrays and create table
    if count > 0
        temp_table = table(repmat({station}, count, 1), snr_vals(1:count), lons(1:count), lats(1:count), depths(1:count), ...
            'VariableNames', {'Station', 'SNR_dB', 'Lon', 'Lat', 'Depth'});
        snr_data = [snr_data; temp_table];
        fprintf('Station %s: %d valid data points\n', station, count);
    else
        fprintf('Station %s: No valid data points\n', station);
    end
end

% Calculate average SNR and depth using groupby
if ~isempty(snr_data)
    [~, ia, ic] = unique(snr_data(:, {'Lon', 'Lat', 'Depth'}), 'rows');
    avg_snr_table = table();
    avg_snr_table.Lon = snr_data.Lon(ia);
    avg_snr_table.Lat = snr_data.Lat(ia);
    avg_snr_table.Depth = snr_data.Depth(ia);
    avg_snr_table.SNR_dB = accumarray(ic, snr_data.SNR_dB, [], @mean);
else
    avg_snr_table = table();
    fprintf('No data available for average SNR plot\n');
end
lonLim = [-130.031 -129.97];
latLim = [45.92 46];
% Create figure with 8 subplots (2 rows, 4 columns)
figure('Position', [100, 100, 1200, 727]);
t = tiledlayout(1, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

% Calculate logarithmic SNR for colormap
if ~isempty(snr_data)
    snr_data.SNR_dB_log = log10(snr_data.SNR_dB); % Add logarithmic SNR column
    avg_snr_table.SNR_dB_log = log10(avg_snr_table.SNR_dB); % Add for average
end
 axial_calderaRim; % Assumes this function exists
% Plot for each station
for s = 1:length(stations)
    nexttile;
   
    plot(calderaRim(:,1), calderaRim(:,2), 'k', 'LineWidth', 3);
    hold on
    station = stations{s};
    if ~isempty(snr_data)
        idx = strcmp(snr_data.Station, station);
        if any(idx)
            scatter3(snr_data.Lon(idx), snr_data.Lat(idx), -snr_data.Depth(idx), 2, snr_data.SNR_dB_log(idx), 'filled');
            view(0,90); % Adjust view angle for better 3D visualization
        else
            text(0.5, 0.5, 'No Data', 'HorizontalAlignment', 'center', 'FontSize', 10);
            fprintf('No plot data for station %s\n', station);
        end
    else
        text(0.5, 0.5, 'No Data', 'HorizontalAlignment', 'center', 'FontSize', 10);
        fprintf('No plot data for station %s (empty snr_data)\n', station);
    end
    title(station);
    colormap(jet);
    if ~isempty(snr_data)
        clim([1, 4]); % Set color limits based on log SNR
    end
    grid on;
    xlabel('Longitude');
    ylabel('Latitude');
    zlabel('Depth');
    if s == 4
        cb = colorbar('Location', 'southoutside');
        cb.Label.String = 'log10(SNR)';
    end
end

% Plot for average SNR
nexttile;
plot(calderaRim(:,1), calderaRim(:,2), 'k', 'LineWidth', 3);
hold on;
if ~isempty(avg_snr_table) % Corrected from avg_snr_data to avg_snr_table
    scatter3(avg_snr_table.Lon, avg_snr_table.Lat, -avg_snr_table.Depth, 2, avg_snr_table.SNR_dB_log, 'filled');
    view(0,90); % Adjust view angle
else
    text(0.5, 0.5, 'No Data', 'HorizontalAlignment', 'center', 'FontSize', 10);
end
title('Average SNR');
colormap(jet);
if ~isempty(snr_data)
    clim([2, 4]); % Set color limits based on log SNR
end
grid on;
xlabel('Longitude');
ylabel('Latitude');
zlabel('Depth');
cb = colorbar('Location', 'southoutside');
cb.Label.String = 'log10(SNR)';

% Adjust layout
title(t, '3D Logarithmic SNR for Felix Events by Station and Average');
set(gca, 'FontSize', 10);
% New figure for SNR vs Depth
figure('Position', [100, 100, 800, 400]);

% Define depth bins (0 to 2 km, 0.1 km intervals)
depth_bins = 0:0.1:2;
snr_log_avg = cell(length(stations) + 1, 1); % Store SNR for each station + average
colors = lines(length(stations) + 1); % Color map for stations + average
hold on;

% Process each station
for s = 1:length(stations)
    station = stations{s};
    idx = strcmp(snr_data.Station, station);
    if any(idx)
        % Bin SNR data by depth
        station_depths = snr_data.Depth(idx);
        station_snr = snr_data.SNR_dB(idx);
        [~, ~, bin_idx] = histcounts(station_depths, depth_bins);
        snr_mean = accumarray(bin_idx(bin_idx > 0), station_snr(bin_idx > 0), [], @mean);
        valid_bins = unique(bin_idx(bin_idx > 0));
        snr_log_avg{s} = nan(length(depth_bins) - 1, 1);
        snr_log_avg{s}(valid_bins) = log10(snr_mean); % Logarithmic SNR
        % Plot
        plot(depth_bins(1:end-1) + 0.05, snr_log_avg{s}, 'LineWidth', 1.5, 'Color', colors(s, :), 'DisplayName', station);
    else
        snr_log_avg{s} = nan(length(depth_bins) - 1, 1);
    end
end

% Process average SNR
if ~isempty(avg_snr_table)
    [~, ~, bin_idx] = histcounts(avg_snr_table.Depth, depth_bins);
    snr_mean = accumarray(bin_idx(bin_idx > 0), avg_snr_table.SNR_dB(bin_idx > 0), [], @mean);
    valid_bins = unique(bin_idx(bin_idx > 0));
    snr_log_avg{end} = nan(length(depth_bins) - 1, 1);
    snr_log_avg{end}(valid_bins) = log10(snr_mean); % Logarithmic SNR
    % Plot
    plot(depth_bins(1:end-1) + 0.05, snr_log_avg{end}, 'LineWidth', 2, 'Color', colors(end, :), 'DisplayName', 'Average');
end

% Customize plot
xlabel('Depth (km)');
ylabel('log10(SNR)');
title('SNR vs Depth by Station and Average');
xlim([0 2]);
grid on;
legend('Location', 'best');
set(gca, 'FontSize', 10);
hold off;
clear
