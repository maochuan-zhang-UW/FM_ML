clc; clear; close all;
dt = 1/100;
P_x = [-0.25 0.25];
P_x1 = P_x(1):dt:P_x(2)-dt;
path = '/Users/mczhang/Documents/GitHub/FM4/02-data/';
path2 = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/';
stations = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};

out_file = [path2, 'D_man/D_man_dB20_polish.mat'];
load(out_file);
% Create a figure for interactive inspection


% Get all station names from Felix structure fields
all_fields = fieldnames(Felix);
wave_fields = all_fields(startsWith(all_fields, 'W_'));
polarity_fields = all_fields(startsWith(all_fields, 'Po_'));

% Remove 'W_' and 'Po_' prefixes to get station names
stations = unique(cellfun(@(x) x(3:end), wave_fields, 'UniformOutput', false));


for event_idx = 1:length(Felix)
    for station_idx = 1:length(stations)
        station = stations{station_idx};
        polarity_field_man = ['Man_', station];
        
        % Initialize Man_ field if it doesn't exist or is empty
        if ~isfield(Felix(event_idx), polarity_field_man) || isempty(Felix(event_idx).(polarity_field_man))
            Felix(event_idx).(polarity_field_man) = [];
        end
    end
end
% Define the three regions
region1 = [0, 67];    % Set polarity UP
region2 = [68, 135];  % Remove wave
region3 = [136, 200]; % Set polarity DOWN
fig = figure('Position', [100, 100, 1200, 400], 'Name', 'Waveform Inspector');
% Process each event in Felix
for event_idx = 1:length(Felix)
    fprintf('Processing event %d/%d\n', event_idx, length(Felix));
    
    % Process each station
    for station_idx = 1:length(stations)
        station = stations{station_idx};
        wave_field = ['W_', station];
        polarity_field = ['Po_', station];
        polarity_field_man = ['Man_', station];
        
        % Check if this station has waveform data for this event
        if isfield(Felix, wave_field) && ~isempty(Felix(event_idx).(wave_field)) && ...
           isfield(Felix, polarity_field) && ~isempty(Felix(event_idx).(polarity_field))
            
            wave_data = Felix(event_idx).(wave_field);
            polarity_data = Felix(event_idx).(polarity_field);
            polarity_data_man = Felix(event_idx).(polarity_field_man);

            if ~isempty(polarity_data_man)
                continue;
            end
            
            % Check if wave data has exactly 200 points
            if length(wave_data) == 200
                
                % Plot the waveform
                clf(fig);
                plot(wave_data, 'b-', 'LineWidth', 1.5);
                hold on;
                
                % Add vertical dashed lines for the three regions
                plot([region1(2) region1(2)], ylim, 'r--', 'LineWidth', 2); % End of region 1
                plot([region2(2) region2(2)], ylim, 'r--', 'LineWidth', 2); % End of region 2
                
                % Add text labels for each region
                text(region1(2)/2, max(ylim)*0.9, 'UP (0-67)', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
                text((region2(1)+region2(2))/2, max(ylim)*0.9, 'REMOVE (68-135)', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
                text((region3(1)+region3(2))/2, max(ylim)*0.9, 'DOWN (136-200)', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
                
                % Add title with station and polarity info
                title(sprintf('Event %d - Station %s - Current Polarity: %d\nClick: [0-67]=UP, [68-135]=REMOVE, [136-200]=DOWN', ...
                    event_idx, station, polarity_data), 'FontSize', 14);
                
                xlabel('Sample Number');
                ylabel('Amplitude');
                grid on;
                
                % Set x-axis limits to show the full 200 points
                xlim([0 200]);
                
                % Wait for user click
                fprintf('Click on the plot to make a decision for %s\n', station);
                [x, ~] = ginput(1);
                
                % Process user input
                if x >= region1(1) && x <= region1(2)
                    % Set polarity to UP (1)
                    Felix(event_idx).(polarity_field_man) = 1;
                    fprintf('  -> Polarity set to UP (1) for station %s\n', station);
                elseif x >= region2(1) && x <= region2(2)
                    % Remove the wave by setting it to empty
                    Felix(event_idx).(wave_field) = [];
                    Felix(event_idx).(polarity_field_man) = 0;
                    fprintf('  -> Wave removed for station %s\n', station);
                elseif x >= region3(1) && x <= region3(2)
                    % Set polarity to DOWN (-1)
                    Felix(event_idx).(polarity_field_man) = -1;
                    fprintf('  -> Polarity set to DOWN (-1) for station %s\n', station);
                else
                    fprintf('  -> Invalid click position, keeping current values for station %s\n', station);
                end
                
            else
                fprintf('  Skipping station %s: waveform length is %d (expected 200)\n', ...
                    station, length(wave_data));
            end
        end
    end
end

% Display completion message
fprintf('Waveform inspection completed!\n');
disp('Modified Felix structure contains your changes.');