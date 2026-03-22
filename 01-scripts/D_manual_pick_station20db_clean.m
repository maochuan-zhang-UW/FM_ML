clc; clear; close all;

%% Load data
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_wave/A_wave_dB20_polish.mat')

%% Clean the data
fields = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};

% Time vector for 100 samples at 100 Hz (after resampling from 200 Hz)
% Original: -0.5 to 0.49 s at 200 Hz = 200 samples
% After resampling: 100 samples at 100 Hz
time_vector = linspace(-0.5, 0.49, 100);

% Define the time window to check for maximum
time_window = [-0.05, 0.15];
window_indices = find(time_vector >= time_window(1) & time_vector <= time_window(2));

% Counter for removed waveforms
removed_count = zeros(1, length(fields));
total_count = zeros(1, length(fields));

% Process each event
for i = 1:length(Felix)
    for kz = 1:length(fields)
        field = fields{kz};
        wave_field = ['W_', field];
        
        % Check if waveform exists and is not zero/empty
        if isfield(Felix(i), wave_field) && ~isequal(Felix(i).(wave_field), 0)
            wave = Felix(i).(wave_field);
            
            if ~isempty(wave) && length(wave) > 1
                total_count(kz) = total_count(kz) + 1;
                
                % Resample from 200 Hz to 100 Hz
                wave_resampled = resample(wave, 1, 2);
                
                % Find the absolute maximum in the entire waveform
                [max_val, max_idx] = max(abs(wave_resampled));
                
                % Check if the maximum is NOT in the specified time window
                if ~ismember(max_idx, window_indices)
                    % Remove this waveform by setting it to 0
                    Felix(i).(wave_field) = 0;
                    
                    % Also remove the polarity field if it exists
                    po_field = ['Po_', field];
                    if isfield(Felix(i), po_field)
                        Felix(i).(po_field) = 0;
                    end
                    
                    removed_count(kz) = removed_count(kz) + 1;
                end
            end
        end
    end
end

%% Display cleaning statistics
fprintf('\n=== Data Cleaning Summary ===\n');
for kz = 1:length(fields)
    fprintf('Station %s: Removed %d out of %d waveforms (%.1f%%)\n', ...
        fields{kz}, removed_count(kz), total_count(kz), ...
        100*removed_count(kz)/total_count(kz));
end
fprintf('Total: Removed %d out of %d waveforms (%.1f%%)\n\n', ...
    sum(removed_count), sum(total_count), 100*sum(removed_count)/sum(total_count));

%% Save cleaned data
save('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_wave/A_wave_dB20_cleaned.mat', 'Felix');
fprintf('Cleaned data saved to: A_wave_dB20_cleaned.mat\n\n');

%% Plot cleaned waveforms
figure('Position', [100, 100, 1800, 600]);
n_events_to_plot = 20;
offset = 2;

for kz = 1:length(fields)
    subplot(1, 7, kz);
    hold on;
    field = fields{kz};
    wave_field = ['W_', field];
    
    % Plot up to 20 events
    count = 0;
    for i = 1:length(Felix)
        if count >= n_events_to_plot
            break;
        end
        
        % Check if waveform exists and is not zero/empty
        if isfield(Felix(i), wave_field) && ~isequal(Felix(i).(wave_field), 0)
            wave = Felix(i).(wave_field);
            if ~isempty(wave) && length(wave) > 1
                % Normalize the waveform
                wave = resample(wave, 1, 2);
                wave_norm = wave / max(abs(wave));
                
                % Plot with offset
                y_offset = count * offset;
                plot([-0.5:0.01:0.49], wave_norm + y_offset, 'b', 'LineWidth', 1.5);
                
                % Highlight the time window [-0.05, 0.2]
                if count == 0
                    patch([time_window(1), time_window(2), time_window(2), time_window(1)], ...
                          [-offset, -offset, count*offset+offset, count*offset+offset], ...
                          'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
                end
                
                % Add polarity label
                if isfield(Felix(i), ['Po_', field])
                    po_value = Felix(i).(['Po_', field]);
                    text(-0.3, y_offset+0.4, sprintf('Po: %d', po_value), ...
                        'FontSize', 10, 'HorizontalAlignment', 'right');
                end
                count = count + 1;
            end
        end
    end
    
    % Formatting
    if kz == 1
        ylabel('Normalized Amplitude');
        xlabel('Time (s)');
    end
    title(sprintf('Station %s', field));
    ylim([-offset, count * offset]);
    set(gca, 'YTick', []);
    grid on;
    box on;
    set(gca, 'FontSize', 12);
    hold off;
end
sgtitle('Cleaned Template Waveforms from All Stations');