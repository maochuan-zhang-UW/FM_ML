clear; close all;

fields = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};
groups = {'W1','W2','W3','W4','W5','W6','W7','W8','E1','E2','E3','E4','E5','E6','E7','E8'};

load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_wave/A_Felix_10000_noise.mat')
%Felix=Felix(1:30);
dt = 1/200; % Hz

%% Load parameters
parameterFile = 'parameter_script_realtimeVer1_MC_focal';
run(parameterFile); % Should load variable 'p'

if ~exist('p', 'var')
    error('Variable "p" not found after running %s', parameterFile);
end

%% Step A: obtain wave of event
P.a.sttime = -3;
P.a.edtime = 7;
P.a.window = [-0.5 0.5];
P.filt = 5; % Use the first filter band (e.g., [3 20] Hz), cuz small earthquakes are good

t_trace = linspace(P.a.sttime, P.a.edtime, 2001)';
idx_start = round((P.a.window(1) - P.a.sttime) / dt) + 1;
idx_end   = round((P.a.window(2) - P.a.sttime) / dt);

overall_tic = tic;

for kz = 1:length(fields)
    field = fields{kz};
    fprintf('Processing field %s...\n', field);
    t1 = tic;

    tempFelix = Felix;  % Copy to avoid modifying inside parfor

    % Use parallel loop for each event
    parfor i = 1:length(Felix)
        temp = 0;
        DDt_field = ['DDt_', field];

        % Check if field exists and get the value
        if isfield(tempFelix(i), DDt_field)
            ddt_value = tempFelix(i).(DDt_field);

            % Ensure it's a scalar and check the condition
            if isscalar(ddt_value) && (ddt_value > -3)
                trace_Z = obtain_waveforms_Z(tempFelix(i), kz, P.a.sttime, P.a.edtime, p);

                if isfield(trace_Z, 'dataFilt') && size(trace_Z.dataFilt, 1) >= P.filt
                    temp = trace_Z.dataFilt(idx_start:idx_end);
                    % Resample from 200 Hz to 100 Hz (downsample by factor of 2)
                    temp = resample(temp, 1, 2);  % Alternative: temp = downsample(temp, 2);
                end
            end
        end

        % Assign result
        tempFelix(i).(['W_', field]) = temp;
    end

    Felix = tempFelix;
    fprintf('Elapsed time for field %s: %.2f seconds\n', field, toc(t1));
end

fprintf('Total processing time: %.2f seconds\n', toc(overall_tic));

% Save the results
save('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_Wave/A_wave_noise_10000.mat', 'Felix');

%% Plot normalized waveforms for all stations
figure('Position', [100, 100, 1800, 600]);

n_events_to_plot = 20;
offset = 2; % Vertical offset between traces

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
                wave_norm = wave / max(abs(wave));
                
                % Plot with offset
                y_offset = count * offset;
                plot(wave_norm + y_offset, 'b', 'LineWidth', 0.5);
                
                % Add polarity label
                % if isfield(Felix(i), ['Po_', field])
                %     po_value = Felix(i).(['Po_', field]);
                %     text(-5, y_offset, sprintf('Po: %d', po_value), ...
                %         'FontSize', 8, 'HorizontalAlignment', 'right');
                % end
                
                count = count + 1;
            end
        end
    end
    
    % Formatting
    xlabel('Sample Index');
    ylabel('Normalized Amplitude + Offset');
    title(sprintf('Station %s', field));
    ylim([-offset, count * offset]);
    grid on;
    hold off;
end

sgtitle(sprintf('Normalized Waveforms for All Stations (%d Events Each, Offset +%d)', ...
    n_events_to_plot, offset));
