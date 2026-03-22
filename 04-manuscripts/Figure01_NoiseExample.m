
clc; clear; close all;%% Plot normalized waveforms for all stations
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_Wave/A_wave_noise_10000.mat');
fields = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};
figure('Position', [100, 100, 1800, 600],'Color','w');

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
                plot([-0.5:0.01:0.49], wave_norm + y_offset, 'b', 'LineWidth', 1.5);
                
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
    

    %xlabel('Sample Index');
    if kz==1
    ylabel('Normalized Amplitude');
    xlabel('Times');
    end
    title(sprintf('Station %s', field));
    ylim([-offset, count * offset]);
    grid on;
    box on;
    %set(gca,'fontSize','14');
    hold off;
end

sgtitle(sprintf('Normalized Noise Waveforms for All Stations'));
