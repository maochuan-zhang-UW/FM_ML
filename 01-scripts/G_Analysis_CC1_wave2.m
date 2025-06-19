% Time parameters
t_start = -0.25;  % seconds
t_end = 1.0;      % seconds
fs = 200;         % Hz
n_points = 250;
t = linspace(t_start, t_end, n_points);

% Index range for plotting -0.25 to 0.25 s
t_plot_end = 0.25;
plot_indices = t >= t_start & t <= t_plot_end;

% Station and fields
station = 'CC1';
po_field = ['Po_' station];       % CC polarity
poml_field = ['PoML1_W_' station]; % ML polarity
poCC_field = [station '_Po'];     % Manual polarity
wave_field = ['W_' station];      % Wave data

% Find events where CC and ML agree but Manual disagrees
disagree_events = [];
for i = 1:length(Felix)
    if isfield(Felix(i), po_field) && isfield(Felix(i), poml_field) && isfield(Felix(i), poCC_field)
        % Extract polarity values
        val_CC = sign(Felix(i).(po_field));
        val_Man = sign(Felix(i).(poCC_field));
        val_ML_raw = Felix(i).(poml_field);
        
        % Convert ML raw value to polarity
        if ischar(val_ML_raw)
            if strcmp(val_ML_raw, 'U')
                val_ML = 1;
            elseif strcmp(val_ML_raw, 'D')
                val_ML = -1;
            else
                val_ML = 0;
            end
        else
            val_ML = val_ML_raw;
        end
        
        % Check if CC and ML agree but Manual disagrees
        if val_CC ~= 0 && val_ML ~= 0 && val_Man ~= 0 && val_CC == val_ML && val_Man ~= val_CC
            disagree_events = [disagree_events, i];
        end
    end
end

% Plot waves for events where Manual disagrees with CC & ML agreement
figure('Position', [100, 100, 400, 786]);
hold on;
for idx = 1:min(20, length(disagree_events))
    i = disagree_events(idx);
    wave = Felix(i).(wave_field);
    if length(wave) == n_points
        % Get polarities
        val_CC = sign(Felix(i).(po_field));
        val_Man = sign(Felix(i).(poCC_field));
        val_ML_raw = Felix(i).(poml_field);
        if ischar(val_ML_raw)
            if strcmp(val_ML_raw, 'U')
                val_ML = 1;
            elseif strcmp(val_ML_raw, 'D')
                val_ML = -1;
            else
                val_ML = 0;
            end
        else
            val_ML = val_ML_raw;
        end
        
        % Normalize wave for better visualization
        if max(abs(wave)) ~= 0
            wave = wave / max(abs(wave));
        end
        
        % Plot with black color and line width 2
        plot(t(plot_indices), wave(plot_indices) + idx * 2, 'k-', 'LineWidth', 2, 'DisplayName', sprintf('Event %d', i));
        hold on;
        plot([0,0], [wave(plot_indices) + idx * 2-0.5,wave(plot_indices) + idx * 2+0.5], 'r', 'LineWidth', 2);
      
        % Label polarities
        polarity_text = sprintf('CC: %s, ML: %s, Man: %s', ...
            ternary(val_CC > 0, 'Up', 'Down'), ...
            ternary(val_ML > 0, 'Up', 'Down'), ...
            ternary(val_Man > 0, 'Up', 'Down'));
        text(t_start + 0.02, idx * 2+0.2, polarity_text, 'VerticalAlignment', 'middle');
    end
end
xlim([-0.25 0.25]);
ylim([0 idx * 2+1]);
% Plot formatting
xlabel('Time (s)');
ylabel('Normalized Amplitude (Offset by Event)');
title(sprintf('Waveforms where Manual Disagrees with CC & ML Agreement for %s', station));
grid on;
%legend('show');
hold off;

% Save the plot
print('-dpng', 'waveforms_man_disagrees_cc_ml.png');

% Helper function for ternary operation
function y = ternary(cond, a, b)
    if cond
        y = a;
    else
        y = b;
    end
end