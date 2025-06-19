clc; clear;
load('/Users/mczhang/Documents/GitHub/FM3/02-data/E_Po/E_All.mat');
path2 = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/';
fields = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};

L = 250;            % waveform length
Fs = 200;           % sampling rate
t = linspace(-0.25, 1, L);  % time vector
start_idx = find(t >= -0.1, 1, 'first');
end_idx = find(t <= 0.4, 1, 'last');

for f = 1:length(fields)
    field = fields{f};
    fprintf('\n=== Processing Field: %s ===\n', field);

    % Find all valid waveform indices
    valid_indices = find(arrayfun(@(x) isfield(x, ['W_', field]) && ~isempty(x.(['W_', field])), Felix));
    
    % Select 1000 random valid indices
    rng(2024);
    selected_indices = valid_indices(randperm(length(valid_indices), min(1000, length(valid_indices))));

    for k = 1:length(selected_indices)
        main_idx = selected_indices(k);
        fprintf('Manual pick %d/%d for %s (ID = %d)\n', k, length(selected_indices), field, Felix(main_idx).ID);

        main_waveform = Felix(main_idx).(['W_', field])(:);
        if length(main_waveform) < end_idx
            continue;
        end
        main_waveform = 2 * (main_waveform - min(main_waveform)) / (max(main_waveform) - min(main_waveform)) - 1;

        main_window = main_waveform(start_idx:end_idx);

        % Cross-correlation alignment
        num_waveforms = length(Felix);
        window_length = end_idx - start_idx + 1;
        cc_values = zeros(num_waveforms, 1);
        cross_corrs = zeros(2 * window_length - 1, num_waveforms);

        parfor i = 1:num_waveforms
            if i == main_idx || ~isfield(Felix(i), ['W_', field]) || isempty(Felix(i).(['W_', field]))
                cc_values(i) = 0;
                continue;
            end
            other_wave = Felix(i).(['W_', field])(:);
            if length(other_wave) < end_idx
                cc_values(i) = 0;
                continue;
            end
            other_window = other_wave(start_idx:end_idx);
            other_window = 2 * (other_window - min(other_window)) / (max(other_window) - min(other_window)) - 1;
            cross_corrs(:, i) = xcorr(main_window, other_window, 'coeff');
        end

        [max_cc, max_idx] = max(cross_corrs);
        cc_values = max_cc(:);
        [~, sort_idx] = sort(cc_values, 'descend');
        top_20_indices = sort_idx(1:20);
        top_20_cc = cc_values(sort_idx(1:20));
        top_20_lags = max_idx(sort_idx(1:20)) - window_length;

        % Plot
        figure; set(gcf, "Position", [744 112 950 700]); hold on;
        plot(t, main_waveform, 'k', 'LineWidth', 2);
        vertical_offset = 1;
        aligned_waveforms = zeros(L, 20);

        for j = 1:20
            idx = top_20_indices(j);
            if isempty(Felix(idx).(['W_', field]))
                continue;
            end
            shift = top_20_lags(j);
            waveform = Felix(idx).(['W_', field])(:);
            waveform = 2 * (waveform - min(waveform)) / (max(waveform) - min(waveform)) - 1;

            if shift > 0
                aligned = [zeros(shift, 1); waveform];
                aligned = aligned(1:L);
            else
                aligned = waveform(abs(shift)+1:end);
                if length(aligned) < L
                    aligned = [aligned; zeros(L - length(aligned), 1)];
                end
            end

            aligned_waveforms(:, j) = aligned;
            plot(t, aligned + j * vertical_offset);
            text(t(1)-0.03, j * vertical_offset, sprintf('CC: %.2f', top_20_cc(j)), 'Color', 'b');
        end

        % Plot stack
        avg_stack = mean(aligned_waveforms, 2);
        plot(t, avg_stack + 21 * vertical_offset, 'r', 'LineWidth', 1.5);
        text(t(1), 21 * vertical_offset, 'Stacked Average', 'Color', 'r', 'FontSize', 10);

        % Decorations
        xlim([-0.25 0.5]); ylim([-1 23]);
        xlabel('Time (s)'); ylabel('Amplitude');
        grid on;
        plot([0, 0], [-1, 23], 'r', 'LineWidth', 2);
        plot([-0.096, -0.096], [-1, 23], 'g'); plot([0.096, 0.096], [-1, 23], 'g');

        % Manual polarity picking
        [x, ~] = ginput(1);
        if x > 0.096 && x < 0.277
            P = 1;
        elseif x >= 0.277 && x < 0.458
            P = 2;
        elseif x >= 0.458
            P = 3;
        elseif x <= -0.096 && x > -0.277
            P = -1;
        elseif x <= -0.277 && x >= -0.458
            P = -2;
        elseif x <= -0.458
            P = -3;
        elseif x > -0.096 && x < 0.096
            P = 0;
        else
            P = 0;
        end

        % Save to structure
        Felix(main_idx).([field, '_CCPo']) = P;

        close;
    end

    % Save only the selected 1000 events
    Felix_selected = Felix(selected_indices);
    out_file = fullfile(path2, sprintf('D_manual_%s_CCPo.mat', field));
    save(out_file, 'Felix_selected');
    fprintf('Saved 1000 manually picked events for %s to: %s\n\n', field, out_file);
end
