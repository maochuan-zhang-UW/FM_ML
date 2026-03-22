clc; clear;close all;

path2 = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_wave/';
fields = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};

L = 200;            % waveform length
Fs = 100;           % sampling rate
t = linspace(-1, 1, L);  % time vector
start_idx = find(t >= -0.1, 1, 'first');
end_idx = find(t <= 0.4, 1, 'last');

for f = 1:length(fields)
    field = fields{f};
    fprintf('\n=== Processing Field: %s ===\n', field);
    out_file = [path2, 'A_wave_dB20.mat'];
    load(out_file);

    for k = 1:length(Felix)
        main_event = Felix(k);
        fprintf('Manual pick %d/%d for %s (ID = %d)\n', k, length(Felix), field, main_event.ID);
        main_waveform = main_event.(['W_', field])(:);
        if length(main_waveform) < end_idx
            continue;
        end
        if isfield(main_event, [field, '_CCPo']) && ~isempty(main_event.([field, '_CCPo']))
            continue;
        end
        main_waveform = 2 * (main_waveform - min(main_waveform)) / (max(main_waveform) - min(main_waveform)) - 1;
        main_window = main_waveform(start_idx:end_idx);

        % Cross-correlation against 1000 selected only
        num_waveforms = length(Felix);
        window_length = end_idx - start_idx + 1;
        cc_values = zeros(num_waveforms, 1);
        cross_corrs = zeros(2 * window_length - 1, num_waveforms);

        parfor i = 1:num_waveforms
            if i == k || ~isfield(Felix(i), ['W_', field]) || isempty(Felix(i).(['W_', field]))
                continue;
            end
            other_wave = Felix(i).(['W_', field])(:);
            if length(other_wave) < end_idx
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
        figure; set(gcf, "Position", [1323    41 788 811]); hold on;
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

        avg_stack = mean(aligned_waveforms, 2);
        plot(t, avg_stack + 21 * vertical_offset, 'r', 'LineWidth', 1.5);
        text(t(1), 21 * vertical_offset, 'Stacked Average', 'Color', 'r', 'FontSize', 10);

        xlim([-0.25 0.5]); ylim([-1 23]);
        xlabel('Time (s)'); ylabel('Amplitude');
        grid on;
        plot([0, 0], [-1, 23], 'r', 'LineWidth', 2);
        plot([-0.096, -0.096], [-1, 23], 'g'); plot([0.096, 0.096], [-1, 23], 'g');

        % Manual picking
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

        % Store result and save
        Felix(k).([field, '_CCPo']) = P;
        close;
    end
    save(out_file, 'Felix');
end
