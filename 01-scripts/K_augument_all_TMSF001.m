clear; close all;

% --- Config ---
outDir = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/K_aug';
if ~exist(outDir,'dir'); mkdir(outDir); end

% --- Load shared data ---
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/H_Noi/H_noise_dB20_snrValue.mat'); % snrValues
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/H_Noi/H_Noise_200.mat');     % Felix noise
stations = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};

desired_total_per_station = 5000;  % <<< CHANGE HERE >>>

% Pre-allocate summary table
summaryTable = table('Size',[numel(stations),6], ...
    'VariableTypes',{'string','double','double','double','double','string'}, ...
    'VariableNames',{'Station','OriginalUsed','n_multi_trace','TotalTraces','AugmentationFactor','Note'});

%% Main loop over stations
for s = 1:numel(stations)
    station = stations{s};
    fprintf('\n=== Processing %s ===\n', station);

    % Load station data
    matFile = fullfile(outDir, [station '.mat']);
    if ~isfile(matFile)
        warning('File not found: %s → skipped', matFile);
        continue;
    end
    load(matFile, station);
    data = eval(station);

    % Use only last 20% as real events (your original logic)
    idx_start = ceil(0.8 * length(data));
    n_original_used = length(data) - idx_start + 1;
    if n_original_used == 0
        warning('No events after 80%% split for %s', station);
        continue;
    end

    % Auto-compute augmentation factor
    n_multi_trace = max(1, round((desired_total_per_station / n_original_used) - 1));
    note = '';
    if strcmp(station, 'ID1')
        n_multi_trace = n_multi_trace * 3;
        note = 'ID1 x3 boost';
    end

    total_expected = n_original_used * (1 + n_multi_trace);
    aug_factor = total_expected / n_original_used;

    fprintf('Original events used   : %d\n', n_original_used);
    fprintf('Noisy copies per event : %d\n', n_multi_trace);
    fprintf('Total traces expected  : %d (×%.2f)\n', total_expected, aug_factor);

    % Store in summary
    %summaryTable{s,:) = {station, n_original_used, n_multi_trace, total_expected, aug_factor, note};

    % Fit lognormal SNR distribution
    snrData = snrValues{s}(snrValues{s} > 0 & isfinite(snrValues{s}));
    pd = fitdist(snrData(:),'Lognormal');

    % Generate synthetic SNRs
    synthetic_SNR = random(pd, n_original_used * n_multi_trace, 1);

    % Field names
    wField   = ['W_'   station];
    manField = ['Man_' station];
    poField  = ['Po_'  station];
    snrField = ['SNR_' station];

    % Pre-define time vector (200 Hz → 200 samples = 1 second)
    t = (0:199)' / 200;  % seconds

    Station_add = [];
    snr_idx     = 1;     % index into synthetic_SNR
    global_id   = 1;     % unique ID2

    for i = idx_start:length(data)
        % --- 1. Keep original clean trace ---
        tmp = struct();
        tmp.ID2   = global_id;  global_id = global_id + 1;
        tmp.ID    = data(i).ID;
        tmp.lon   = data(i).lon;
        tmp.lat   = data(i).lat;
        tmp.depth = data(i).depth;
        tmp.(wField)   = data(i).(wField);
        tmp.(manField) = data(i).(manField);
        tmp.(poField)  = data(i).(poField);

        wf = tmp.(wField);
        tmp.(snrField) = 20*log10( rms(wf(81:160)) / (rms(wf(1:80)) + eps) );
        Station_add = [Station_add tmp];

        % --- 2. Create n_multi_trace noisy + time-shifted versions ---
        signal_clean = data(i).(wField)(1:200);  % ensure length 200

        for k = 1:n_multi_trace
            % Random time shift: N(0, 0.01²) seconds
            dt_sec = 0.01 * randn();              % std = 0.01 s
            t_shifted = t + dt_sec;

            % High-quality shift via spline interpolation
            signal_shifted = interp1(t, signal_clean, t_shifted, 'spline', 'extrap');
            if any(isnan(signal_shifted))
                signal_shifted = interp1(t, signal_clean, t_shifted, 'linear', 0);
            end

            % Target SNR
            target_snr_dB = synthetic_SNR(snr_idx);
            snr_idx = snr_idx + 1;
            rms_signal = rms(signal_shifted(81:160));
            rms_noise_target = rms_signal / (10^(target_snr_dB/20));

            % Pick real Felix noise
            valid = false; attempts = 0;
            while ~valid && attempts < 200
                idx_n = randi(numel(Felix));
                if isfield(Felix(idx_n), wField) && ~isempty(Felix(idx_n).(wField))
                    noise_cand = Felix(idx_n).(wField);
                    if numel(noise_cand) >= 200 && any(noise_cand(1:200)~=0)
                        noise_raw = noise_cand(1:200);
                        valid = true;
                    end
                end
                attempts = attempts + 1;
            end
            if ~valid
                noise_raw = randn(200,1) * 0.01;
            end

            % Scale noise to target RMS
            scale = rms_noise_target / (rms(noise_raw) + eps);
            scaled_noise = scale * noise_raw;

            % Final synthetic trace
            synthetic_trace = signal_shifted + scaled_noise;

            % Build struct
            tmp2 = struct();
            tmp2.ID2   = global_id;  global_id = global_id + 1;
            tmp2.ID    = data(i).ID;
            tmp2.lon   = data(i).lon;
            tmp2.lat   = data(i).lat;
            tmp2.depth = data(i).depth;
            tmp2.(wField)   = synthetic_trace;
            tmp2.(manField) = data(i).(manField);
            tmp2.(poField)  = data(i).(poField);
            tmp2.(snrField) = 20*log10( rms(synthetic_trace(81:160)) / (rms(synthetic_trace(1:80))+eps) );
            tmp2.time_shift_sec = dt_sec;  % optional: keep track

            Station_add = [Station_add tmp2];
        end
    end

    % Save
    varName = [station '_add'];
    eval([varName ' = Station_add;']);
    save(fullfile(outDir, [varName '_V5_TMSF.mat']), varName, '-v7.3');
    fprintf('→ Saved %s with %d traces\n', varName, numel(Station_add));
end

%% Final Summary
disp(' ');
disp('================== AUGMENTATION SUMMARY (with Time Shift) ==================');
%disp(summaryTable);
disp('===================================================================');

%% Plot SNR distributions
figure('Position',[100 100 1100 700]);
tiledlayout(3,3,'TileSpacing','compact');
for s = 1:numel(stations)
    station = stations{s};
    snrData = snrValues{s};
    snrData = snrData(snrData > 0 & isfinite(snrData));
    if isempty(snrData); continue; end
    pd = fitdist(snrData(:),'Lognormal');
    nexttile;
    histogram(snrData,'Normalization','pdf','BinWidth',1); hold on;
    xval = linspace(0, max(snrData)+10, 300);
    plot(xval, pdf(pd,xval),'r-','LineWidth',2);
    title(sprintf('%s (n=%d)\nμ=%.2f σ=%.2f',station,numel(snrData),pd.mu,pd.sigma));
    xlabel('SNR (dB)'); ylabel('PDF'); grid on;
end
sgtitle('Original SNR Distributions (Lognormal Fit)');

%% Optional: Show one example of time-shift augmentation
if exist('signal_clean','var')
    figure;
    plot(t, signal_clean, 'b', 'LineWidth',1.5, 'DisplayName','Original');
    hold on;
    plot(t, signal_shifted, 'g--', 'LineWidth',1.5, 'DisplayName','Time-shifted');
    plot(t, synthetic_trace, 'r', 'LineWidth',1.2, 'DisplayName','+ Noise (final)');
    legend('Location','best');
    grid on; xlabel('Time (s)'); ylabel('Amplitude');
    title(sprintf('Example Augmentation — Shift = %.3f s, SNR = %.1f dB', dt_sec, target_snr_dB));
end