clear; close all;

% --- Config ---
outDir = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/K_aug';
if ~exist(outDir,'dir'); mkdir(outDir); end

% --- Load shared data ---
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/H_Noi/H_noise_dB20_snrValue.mat'); % snrValues
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/H_Noi/H_Noise_200.mat');           % Felix noise

stations = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};

% ============================================================
% === Set a target number of waveforms for ALL STATIONS ======
% ============================================================
target_total = 10000;        % <--- YOU SET THIS
station_target_SNR = cell(7,1);

for s = 1:numel(stations)
    station = stations{s};
    fprintf('\nProcessing %s ...\n', station);

    % --- Load station-specific data ---
    matFile = fullfile(outDir, [station '.mat']);
    load(matFile, station);   % loads struct with same name
    data = eval(station);

    % ============================================================
    % === Compute how many noisy copies are needed for THIS station
    % ============================================================
    N0 = length(data);
    local_multi_trace = max(1, floor(target_total / N0));   % auto-balanced
    fprintf('Original N = %d → multiplier = %d → final ≈ %d\n', ...
            N0, local_multi_trace, N0*(local_multi_trace+1));


    % =====================================================
    % --- Create Step SNR Distribution for THIS Station ---
    % =====================================================
    numSamples = length(data) * local_multi_trace;

    u = rand(numSamples, 1);
    target_SNR = zeros(numSamples,1);

    idx_low  = (u <= 0.10);                    % 10% in [0–5]
    idx_mid  = (u > 0.10 & u <= 0.90);         % 80% in [5–35]
    idx_high = (u > 0.90);                     % 10% in [35–50]

    target_SNR(idx_low)  = 0  + 5  * rand(sum(idx_low), 1);
    target_SNR(idx_mid)  = 5  + 30 * rand(sum(idx_mid), 1);
    target_SNR(idx_high) = 35 + 15 * rand(sum(idx_high),1);

    % store for plotting later
    station_target_SNR{s} = target_SNR;

    % --- Prepare field names ---
    wField   = ['W_' station];
    manField = ['Man_' station];
    poField  = ['Po_' station];
    snrField = ['SNR_' station];

    % --- Augment data ---
    Station_add = [];
    j = 1;  
    n = 1;  

    for i = ceil(length(data)*0.8):length(data)

        % ---- original waveform ----
        tmp = struct();
        tmp.ID2    = n; n = n + 1;
        tmp.ID     = data(i).ID;
        tmp.lon    = data(i).lon;
        tmp.lat    = data(i).lat;
        tmp.depth  = data(i).depth;
        tmp.(wField)   = data(i).(wField);
        tmp.(manField) = data(i).(manField);
        tmp.(poField)  = data(i).(poField);

        wf = tmp.(wField);
        noise_seg  = wf(1:80);
        signal_seg = wf(81:160);
        tmp.(snrField) = 20*log10(rms(signal_seg)/rms(noise_seg));

        Station_add = [Station_add tmp]; %#ok<AGROW>

        % ---- noisy copies ----
        for k = 1:local_multi_trace
            signal = data(i).(wField);
            rms_signal = rms(signal);

            target_snr_dB = target_SNR(j);
            j = j + 1;

            rms_noise_target = rms_signal / (10^(target_snr_dB/20));

            valid = false;
            while ~valid
                noisenum = randi([1, length(Felix)]);
                if isfield(Felix(noisenum),wField) && ~isempty(Felix(noisenum).(wField))
                    noise = Felix(noisenum).(wField);
                    if length(noise) == 200 && any(noise)
                        valid = true;
                    end
                end
            end

            rms_noise_actual = rms(noise);
            scale = rms_noise_target / rms_noise_actual;
            synthetic = signal + scale * noise;

            tmp2 = struct();
            tmp2.ID2    = n; n = n + 1;
            tmp2.ID     = data(i).ID;
            tmp2.lon    = data(i).lon;
            tmp2.lat    = data(i).lat;
            tmp2.depth  = data(i).depth;
            tmp2.(wField)   = synthetic;
            tmp2.(manField) = data(i).(manField);
            tmp2.(poField)  = data(i).(poField);

            noise_seg  = synthetic(1:80);
            signal_seg = synthetic(81:160);
            tmp2.(snrField) = 20*log10(rms(signal_seg)/rms(noise_seg));

            Station_add = [Station_add tmp2]; %#ok<AGROW>
        end
    end

    % --- Save station-specific dataset ---
    varName = [station '_add'];
    eval([varName ' = Station_add;']);
    save(fullfile(outDir, [varName '_V5_TMSF.mat']), varName);
    fprintf('Saved %s.mat with %d entries\n', varName, numel(Station_add));
end

% ============================================================
% --- Plot Designed Target SNR Distribution for 7 Stations ---
% ============================================================
figure;
set(gcf,"Position",[700 300 900 700]);
tiledlayout(3,3);

binEdges = 0:1:50;

for s = 1:numel(stations)
    nexttile;

    snrS = station_target_SNR{s};   % SNR distribution for station s
    histogram(snrS, 'BinEdges', binEdges, 'Normalization','probability');
    hold on;

    % vertical lines for boundaries
    xline(5,'r--','5 dB');
    xline(35,'b--','35 dB');

    xlabel('Target SNR (dB)');
    ylabel('Probability');
    title(sprintf('%s Target SNR', stations{s}));
    grid on;

    % Print proportions
    low  = mean(snrS >= 0  & snrS < 5);
    mid  = mean(snrS >= 5  & snrS < 35);
    high = mean(snrS >= 35 & snrS <= 50);

    fprintf('\n%s proportions:\n', stations{s});
    fprintf('   [0,5)   : %.3f\n', low);
    fprintf('   [5,35)  : %.3f\n', mid);
    fprintf('   [35,50] : %.3f\n', high);
end

sgtitle('Custom Target SNR Distribution for All 7 Stations (10/80/10)');
set(gcf,'Color','w');
