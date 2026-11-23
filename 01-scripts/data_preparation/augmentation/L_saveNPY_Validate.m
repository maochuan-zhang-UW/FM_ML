clear; close all; clc;

%% ==========================================
%                  CONFIG
% ===========================================
outDir = '/Users/mczhang/Documents/GitHub/FM_ML/02-data/K_aug/Val';
if ~exist(outDir,'dir'); mkdir(outDir); end

target_total = 5000;   % desired per station

% --- Load shared data ---
load('/Users/mczhang/Documents/GitHub/FM_ML/02-data/H_noi/H_noise_dB20_snrValue.mat'); 
load('/Users/mczhang/Documents/GitHub/FM_ML/02-data/H_noi/H_Noise_200.mat');           
load('/Users/mcZhang/Documents/GitHub/FM_ML/02-data/K_aug/Template_divide.mat')

stations = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};

%% ==========================================
%     ACCUMULATORS FOR FULL TEST SET
% ==========================================
X_all = [];
y_all = [];

%% ==========================================
%              MAIN LOOP
% ==========================================
for s = 1:numel(stations)

    station = stations{s};
    station_template = [station '_V'];
    fprintf('\n============================\nProcessing %s\n', station_template);

    data = eval(station_template);
    N_orig = numel(data);

    n_multi_trace = target_total / N_orig;
    fprintf('Original %d → multiplier %.2f\n', N_orig, n_multi_trace);

    %% --- Fit SNR distro ---
    snrData = snrValues{s};
    snrData = snrData(snrData>0);
    pd = fitdist(snrData(:),'Lognormal');

    numSamples = round(N_orig*n_multi_trace);
    synthetic_SNR = random(pd, numSamples, 1);

    %% --- Field names ---
    wField   = ['W_' station];
    manField = ['Man_' station];
    poField  = ['Po_' station];
    snrField = ['SNR_' station];

    Station_add = [];
    j = 1; 
    n = 1;

    %% -------------------------
    %   AUGMENTATION
    % -------------------------
    for i = 1:N_orig

        %% ---- Original ----
        tmp = struct();
        tmp.ID2   = n; n=n+1;
        tmp.ID    = data(i).ID;
        tmp.lon   = data(i).lon;
        tmp.lat   = data(i).lat;
        tmp.depth = data(i).depth;

        wf = data(i).(wField);
        tmp.(wField)   = wf;
        tmp.(manField) = data(i).(manField);
        tmp.(poField)  = data(i).(poField);
        tmp.(snrField) = 20*log10(rms(wf(81:160))/rms(wf(1:80)));

        Station_add = [Station_add tmp];

        %% ---- Synthetic copies ----
        for k = 1:n_multi_trace

            signal = data(i).(wField);
            rms_signal = rms(signal);

            target_snr_dB = synthetic_SNR(j);
            j = j + 1;

            rms_noise_target = rms_signal / (10^(target_snr_dB/20));

            valid = false;
            while ~valid
                noisenum = randi([1 length(Felix)]);
                if isfield(Felix(noisenum), wField)
                    noise = Felix(noisenum).(wField);
                    if length(noise)==200 && any(noise)
                        valid = true;
                    end
                end
            end

            scale = rms_noise_target / rms(noise);
            synthetic = signal + scale*noise;

            tmp2 = struct();
            tmp2.ID2   = n; n=n+1;
            tmp2.ID    = data(i).ID;
            tmp2.lon   = data(i).lon;
            tmp2.lat   = data(i).lat;
            tmp2.depth = data(i).depth;

            tmp2.(wField)   = synthetic;
            tmp2.(manField) = data(i).(manField);
            tmp2.(poField)  = data(i).(poField);
            tmp2.(snrField) = 20*log10(rms(synthetic(81:160))/rms(synthetic(1:80)));

            Station_add = [Station_add tmp2];
        end
    end

    %% SAVE AUGMENTED STATION (OPTIONAL)
    %save(fullfile(outDir,[station_template '_add.mat']),"Station_add");

    %% -----------------------------------
    %    EXTRACT 64-SAMPLE + NORMALIZE
    % -----------------------------------
    for i = 1:numel(Station_add)
        w = Station_add(i).(wField);

        center = 100;
        w64 = w(center-31:center+32);

        maxv = max(abs(w64));
        if maxv==0, maxv=1; end
        w64 = w64/maxv;

        pol = Station_add(i).(manField);
        pol = (pol == 1);   % -1→0, +1→1

        X_all = [X_all; reshape(w64,1,64)];
        y_all = [y_all; pol];
    end

    fprintf('  → Added %d samples to dataset\n', numel(Station_add));

end

%% ==========================================
%      FINAL OUTPUT (ALL ARE TEST)
% ==========================================
timeseries = X_all;
polarities = y_all;

fprintf('\nTOTAL validated samples: %d\n', size(timeseries,1));

%% --- save as NPZ (Python-compatible)
save(fullfile(outDir,'timeseries.npy'), 'timeseries','-v7.3');
save(fullfile(outDir,'polarities.npy'),'polarities','-v7.3');

fprintf('\n=====================================\n');
fprintf(' ✔ Saved validated.npy\n ✔ Saved polarities.npy\n');
fprintf('=====================================\n');
