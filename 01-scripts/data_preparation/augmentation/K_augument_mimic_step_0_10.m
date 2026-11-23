clear; close all; clc;

%% ================================
%              CONFIG
% ================================
outDir = '/Users/mczhang/Documents/GitHub/FM_ML/02-data/K_aug/STEP010';
if ~exist(outDir,'dir'); mkdir(outDir); end

target_total = 15000;   % Desired total samples per station

% --- Load shared data ---
load('/Users/mczhang/Documents/GitHub/FM_ML/02-data/H_noi/H_noise_dB20_snrValue.mat'); % snrValues
load('/Users/mczhang/Documents/GitHub/FM_ML/02-data/H_noi/H_Noise_200.mat');           % Felix noise
load('/Users/mcZhang/Documents/GitHub/FM_ML/02-data/K_aug/Template_divide.mat')

% --- Station list ---
stations = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};

%% ================================
%       MAIN AUGMENTATION LOOP
% ================================
for s = 1:numel(stations)

    %% --------------------
    % Load station template
    %% --------------------
    station_name     = stations{s};
    station_template = [station_name '_T'];   % Auto-generate AS1_T, AS2_T, ...
    fprintf('\nProcessing %s ...\n', station_template);

    data = eval(station_template);
    N_orig = length(data);

    % Compute multiplier
    n_multi_trace = target_total / N_orig;
    fprintf('Original=%d, target=%d → multiplier = %.2f\n', ...
            N_orig, target_total, n_multi_trace);

    %% -------------------------
    % Fit SNR distribution model
    %% -------------------------
    snrData = snrValues{s};
    snrData = snrData(snrData > 0);
    %% -------------------------
%  Generate SNR from 10 to 30 dB (uniform)
%% -------------------------
%% -------------------------
%  Generate SNR from 0 to 10 dB (uniform)
%% -------------------------
numSamples = round(N_orig * n_multi_trace);

% Draw synthetic SNRs uniformly between [0, 10] dB
synthetic_SNR = 0 + (10 - 0) * rand(numSamples, 1);


    % pd = fitdist(snrData(:),'Lognormal');
    % 
    % numSamples = round(N_orig * n_multi_trace);
    % synthetic_SNR = random(pd, numSamples, 1);

    %% -------------------------
    % Prepare field names
    %% -------------------------
    wField   = ['W_'   station_name];
    manField = ['Man_' station_name];
    poField  = ['Po_'  station_name];
    snrField = ['SNR_' station_name];

    %% -------------------------
    % Start augmentation
    %% -------------------------
    Station_add = [];
    j = 1;   % index into synthetic_SNR
    n = 1;   % running ID2 counter

    % Loop over ALL samples (you changed to "1 : N_orig")
    for i = 1 : N_orig

        %% ---- Original waveform ----
        tmp = struct();
        tmp.ID2    = n; n = n + 1;
        tmp.ID     = data(i).ID;
        tmp.lon    = data(i).lon;
        tmp.lat    = data(i).lat;
        tmp.depth  = data(i).depth;
        tmp.(wField)   = data(i).(wField);
        tmp.(manField) = data(i).(manField);
        tmp.(poField)  = data(i).(poField);

        % Compute original SNR
        wf = tmp.(wField);
        tmp.(snrField) = 20*log10(rms(wf(81:160))/rms(wf(1:80)));

        Station_add = [Station_add tmp]; %#ok<AGROW>

        %% ---- Synthetic noisy copies ----
        for k = 1:n_multi_trace
            signal = data(i).(wField);
            rms_signal = rms(signal);

            % Target SNR
            target_snr_dB = synthetic_SNR(j);
            j = j + 1;

            % Convert SNR dB to noise RMS
            rms_noise_target = rms_signal / (10^(target_snr_dB/20));

            % Random valid Felix noise
            valid = false;
            while ~valid
                noisenum = randi([1, length(Felix)]);
                if isfield(Felix(noisenum), wField)
                    noise = Felix(noisenum).(wField);
                    if length(noise) == 200 && any(noise)
                        valid = true;
                    end
                end
            end

            % Scale noise
            rms_noise_actual = rms(noise);
            scale = rms_noise_target / rms_noise_actual;
            scaled_noise = scale * noise;

            % Create synthetic waveform
            synthetic = signal + scaled_noise;

            tmp2 = struct();
            tmp2.ID2    = n; n = n + 1;
            tmp2.ID     = data(i).ID;
            tmp2.lon    = data(i).lon;
            tmp2.lat    = data(i).lat;
            tmp2.depth  = data(i).depth;
            tmp2.(wField)   = synthetic;
            tmp2.(manField) = data(i).(manField);
            tmp2.(poField)  = data(i).(poField);

            % Compute synthetic SNR
            tmp2.(snrField) = 20*log10(rms(synthetic(81:160)) / rms(synthetic(1:80)));

            Station_add = [Station_add tmp2]; %#ok<AGROW>
        end
    end

    %% -------------------------
    % Save result
    %% -------------------------
    varName = [station_template '_add'];  % e.g., AS1_T_add
    eval([varName ' = Station_add;']);
    save(fullfile(outDir, [varName '_ST.mat']), varName);
    fprintf('Saved %s with %d entries\n', varName, numel(Station_add));
end

%% ================================
%       VISUALIZE SNR PDFs
% ================================
figure; set(gcf,"Position",[743 313 822 637]);
tiledlayout(3,3);

for s = 1:numel(stations)
    station_name = stations{s};
    snrData = snrValues{s};
    snrData = snrData(snrData > 0);

    pd = fitdist(snrData(:), 'Lognormal');

    nexttile;
histogram(synthetic_SNR,50,'Normalization','pdf')
title(station_name)
xlabel('SNR (dB)')
ylabel('PDF')
end
