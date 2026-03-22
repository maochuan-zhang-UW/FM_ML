clear; close all;

% --- Config ---
outDir = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/K_aug';
if ~exist(outDir,'dir'); mkdir(outDir); end

% --- Load shared data ---
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/H_Noi/H_noise_dB20_snrValue.mat'); % snrValues
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/H_Noi/H_Noise_200.mat');           % Felix noise

stations = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};
n_multi_trace=10;
for s = 1:numel(stations)
    station = stations{s};
    fprintf('\nProcessing %s ...\n', station);

    % --- Load station-specific data ---
    matFile = fullfile(outDir, [station '.mat']);
    load(matFile, station);   % loads struct named same as station
    data = eval(station);

    % --- Fit lognormal distribution to SNR data ---
    snrData = snrValues{s};
    snrData = snrData(snrData > 0);
    pd = fitdist(snrData(:),'Lognormal');

    % --- Generate synthetic SNR values (dB) ---
    if s==7
        n_multi_trace=n_multi_trace*3 ;
    end
    numSamples = length(data) * n_multi_trace;  % 10 noisy copies per waveform%30for ID1
    synthetic_SNR = random(pd, numSamples, 1);

    % --- Prepare field names ---
    wField   = ['W_' station];
    manField = ['Man_' station];
    poField  = ['Po_' station];
    snrField = ['SNR_' station];

    % --- Augment data ---
    Station_add = [];
    j = 1;  % index into synthetic_SNR
    n = 1;  % running ID2 counter

    %for i = 1:length(data)
    %for i = 1:ceil(length(data)*0.8)
    for i = ceil(length(data)*0.8):length(data)
        % --- Original ---
        tmp = struct();
        tmp.ID2    = n; n = n + 1;
        tmp.ID     = data(i).ID;
        tmp.lon    = data(i).lon;
        tmp.lat    = data(i).lat;
        tmp.depth  = data(i).depth;
        tmp.(wField)   = data(i).(wField);
        tmp.(manField) = data(i).(manField);
        tmp.(poField)  = data(i).(poField);

        % Compute SNR
        wf = tmp.(wField);
        noise_seg   = wf(1:80);
        signal_seg  = wf(81:160);
        tmp.(snrField) = 20*log10(rms(signal_seg)/rms(noise_seg));

        Station_add = [Station_add tmp]; %#ok<AGROW>

        % --- Noisy copies ---
        for k = 1:n_multi_trace
            signal = data(i).(wField);
            rms_signal = rms(signal);

            % Target SNR in dB
            target_snr_dB = synthetic_SNR(j); 
            j = j + 1;

            % Convert dB → noise RMS
            rms_noise_target = rms_signal / (10^(target_snr_dB/20));

            % Pick valid noise
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

            % Scale noise
            rms_noise_actual = rms(noise);
            scale = rms_noise_target / rms_noise_actual;
            scaled_noise = scale * noise;

            % Synthetic waveform
            synthetic = signal + scaled_noise;

            % Store
            tmp2 = struct();
            tmp2.ID2    = n; n = n + 1;
            tmp2.ID     = data(i).ID;
            tmp2.lon    = data(i).lon;
            tmp2.lat    = data(i).lat;
            tmp2.depth  = data(i).depth;
            tmp2.(wField)   = synthetic;
            tmp2.(manField) = data(i).(manField);
            tmp2.(poField)  = data(i).(poField);

            % Compute SNR
            noise_seg   = synthetic(1:80);
            signal_seg  = synthetic(81:160);
            tmp2.(snrField) = 20*log10(rms(signal_seg)/rms(noise_seg));

            Station_add = [Station_add tmp2]; %#ok<AGROW>
        end
    end

    % --- Save station-specific dataset ---
    varName = [station '_add'];
    eval([varName ' = Station_add;']);
    %save(fullfile(outDir, [varName '_V3_for_test.mat']), varName);

    fprintf('Saved %s.mat with %d entries\n', varName, numel(Station_add));
end

% --- Visualize SNR (noise) distribution for each station ---
figure; set(gcf,"Position",[743   313   822   637]);
tiledlayout(3,3);  % 7 stations -> a few empty tiles is fine

for s = 1:numel(stations)
    station = stations{s};

    % Get SNR data
    snrData = snrValues{s};
    snrData = snrData(snrData > 0);

    % Fit lognormal
    pd = fitdist(snrData(:), 'Lognormal');

    % Plot histogram + fitted PDF
    nexttile;
    histogram(snrData, 'Normalization','pdf'); hold on;
    x = linspace(min(snrData), max(snrData), 100);
    plot(x, pdf(pd, x), 'LineWidth', 2);

    title(sprintf('%s (\\mu=%.2f, \\sigma=%.2f)', ...
          station, pd.mu, pd.sigma));
    xlabel('SNR (dB)');
    ylabel('PDF');
end
