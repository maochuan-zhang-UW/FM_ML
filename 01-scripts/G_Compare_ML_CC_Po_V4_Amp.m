clc; clear;

load('/Users/mczhang/Documents/GitHub/FM3/02-data/E_Po/E_All_polished_updated_V2.mat');


stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
snr_data = [];

for s = 1:length(stations)
    station = stations{s};
    po_field = ['Po_' station];         % true polarity
    poml_field = ['PoML_W_' station];   % predicted polarity
    nsp_field = ['NSP_' station];       % [noise, s, p]

    for i = 1:length(Felix)
        if isempty(Felix(i).(po_field)) || isempty(Felix(i).(poml_field)) || isempty(Felix(i).(nsp_field))
            continue;
        end

        po = Felix(i).(po_field);
        poml = Felix(i).(poml_field);
        nsp = Felix(i).(nsp_field);

        if length(nsp) < 3 || po == 0
            continue;
        end

        % Convert poml to numeric
        if ischar(poml)
            if strcmp(poml, 'U')
                poml = 1;
            elseif strcmp(poml, 'D')
                poml = -1;
            else
                poml = 0;
            end
        end

        % Calculate SNR only if poml is valid
        if poml == 0
            continue;
        end

        noise_val = nsp(1);
        p_val = nsp(3);

        if noise_val <= 0 || p_val <= 0 || noise_val==1
            continue;
        end

        %snr_db = 10 * log10(p_val / noise_val);
        snr_db = p_val;

        match = double(po == poml);

        snr_data = [snr_data; struct( ...
            'Station', station, ...
            'SNR_dB', snr_db, ...
            'Match', match)];
    end
end

% Convert to table for analysis
SNR_table = struct2table(snr_data);

% Bin the SNR and calculate accuracy per bin
edges = 0:2:20;
SNR_table.SNR_bin = discretize(SNR_table.SNR_dB, edges);

bin_centers = edges(1:end-1) + diff(edges)/2;
accuracy = zeros(size(bin_centers));
for b = 1:length(bin_centers)
    idx = SNR_table.SNR_bin == b;
    if sum(idx) > 0
        accuracy(b) = mean(SNR_table.Match(idx));
    else
        accuracy(b) = NaN;
    end
end

% Plot accuracy vs. SNR
figure;
plot(bin_centers, accuracy * 100, '-o');
xlabel('SNR (dB)');
ylabel('Accuracy (%)');
title('Accuracy vs. SNR (dB)');
grid on;


% Convert struct array to table
SNR_table = struct2table(snr_data);

% Define SNR bins
edges = 0:20:200;
bin_centers = edges(1:end-1) + diff(edges)/2;

stations = unique(SNR_table.Station);

figure;
hold on;

% Loop through stations
for i = 1:length(stations)
    station = stations{i};

    % Filter data for this station
    T = SNR_table(strcmp(SNR_table.Station, station), :);

    % Bin SNR and compute accuracy in each bin
    T.SNR_bin = discretize(T.SNR_dB, edges);
    acc = nan(size(bin_centers));
    for b = 1:length(bin_centers)
        idx = T.SNR_bin == b;
        if any(idx)
            acc(b) = mean(T.Match(idx));
        end
    end

    % Plot
    plot(bin_centers, acc * 100, '-o', 'DisplayName', station);
end

xlabel('SNR (dB)');
ylabel('Accuracy (%)');
title('Accuracy vs. SNR by Station');
legend('Location', 'best');
grid on;


%% SNR to accurancy in each station
SNR_table = struct2table(snr_data);

% Define SNR bins
edges = 0:20:4000;
bin_centers = edges(1:end-1) + diff(edges)/2;

stations = unique(SNR_table.Station);

figure;
hold on;

% Loop through stations
for i = 1:length(stations)
    station = stations{i};

    % Filter data for this station
    T = SNR_table(strcmp(SNR_table.Station, station), :);

    % Bin SNR and compute accuracy in each bin
    T.SNR_bin = discretize(T.SNR_dB, edges);
    acc = nan(size(bin_centers));
    for b = 1:length(bin_centers)
        idx = T.SNR_bin == b;
        if any(idx)
            acc(b) = mean(T.Match(idx));
        end
    end

    % Plot
    plot(bin_centers, acc * 100, '-o', 'DisplayName', station);
end

xlabel('amplitude');
ylabel('Accuracy (%)');
title('Accuracy vs. AMP by Station');
legend('Location', 'best');
grid on;

%% mean SNR in each station
SNR_table = struct2table(snr_data);

% Get unique station names
stations = unique(SNR_table.Station);

% Initialize result storage
avg_snr = zeros(length(stations), 1);
std_snr = zeros(length(stations), 1);

% Loop to compute average and std for each station
for i = 1:length(stations)
    station = stations{i};
    idx = strcmp(SNR_table.Station, station);
    avg_snr(i) = mean(SNR_table.SNR_dB(idx), 'omitnan');
    std_snr(i) = std(SNR_table.SNR_dB(idx), 'omitnan');
end

% Display table
T = table(stations, avg_snr, std_snr, 'VariableNames', {'Station', 'Mean_SNR_dB', 'Std_SNR_dB'});
disp(T);

% Optional: bar plot of mean SNR
figure;
bar(categorical(stations), avg_snr);
ylabel('Mean SNR (dB)');
xlabel('Station');
title('Average SNR per Station');
grid on;


 % Station    Mean_SNR_dB    Std_SNR_dB
 %    _______    ___________    __________
 % 
 %    {'AS1'}      7.2669         4.4142  
 %    {'AS2'}      7.3414         4.5629  
 %    {'CC1'}      8.0694         5.1612  
 %    {'EC1'}      7.4552         4.8745  
 %    {'EC2'}      7.4255         4.5829  
 %    {'EC3'}      6.8979         4.5136  
 %    {'ID1'}      6.7386         4.3471  