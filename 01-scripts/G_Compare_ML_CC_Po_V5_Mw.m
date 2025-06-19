clc; clear;

% Load Felix struct
load('/Users/mczhang/Documents/GitHub/FM3/02-data/E_Po/E_All_polished_updated_V2.mat');

% Load Kaiwen's magnitude file
filename = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/Axia_kaiwen.txt';
data = readmatrix(filename);

% Extract IDs and magnitudes
IDs = data(:, end);
Mags = data(:, end-2);  % assuming magnitude is 3rd last column

% Add Mw to Felix
id_list = [Felix.ID];
[common_IDs, ia, ib] = intersect(id_list, IDs);
for i = 1:length(ia)
    Felix(ia(i)).Mw = Mags(ib(i));
end

% Now analyze accuracy vs magnitude per station
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
mag_bins = -0.6:0.2:2.5;
bin_centers = mag_bins(1:end-1) + 0.1;

figure;
hold on;

for s = 1:length(stations)
    station = stations{s};
    po_field = ['Po_' station];
    poml_field = ['PoML_W_' station];

    mags = [];
    matches = [];

    for i = 1:length(Felix)
        if ~isfield(Felix(i), 'Mw') || isempty(Felix(i).(po_field)) || isempty(Felix(i).(poml_field))
            continue;
        end

        po = Felix(i).(po_field);
        poml = Felix(i).(poml_field);
        mw = Felix(i).Mw;

        if po == 0 || isempty(mw) 
            continue;
        end

        if ischar(poml)
            if strcmp(poml, 'U')
                poml = 1;
            elseif strcmp(poml, 'D')
                poml = -1;
            else
                poml = 0;
            end
        end

        if poml == 0
            continue;
        end

        mags(end+1) = mw;
        matches(end+1) = double(po == poml);
    end

    % Bin by magnitude
    accuracy = zeros(size(bin_centers));
    for b = 1:length(mag_bins)-1
        idx = mags >= mag_bins(b) & mags < mag_bins(b+1);
        if sum(idx) > 0
            accuracy(b) = mean(matches(idx));
        else
            accuracy(b) = NaN;
        end
    end

    plot(bin_centers, accuracy * 100, '-o', 'DisplayName', station);
end

xlabel('Magnitude (Mw)');
ylabel('Accuracy (%)');
title('Polarity Accuracy vs. Magnitude');
legend('show');
grid on;
