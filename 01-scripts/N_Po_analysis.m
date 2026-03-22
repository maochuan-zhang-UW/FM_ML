% List of stations
clear;close all;
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};

% Base path where your files are stored
basePath = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/N_Po/';

% Preallocate results
results = cell(length(stations), 4);

for k = 1:length(stations)
    st = stations{k};
    
    % Build filename and load
    filename = fullfile(basePath, ['N_Po' st 'V4.mat']);
    load(filename, 'Felix');
    
    % Field names
    ccField  = ['CC_'  st];
    manField = ['Man_' st];

    % Initialize counters
    m = 0; % valid
    n = 0; % same

    % Loop through structs
    for i = 1:length(Felix)
        if isfield(Felix, ccField) && isfield(Felix, manField)
            ccVal  = Felix(i).(ccField);
            manVal = Felix(i).(manField);

            if ~isempty(ccVal) && ccVal ~= 0
                m = m + 1; % valid
                if ccVal == manVal
                    n = n + 1; % match
                end
            end
        end
    end

    % Compute ratio
    ratio = n / m;

    % Save results
    results{k,1} = st;
    results{k,2} = m;
    results{k,3} = n;
    results{k,4} = ratio;

    % Print
    fprintf('%s -> Total valid = %d, Same = %d, Ratio = %.4f\n', ...
        st, m, n, ratio);
end

% Convert to table for easier handling
T = cell2table(results, 'VariableNames', {'Station','TotalValid','Same','Ratio'});
disp(T);
