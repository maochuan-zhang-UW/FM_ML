% Script to multiply non-zero PoML_W_* and EQP_* fields by -1 for each station
% Input: Felix struct array with fields for 7 stations (AS1, AS2, CC1, EC1, EC2, EC3, ID1)
% Output: Modified Felix struct saved to a new .mat file

% Load the data
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/F_ML/A_wave_dB15_DT_CFM_EQP.mat')

% Define stations and fields to modify
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
fields = {'PoML_W_', 'EQP_'};

% Loop through all entries in Felix
for i = 1:length(Felix)
    for s = 1:length(stations)
        station = stations{s};
        for f = 1:length(fields)
            field_name = [fields{f} station];
            if Felix(i).(field_name) ~= 0
                Felix(i).(field_name) = Felix(i).(field_name) * -1;
            end
        end
    end
end

% Save the modified Felix struct to a new .mat file
save('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/F_ML/A_wave_dB15_DT_CFM_EQP_modified.mat', 'Felix');

% Display confirmation
fprintf('Non-zero PoML_W_* and EQP_* fields have been multiplied by -1.\n');
fprintf('Modified data saved to: %s\n', '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/F_ML/A_wave_dB15_DT_CFM_EQP_modified.mat');