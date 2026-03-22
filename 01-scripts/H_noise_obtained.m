% Create a new Felix structure with truncated waveforms
clear;
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_wave/A_wave_forML1sec200Hz.mat');
% Define the time window and total length
% List of all waveform fields
waveform_fields = {'W_AS1', 'W_AS2', 'W_CC1', 'W_EC1', 'W_EC2', 'W_EC3', 'W_ID1'};
% Define the time window and total length
start_sample = 1;      % Start at sample 1 (0 in your notation)
end_sample = 200;      % End at sample 200
% Process each element in the Felix structure
for i = 1:length(Felix)
    % Process each waveform field
    for j = 1:length(waveform_fields)
        field_name = waveform_fields{j};     
        if isempty(Felix(i).(field_name)) || length(Felix(i).(field_name))<100
            continue;
        else
            original_waveform = Felix(i).(field_name);
            Felix(i).(field_name)=[];
            truncated_waveform = original_waveform(start_sample:end_sample);
            Felix(i).(field_name)=truncated_waveform;
        end
    end
end