clear; % Clear workspace
% Load data
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_wave/A_wave_forML1sec100hz.mat');
numStructs = length(Felix);
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
numStations = length(stations);
snrThreshold = 15; % dB threshold

for i = 1:numStructs
    for j = 1:numStations
        nspField = ['NSP_' stations{j}];
        wField   = ['W_' stations{j}];
        
        if isfield(Felix(i), nspField) && isfield(Felix(i), wField)
            nsp = Felix(i).(nspField);
            
            % Check valid noise & signal power
            if length(nsp) >= 3 && nsp(1) > 0 && nsp(3) > 0
                SNR_dB = 10 * log10(nsp(3) / nsp(1));
                
                % If SNR < threshold, remove this waveform
                if SNR_dB < snrThreshold
                    Felix(i).(wField)   = [];   % clear waveform
                    Felix(i).(nspField) = [];   % optional: also clear NSP
                end
            end
        end
    end
end

% --- Remove Felix entries if ALL W-stations are empty ---
keepMask = true(1, numStructs); % logical mask to track which to keep
for i = 1:numStructs
    allEmpty = true;
    for j = 1:numStations
        wField = ['W_' stations{j}];
        if isfield(Felix(i), wField) && ~isempty(Felix(i).(wField))
            allEmpty = false;
            break;
        end
    end
    if allEmpty
        keepMask(i) = false;
    end
end

Felix = Felix(keepMask); % keep only those with at least one W_station

