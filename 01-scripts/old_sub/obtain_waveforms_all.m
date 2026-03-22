function [traceZ, traceN, traceE] = obtain_waveforms_all(ph2dt, ind_station, st1, st2, p)
% Obtain waveforms for Z, N, E channels in one call
stations = {'AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3', 'AXID1'};
channelsZ = {'EHZ', 'EHZ', 'HHZ', 'EHZ', 'HHZ', 'EHZ', 'EHZ'};
channelsN = {'EHN', 'EHN', 'HHN', 'EHN', 'HHN', 'EHN', 'EHN'};
channelsE = {'EHE', 'EHE', 'HHE', 'EHE', 'HHE', 'EHE', 'EHE'};

% Select station and channels
station = stations{ind_station};
channelZ = channelsZ{ind_station};
channelN = channelsN{ind_station};
channelE = channelsE{ind_station};

% Construct file path
d = ph2dt.on;
fileData = fullfile(p.dir.data, datestr(d, 'yyyy'), datestr(d, 'mm'), ...
    [datestr(d, 'yyyy-mm-dd-HH-00-00') '.mat']);

% Load data
if exist(fileData, 'file') == 2
    trace1 = load(fileData);
    trace1 = trace1.trace;
else
    trace1 = [];
end

% Process waveforms
if ~isempty(trace1)
    phaseT = ph2dt.(['DDt_', station(3:end)]);
    tlim = [st1 st2];
    timeWindow = ph2dt.on + (phaseT + tlim) / 86400;
    
    % Subset all channels at once
    traceZ = subset_trace(trace1, timeWindow, station, channelZ, [], 0);
    traceN = subset_trace(trace1, timeWindow, station, channelN, [], 0);
    traceE = subset_trace(trace1, timeWindow, station, channelE, [], 0);
    
    clear trace1; % Free memory
    
    % Validate traces
    if isempty(traceZ) || isempty(traceN) || isempty(traceE) || ...
            length(traceZ.data) < 700 || length(traceN.data) < 700 || length(traceE.data) < 700
        traceZ = []; traceN = []; traceE = [];
        return;
    end
    
    % Apply filtering and demeaning
    for i = 1:length(traceZ)
        for j = 1:length(p.filt)
            try
                traceZ(i).dataFilt(:, j) = trace_filter(traceZ(i).data, p.filt(j), traceZ(i).sampleRate);
                traceN(i).dataFilt(:, j) = trace_filter(traceN(i).data, p.filt(j), traceN(i).sampleRate);
                traceE(i).dataFilt(:, j) = trace_filter(traceE(i).data, p.filt(j), traceE(i).sampleRate);
                
                % Demean
                traceZ(i).dataFilt(:, j) = traceZ(i).dataFilt(:, j) - mean(traceZ(i).dataFilt(:, j));
                traceN(i).dataFilt(:, j) = traceN(i).dataFilt(:, j) - mean(traceN(i).dataFilt(:, j));
                traceE(i).dataFilt(:, j) = traceE(i).dataFilt(:, j) - mean(traceE(i).dataFilt(:, j));
            catch
                traceZ = []; traceN = []; traceE = [];
                return;
            end
        end
    end
else
    traceZ = []; traceN = []; traceE = [];
end
end