function traceZ = obtain_waveforms_Z(ph2dt, ind_station, st1, st2, p)
% Obtain waveform for Z channel only
stations = {'AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3', 'AXID1'};
channelsZ = {'EHZ', 'EHZ', 'HHZ', 'EHZ', 'HHZ', 'EHZ', 'EHZ'};

% Select station and channel
station = stations{ind_station};
channelZ = channelsZ{ind_station};

% Construct file path
d = ph2dt.on;
fileData = fullfile(p.dir.data, datestr(d, 'yyyy'), datestr(d, 'mm'), ...
    [datestr(d, 'yyyy-mm-dd-HH-00-00') '.mat']);

% Load data
if exist(fileData, 'file') == 2
    trace1 = load(fileData);
    trace1 = trace1.trace;
else
    traceZ = [];
    return;
end

% Process waveform
if ~isempty(trace1)
    phaseT = ph2dt.(['DDt_', station(3:end)]);
    tlim = [st1 st2];
    timeWindow = ph2dt.on + (phaseT + tlim) / 86400;

    % Subset Z channel only
    traceZ = subset_trace(trace1, timeWindow, station, channelZ, [], 0);
    clear trace1;

    % Validate trace
    if isempty(traceZ) || length(traceZ.data) < 700
        traceZ = [];
        return;
    end

    % Apply filtering and demeaning
    % for i = 1%:length(traceZ)
    %     for j = 1%:length(p.filt)
                traceZ.dataFilt= trace_filter(traceZ.data, p.filt(1), traceZ.sampleRate);
                traceZ.dataFilt= traceZ.dataFilt - mean(traceZ.dataFilt);
    %     end
    % end
else
    traceZ = [];
end
end
