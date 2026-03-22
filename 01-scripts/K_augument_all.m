clear; close all;
% Load data
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/D_man/D_man_dB20_polish.mat');

numStructs = length(Felix);
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};

outDir = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/K_aug';
if ~exist(outDir,'dir')
    mkdir(outDir);
end

for j = 1:numel(stations)
    station = stations{j};
    wField   = ['W_' station];
    manField = ['Man_' station];
    poField  = ['Po_' station];
    
    AugData = [];   % start empty
    n = 1;          % running index for ID2
    
    for i = 1:numStructs
        if isfield(Felix(i), wField) && ~isempty(Felix(i).(wField))
            % --- Original ---
            tmp = struct();
            tmp.ID2       = n; n = n + 1;
            tmp.ID        = Felix(i).ID;
            tmp.lon       = Felix(i).lon;
            tmp.lat       = Felix(i).lat;
            tmp.depth     = Felix(i).depth;
            tmp.(wField)  = Felix(i).(wField);
            tmp.(manField)= Felix(i).(manField);
            tmp.(poField) = Felix(i).(poField);
            AugData = [AugData tmp];   %#ok<AGROW>
            
            % --- Copy & flip waveform + Man polarity ---
            tmp2 = struct();
            tmp2.ID2       = n; n = n + 1;
            tmp2.ID        = Felix(i).ID;
            tmp2.lon       = Felix(i).lon;
            tmp2.lat       = Felix(i).lat;
            tmp2.depth     = Felix(i).depth;
            tmp2.(wField)  = -Felix(i).(wField);
            tmp2.(manField)= -Felix(i).(manField);
            tmp2.(poField) = Felix(i).(poField);
            AugData = [AugData tmp2];  %#ok<AGROW>
        end
    end
    
    % Save this station’s struct (only its fields)
    eval([station ' = AugData;']);
    save(fullfile(outDir, [station '.mat']), station);
    
    fprintf('Saved %s.mat with %d entries\n', station, numel(AugData));
end
