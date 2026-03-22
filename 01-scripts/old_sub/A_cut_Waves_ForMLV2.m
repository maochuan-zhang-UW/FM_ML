clear; close all;

fields = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};
groups = {'W1','W2','W3','W4','W5','W6','W7','W8','E1','E2','E3','E4','E5','E6','E7','E8'};

load('/Users/mczhang/Documents/GitHub/FM3/02-data/E_Po/E_All_simple.mat')

dt = 1/200; % Hz

%% Load parameters
parameterFile = 'parameter_script_realtimeVer1_MC_focal';
run(parameterFile); % Should load variable 'p'

if ~exist('p', 'var')
    error('Variable "p" not found after running %s', parameterFile);
end

%% Step A: obtain wave of event
P.a.sttime = -3;
P.a.edtime = 7;
P.a.window = [-1 1];
P.filt = 1; % Use the first filter band (e.g., [4 50] Hz)

% Initialize missing waveform fields
for i = 1:length(Felix)
    for kz = 1:length(fields)
        fname = ['W_', fields{kz}];
        if ~isfield(Felix(i), fname)
            Felix(i).(fname) = 0;  % or zeros(257,1) if you know the length
        end
    end
end

t_trace = linspace(P.a.sttime, P.a.edtime, 2001)';
idx_start = round((P.a.window(1) - P.a.sttime) / dt) + 1;
idx_end   = round((P.a.window(2) - P.a.sttime) / dt);

overall_tic = tic;
for kz = 1:length(fields)
    field = fields{kz};
    fprintf('Processing field %s...\n', field);

    t1 = tic;  % Timer for each field
    tempFelix = Felix;  % Avoid dynamic field assignment in loop

    for i = 1:100%length(Felix)
        temp = 0;  % default to zero

        DDt_field = ['DDt_', field];
        if isfield(Felix(i), DDt_field) & (Felix(i).(DDt_field) > -3)
            trace_Z = obtain_waveforms_Z(Felix(i), kz, P.a.sttime, P.a.edtime, p);

            if isfield(trace_Z, 'dataFilt') && size(trace_Z.dataFilt, 2) >= P.filt

                temp = trace_Z.dataFilt(idx_start:idx_end, P.filt);
            end
        else
            continue;
        end
        tempFelix(i).(['W_', field]) = temp;
    end

    Felix = tempFelix;
    fprintf('Elapsed time for field %s: %.2f seconds\n', field, toc(t1));
end

fprintf('Total processing time: %.2f seconds\n', toc(overall_tic));

% Save the results
save('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/E_wave_forML.mat', 'Felix');
