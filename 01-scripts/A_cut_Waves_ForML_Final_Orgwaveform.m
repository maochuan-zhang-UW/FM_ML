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
P.a.window = [-3 3];
P.filt = 5; % Use the first filter band (e.g., [3 20] Hz), cuz small earthquakes are good 

% Initialize missing waveform fields
for i = 1000:length(Felix)
    for kz = 3:length(fields)
        fname = ['W_', fields{kz}];
        if ~isfield(Felix(i), fname)
            Felix(i).(fname) = 0;
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
    t1 = tic;

    tempFelix = Felix;  % Copy to avoid modifying inside parfor

    % Use parallel loop for each event
    parfor i = 1:length(Felix)
        temp = 0;

        DDt_field = ['DDt_', field];
        if isfield(Felix(i), DDt_field) & (Felix(i).(DDt_field) > -3)
            trace_Z = obtain_waveforms_Z(Felix(i), kz, P.a.sttime, P.a.edtime, p);
            if isfield(trace_Z, 'dataFilt') && size(trace_Z.dataFilt, 2) >= P.filt
                temp = trace_Z.dataFilt(idx_start:idx_end, P.filt);
            end
        end
        % Assign result
        tempFelix(i).(['W_', field]) = temp;
    end

    Felix = tempFelix;
    fprintf('Elapsed time for field %s: %.2f seconds\n', field, toc(t1));
end

fprintf('Total processing time: %.2f seconds\n', toc(overall_tic));

% Save the results
save('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_Wave/A_wave_forML_org.mat', 'Felix');

% %% downsample
% % Downsample waveforms from 200 Hz to 100 Hz (factor of 2)
% downsample_factor = 2;
% 
% for i = 1:length(Felix)
%     for kz = 1:length(fields)
%         fieldname = ['W_', fields{kz}];
%         waveform = Felix(i).(fieldname);
% 
%         if isnumeric(waveform) && ~isscalar(waveform) && size(waveform, 1) > 1
%             % Use downsample or resample (use resample if you want anti-aliasing filter)
%             waveform_ds = downsample(waveform, downsample_factor);  % or: resample(waveform, 1, 2);
%             Felix(i).(fieldname) = waveform_ds;
%         end
%     end
% end
% 
% % Save the downsampled Felix
% save('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/E_wave_forML_downsampled.mat', 'Felix');
% 
% %% cut for waveform from [-0.64 0.64]
% 
% load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/E_wave_forML_downsampled.mat')
% 
% % Sampling info after downsampling
% fs = 100;                   % Hz
% dt = 1/fs;                 
% total_samples = 2 * fs;  % From -1 to 1 sec, e.g., 201 samples
% t = linspace(-1, 1, total_samples);
% 
% % Indices for [-0.64, 0.64]s
% start_idx = find(t >= -0.64, 1, 'first');
% end_idx = find(t <= 0.64, 1, 'last');
% 
% for i = 1:length(Felix)
%     for kz = 1:length(fields)
%         fieldname = ['W_', fields{kz}];
%         waveform = Felix(i).(fieldname);
% 
%         if isnumeric(waveform) && ~isscalar(waveform) && size(waveform,1) >= end_idx
%             Felix(i).(fieldname) = waveform(start_idx:end_idx, :);
%         end
%     end
% end
% 
% % Save cropped data
% save('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/E_wave_forML_downsampled_cropped.mat', 'Felix');

