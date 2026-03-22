% Load your file
clear;close all;
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/D_man/D_CC1_CCPo.mat');

N = length(Felix);          % Number of waveforms
data = zeros(N, 600);       % Preallocate resampled waveforms
labels = zeros(N, 1);       % Preallocate polarity labels (0 or 1)

for i = 1:N
    x = Felix(i).W1_CC1;     % Original 128-sample waveform
    if isempty(x) || length(x) ~= 128
        continue  % skip if invalid
    end
    
    % Resample to 600 points
    data(i, :) = resample(x, 600, 128);  % resample(new_num, original_num)
    
    % Get polarity from sign(CCPo): 1 → up (0), -1 → down (1)
    p = sign(Felix(i).CC1_CCPo);
    if p == 1
        labels(i) = 1;
    elseif p == -1
        labels(i) = 0;
    else
        labels(i) = -1;  % unknown or ambiguous
    end
end

% Remove invalid (unlabeled) rows
%valid = labels ~= -1;
%data = data(valid, :);
%labels = labels(valid);

% Save to .mat for Python use
save('W1_CC1_resampled.mat', 'data', 'labels');
