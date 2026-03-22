load('D_CC1_CCPo.mat');

N = length(Felix);
data = zeros(N, 400);
labels = zeros(N, 1);

for i = 1:N
    x = Felix(i).W1_CC1;
    if isempty(x) || length(x) ~= 128
        continue
    end

    % Upsample from 128 → 400
    x_resampled = resample(x, 400, 128);

    % Normalize to [-1, 1]
    x_resampled = x_resampled / max(abs(x_resampled));

    data(i, :) = x_resampled;

    % Label: 1 = Up, 0 = Down
    p = sign(Felix(i).CC1_CCPo);
    if p == 1
        labels(i) = 1;  % Up
    elseif p == -1
        labels(i) = 0;  % Down
    else
        labels(i) = -1; % Unknown
    end
end

% Keep only valid (labeled) data
% valid = labels >= 0;
% data = data(valid, :);
% labels = labels(valid);

save('Ross2018_input_data.mat', 'data', 'labels');
