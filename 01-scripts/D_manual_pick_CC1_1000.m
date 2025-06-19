clc; clear; close all;
dt = 1/200;
P_x = [-0.25 0.25];
P_x1 = P_x(1):dt:P_x(2)-dt;
path = '/Users/mczhang/Documents/GitHub/FM4/02-data/';
path2 = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/';
fields = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};

rng('shuffle'); % Ensure randomness

% Load already picked 200
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/D_AS1.mat');
Felix_already = Felix; clear Felix;

% Load full dataset
load('/Users/mczhang/Documents/GitHub/FM3/02-data/E_Po/E_All.mat');  % loads Felix

results = struct();        % Store stats for each field
all_stats = {};            % For text summary

for kz = 1 % Only field CC1
    fprintf('\n | Field: %s\n', fields{kz});
    fieldname = ['W_', fields{kz}];

    % Step 1: Filter valid waveforms
    valid_idx = find(arrayfun(@(x) isfield(x, fieldname) && ~isempty(getfield(x, fieldname)), Felix));

    % Step 2: Exclude already used IDs
    all_ids = [Felix.ID];
    already_ids = [Felix_already.ID];
    new_idx = valid_idx(~ismember([Felix(valid_idx).ID], already_ids));

    if numel(new_idx) < 800
        fprintf('Not enough new waveforms for %s. Only %d available.\n', fields{kz}, numel(new_idx));
        continue;
    end

    % Step 3: Randomly select 800 new waveforms
    sel_idx = randsample(new_idx, 800);
    wave = Felix(sel_idx);  % 800 new waves

    total_picked = 0;
    non_zero_count = 0;

    % Step 4: Manual picking
    for j = 1:length(wave)
        figure(1); clf; set(gcf,'position',[600,500,800,400]);
        eval(strcat('a = wave(j).W_',fields{kz},';'));
        a = a(1:100);

        plot(P_x1, normalize(a,'range',[-1 1]), 'b', 'LineWidth', 3); hold on;
        title(['No:' num2str(wave(j).ID) ' | Field: ' fields{kz} ]);
        plot([0,0], [-0.5,0.5], 'r', 'LineWidth', 2);
        plot([-0.0375,-0.0375], [-1,1], 'g', 'LineWidth', 2);
        plot([0.0375,0.0375], [-1,1], 'g', 'LineWidth', 2);
        plot([-0.1083,-0.1083], [-1,1], 'y', 'LineWidth', 4);
        plot([-0.1791,-0.1791], [-1,1], 'k', 'LineWidth', 5);
        plot([0.1083,0.1083], [-1,1], 'y', 'LineWidth', 4);
        plot([0.1791,0.1791], [-1,1], 'k', 'LineWidth', 5);

        [x,~] = ginput(1);

        % Classify polarity
        if x > 0.0375 && x < 0.1083; P = 1;
        elseif x >= 0.1083 && x < 0.1791; P = 2;
        elseif x >= 0.1791 && x < 0.25; P = 3;
        elseif x <= -0.0375 && x > -0.1083; P = -1;
        elseif x <= -0.1083 && x >= -0.1791; P = -2;
        elseif x <= -0.1791 && x > -0.25; P = -3;
        elseif x > -0.0375 && x < 0.0375; P = 0;
        elseif x <= -0.25; P = -4;
        elseif x >= 0.25; P = 4;
        else; P = 0;
        end

        eval(strcat('wave(j).',fields{kz},'_Po = P;'));

        total_picked = total_picked + 1;
        if P ~= 0
            non_zero_count = non_zero_count + 1;
        end
    end

    % Step 5: Combine with already picked ones
    Felix = [Felix_already, wave];

    % Step 6: Save result
    save([path2, 'D_',fields{kz},'.mat'], 'Felix');

    % Count polarity types
    P_values = arrayfun(@(x) getfield(x, [fields{kz},'_Po']), wave);
    num_positive = sum(P_values > 0);
    num_negative = sum(P_values < 0);
    num_uncertain = sum(P_values == 0);

    results.(fields{kz}).total = total_picked;
    results.(fields{kz}).positive = num_positive;
    results.(fields{kz}).negative = num_negative;
    results.(fields{kz}).uncertain = num_uncertain;

    fprintf('=== Result: %s ===\n', fields{kz});
    fprintf('Picked: %d | Positive: %d | Negative: %d | Uncertain: %d\n\n', ...
        total_picked, num_positive, num_negative, num_uncertain);

    all_stats(end+1,:) = {fields{kz}, total_picked, num_positive, num_negative, num_uncertain}; %#ok<SAGROW>
end

% Save result summary
save([path2, 'Pick_Statistics.mat'], 'results');

txtfile = fopen([path, 'D_man/Pick_Statistics.txt'], 'w');
fprintf(txtfile, 'Field\tTotal\tPositive\tNegative\tUncertain\n');
for i = 1:size(all_stats,1)
    fprintf(txtfile, '%s\t%d\t%d\t%d\t%d\n', all_stats{i,1}, all_stats{i,2}, all_stats{i,3}, all_stats{i,4}, all_stats{i,5});
end
fclose(txtfile);


