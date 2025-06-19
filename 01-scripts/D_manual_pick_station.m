clc; clear; close all;
dt = 1/200;
P_x = [-0.25 0.25];
P_x1 = P_x(1):dt:P_x(2)-dt;
path = '/Users/mczhang/Documents/GitHub/FM4/02-data/';
path2 = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/';
fields = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};

rng('shuffle'); % Ensure randomness
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/D_CC1.mat');
Felix_already=Felix;clear Felix;

load('/Users/mczhang/Documents/GitHub/FM3/02-data/E_Po/E_All.mat')  % loads Felix

results = struct();        % Store stats for each field
all_stats = {};            % For text summary

for kz = 3%1:length(fields)
    fprintf('\n | Field: %s\n', fields{kz});
    fieldname = ['W_', fields{kz}];

    % Filter Felix entries with valid waveform in this field
    valid_idx = find(arrayfun(@(x) isfield(x, fieldname) && ~isempty(getfield(x, fieldname)), Felix));
    
    if numel(valid_idx) < 200
        fprintf('Not enough waveforms for %s. Only %d available.\n', fields{kz}, numel(valid_idx));
        continue;
    end
    
    sel_idx = randsample(valid_idx, 200);  % randomly select 200
    wave = Felix(sel_idx);  % selected waveforms

    total_picked = 0;
    non_zero_count = 0;

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
    clear Felix;
    Felix=wave;
    % Save wave data
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
