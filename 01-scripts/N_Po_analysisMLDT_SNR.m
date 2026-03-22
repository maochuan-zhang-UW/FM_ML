clear
stations = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};

% Define SNR bins
snr_edges = [0 5 10 15 20 25 inf];
snr_labels = {'0–5','5–10','10–15','15–20','20–25','>25'};

figure('Name','Right Ratio vs SNR','Position',[100,100,1600,600]);

for i = 1:numel(stations)
    st = stations{i};
    fname = ['/Users/mczhang/Documents/GitHub/FM5_ML/02-data/N_Po/N_Po' st '_DT.mat'];
    load(fname,'Felix');

    % Build field names
    cfm_field = ['PoML_W_' st];
    man_field = ['Man_' st];
    snr_field = ['SNR_' st];

    % Extract values
    cfm = [Felix.(cfm_field)]';
    man = [Felix.(man_field)]';
    snr = [Felix.(snr_field)]';

    % Mask for valid entries
    mask = cfm ~= 0 & ~isnan(snr);
    cfm_valid = cfm(mask);
    man_valid = man(mask);
    snr_valid = snr(mask);

    % Bin SNR
    [~,~,bin] = histcounts(snr_valid, snr_edges);

    right_ratio = nan(1,numel(snr_labels));
    counts = zeros(1,numel(snr_labels));
    for b = 1:numel(snr_labels)
        idx = (bin == b);
        counts(b) = sum(idx);
        if counts(b) > 0
            right_ratio(b) = sum(cfm_valid(idx) == man_valid(idx)) / counts(b);
        end
    end

    % Plot per station
    subplot(2,4,i);
    yyaxis left
    plot(1:numel(snr_labels), right_ratio, '-o','LineWidth',1.5,'MarkerSize',6);
    ylim([0 1]); ylabel('Right Ratio');
    xticks(1:numel(snr_labels)); xticklabels(snr_labels);

    yyaxis right
    bar(1:numel(snr_labels), counts, 'FaceAlpha',0.3,'FaceColor',[0.8 0.4 0.2]);
    ylabel('Number of Events');

    title(sprintf('Station %s', st));
    grid on;
end

sgtitle('Polarity Match Ratio (Left Y) and Event Counts (Right Y) vs SNR');
