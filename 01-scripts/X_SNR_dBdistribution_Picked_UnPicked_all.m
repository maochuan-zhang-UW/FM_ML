% List of station files
files = { ...
    'D_manual_AS1_CCPo.mat', ...
    'D_manual_AS2_CCPo.mat', ...
    'D_manual_CC1_CCPo.mat', ...
    'D_manual_EC1_CCPo.mat', ...
    'D_manual_EC2_CCPo.mat', ...
    'D_manual_EC3_CCPo.mat', ...
    'D_manual_ID1_CCPo.mat'};

stations = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};

% Storage for combined SNR
all_snr_all = [];
all_snr_picked = [];

figure('Color','w'); set(gcf,'Position', [744   419   998   531]);


for f = 1:length(files)
    % Load file
    data = load(files{f});
    Felix = data.Felix;

    % Init
    snr_all = [];
    snr_picked = [];

    % Loop over structs
    for i = 1:length(Felix)
        fieldNSP = ['NSP_' stations{f}];
        if f==3
            fieldPo  = [stations{f} '_Po'];  % e.g., CC1_Po
        else
        fieldPo  = [stations{f} '_CCPo'];  % e.g., CC1_Po
        end

        if isfield(Felix(i), fieldNSP) && isfield(Felix(i), fieldPo)
            nsp = Felix(i).(fieldNSP);
            po  = Felix(i).(fieldPo);

            if ~isempty(nsp) && length(nsp) >= 3
                noise_val  = nsp(1);
                signal_val = nsp(3);

                if noise_val > 0 && signal_val > 0
                    snr_val_dB = 20*log10(signal_val / noise_val);

                    % Skip SNR = 0 dB
                    if snr_val_dB < 1
                        continue;
                    end

                    snr_all(end+1) = snr_val_dB;
                    if po ~= 0
                        snr_picked(end+1) = snr_val_dB;
                    end
                end
            end
        end
    end

    % Store for combined
    all_snr_all = [all_snr_all snr_all];
    all_snr_picked = [all_snr_picked snr_picked];

    % Subplot
    subplot(2,4,f); hold on;
    histogram(snr_all, 'BinWidth', 2, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none');
    histogram(snr_picked, 'BinWidth', 2, 'FaceColor', [0.2 0.4 0.8], 'EdgeColor', 'none');
    xlabel('SNR (dB)');
    ylabel('Count');
    title(stations{f});
    legend({'All','Picked'}, 'Location','best');
    grid on;
end

% Combined subplot
subplot(2,4,8); hold on;
histogram(all_snr_all, 'BinWidth', 2, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none');
histogram(all_snr_picked, 'BinWidth', 2, 'FaceColor', [0.2 0.4 0.8], 'EdgeColor', 'none');
xlabel('SNR (dB)');
ylabel('Count');
title('All Stations');
legend({'All','Picked'}, 'Location','best');
grid on;

%sgtitle('SNR Distribution at CC1 (in dB, excluding 0 dB)');
% --- Combined figure (separate from subplots) ---
figure('Color','w'); hold on;

% Histograms
histogram(all_snr_all, 'BinWidth', 2, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none');
histogram(all_snr_picked, 'BinWidth', 2, 'FaceColor', [0.2 0.4 0.8], 'EdgeColor', 'none');

xlabel('SNR (dB)');
ylabel('Count');
title('All Stations SNR Distribution');
legend({'All','Picked'}, 'Location','best');
grid on;

% --- Calculate % picked at thresholds ---
thresholds = [0 5 10 15 20];
percents = zeros(size(thresholds));

for k = 1:length(thresholds)
    total_above = sum(all_snr_all >= thresholds(k));
    picked_above = sum(all_snr_picked >= thresholds(k));

    if total_above > 0
        percents(k) = 100 * picked_above / total_above;
    else
        percents(k) = NaN;
    end
end

% --- Overlay as text on the figure ---
yl = ylim;  % current y-axis limits
for k = 1:length(thresholds)
    xloc = thresholds(k);
    yloc = yl(2)*0.9 - (k-1)*0.08*yl(2);  % stagger text vertically
    text(xloc, yloc, sprintf('≥ %d dB: %.1f%% picked', thresholds(k), percents(k)), ...
        'FontSize', 10, 'FontWeight','bold', 'Color','k');
end
