clear; close all;
% Load data
% load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_wave/A_wave_forML1sec100hz.mat');
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/D_man/D_man_dB20_polish.mat');

numStructs = length(Felix);
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};

AS1 = [];  % start empty

for i = 1:numStructs
    % only first station (AS1)
    wField   = ['W_' stations{1}];   % 'W_AS1'
    manField = ['Man_' stations{1}]; % 'Man_AS1'
    poField  = ['Po_' stations{1}];  % 'Po_AS1'
    
    if isfield(Felix(i), wField) && ~isempty(Felix(i).(wField))
        % --- Original ---
        tmp.ID        = Felix(i).ID;
        tmp.lon       = Felix(i).lon;
        tmp.lat       = Felix(i).lat;
        tmp.depth     = Felix(i).depth;
        tmp.(wField)  = Felix(i).(wField);
        tmp.(manField)= Felix(i).(manField);
        tmp.(poField) = Felix(i).(poField);
        AS1 = [AS1 tmp];   % append

        % --- Copy & flip both waveform and Man_AS1 ---
        tmp2.ID        = Felix(i).ID;
        tmp2.lon       = Felix(i).lon;
        tmp2.lat       = Felix(i).lat;
        tmp2.depth     = Felix(i).depth;
        tmp2.(wField)  = -Felix(i).(wField);       % flip waveform
        tmp2.(manField)= -Felix(i).(manField);     % flip manual polarity
        tmp2.(poField) = Felix(i).(poField);
        AS1 = [AS1 tmp2];  % append
    end
end

% check result
AS1(1)



% 
% % --- Step 2: Define SNR range for x-axis ---
% xVals = 0:1:50; % dB values (adjust if needed)
% markDB = [5 10 15 20]; % annotate at these dB values
% 
% % --- Step 3: Plot 8 subplots ---
% figure('Position', [100, 100, 1200, 900]);
% 
% for j = 1:numStations+1
%     subplot(4, 2, j);
%     if j <= numStations
%         snrData = snrValues{j};
%         titleStr = ['Station ' stations{j}];
%     else
%         snrData = allSnr;
%         titleStr = 'All Stations Total';
%     end
% 
%     % compute survival counts (# with SNR > threshold)
%     yVals = arrayfun(@(x) sum(snrData > x), xVals);
% 
%     plot(xVals, yVals, 'LineWidth', 1.5);
%     xlabel('SNR (dB)');
%     ylabel('Count');
%     title(titleStr);
%     grid on;
%     hold on;
% 
%     % annotate specific dB values
%     for k = 1:length(markDB)
%         dbVal = markDB(k);
%         yVal = sum(snrData > dbVal);
%         plot(dbVal, yVal, 'ro', 'MarkerSize', 6, 'HandleVisibility','off'); % mark point
%         text(dbVal, yVal, sprintf('%d', yVal), ...
%             'VerticalAlignment','bottom','HorizontalAlignment','left', 'FontSize', 8);
%     end
% end
% 
% sgtitle('SNR Distribution: Count of W_{station} > Threshold (dB)');
% set(gcf, 'Color', 'w');
% 
% 
% % --- Lognormal survival-style fit for AS1 ---
% snrData = snrValues{1};  % AS1
% snrData = snrData(snrData > 0);  % lognormal requires > 0
% 
% % Step 1: Fit lognormal using MLE
% pd_log = fitdist(snrData(:), 'Lognormal');
% mu_log = pd_log.mu;
% sigma_log = pd_log.sigma;
% 
% % Step 2: Compute survival-style y-values (counts > x)
% xVals = 0:1:50;
% yVals_AS1 = arrayfun(@(x) sum(snrData > x), xVals);
% 
% % Step 3: Create fitted lognormal survival curve
% xFit = linspace(0.5, 50, 300);
% cdf_log = cdf(pd_log, xFit);
% survival = 1 - cdf_log;
% 
% % Scale to match counts
% scaleFactor = max(yVals_AS1) / max(survival);
% fitCounts = survival * scaleFactor;
% 
% % Step 4: Plot
% figure;
% plot(xVals, yVals_AS1, 'b-', 'LineWidth', 1.5); hold on;
% plot(xFit, fitCounts, 'm--', 'LineWidth', 2);
% 
% xlabel('SNR (dB)');
% ylabel('Count');
% title('AS1 Lognormal Fit (Survival Curve Style)');
% legend('Observed Counts', 'Lognormal Fit');
% grid on;
% 
% % Step 5: Add formula annotation
% % Create LaTeX-formatted formula string with fitted mu and sigma
% textStr = sprintf('$\\mathrm{Counts}(x) = A \\cdot \\left[1 - F(x)\\right] = A \\cdot \\left[1 - \\Phi\\left(\\frac{\\ln x - %.2f}{%.2f} \\right)\\right]$', mu_log, sigma_log);
% 
% % Add text box to figure
% xPos = 5; % adjust based on your data
% yPos = max(yVals_AS1) * 0.7; % vertical placement
% text(xPos, yPos, textStr, 'Interpreter', 'latex', 'FontSize', 12, ...
%      'BackgroundColor', 'w', 'EdgeColor', 'k', 'Margin', 6);
% 
% fprintf('AS1 Lognormal Fit (Survival Style): mu = %.2f, sigma = %.2f\n', mu_log, sigma_log);
