clc; clear; close all;

%% ==============================
%  GLOBAL SETTINGS
%% ==============================
% Controllable font size settings
TITLE_FONT_SIZE = 28;
AXIS_FONT_SIZE = 22;
LEGEND_FONT_SIZE = 20;
LABEL_FONT_SIZE = 26;

% Pastel colors (consistent across all figures)
c1 = [0.95 0.80 0.45];   % yellow
c2 = [0.95 0.60 0.60];   % pink
c3 = [0.60 0.90 0.90];   % cyan
c4 = [0.60 0.80 0.60];   % green
c5 = [0.75 0.65 0.95];   % purple

%% ==============================
%  COMMON STATION NAMES
%% ==============================
stations = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};
stations_avg = [stations, {'Average'}];

% Accuracy values
DiTIng     = [97.03 78.23 97.86 89.10 88.04 77.03 73.23];
CFM        = [97.03 80.98 97.35 88.27 87.12 82.91 74.46];
EQPolarity = [92.85 76.57 90.85 85.16 83.09 80.64 72.49];
PolCap     = [0.7942 0.7270 0.8186 0.7699 0.8307 0.7617 0.6857] * 100;
CC_ratio   = [0.99972 0.95991 0.99134 0.93391 0.84833 0.87012 0.80463] * 100;

% Add averages
DiTIng_A     = [DiTIng     mean(DiTIng)];
CFM_A        = [CFM        mean(CFM)];
EQPolarity_A = [EQPolarity mean(EQPolarity)];
PolCap_A     = [PolCap     mean(PolCap)];
CC_ratio_A   = [CC_ratio   mean(CC_ratio)];

%% ==============================
%  FIGURE 1: Histogram Bar Plot (Original Z_AGU2025.m - second figure)
%% ==============================
figure('Color','w', 'Position', [100 100 900 650]);
hold on;

bar_data = [DiTIng_A;
            CFM_A;
            EQPolarity_A;
            PolCap_A;
            CC_ratio_A]';

b = bar(bar_data, 'LineWidth',1.5);

% Apply CNN pastel colors to bars
b(1).FaceColor = c1;
b(2).FaceColor = c2;
b(3).FaceColor = c3;
b(4).FaceColor = c4;
b(5).FaceColor = c5;

xticks(1:length(stations_avg));
xticklabels(stations_avg);
xtickangle(0);

ylabel('Accuracy (%)','FontSize',LABEL_FONT_SIZE,'FontWeight','bold');
xlabel('Stations','FontSize',LABEL_FONT_SIZE,'FontWeight','bold');
title('Accuracy of Existing DL Models vs Cross-Correlation','FontSize',TITLE_FONT_SIZE,'FontWeight','bold');

legend({'DiTing','CFM','EQPolarity','PolCap','CC'}, ...
       'Location',"northeast",'NumColumns',3,'FontSize',LEGEND_FONT_SIZE);

set(gca,'FontSize',AXIS_FONT_SIZE,'LineWidth',1.5);
ylim([60 100]);
grid on;

%% ==============================
%  FIGURE 2: All-Stations vs LOSO vs Fine-Tuned (Z_AGU2025_2.m)
%% ==============================
% Accuracy: trained on all
acc_all = [0.9935 0.9858 0.9876 0.9867 0.9868 0.9871 0.9644];
acc_all_A = [acc_all mean(acc_all)];

% Accuracy: LOSO
acc_loso = [0.9964 0.9855 0.9902 0.9869 0.9854 0.9877 0.9653];
acc_loso_A = [acc_loso mean(acc_loso)];

% Fine-Tuned Accuracies
acc_fine = [0.9829 0.9484 0.9653 0.9585 0.9534 0.9386 0.9013 0.9570];

% Combine into matrix
bar_data = [acc_all_A(:)*100, acc_loso_A(:)*100, acc_fine(:)*100];

% Create figure
figure('Color','w','Position',[100 100 900 650]);

b = bar(bar_data, 'grouped', 'LineWidth',1.7);

% Color settings
b(1).FaceColor = c1;   % Trained on All Stations
b(2).FaceColor = c2;   % LOSO
b(3).FaceColor = c3;   % Fine-Tuned

% Axis formatting
xticks(1:length(stations_avg));
xticklabels(stations_avg);
xtickangle(0);
xlabel('Stations','FontSize',LABEL_FONT_SIZE,'FontWeight','bold');
ylabel('Accuracy (%)','FontSize',LABEL_FONT_SIZE,'FontWeight','bold');
title(' All-Stations vs LOSO vs Transfer Learning Accuracy','FontSize',TITLE_FONT_SIZE,'FontWeight','bold');

legend({'Trained on All Stations','LOSO','Transfer Learning'}, ...
    'FontSize',LEGEND_FONT_SIZE,'Location','north','NumColumns',3);

set(gca,'FontSize',AXIS_FONT_SIZE,'LineWidth',1.5);
ylim([90 100]);
grid on;
%% ==============================
%  FIGURE 3: SNR Comparison (Z_AGU2025_3.m)
%% ==============================
% Step 6: Fine-Tune (Multiply 100)
B6 = [0.9827 0.9425 0.9682 0.9617 0.9558 0.9437 0.9184 0.9534] * 100;
A6 = [0.9869 0.9555 0.9706 0.9621 0.9583 0.9507 0.9334 0.9596] * 100;
C6 = [0.9879 0.9537 0.9667 0.9693 0.9567 0.9555 0.9385 0.9612] * 100;
data6 = [B6(:), A6(:), C6(:)];

% Step 5: Training Results (Multiply 100)
A5 = [0.9935 0.9858 0.9876 0.9867 0.9868 0.9871 0.9644 0.9850] * 100;
B5 = [0.9859 0.9505 0.9739 0.9584 0.9623 0.9569 0.9341 0.9604] * 100;
C5 = [0.9940 0.9731 0.9868 0.9760 0.9690 0.9850 0.9770 0.9801] * 100;
data5 = [B5(:), A5(:), C5(:)];

% Create figure with 2 subplots
figure('Color','w','Position',[100 100 900 650]);

% SUBPLOT (1): Training Accuracy
subplot(2,1,1);
b1 = bar(data5, 'grouped', 'LineWidth',1.7);
b1(1).FaceColor = c1;
b1(2).FaceColor = c2;
b1(3).FaceColor = c3;

xticks(1:length(stations_avg));
xticklabels(stations_avg);
xtickangle(0);

ylabel('Accuracy (%)','FontSize',LABEL_FONT_SIZE,'FontWeight','bold');
title('Training Accuracy Across Stations','FontSize',TITLE_FONT_SIZE,'FontWeight','bold');

legend({'High-SNR','Orig-SNR','Lower-SNR'}, ...
       'FontSize',LEGEND_FONT_SIZE,'Location','north','NumColumns',3);

set(gca,'FontSize',AXIS_FONT_SIZE,'LineWidth',1.5);
ylim([90 101]);
grid on;

% SUBPLOT (2): Fine-Tuned Accuracy
subplot(2,1,2);
b2 = bar(data6, 'grouped', 'LineWidth',1.7);
b2(1).FaceColor = c1;
b2(2).FaceColor = c2;
b2(3).FaceColor = c3;

xticks(1:length(stations_avg));
xticklabels(stations_avg);
xtickangle(0);

ylabel('Accuracy (%)','FontSize',LABEL_FONT_SIZE,'FontWeight','bold');
xlabel('Stations','FontSize',LABEL_FONT_SIZE,'FontWeight','bold');
title('Transfer Learning Accuracy Across Stations','FontSize',TITLE_FONT_SIZE,'FontWeight','bold');

legend({'High-SNR','Orig-SNR','Low-SNR'}, ...
       'FontSize',LEGEND_FONT_SIZE,'Location','north','NumColumns',3);

set(gca,'FontSize',AXIS_FONT_SIZE,'LineWidth',1.5);
ylim([90 100]);
grid on;



%% ==============================
%  FIGURE 4: Time Shift Comparison (New Figure)
%% -----------------------------
% Fine-tuned model performance (new table) ×100
%% -----------------------------
FT_02 = [0.8278 0.7496 0.8066 0.7712 0.7533 0.7358 0.7455 0.7700] * 100;
FT_01 = [0.9524 0.8793 0.9167 0.9123 0.8909 0.8689 0.8678 0.8984] * 100;
FT_00 = [0.9869 0.9555 0.9706 0.9621 0.9583 0.9507 0.9334 0.9596] * 100;

data_finetune = [FT_00(:), FT_01(:), FT_02(:)];

%% -----------------------------
% Training model performance ×100
%% -----------------------------
A_02 = [0.8335 0.7520 0.8133 0.7512 0.7288 0.7271 0.7393 0.7636] * 100;
A_01 = [0.9704 0.8992 0.9415 0.9310 0.9022 0.9007 0.8991 0.9207] * 100;
A_00 = [0.9935 0.9858 0.9876 0.9867 0.9868 0.9871 0.9644 0.9850] * 100;

data_training = [A_00(:), A_01(:), A_02(:)];

%% -----------------------------
% CNN pastel colors
%% -----------------------------
c1 = [0.95 0.80 0.45];   % yellow (timeshift 0.00)
c2 = [0.95 0.60 0.60];   % pink   (timeshift 0.01)
c3 = [0.60 0.90 0.90];   % cyan   (timeshift 0.02)

%% -----------------------------
% Create figure 4
%% -----------------------------
figure('Color','w','Position',[100 100 900 650]);

%% ======= SUBPLOT (211): Fine-Tuned Model =======
subplot(2,1,2); hold on;

b = bar(data_finetune, 'grouped', 'LineWidth',1.7);
b(1).FaceColor = c1;   % 0.00
b(2).FaceColor = c2;   % 0.01
b(3).FaceColor = c3;   % 0.02

xticks(1:length(stations_avg));
xticklabels(stations_avg);
xtickangle(0);

ylabel('Accuracy (%)','FontSize',26,'FontWeight','bold');
title('Transfer Learning Performance vs Time Shift','FontSize',26,'FontWeight','bold');
xlabel('Stations','FontSize',LABEL_FONT_SIZE,'FontWeight','bold');
legend({'Time Shift = 0.00','Time Shift = 0.01','Time Shift = 0.02'}, ...
       'FontSize',20,'Location','north','NumColumns',3);

set(gca,'FontSize',20,'LineWidth',1.5);
ylim([70 103]);
grid on;
box on;

%% ======= SUBPLOT (212): Trained Model =======
subplot(2,1,1); hold on;

b2 = bar(data_training, 'grouped', 'LineWidth',1.7);
b2(1).FaceColor = c1;
b2(2).FaceColor = c2;
b2(3).FaceColor = c3;

xticks(1:length(stations_avg));
xticklabels(stations_avg);
xtickangle(0);

ylabel('Accuracy (%)','FontSize',26,'FontWeight','bold');
title('Training Performance vs Time Shift','FontSize',26,'FontWeight','bold');

legend({'Time Shift = 0.00','Time Shift = 0.01','Time Shift = 0.02'}, ...
       'FontSize',20,'Location','north','NumColumns',3);

set(gca,'FontSize',20,'LineWidth',1.5);
ylim([70 105]);
grid on;
box on;
%% ==============================
%% -----------------------------
% Training model (from screenshot)
%% -----------------------------
A_train = [0.9942 0.9483 0.9789 0.9668 0.9598 0.9631 0.9466 0.9655] * 100;
B_train = [0.9722 0.9052 0.9389 0.9244 0.9192 0.9254 0.8892 0.9250] * 100;
C_train = [0.9948 0.9407 0.9840 0.9674 0.9640 0.9472 0.9311 0.9615] * 100;

data_training = [A_train(:), B_train(:), C_train(:)];

%% -----------------------------
% Fine-tuned model (from screenshot)
%% -----------------------------
A_ft = [0.9776 0.8992 0.9438 0.9351 0.9294 0.8945 0.8884 0.9241] * 100;
B_ft = [0.8976 0.8115 0.8711 0.8353 0.8251 0.7875 0.7963 0.8322] * 100;
C_ft = [0.9768 0.8583 0.9498 0.9128 0.9167 0.8613 0.8354 0.9020] * 100;

data_finetune = [A_ft(:), B_ft(:), C_ft(:)];

%% -----------------------------
% Colors
%% -----------------------------
c1 = [0.95 0.80 0.45];
c2 = [0.95 0.60 0.60];
c3 = [0.60 0.90 0.90];

%% -----------------------------
% Create figure 5
%% -----------------------------
figure('Color','w','Position',[100 100 900 650]);

%% ======= TOP: Training =======
subplot(2,1,1); hold on;
b1 = bar(data_training,'grouped','LineWidth',1.5);

b1(1).FaceColor = c1;
b1(2).FaceColor = c2;
b1(3).FaceColor = c3;

xticks(1:length(stations_avg));
xticklabels(stations_avg);
xtickangle(0);
ylabel('Accuracy (%)','FontSize',26,'FontWeight','bold');
title('Training Performance','FontSize',26,'FontWeight','bold');

legend({ ...
    'Train 0.01 s, test 0.01 s', ...
    'Train 0.01 s, test 0.02 s', ...
    'Train 0.02 s, test 0.01 s'}, ...
    'FontSize',15,'Location','north','NumColumns',3,'Box','on');

ylim([75 104]);
set(gca,'FontSize',22,'LineWidth',1.5);
grid on;
box on;
%% ======= BOTTOM: Fine-tuned =======
subplot(2,1,2); hold on;
b2 = bar(data_finetune,'grouped','LineWidth',1.5);

b2(1).FaceColor = c1;
b2(2).FaceColor = c2;
b2(3).FaceColor = c3;

xticks(1:length(stations_avg));
xticklabels(stations_avg);
xtickangle(0);
ylabel('Accuracy (%)','FontSize',26,'FontWeight','bold');
title('Transfer Learning Performance','FontSize',26,'FontWeight','bold');
xlabel('Stations','FontSize',LABEL_FONT_SIZE,'FontWeight','bold');
legend({ ...
    'FT 0.01 s, test 0.01 s', ...
    'FT 0.01 s, test 0.02 s', ...
    'FT 0.02 s, test 0.01 s'}, ...
    'FontSize',15,'Location','north','NumColumns',3,'Box','on');

ylim([75 102]);
set(gca,'FontSize',22,'LineWidth',1.5);
grid on;
box on;

%% ==============================
%  SUMMARY
%% ==============================
fprintf('All 6 figures have been generated with consistent font settings:\n');
fprintf('  Title Font Size: %d\n', TITLE_FONT_SIZE);
fprintf('  Axis Font Size: %d\n', AXIS_FONT_SIZE);
fprintf('  Legend Font Size: %d\n', LEGEND_FONT_SIZE);
fprintf('  Label Font Size: %d\n', LABEL_FONT_SIZE);
fprintf('\nFigures generated:\n');
fprintf('  1. Per-Station Polarity Accuracy Comparison (Line Plot)\n');
fprintf('  2. Accuracy Comparison by Station (Bar Plot)\n');
fprintf('  3. Train vs Fine-Tune on Timeshift Dataset\n');
fprintf('  4. SNR Comparison (Training vs Fine-Tuned)\n');
fprintf('  5. All-Stations vs LOSO vs Fine-Tuned\n');
fprintf('  6. Time Shift Comparison (Training vs Fine-Tuned)\n');
fprintf('\nTo change font sizes, modify the GLOBAL SETTINGS section at the top.\n');

%% ==============================
%  SAVE ALL FIGURES INTO ONE PDF
%% ==============================
pdf_name = 'All_Figures_Combined.pdf';

% Delete if already exists (avoid appending old content)
if exist(pdf_name, 'file')
    delete(pdf_name);
end

% Get all open figure handles
figs = findall(0, 'Type', 'figure');
% Sort by figure number so order is consistent
[~, idx] = sort([figs.Number]);
figs = figs(idx);

% Append each figure to the PDF
for i = 1:length(figs)
    exportgraphics(figs(i), pdf_name, 'Append', true);
end

fprintf('Saved all %d figures into: %s\n', length(figs), pdf_name);
