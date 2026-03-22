clc; clear; close all;

%% -----------------------------
% Station names
%% -----------------------------
stations = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1','Average'};

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
% Create figure
%% -----------------------------
figure('Color','w','Position',[100 100 734 724]);

%% ======= SUBPLOT (211): Fine-Tuned Model =======
subplot(2,1,2); hold on;

b = bar(data_finetune, 'grouped', 'LineWidth',1.7);
b(1).FaceColor = c1;   % 0.00
b(2).FaceColor = c2;   % 0.01
b(3).FaceColor = c3;   % 0.02

xticks(1:length(stations));
xticklabels(stations);
xtickangle(0);

ylabel('Accuracy (%)','FontSize',26,'FontWeight','bold');
title('Fine-Tuned Model Performance at Different Time Shifts','FontSize',26,'FontWeight','bold');

legend({'Time Shift = 0.00','Time Shift = 0.01','Time Shift = 0.02'}, ...
       'FontSize',20,'Location','north','NumColumns',3);

set(gca,'FontSize',20,'LineWidth',1.5);
ylim([70 100]);
grid on;

%% ======= SUBPLOT (212): Trained Model =======
subplot(2,1,1); hold on;

b2 = bar(data_training, 'grouped', 'LineWidth',1.7);
b2(1).FaceColor = c1;
b2(2).FaceColor = c2;
b2(3).FaceColor = c3;

xticks(1:length(stations));
xticklabels(stations);
xtickangle(0);

ylabel('Accuracy (%)','FontSize',26,'FontWeight','bold');
title('Training Model Performance at Different Time Shifts','FontSize',26,'FontWeight','bold');

legend({'Time Shift = 0.00','Time Shift = 0.01','Time Shift = 0.02'}, ...
       'FontSize',20,'Location','north','NumColumns',3);

set(gca,'FontSize',20,'LineWidth',1.5);
ylim([70 105]);
grid on;
