clc; clear; close all;

%% -----------------------------
% Station names
%% -----------------------------
stations = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1','Average'};

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
% Create figure
%% -----------------------------
figure('Color','w','Position',[100 100 780 900]);

%% ======= TOP: Training =======
subplot(2,1,1); hold on;
b1 = bar(data_training,'grouped','LineWidth',1.5);

b1(1).FaceColor = c1;
b1(2).FaceColor = c2;
b1(3).FaceColor = c3;

xticks(1:numel(stations));
xticklabels(stations);
xtickangle(0);
ylabel('Accuracy (%)','FontSize',26,'FontWeight','bold');
title('Training Model Performance','FontSize',26,'FontWeight','bold');

legend({ ...
    'Train 0.01 s, test 0.01 s', ...
    'Train 0.01 s, test 0.02 s', ...
    'Train 0.02 s, test 0.01 s'}, ...
    'FontSize',15,'Location','north','NumColumns',3,'Box','off');

ylim([75 102]);
set(gca,'FontSize',22,'LineWidth',1.5);
grid on;

%% ======= BOTTOM: Fine-tuned =======
subplot(2,1,2); hold on;
b2 = bar(data_finetune,'grouped','LineWidth',1.5);

b2(1).FaceColor = c1;
b2(2).FaceColor = c2;
b2(3).FaceColor = c3;

xticks(1:numel(stations));
xticklabels(stations);
xtickangle(0);
ylabel('Accuracy (%)','FontSize',26,'FontWeight','bold');
title('Fine-Tuned Model Performance','FontSize',26,'FontWeight','bold');

legend({ ...
    'FT 0.01 s, test 0.01 s', ...
    'FT 0.01 s, test 0.02 s', ...
    'FT 0.02 s, test 0.01 s'}, ...
    'FontSize',15,'Location','north','NumColumns',3,'Box','off');

ylim([75 102]);
set(gca,'FontSize',22,'LineWidth',1.5);
grid on;
