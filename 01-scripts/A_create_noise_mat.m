% create 10000 random noise from 2016 to 2017 structure so that I can run
% pararell to get it 
clc;clear;
% Generate 10000 Felix structures
% Set random seed for reproducibility
rng(42);  % You can change this seed value to get different random sequences

% Initialize the structure array
Felix = struct();

% Date range for 2016
start_date = datenum('2016-01-01 00:01:01');
end_date = datenum('2016-12-31 23:59:59');

% Generate 10000 structures
for i = 1:10000
    % Basic fields
    Felix(i).ID = i;
    Felix(i).on = start_date + (end_date - start_date) * rand();
    Felix(i).lon = 0;
    Felix(i).lat = 0;
    Felix(i).depth = 0;
    
    % DDt fields - randomly select from 0.1 to 2.1 with format %.4f
    Felix(i).DDt_AS1 = round(0.1 + (2.1 - 0.1) * rand(), 4);
    Felix(i).DDt_AS2 = round(0.1 + (2.1 - 0.1) * rand(), 4);
    Felix(i).DDt_CC1 = round(0.1 + (2.1 - 0.1) * rand(), 4);
    Felix(i).DDt_EC1 = round(0.1 + (2.1 - 0.1) * rand(), 4);
    Felix(i).DDt_EC2 = round(0.1 + (2.1 - 0.1) * rand(), 4);
    Felix(i).DDt_EC3 = round(0.1 + (2.1 - 0.1) * rand(), 4);
    Felix(i).DDt_ID1 = round(0.1 + (2.1 - 0.1) * rand(), 4);
    
    % NSP fields - set to zeros
    Felix(i).NSP_AS1 = [0 0 0];
    Felix(i).NSP_AS2 = [0 0 0];
    Felix(i).NSP_CC1 = [0 0 0];
    Felix(i).NSP_EC1 = [0 0 0];
    Felix(i).NSP_EC2 = [0 0 0];
    Felix(i).NSP_EC3 = [0 0 0];
    Felix(i).NSP_ID1 = [0 0 0];
    
    % SP fields - set to zeros
    Felix(i).SP_AS1 = 0;
    Felix(i).SP_AS2 = 0;
    Felix(i).SP_CC1 = 0;
    Felix(i).SP_EC1 = 0;
    Felix(i).SP_EC2 = 0;
    Felix(i).SP_EC3 = 0;
    Felix(i).SP_ID1 = 0;
    Felix(i).SP_All = 0;
    
    % Po fields - set to zeros
    Felix(i).Po_AS1 = 0;
    Felix(i).Po_AS2 = 0;
    Felix(i).Po_CC1 = 0;
    Felix(i).Po_EC1 = 0;
    Felix(i).Po_EC2 = 0;
    Felix(i).Po_EC3 = 0;
    Felix(i).Po_ID1 = 0;
    Felix(i).PoALL = 0;
    
    % W fields - set to zero vectors
    Felix(i).W_AS1 = zeros(128, 1);
    Felix(i).W_AS2 = zeros(128, 1);
    Felix(i).W_CC1 = zeros(128, 1);
    Felix(i).W_EC1 = zeros(128, 1);
    Felix(i).W_EC2 = zeros(128, 1);
    Felix(i).W_EC3 = zeros(128, 1);
    Felix(i).W_ID1 = zeros(128, 1);
end

% Save to .mat file
save('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_wave/A_Felix_10000_noise.mat', 'Felix');

fprintf('Successfully generated 10000 Felix structures.\n');
fprintf('Random seed used: 42\n');
fprintf('Saved to: Felix_10000.mat\n');

% Display first and last entries as verification
fprintf('\nFirst entry (Felix(1)):\n');
disp(Felix(1));
fprintf('\nLast entry (Felix(10000)):\n');
disp(Felix(10000));