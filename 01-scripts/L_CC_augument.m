% Optimized Cross-correlation analysis
clc; clear; close all;

% Configuration
path = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/';
fields = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
groups = {'All'};
dt_window = [0.2, 0.4]; % Short and long windows
windowstring = {'s', 'l'};
dt = 1/100; % Sampling interval (Hz)
P.a.window = [-1 1]; % Window based on P picks
P.b.ccthreshold = 0.7; % CC threshold for pairs
P.b.lagahead = 0.1; % 0.1s before P pick
P.c.ccthreshold = 0.7; % CC threshold for matrix
P.c.timelag = 0.04; % Timelag threshold (seconds)
CC_threshold = P.c.ccthreshold;
TL_threshold = P.c.timelag;

% Initialize parallel pool once
if isempty(gcp('nocreate'))
    parpool(8);
end

for gp = 1%length(groups)
    tic;
    for kz = 4:length(fields)
        % Load data
        load([path, '/K_aug/' fields{kz} '_add_V4_STEP.mat'])
        eval(strcat('Felix = ', fields{kz}, '_add;'));

        for wd = 1:length(dt_window)
            % Pre-allocate signal matrix
            window_samples = floor((dt_window(wd) + P.b.lagahead) / dt);
            max_signals = length(Felix);
            MAT = zeros(max_signals, window_samples);
            ID_T = zeros(max_signals, 1);
            n = 1;

            % Populate signal matrix
            for i = 1:length(Felix)
                fieldName = sprintf('Felix(%d).W_%s', i, fields{kz});
                if ~isempty(eval(fieldName))
                    signal = eval(fieldName);
                    if length(signal)<10
                        continue;
                    end
                    start_idx = floor((abs(P.a.window(1)) - P.b.lagahead) / dt);
                    end_idx = start_idx + window_samples - 1;
                    MAT(n, :) = signal(start_idx:end_idx);
                    ID_T(n) = Felix(i).ID2;
                    n = n + 1;
                end
            end
            MAT = MAT(1:n-1, :); % Trim unused rows
            ID_T = ID_T(1:n-1);

            % Cross-correlation computation
            signalMatrix = MAT;
            [numberOfTimeSeries, ~] = size(signalMatrix);
            max_pairs = numberOfTimeSeries * (numberOfTimeSeries - 1) / 2;
            resultsCell = cell(max_pairs, 1);

            % Temporary storage for parfor results
            tempResultsAll = cell(numberOfTimeSeries-1, 1);

            parfor i = 1:(numberOfTimeSeries-1)
                tempResults = cell(numberOfTimeSeries - i, 1);
                for j = (i+1):numberOfTimeSeries
                    % Compute normalized cross-correlation
                    [correlation, lags] = xcorr(signalMatrix(i, :), signalMatrix(j, :), 'normalized');
                    [maxCorrelation, maxIdx] = max(correlation);
                    [minCorrelation, minIdx] = min(correlation);

                    % Store results if threshold is met
                    if abs(maxCorrelation) > CC_threshold || abs(minCorrelation) > CC_threshold
                        tempResults{j-i} = struct(...
                            'series1', i, ...
                            'series2', j, ...
                            'series3', ID_T(i), ...
                            'series4', ID_T(j), ...
                            'maxCorr', maxCorrelation, ...
                            'maxLag', lags(maxIdx) * dt, ...
                            'minCorr', minCorrelation, ...
                            'minLag', lags(minIdx) * dt);
                    end
                end
                % Store non-empty results for this iteration
                tempResultsAll{i} = tempResults(~cellfun(@isempty, tempResults));
            end

            % Combine results from all iterations
            resultCounter = 1;
            for i = 1:length(tempResultsAll)
                tempResults = tempResultsAll{i};
                resultsCell(resultCounter:(resultCounter + length(tempResults) - 1)) = tempResults;
                resultCounter = resultCounter + length(tempResults);
            end

            % Trim and convert to matrix
            resultsCell = resultsCell(1:resultCounter-1);
            resultsMatrix = zeros(length(resultsCell), 8);
            for k = 1:length(resultsCell)
                resultsMatrix(k, :) = [...
                    resultsCell{k}.series1, ...
                    resultsCell{k}.series2, ...
                    resultsCell{k}.series3, ...
                    resultsCell{k}.series4, ...
                    resultsCell{k}.maxCorr, ...
                    resultsCell{k}.maxLag, ...
                    resultsCell{k}.minCorr, ...
                    resultsCell{k}.minLag];
            end

            % Process resultsMatrix
            maxCorr = resultsMatrix(:, 5);
            minCorr = resultsMatrix(:, 7);

            % Compute dominant correlation (max or min) with original sign
            %resultsMatrix(:, 9) = maxCorr .* (abs(maxCorr) >= abs(minCorr)) + minCorr .* (abs(minCorr) > abs(maxCorr));
            resultsMatrix(:,9) = arrayfun(@(x, y) sign(x * (abs(x) >= abs(y)) + y * (abs(y) > abs(x))) * abs(abs(x) - abs(y)), maxCorr, minCorr);
            % Assign dominant correlation and lag to columns 10 and 11
            idx = maxCorr > abs(minCorr);
            resultsMatrix(idx, 10) = resultsMatrix(idx, 5); % maxCorr
            resultsMatrix(idx, 11) = resultsMatrix(idx, 6); % maxLag
            resultsMatrix(~idx, 10) = resultsMatrix(~idx, 7); % minCorr
            resultsMatrix(~idx, 11) = resultsMatrix(~idx, 8); % minLag

            % Filter by timelag threshold
            resultsMatrix(abs(resultsMatrix(:, 11)) >= TL_threshold, :) = [];

            % Remove unnecessary columns
            resultsMatrix(:, 1:2) = [];
            resultsMatrix(:,3:6)=[];
            % Store results for short/long window
            eval(['resultsMatrix_', windowstring{wd}, '=resultsMatrix;']);
        end

        % Combine short and long window results
        [commonRows, index5, index2] = intersect(resultsMatrix_l(:,1:2), resultsMatrix_s(:,1:2), 'rows');
        resultsMatrix_cb=resultsMatrix_l(index5,:);
        resultsMatrix_cb(:,6)=resultsMatrix_s(index2,4);
        resultsMatrix_cb(:,7)=resultsMatrix_s(index2,5);
        index3= abs(resultsMatrix_cb(:,7)-resultsMatrix_cb(:,5))>0.01;
        resultsMatrix_cb(index3,:)=[];
        %resultsMatrix_cb_bad=resultsMatrix_cb(index3,:);

        filteredResultsMatrix = resultsMatrix_cb;
        filteredResultsMatrix(:,4:7)=[];
        % Save results
        save([path, 'L_CC/L_', groups{gp}, '_', fields{kz}, '.mat'], 'filteredResultsMatrix','resultsMatrix_cb');
        
        % Display timing
        fprintf('Processed group %s, field %s: %.2f seconds\n', groups{gp}, fields{kz}, toc);
        clear Felix;
    end
end

% Clean up
delete(gcp('nocreate')); % Close parallel pool
disp('Cross-correlation analysis completed.');