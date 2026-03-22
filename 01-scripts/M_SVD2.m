cclc;clear;close all;
path='/Users/mczhang/Documents/GitHub/FM5_ML/02-data/';
fields={'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};
groups = {'All'};

% Create progress bar
h = waitbar(0, 'Processing stations...');

for gp = 1 % :length(groups)
    for kz = 1:length(fields)
        result = [];
        load([path,'L_CC/L_All_',fields{kz},'.mat']);
        result = [result; filteredResultsMatrix];
        clear filteredResultsMatrix;
        filteredResultsMatrix = result; clear result;

        % Now extract series3, series4, and values
        series3 = filteredResultsMatrix(:, 1);
        series4 = filteredResultsMatrix(:, 2);
        values  = filteredResultsMatrix(:, 3);

        % Create a list of unique values
        uniqueValues = unique([series3; series4]);

        % Create a map from unique values to indices
        map = containers.Map(uniqueValues, 1:length(uniqueValues));

        % Map series3 and series4 to new indices
        series3_new = cell2mat(map.values(num2cell(series3)));
        series4_new = cell2mat(map.values(num2cell(series4)));

        % Save unique values and their corresponding indices
        uniqueValues_new = zeros(length(uniqueValues), 2);
        for i = 1:length(uniqueValues)
            uniqueValues_new(i,1) = uniqueValues(i);
            uniqueValues_new(i,2) = map(uniqueValues(i));
        end

        % Create sparse matrix
        sparseMatrix = sparse(series3_new, series4_new, values);

        % Make square
        [nRows, nCols] = size(sparseMatrix);
        if nRows > nCols
            sparseMatrix = [sparseMatrix, sparse(nRows, nRows - nCols)];
        elseif nCols > nRows
            sparseMatrix = [sparseMatrix; sparse(nCols - nRows, nCols)];
        end

        % Make symmetric
        sparseMatrix = sparseMatrix + sparseMatrix' - diag(diag(sparseMatrix));

        % Perform singular value decomposition (only 1 vector)
        [U,~,~] = svds(sparseMatrix,1);

        % Create SVD_result with ID_T, U(:,1), and sign
        SVD_result = [uniqueValues_new(:,1), U(:,1), sign(U(:,1))];

        % Sign matrix (keep sparse)
        SignMatrix = sign(sparseMatrix);

        % Predicted sign matrix
        predictedSign = sign(U(:,1)) * sign(U(:,1))';

        % Compute matches (vectorized)
        mask = SignMatrix ~= 0;
        same = (predictedSign == SignMatrix) & mask;
        diff = (predictedSign ~= SignMatrix) & mask;

        SVD_result(:,4) = sum(same,2);
        SVD_result(:,5) = sum(diff,2);

        %% Determine the SVD_threshold using elbow method
        SVD_num = max(abs((floor(log10(abs(SVD_result(:,2)))))));
        thresholds = zeros(1,SVD_num);
        for ith=1:SVD_num
            thresholds(ith)=10^-(ith);
        end
        counts = zeros(size(thresholds));
        for i=1:numel(thresholds)
            counts(i) = sum(abs(SVD_result(:,2)) > thresholds(i));
        end
        x = log10(thresholds);
        p = polyfit([x(1) x(end)], [counts(1) counts(end)], 1);
        line_values = polyval(p, x);
        A = p(1); B = -1; C = p(2);
        distances = abs(A*x + B*counts + C) / sqrt(A^2 + B^2);
        [~, idx] = max(distances);
        SVD_threshold = x(idx);
        SVD_result(:,6) = SVD_threshold;

        % Save results
        save([path,'M_SVD/M_',fields{kz},'.mat'], 'SVD_result');

        % Update progress bar
        waitbar(kz/length(fields), h, ...
            sprintf('Processing %s (%d/%d)', fields{kz}, kz, length(fields)));
    end
end

% Close progress bar
close(h);
