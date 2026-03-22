clc;clear;close all;
path='/Users/mczhang/Documents/GitHub/FM5_ML/02-data/';
fields={'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};
groups = {'All'};

for gp=1%:length(groupsload)
    for kz=1:length(fields)   
        result=[];
        load([path,'L_CC/L_All_',fields{kz},'.mat']);
        result=[result;filteredResultsMatrix];
        clear filteredResultsMatrix;
        filteredResultsMatrix=result;clear result;
        % Now extract series3, series4, and values from the filteredResultsMatrix
        series3 = filteredResultsMatrix(:, 1);
        series4 = filteredResultsMatrix(:, 2);
        values = filteredResultsMatrix(:, 3);
        % Create a list of unique values in series3 and series4
        uniqueValues = unique([series3; series4]);

        % Create a map from the unique values to a set of indices
        map = containers.Map(uniqueValues, 1:length(uniqueValues));

        % Map series3 and series4 to the new indices
        series3_new = cell2mat(map.values(num2cell(series3)));
        series4_new = cell2mat(map.values(num2cell(series4)));

        %% Save unique values and their corresponding small indices
        uniqueValues_new = zeros(length(uniqueValues), 2);
        for i = 1:length(uniqueValues)
            uniqueValues_new(i, 1) = uniqueValues(i);
            uniqueValues_new(i, 2) = map(uniqueValues(i));
        end
        % Now you can use series3_new and series4_new to create your sparse matrix
        sparseMatrix = sparse(series3_new, series4_new, values);

        % Get the size of the sparseMatrix
        [nRows, nCols] = size(sparseMatrix);

        % Make the matrix square by padding with zeros if necessary
        if nRows > nCols
            sparseMatrix = [sparseMatrix, sparse(nRows, nRows - nCols)];
        elseif nCols > nRows
            sparseMatrix = [sparseMatrix; sparse(nCols - nRows, nCols)];
        end
        % Now you can make the matrix symmetric along the diagonal
        sparseMatrix = sparseMatrix + sparseMatrix' - diag(diag(sparseMatrix));
        % Perform singular value decomposition
        [U, S, V] = svds(sparseMatrix);clear S V;
        % Create SVD_result with ID_T as the first column and U(:,1) as the second column
        SVD_result = [uniqueValues_new(:, 1) U(:,1) sign(U(:,1))];

        % plot the full Matrix
        fullMatrix = full(sparseMatrix);

        % sign matrix
        SignMatrix=sign(fullMatrix);
        % predict sign matrix
        predictedSign = sign(U(:,1))*sign(U(:,1))';

        % Loop over each row
        for i = 1:size(SignMatrix, 1)
            % Find the non-zero elements in the current row of SignMatrix
            nonZeroLocations = SignMatrix(i, :) ~= 0;

            % Extract the elements in the current row of predictedSign at the non-zero locations
            predictedSignElements = predictedSign(i, nonZeroLocations);
            SignMatrixElements = SignMatrix(i, nonZeroLocations);

            % Count the number of elements that are the same and different
            SVD_result(i,4) = sum(predictedSignElements == SignMatrixElements);
            SVD_result(i,5) = sum(predictedSignElements ~= SignMatrixElements);
        end
        %% determine the SVD_threshold using elbow method
        SVD_num=max(abs((floor(log10(abs(SVD_result(:,2)))))));
        for ith=1:SVD_num
            thresholds(ith)=10^-(ith);
        end
        counts = zeros(size(thresholds));
        % Calculate the counts
        for i = 1:numel(thresholds)
            counts(i) = sum(abs(SVD_result(:,2)) > thresholds(i));
        end
        % Define the colors for the bars
        colors = [0.8 0.2 0.2; 0.2 0.8 0.2; 0.2 0.2 0.8; 0.8 0.8 0.2; 0.2 0.8 0.8; 0.8 0.2 0.8; 0.2 0.2 0.2];

        % Create the bar plot
        arr = counts;
        % Create indices for x-axis
        x = log10(thresholds);
        % Calculate line coefficients (polyfit)
        p = polyfit([x(1) x(end)], [arr(1) arr(end)], 1);
        % Calculate the line values
        line_values = polyval(p, x);
        % Line equation coefficients (A, B, C) for Ax + By + C = 0
        A = p(1);
        B = -1;
        C = p(2);
        % Calculate the perpendicular distances from the points to the line
        distances = abs(A*x + B*arr + C) / sqrt(A^2 + B^2);
        % Find the index of the maximum distance
        [~, idx] = max(distances);
        SVD_threshold=x(idx);
        SVD_result(:,6)=SVD_threshold;
        save([path,'M_SVD/M_',fields{kz},'.mat'], 'SVD_result');
        % [S,I]=sort(SVD_result(:,2),"descend");
        % SVD_result=SVD_result(I,:);
        % SVD_man=SVD_result(1:10,:);
        % save(['D_man/D_',groups{gp},'_',fields{kz},'.mat'], 'SVD_man');
    end
end
