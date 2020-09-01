function [ A, k ] = xvalidation( numFolds, numK, dataSetNr )

[X, D, L] = loadDataSet( dataSetNr );

numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select samples at random, false = select the first features

[XBins, ~, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numFolds, selectAtRandom);

A = zeros(1, numFolds);

% Construct a matrix of different combinations
% of Bins [2 3 4 5..; 1 3 4 5..; ..]. The ex-
% cluded number will be used as test bin.
mBin = zeros(numFolds, numFolds);

for i = 1:numFolds
    mBin(i,:) = 1:numFolds;
end

mBin = mBin - diag(diag(mBin));
mBin = reshape(nonzeros(mBin'), size(mBin, 2)-1, [])';

for i = 1:numK
    
    accuricies = 0;
    
    for n = 1:numFolds
        XT = combineBins(XBins, mBin(n,:));
        LT = combineBins(LBins, mBin(n,:));
        
        % Becuase the structure of the Bins matrix, XBins{n} 
        % and LBins{n} contains the excluded value.
        pred = kNN(XBins{n}, i, XT, LT);
        
        cM = calcConfusionMatrix(pred, LBins{n});
        
        acc = calcAccuracy(cM);
        
        % Stores the total accuracy for the different
        % bins.
        accuricies = accuricies + acc;
    end
   
    A(i) = accuricies;
end

% Normalize the accuracy values with numBins
% Returning a vector help plot the accuracy.
A = A./numFolds;

% The index in vector a correspons to the
% k - value, extracing the one with highest
% accuracy.
[~, k] = max(A);

end

