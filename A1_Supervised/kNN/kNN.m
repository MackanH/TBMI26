function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)


% Euclidean distance matrix D between data set
% X and training set XTrain.
D = pdist2(X, XTrain);

% Index vector I stores the indices in sorted
% order of distances (in increasing order).
[~, I] = sort(D, 2);

% Retreive the labels for the k smallest dist-
% ances  using I.
kNearest = LTrain( I(:, 1:k) );

% Assign the prediction to LPred. Mode returns
% the most freq. occuring value for each row.
LPred = mode(kNearest, 2);

end

