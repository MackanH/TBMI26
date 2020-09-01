function [ acc ] = calcAccuracy( cM )
% CALCACCURACY Takes a confusion matrix amd calculates the accuracy

% (TP + TN) / (TP + FP + TN + FN)
acc = sum( diag(cM) ) / sum( sum(cM) );

end

