function [ cM ] = calcConfusionMatrix( LPred, LTrue )
% CALCCONFUSIONMATRIX returns the confusion matrix of the predicted labels

cls = unique(LTrue);

noc = length(cls);

cM = zeros(noc);

%       Prediction
%     ______________
% T  |              |
% r  |              |
% u  |              |
% e  |              |
%    |______________|

%cM = confusionmat(LPred, LTrue);

for i = 1 : size(LPred,1)
    cM( LPred(i), LTrue(i) ) = cM( LPred(i), LTrue(i) ) + 1;
end

