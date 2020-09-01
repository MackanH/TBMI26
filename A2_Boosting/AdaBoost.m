%% Hyper-parameters

% Number of randomized Haar-features
nbrHaarFeatures = 300;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 1000;
% Number of weak classifiers
nbrWeakClassifiers = 100;

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

%% Create image sets (do not modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError

d = ones(nbrTrainImages, 1)./nbrTrainImages;

%[tao, p, f, alpha]
data = zeros(nbrWeakClassifiers,4);

loading = 0; tic
for clsTe = 1:nbrWeakClassifiers % For all weak classifiers...
    
    e_min = inf;
    t_min = 0;
    p_min = 0;
    f_min = 0;
    a_min = 0;
    h = 0;
    
    for f=1:nbrHaarFeatures % For all features...
        
        tao = xTrain(f,:);
        
        for t = tao % For all thresholds...
            p = 1.0;
            
            x = WeakClassifier(t, p, xTrain(f,:));
            
            e = WeakClassifierError(x, d, yTrain);

            if e > 0.5
                p = -1;
                e = 1-e;
            end

            if e < e_min
                e_min = e;
                
                alpha = 0.5 * log( (1-e_min) / e_min );
                
                t_min = t;
                p_min = p;
                f_min = f;
                a_min = alpha;
                
                h = p*x;
            end
        end
    end
    
    d = d .* exp(-a_min * yTrain .* h)';
    d = d ./ sum(d);
    
    data(clsTe,1) = t_min;
    data(clsTe,2) = p_min;
    data(clsTe,3) = f_min;
    data(clsTe,4) = a_min;

    loading = loading + (100/nbrWeakClassifiers);
    
    disp([num2str(round(loading)) '%']);
end
 
trainingTime = toc;
 
disp(['Total time training: ' num2str(trainingTime) ' sec'])
 

%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.

classifierTr = zeros(nbrWeakClassifiers, size(yTrain,2));


for f = 1:nbrWeakClassifiers
    
    classifierTr(f,:) = data(f,4) ...
        * WeakClassifier(data(f, 1), data(f, 2), xTrain(data(f, 3),:));
    
    clsTr = signed( sum(classifierTr(1:f,:),1) );
    
    accTr(f) = 1 - mean( abs(clsTr - yTrain) ) / 2;
end

%%

classifierTe = zeros(nbrWeakClassifiers, size(yTest,2));


for f = 1:nbrWeakClassifiers
    
    classifierTe(f,:) = data(f,4) ...
        * WeakClassifier(data(f, 1), data(f, 2), xTest(data(f, 3),:));
    
    clsTe = sign( sum(classifierTe(1:f,:),1) );
    
    accTe(f) = 1 - mean( abs(clsTe - yTest) ) / 2;
end

[~, I] = max(accTe);

%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.

figure(4)
subplot(1,2,1)
plot(1:nbrWeakClassifiers,accTr)
title('Accuracy of training data')

subplot(1,2,2)
plot(1:nbrWeakClassifiers,accTe)
title('Accuracy of test data')


%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.

figure(5);
colormap gray;
i = 1;
k = 1;
while k < (25 + 1)
 
    if clsTe(i) ~= yTest(i)
        
        subplot(5,5,k), imagesc(testImages(:,:,i));
        axis image;
        axis off;
        k = k + 1;
    end
    
    i = i + 1;
end

figure(6);
colormap gray;
i = nbrTestImages;
k = 1;
while k < (25 + 1)
 
    if clsTe(i) ~= yTest(i)
        
        subplot(5,5,k), imagesc(testImages(:,:,i));
        axis image;
        axis off;
        k = k + 1;
    end
    
    i = i - 1;
end

%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.

figure(7);
colormap gray;
k = 1;
for i = 1 : size( data(:, 3) )
    
    if k > 25 
        break;
    end
        
    subplot(5,5,k), imagesc(haarFeatureMasks(:,:,data(i, 3)));
    axis image;
    axis off;
    
    k = k + 1;  
end

