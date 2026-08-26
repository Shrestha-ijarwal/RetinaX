clc;
clear;
close all;

%% 1. LOAD DATASET
dataFolder = 'C:\Users\sijar\Downloads\TRUST_DR\organized_dataset';
imds = imageDatastore( ...
    dataFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

disp('FULL DATASET:');
disp(countEachLabel(imds));

%% 2. SPLIT DATASET
% 70% Training
% 15% Validation
% 15% Testing
[imdsTrain, imdsRemaining] = splitEachLabel( ...
    imds, 0.70, 'randomized');
[imdsValidation, imdsTest] = splitEachLabel( ...
    imdsRemaining, 0.50, 'randomized');

disp('TRAINING DATA:');
disp(countEachLabel(imdsTrain));
disp('VALIDATION DATA:');
disp(countEachLabel(imdsValidation));
disp('TEST DATA:');
disp(countEachLabel(imdsTest));

%% 3. LOAD RESNET-50
net = resnet50;
inputSize = net.Layers(1).InputSize;
disp('Input size:');
disp(inputSize);

%% 4. DATA AUGMENTATION
augmenter = imageDataAugmenter( ...
    'RandRotation', [-10 10], ...
    'RandXReflection', true, ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10]);

augimdsTrain = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsTrain, ...
    'DataAugmentation', augmenter);

augimdsValidation = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsValidation);

augimdsTest = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsTest);

% --- CLAHE: wrap all three datastores so every image gets contrast-enhanced
% before it reaches the network. Train/val/test must all be consistent.
augimdsTrain = transform(augimdsTrain, @preprocessCLAHEWrapper);
augimdsValidation = transform(augimdsValidation, @preprocessCLAHEWrapper);
augimdsTest = transform(augimdsTest, @preprocessCLAHEWrapper);

%% 5. MODIFY RESNET-50 FOR 5 CLASSES
lgraph = layerGraph(net);
numClasses = 5;

newFC = fullyConnectedLayer( ...
    numClasses, ...
    'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);

newSoftmax = softmaxLayer( ...
    'Name', 'new_softmax');

% --- Class weights: computed from TRAINING set label counts (inverse frequency)
% This forces the loss to penalize mistakes on rare classes (Severe/Proliferative)
% more heavily, instead of the model just favoring the majority class.
labelCounts = countEachLabel(imdsTrain);
totalImages = sum(labelCounts.Count);
classWeights = totalImages ./ (numClasses * labelCounts.Count);

disp('CLASS WEIGHTS (higher = rarer class, penalized more):');
disp(table(labelCounts.Label, classWeights, 'VariableNames', {'Class','Weight'}));

newClassLayer = classificationLayer( ...
    'Name', 'new_classoutput', ...
    'Classes', labelCounts.Label, ...
    'ClassWeights', classWeights);

lgraph = replaceLayer(lgraph, 'fc1000', newFC);
lgraph = replaceLayer( ...
    lgraph, ...
    'fc1000_softmax', ...
    newSoftmax);
lgraph = replaceLayer( ...
    lgraph, ...
    'ClassificationLayer_fc1000', ...
    newClassLayer);

%% 6. TRAINING SETTINGS
options = trainingOptions( ...
    'adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

%% 7. START TRAINING
trainedNet = trainNetwork( ...
    augimdsTrain, ...
    lgraph, ...
    options);

%% 8. SAVE MODEL
modelPath = 'C:\Users\sijar\Downloads\TRUST_DR\models\trained_DR_model_v2.mat';
save(modelPath, 'trainedNet');
disp('MODEL SAVED SUCCESSFULLY!');

%% 9. TEST MODEL
predictedLabels = classify( ...
    trainedNet, ...
    augimdsTest);
trueLabels = imdsTest.Labels;
accuracy = mean(predictedLabels == trueLabels);
fprintf('\nTEST ACCURACY: %.2f%%\n', accuracy * 100);

%% 10. CONFUSION MATRIX
figure;
confusionchart(trueLabels, predictedLabels);
title('DR Classification Confusion Matrix (Class-Weighted + CLAHE)');

%% 11. REFERABLE DR METRICS
trueReferable = ismember( ...
    string(trueLabels), ...
    ["Moderate","Severe","Proliferative"]);
predReferable = ismember( ...
    string(predictedLabels), ...
    ["Moderate","Severe","Proliferative"]);

TP = sum(predReferable & trueReferable);
TN = sum(~predReferable & ~trueReferable);
FP = sum(predReferable & ~trueReferable);
FN = sum(~predReferable & trueReferable);

sensitivity = TP / (TP + FN);
specificity = TN / (TN + FP);

fprintf('\nREFERABLE DR RESULTS:\n');
fprintf('Sensitivity: %.2f%%\n', sensitivity * 100);
fprintf('Specificity: %.2f%%\n', specificity * 100);


%% ===================== HELPER FUNCTIONS =====================
% NOTE: Local functions must be at the END of the script file in MATLAB.

function dataOut = preprocessCLAHEWrapper(data)
    % 'data' is a table coming from the augmentedImageDatastore transform.
    % Column 1 holds the image(s) for this batch/row.
    dataOut = data;
    for i = 1:size(data,1)
        img = data.input{i};
        dataOut.input{i} = applyCLAHE(img);
    end
end

function imgOut = applyCLAHE(img)
    % Applies CLAHE on the L channel (Lab color space) so color balance
    % of the retina image is preserved while contrast is enhanced.
    if size(img,3) == 3
        labImg = rgb2lab(img);
        L = labImg(:,:,1) / 100;
        L = adapthisteq(L);
        labImg(:,:,1) = L * 100;
        imgOut = lab2rgb(labImg);
        imgOut = im2uint8(imgOut);
    else
        imgOut = adapthisteq(img);
    end
end