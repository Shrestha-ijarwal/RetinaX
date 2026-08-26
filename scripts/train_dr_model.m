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

newClassLayer = classificationLayer( ...
    'Name', 'new_classoutput');


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

modelPath = 'C:\Users\sijar\Downloads\TRUST_DR\models\trained_DR_model.mat';

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

title('APTOS DR Classification Confusion Matrix');


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