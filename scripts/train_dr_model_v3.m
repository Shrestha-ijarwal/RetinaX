clc;
clear;
close all;

%% ============================================================
% V3 - Improved DR Classification Model
% CLAHE + Mild Class Weights + Stronger Augmentation +
% Longer Fine-Tuning + Learning Rate Decay
%% ============================================================

rng(42);

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

disp('ResNet-50 Input Size:');
disp(inputSize);


%% 4. DATA AUGMENTATION

augmenter = imageDataAugmenter( ...
    'RandRotation', [-15 15], ...
    'RandXReflection', true, ...
    'RandXTranslation', [-15 15], ...
    'RandYTranslation', [-15 15], ...
    'RandScale', [0.9 1.1]);


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


%% 5. APPLY CLAHE PREPROCESSING

augimdsTrain = transform( ...
    augimdsTrain, ...
    @preprocessCLAHEWrapper);

augimdsValidation = transform( ...
    augimdsValidation, ...
    @preprocessCLAHEWrapper);

augimdsTest = transform( ...
    augimdsTest, ...
    @preprocessCLAHEWrapper);


%% 6. MODIFY RESNET-50 FOR 5 CLASSES

lgraph = layerGraph(net);

numClasses = 5;


% New fully connected layer

newFC = fullyConnectedLayer( ...
    numClasses, ...
    'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);


newSoftmax = softmaxLayer( ...
    'Name', 'new_softmax');


%% 7. MILD CLASS WEIGHTS

labelCounts = countEachLabel(imdsTrain);

totalImages = sum(labelCounts.Count);


% Square-root inverse frequency
% Less aggressive than V2

classWeights = sqrt( ...
    totalImages ./ ...
    (numClasses * labelCounts.Count));


% Normalize so average weight is approximately 1

classWeights = classWeights / mean(classWeights);


disp('V3 MILD CLASS WEIGHTS:');

disp(table( ...
    labelCounts.Label, ...
    classWeights, ...
    'VariableNames', {'Class','Weight'}));


%% CLASSIFICATION LAYER

newClassLayer = classificationLayer( ...
    'Name', 'new_classoutput', ...
    'Classes', labelCounts.Label, ...
    'ClassWeights', classWeights);


%% REPLACE RESNET-50 FINAL LAYERS

lgraph = replaceLayer( ...
    lgraph, ...
    'fc1000', ...
    newFC);

lgraph = replaceLayer( ...
    lgraph, ...
    'fc1000_softmax', ...
    newSoftmax);

lgraph = replaceLayer( ...
    lgraph, ...
    'ClassificationLayer_fc1000', ...
    newClassLayer);


%% 8. TRAINING SETTINGS

options = trainingOptions( ...
    'adam', ...
    'InitialLearnRate', 3e-5, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.2, ...
    'LearnRateDropPeriod', 5, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');


%% ============================================================
% 9. START TRAINING
%% ============================================================

disp('STARTING V3 TRAINING...');

trainedNet = trainNetwork( ...
    augimdsTrain, ...
    lgraph, ...
    options);


%% ============================================================
% 10. SAVE MODEL
%% ============================================================

modelPath = ...
    'C:\Users\sijar\Downloads\TRUST_DR\models\trained_DR_model_v3.mat';

save(modelPath, 'trainedNet');

disp('V3 MODEL SAVED SUCCESSFULLY!');


%% ============================================================
% 11. TEST MODEL
%% ============================================================

predictedLabels = classify( ...
    trainedNet, ...
    augimdsTest);

trueLabels = imdsTest.Labels;


accuracy = mean(predictedLabels == trueLabels);

fprintf('\n========================================\n');

fprintf('V3 TEST ACCURACY: %.2f%%\n', ...
    accuracy * 100);

fprintf('========================================\n');


%% ============================================================
% 12. CONFUSION MATRIX
%% ============================================================

figure;

confusionchart( ...
    trueLabels, ...
    predictedLabels);

title('DR Classification Confusion Matrix - V3');


%% ============================================================
% 13. REFERABLE DR METRICS
%% ============================================================

% Referable DR:
% Moderate + Severe + Proliferative

trueReferable = ismember( ...
    string(trueLabels), ...
    ["Moderate", "Severe", "Proliferative"]);


predReferable = ismember( ...
    string(predictedLabels), ...
    ["Moderate", "Severe", "Proliferative"]);


TP = sum(predReferable & trueReferable);

TN = sum(~predReferable & ~trueReferable);

FP = sum(predReferable & ~trueReferable);

FN = sum(~predReferable & trueReferable);


sensitivity = TP / (TP + FN);

specificity = TN / (TN + FP);


fprintf('\nREFERABLE DR RESULTS - V3:\n');

fprintf('Sensitivity: %.2f%%\n', ...
    sensitivity * 100);

fprintf('Specificity: %.2f%%\n', ...
    specificity * 100);

fprintf('\n========================================\n');

fprintf('V3 TRAINING COMPLETE\n');

fprintf('5-Class Accuracy: %.2f%%\n', ...
    accuracy * 100);

fprintf('Referable DR Sensitivity: %.2f%%\n', ...
    sensitivity * 100);

fprintf('Referable DR Specificity: %.2f%%\n', ...
    specificity * 100);

fprintf('========================================\n');


%% ============================================================
% HELPER FUNCTIONS
% These MUST stay at the end of the script
%% ============================================================

function dataOut = preprocessCLAHEWrapper(data)

    dataOut = data;

    for i = 1:size(data,1)

        img = data.input{i};

        dataOut.input{i} = applyCLAHE(img);

    end

end


function imgOut = applyCLAHE(img)

    % Apply CLAHE on L channel of Lab color space

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