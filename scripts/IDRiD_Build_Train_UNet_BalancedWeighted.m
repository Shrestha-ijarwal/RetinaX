%% IDRiD_Build_Train_UNet_BalancedWeighted.m
% IDRiD Multiclass Retinal Lesion Segmentation
% Balanced Weighted U-Net Training
%
% Classes:
% 0 = Background
% 1 = Microaneurysms (MA)
% 2 = Haemorrhages (HE)
% 3 = Hard Exudates (EX)
% 4 = Soft Exudates (SE)
% 5 = Optic Disc (OD)
%
% Experiment 3:
% Moderate class weighting to avoid:
% 1. Background-only predictions
% 2. Excessive lesion predictions

clear;
clc;
close all;


%% ============================================================
% 1. DATASET PATHS
% =============================================================

basePath = ...
    'C:\Users\sijar\Downloads\TRUST_DR\datasets\A. Segmentation\A. Segmentation';

trainImagePath = fullfile( ...
    basePath, ...
    '1. Original Images', ...
    'a. Training Set');

maskPath = fullfile( ...
    basePath, ...
    '3. Combined Training Masks');


%% ============================================================
% 2. DEFINE SEGMENTATION CLASSES
% =============================================================

classNames = [
    "Background"
    "MA"
    "HE"
    "EX"
    "SE"
    "OD"
];

labelIDs = [0 1 2 3 4 5];

numClasses = numel(classNames);

inputSize = [256 256];


fprintf('\n============================================\n');
fprintf('IDRiD BALANCED WEIGHTED U-NET TRAINING\n');
fprintf('============================================\n');

fprintf('\nSegmentation classes:\n');

for i = 1:numClasses
    fprintf('%d = %s\n', ...
        labelIDs(i), ...
        classNames(i));
end


%% ============================================================
% 3. CREATE IMAGE DATASTORE
% =============================================================

imds = imageDatastore( ...
    trainImagePath, ...
    'FileExtensions', {'.jpg'});


%% ============================================================
% 4. CREATE PIXEL LABEL DATASTORE
% =============================================================

pxds = pixelLabelDatastore( ...
    maskPath, ...
    classNames, ...
    labelIDs, ...
    'FileExtensions', {'.png'});


fprintf('\nNumber of images: %d\n', ...
    numel(imds.Files));

fprintf('Number of masks : %d\n', ...
    numel(pxds.Files));


%% ============================================================
% 5. VERIFY IMAGE-MASK PAIRING
% =============================================================

fprintf('\nVerifying image-mask pairing...\n');

for i = 1:numel(imds.Files)

    [~, imageName, ~] = fileparts(imds.Files{i});
    [~, maskName, ~] = fileparts(pxds.Files{i});

    expectedMaskName = [imageName '_mask'];

    if ~strcmp(maskName, expectedMaskName)

        error( ...
            'Image-mask mismatch: %s and %s', ...
            imageName, ...
            maskName);

    end

end

fprintf('SUCCESS: Image-mask pairing verified.\n');


%% ============================================================
% 6. TRAIN / VALIDATION SPLIT
% =============================================================

rng(42);

numImages = numel(imds.Files);

indices = randperm(numImages);

numTrain = round(0.8 * numImages);

trainIdx = indices(1:numTrain);
valIdx = indices(numTrain+1:end);


imdsTrain = subset(imds, trainIdx);
pxdsTrain = subset(pxds, trainIdx);

imdsVal = subset(imds, valIdx);
pxdsVal = subset(pxds, valIdx);


fprintf('\n============================================\n');
fprintf('TRAIN / VALIDATION SPLIT\n');
fprintf('============================================\n');

fprintf('Total images      : %d\n', numImages);

fprintf('Training images   : %d\n', ...
    numel(imdsTrain.Files));

fprintf('Validation images : %d\n', ...
    numel(imdsVal.Files));


%% ============================================================
% 7. CALCULATE CLASS FREQUENCIES AFTER RESIZING
% ============================================================
%
% Calculate frequencies at the SAME 256 x 256 resolution
% used during training.
%
% This is better than calculating frequencies using the
% original 2848 x 4288 masks.
% =============================================================

fprintf('\n============================================\n');
fprintf('CALCULATING RESIZED CLASS FREQUENCIES\n');
fprintf('============================================\n');


classPixelCounts = zeros(numClasses, 1);


for i = 1:numel(pxdsTrain.Files)

    % Read original categorical mask
    mask = readimage(pxdsTrain, i);

    % Resize to training resolution
    mask = imresize( ...
        mask, ...
        inputSize, ...
        'nearest');


    % Count pixels for every class
    for c = 1:numClasses

        classPixelCounts(c) = ...
            classPixelCounts(c) + ...
            sum(mask(:) == classNames(c));

    end


    fprintf('Processed mask %d/%d\n', ...
        i, ...
        numel(pxdsTrain.Files));

end


totalPixels = sum(classPixelCounts);

classFrequencies = ...
    classPixelCounts ./ totalPixels;


%% ============================================================
% 8. CALCULATE BALANCED CLASS WEIGHTS
% =============================================================
%
% We use square-root inverse frequency weighting:
%
% weight = sqrt(medianFrequency / frequency)
%
% This is much less aggressive than direct inverse-frequency
% weighting.
%
% Then:
% - Normalize weights
% - Apply a minimum background weight
% - Cap maximum rare-class weights
% =============================================================

nonzeroFrequencies = ...
    classFrequencies(classFrequencies > 0);

medianFrequency = median(nonzeroFrequencies);


% Moderate square-root weighting
rawWeights = sqrt( ...
    medianFrequency ./ ...
    (classFrequencies + eps));


% Normalize weights to mean = 1
classWeights = ...
    rawWeights ./ mean(rawWeights);


%% ------------------------------------------------------------
% MANUAL STABILITY LIMITS
% ------------------------------------------------------------

% Background must still contribute to the loss
backgroundWeightFloor = 0.25;

% Rare classes cannot dominate excessively
maximumClassWeight = 3.0;


% Apply background floor
classWeights(1) = max( ...
    classWeights(1), ...
    backgroundWeightFloor);


% Apply maximum weight cap
classWeights = min( ...
    classWeights, ...
    maximumClassWeight);


% Normalize again
classWeights = ...
    classWeights ./ mean(classWeights);


% Convert to single
classWeights = single(classWeights(:));


%% ============================================================
% 9. DISPLAY CLASS STATISTICS
% =============================================================

fprintf('\n============================================\n');
fprintf('BALANCED CLASS STATISTICS AND WEIGHTS\n');
fprintf('============================================\n');

fprintf('\n%-15s %-15s %-15s %-15s\n', ...
    'Class', ...
    'Pixel Count', ...
    'Frequency', ...
    'Weight');

fprintf('---------------------------------------------------------------\n');


for c = 1:numClasses

    fprintf('%-15s %-15d %-15.8f %-15.4f\n', ...
        char(classNames(c)), ...
        classPixelCounts(c), ...
        classFrequencies(c), ...
        classWeights(c));

end


fprintf('\nWeighting strategy:\n');
fprintf('- Square-root inverse-frequency weighting\n');
fprintf('- Background weight protected with minimum floor\n');
fprintf('- Rare-class weights capped for stability\n');


%% ============================================================
% 10. COMBINE DATASTORES
% =============================================================

dsTrain = combine(imdsTrain, pxdsTrain);

dsVal = combine(imdsVal, pxdsVal);


%% ============================================================
% 11. RESIZE TRAINING AND VALIDATION DATA
% =============================================================

dsTrainResized = transform( ...
    dsTrain, ...
    @(data) resizeData(data, inputSize));


dsValResized = transform( ...
    dsVal, ...
    @(data) resizeData(data, inputSize));


fprintf('\nNetwork input size: %d x %d\n', ...
    inputSize(1), ...
    inputSize(2));


%% ============================================================
% 12. BUILD U-NET
% =============================================================

fprintf('\n============================================\n');
fprintf('BUILDING U-NET\n');
fprintf('============================================\n');


net = unet( ...
    [inputSize 3], ...
    numClasses);


fprintf('U-Net created successfully.\n');


%% ============================================================
% 13. DEFINE BALANCED WEIGHTED LOSS
% =============================================================

lossFcn = @(Y,T) crossentropy( ...
    Y, ...
    T, ...
    classWeights, ...
    NormalizationFactor = "all-elements", ...
    WeightsFormat = "C");


fprintf('Balanced weighted cross-entropy configured.\n');


%% ============================================================
% 14. TRAINING OPTIONS
% =============================================================

options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 2, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', dsValResized, ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'cpu');


%% ============================================================
% 15. START TRAINING
% =============================================================

fprintf('\n============================================\n');
fprintf('STARTING BALANCED WEIGHTED U-NET TRAINING\n');
fprintf('============================================\n');


net = trainnet( ...
    dsTrainResized, ...
    net, ...
    lossFcn, ...
    options);


%% ============================================================
% 16. SAVE TRAINED MODEL
% =============================================================

modelPath = fullfile( ...
    basePath, ...
    'IDRiD_UNet_BalancedWeighted_256.mat');


save(modelPath, ...
    'net', ...
    'classNames', ...
    'labelIDs', ...
    'inputSize', ...
    'classWeights', ...
    'classFrequencies');


fprintf('\n============================================\n');
fprintf('BALANCED WEIGHTED TRAINING COMPLETE\n');
fprintf('============================================\n');


fprintf('\nModel saved to:\n%s\n', ...
    modelPath);


%% ============================================================
% LOCAL FUNCTION
% =============================================================

function data = resizeData(data, inputSize)

    % Resize RGB retinal image
    data{1} = imresize( ...
        data{1}, ...
        inputSize);


    % Resize segmentation mask using nearest neighbor
    % so class labels remain unchanged
    data{2} = imresize( ...
        data{2}, ...
        inputSize, ...
        'nearest');

end