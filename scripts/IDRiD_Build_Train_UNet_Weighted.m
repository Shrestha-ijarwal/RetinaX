%% IDRiD_Build_Train_UNet_Weighted.m
% IDRiD Multiclass Retinal Lesion Segmentation
% Weighted Cross-Entropy U-Net Training
%
% Classes:
% 0 = Background
% 1 = Microaneurysms (MA)
% 2 = Haemorrhages (HE)
% 3 = Hard Exudates (EX)
% 4 = Soft Exudates (SE)
% 5 = Optic Disc (OD)

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
% 2. CREATE IMAGE DATASTORE
% =============================================================

imds = imageDatastore( ...
    trainImagePath, ...
    'FileExtensions', {'.jpg'});


%% ============================================================
% 3. DEFINE SEGMENTATION CLASSES
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


fprintf('\n============================================\n');
fprintf('IDRiD WEIGHTED U-NET TRAINING\n');
fprintf('============================================\n');

fprintf('\nSegmentation classes:\n');

for i = 1:numClasses
    fprintf('%d = %s\n', ...
        labelIDs(i), ...
        classNames(i));
end


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
% 7. CALCULATE CLASS PIXEL FREQUENCIES
% =============================================================

fprintf('\n============================================\n');
fprintf('CALCULATING CLASS FREQUENCIES\n');
fprintf('============================================\n');


classPixelCounts = zeros(numClasses, 1);


for i = 1:numel(pxdsTrain.Files)

    % Read categorical ground-truth mask
    mask = readimage(pxdsTrain, i);


    % Count pixels belonging to each class
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
% 8. COMPUTE CLASS WEIGHTS
% =============================================================
%
% Rare classes receive larger weights.
% Background receives a smaller weight.
%
% Median-frequency balancing:
%
% weight = median(nonzero frequencies) / class frequency
%
% We then normalize weights so the average weight is 1.
% Finally, cap extreme weights for stable CPU training.
% =============================================================

nonzeroFrequencies = ...
    classFrequencies(classFrequencies > 0);

medianFrequency = median(nonzeroFrequencies);


classWeights = ...
    medianFrequency ./ ...
    (classFrequencies + eps);


% Normalize so mean weight = 1
classWeights = ...
    classWeights ./ mean(classWeights);


% Prevent extremely large weights from destabilizing training
maxWeight = 10;

classWeights = min(classWeights, maxWeight);


% Convert to single precision
classWeights = single(classWeights(:));


%% ============================================================
% 9. DISPLAY CLASS STATISTICS
% =============================================================

fprintf('\n============================================\n');
fprintf('CLASS STATISTICS AND WEIGHTS\n');
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


fprintf('\nIMPORTANT:\n');

fprintf('Rare lesion classes should have higher weights.\n');

fprintf('Background should have a relatively lower weight.\n');


%% ============================================================
% 10. COMBINE DATASTORES
% =============================================================

dsTrain = combine(imdsTrain, pxdsTrain);

dsVal = combine(imdsVal, pxdsVal);


%% ============================================================
% 11. NETWORK INPUT SIZE
% =============================================================

inputSize = [256 256];


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
% 13. DEFINE WEIGHTED CROSS-ENTROPY LOSS
% =============================================================
%
% This is the key change from the previous model.
%
% The old model used:
%
%     'crossentropy'
%
% The new model uses classWeights so rare lesions
% contribute more strongly to the training loss.
% =============================================================

lossFcn = @(Y,T) crossentropy( ...
    Y, ...
    T, ...
    classWeights, ...
    NormalizationFactor = "all-elements", ...
    WeightsFormat = "C");


fprintf('\nWeighted cross-entropy loss configured.\n');


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

fprintf('STARTING WEIGHTED IDRiD U-NET TRAINING\n');

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
    'IDRiD_UNet_Weighted_256.mat');


save(modelPath, ...
    'net', ...
    'classNames', ...
    'labelIDs', ...
    'inputSize', ...
    'classWeights', ...
    'classFrequencies');


fprintf('\n============================================\n');

fprintf('WEIGHTED TRAINING COMPLETE\n');

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


    % Resize categorical segmentation mask
    % Nearest-neighbor interpolation preserves labels
    data{2} = imresize( ...
        data{2}, ...
        inputSize, ...
        'nearest');

end