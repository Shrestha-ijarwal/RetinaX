%% IDRiD_Build_Train_UNet.m
% IDRiD Multiclass Retinal Lesion Segmentation
% CPU-optimized baseline
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


%% ============================================================
% 4. CREATE PIXEL LABEL DATASTORE
% =============================================================

pxds = pixelLabelDatastore( ...
    maskPath, ...
    classNames, ...
    labelIDs, ...
    'FileExtensions', {'.png'});


%% ============================================================
% 5. VERIFY IMAGE-MASK PAIRING
% =============================================================

for i = 1:numel(imds.Files)

    [~, imageName, ~] = fileparts(imds.Files{i});
    [~, maskName, ~] = fileparts(pxds.Files{i});

    expectedMaskName = [imageName '_mask'];

    if ~strcmp(maskName, expectedMaskName)
        error('Image-mask mismatch: %s and %s', ...
            imageName, maskName);
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

fprintf('Training images   : %d\n', numel(imdsTrain.Files));
fprintf('Validation images : %d\n', numel(imdsVal.Files));


%% ============================================================
% 7. COMBINE DATASTORES
% =============================================================

dsTrain = combine(imdsTrain, pxdsTrain);
dsVal = combine(imdsVal, pxdsVal);


%% ============================================================
% 8. NETWORK INPUT SIZE
% =============================================================

% Reduced from 512x512 for faster CPU training
inputSize = [256 256];

dsTrainResized = transform( ...
    dsTrain, ...
    @(data) resizeData(data, inputSize));

dsValResized = transform( ...
    dsVal, ...
    @(data) resizeData(data, inputSize));

fprintf('Network input size: %d x %d\n', ...
    inputSize(1), inputSize(2));


%% ============================================================
% 9. BUILD U-NET
% =============================================================

fprintf('\n============================================\n');
fprintf('BUILDING U-NET\n');
fprintf('============================================\n');

net = unet( ...
    [256 256 3], ...
    numClasses);

fprintf('U-Net created successfully.\n');


%% ============================================================
% 10. TRAINING OPTIONS
% =============================================================

options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 2, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', dsValResized, ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'cpu');


%% ============================================================
% 11. START TRAINING
% =============================================================

fprintf('\n============================================\n');
fprintf('STARTING IDRiD U-NET TRAINING\n');
fprintf('============================================\n');

net = trainnet( ...
    dsTrainResized, ...
    net, ...
    'crossentropy', ...
    options);


%% ============================================================
% 12. SAVE TRAINED MODEL
% =============================================================

modelPath = fullfile( ...
    basePath, ...
    'IDRiD_UNet_Segmentation_256.mat');

save(modelPath, ...
    'net', ...
    'classNames', ...
    'labelIDs', ...
    'inputSize');

fprintf('\n============================================\n');
fprintf('TRAINING COMPLETE\n');
fprintf('============================================\n');

fprintf('Model saved to:\n%s\n', modelPath);


%% ============================================================
% LOCAL FUNCTION
% =============================================================

function data = resizeData(data, inputSize)

    % Resize RGB image
    data{1} = imresize(data{1}, inputSize);

    % Resize categorical mask
    data{2} = imresize(data{2}, inputSize, 'nearest');

end