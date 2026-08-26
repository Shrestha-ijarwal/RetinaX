%% IDRiD_Train_Segmentation.m
% Prepare IDRiD dataset for multiclass semantic segmentation
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

fprintf('\n============================================\n');
fprintf('IDRiD SEGMENTATION DATA PREPARATION\n');
fprintf('============================================\n');

fprintf('\nNumber of images: %d\n', numel(imds.Files));


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

fprintf('\nSegmentation classes:\n');

for i = 1:numel(classNames)
    fprintf('%d = %s\n', i-1, classNames(i));
end


%% ============================================================
% 4. CREATE PIXEL LABEL DATASTORE
% =============================================================

pxds = pixelLabelDatastore( ...
    maskPath, ...
    classNames, ...
    labelIDs, ...
    'FileExtensions', {'.png'});

fprintf('\nNumber of masks: %d\n', numel(pxds.Files));


%% ============================================================
% 5. VERIFY IMAGE-MASK PAIRING
% =============================================================

fprintf('\nVerifying image-mask pairing...\n');

for i = 1:numel(imds.Files)

    % Get image filename
    [~, imageName, ~] = fileparts(imds.Files{i});

    % Get mask filename
    [~, maskName, ~] = fileparts(pxds.Files{i});

    % Expected mask name
    expectedMaskName = [imageName '_mask'];

    % Verify pairing
    if ~strcmp(maskName, expectedMaskName)

        error( ...
            'Mismatch found!\nImage: %s\nMask: %s', ...
            imageName, maskName);

    end

end

fprintf('SUCCESS: All %d images are correctly paired with masks.\n', ...
    numel(imds.Files));


%% ============================================================
% 6. TRAINING / VALIDATION SPLIT
% =============================================================

rng(42);

numImages = numel(imds.Files);

% Randomly shuffle image indices
indices = randperm(numImages);

% 80 percent training
numTrain = round(0.8 * numImages);

trainIdx = indices(1:numTrain);
valIdx   = indices(numTrain+1:end);

% Training datastores
imdsTrain = subset(imds, trainIdx);
pxdsTrain = subset(pxds, trainIdx);

% Validation datastores
imdsVal = subset(imds, valIdx);
pxdsVal = subset(pxds, valIdx);


fprintf('\n============================================\n');
fprintf('TRAIN / VALIDATION SPLIT\n');
fprintf('============================================\n');

fprintf('Total images      : %d\n', numImages);
fprintf('Training images   : %d\n', numel(imdsTrain.Files));
fprintf('Validation images : %d\n', numel(imdsVal.Files));


%% ============================================================
% 7. COMBINE IMAGE AND LABEL DATASTORES
% =============================================================

dsTrain = combine(imdsTrain, pxdsTrain);
dsVal   = combine(imdsVal, pxdsVal);

fprintf('\nImage and label datastores combined successfully.\n');


%% ============================================================
% 8. DEFINE NETWORK INPUT SIZE
% =============================================================

inputSize = [512 512];

fprintf('\nNetwork input size: %d x %d\n', ...
    inputSize(1), inputSize(2));


%% ============================================================
% 9. RESIZE TRAINING DATA
% =============================================================

dsTrainResized = transform( ...
    dsTrain, ...
    @(data) resizeData(data, inputSize));


%% ============================================================
% 10. RESIZE VALIDATION DATA
% =============================================================

dsValResized = transform( ...
    dsVal, ...
    @(data) resizeData(data, inputSize));


fprintf('\n============================================\n');
fprintf('DATA PREPARATION COMPLETE\n');
fprintf('============================================\n');

fprintf('Training samples   : %d\n', numel(imdsTrain.Files));
fprintf('Validation samples : %d\n', numel(imdsVal.Files));

fprintf('\nAll images and masks will be resized to:\n');
fprintf('%d x %d\n', inputSize(1), inputSize(2));


%% ============================================================
% 11. VERIFY ONE RESIZED IMAGE-MASK PAIR
% =============================================================

data = read(dsTrainResized);

resizedImage = data{1};
resizedMask  = data{2};

fprintf('\nResized image size:\n');
disp(size(resizedImage));

fprintf('Resized mask size:\n');
disp(size(resizedMask));

fprintf('Classes present in this mask:\n');
disp(unique(resizedMask));


%% ============================================================
% 12. VISUALIZE RESIZED IMAGE AND MASK
% =============================================================

figure;

subplot(1,2,1);
imshow(resizedImage);
title('Resized IDRiD Image');

subplot(1,2,2);
imshow(labeloverlay(resizedImage, resizedMask));
title('IDRiD Image with Segmentation Labels');


%% ============================================================
% LOCAL FUNCTION
% =============================================================

function data = resizeData(data, inputSize)

    % Resize RGB fundus image
    data{1} = imresize(data{1}, inputSize);

    % Resize categorical segmentation mask
    % Nearest interpolation preserves class labels
    data{2} = imresize(data{2}, inputSize, 'nearest');

end