clc;
clear;
close all;

%% ============================================
% TRUST-DR: RETINAL VESSEL SEGMENTATION (V2)
% DRIVE DATASET
% Adds: synchronized augmentation + class-weighted loss
% ============================================

fprintf('\n============================================\n');
fprintf(' TRUST-DR: VESSEL SEGMENTATION TRAINING (V2)\n');
fprintf('============================================\n\n');

%% 1. DEFINE PATHS

projectFolder = ...
    'C:\Users\sijar\Downloads\TRUST_DR';

imageFolder = fullfile( ...
    projectFolder, ...
    'prepared_drive', ...
    'images');

maskFolder = fullfile( ...
    projectFolder, ...
    'prepared_drive', ...
    'masks');

modelFolder = fullfile( ...
    projectFolder, ...
    'models');

if ~isfolder(modelFolder)
    mkdir(modelFolder);
end

%% 2. CREATE IMAGE DATASTORE

imds = imageDatastore(imageFolder);

fprintf('Number of fundus images: %d\n', numel(imds.Files));

%% 3. CREATE MASK DATASTORE

classNames = ["background","vessel"];

labelIDs = [0 255];

pxds = pixelLabelDatastore( ...
    maskFolder, ...
    classNames, ...
    labelIDs);

fprintf('Number of vessel masks: %d\n\n', ...
    numel(pxds.Files));

%% 3b. CHECK CLASS IMBALANCE + COMPUTE CLASS WEIGHTS

numClasses = 2;

labelCounts = countEachLabel(pxds);
totalPixels = sum(labelCounts.PixelCount);
classWeights = totalPixels ./ (numClasses * labelCounts.PixelCount);

fprintf('CLASS WEIGHTS (higher = rarer class, penalized more):\n');
disp(table(labelCounts.Name, labelCounts.PixelCount, classWeights, ...
    'VariableNames', {'Class','PixelCount','Weight'}));

%% 4. CHECK IMAGE/MASK PAIRING

if numel(imds.Files) ~= numel(pxds.Files)
    error('Number of images and masks does not match!');
end

%% 5. RANDOMIZE DATA

rng(42);

numImages = numel(imds.Files);

indices = randperm(numImages);

%% 6. SPLIT DATA
% 70% Training
% 15% Validation
% 15% Testing

numTrain = round(0.70 * numImages);
numValidation = round(0.15 * numImages);

trainIdx = indices(1:numTrain);

validationIdx = indices( ...
    numTrain+1 : numTrain+numValidation);

testIdx = indices( ...
    numTrain+numValidation+1 : end);

%% Create subsets

imdsTrain = subset(imds, trainIdx);
pxdsTrain = subset(pxds, trainIdx);

imdsValidation = subset(imds, validationIdx);
pxdsValidation = subset(pxds, validationIdx);

imdsTest = subset(imds, testIdx);
pxdsTest = subset(pxds, testIdx);

fprintf('\nDataset split:\n');
fprintf('Training   : %d images\n', numel(trainIdx));
fprintf('Validation : %d images\n', numel(validationIdx));
fprintf('Testing    : %d images\n\n', numel(testIdx));

%% 7. DEFINE IMAGE SIZE

imageSize = [256 256 3];

%% 8. CREATE U-NET NETWORK

fprintf('Creating U-Net segmentation network...\n');

lgraph = unet( ...
    imageSize, ...
    numClasses, ...
    'EncoderDepth', 3);

%% 9. CREATE DATASTORES

dsTrain = combine( ...
    imdsTrain, ...
    pxdsTrain);

dsValidation = combine( ...
    imdsValidation, ...
    pxdsValidation);

%% 9b. SYNCHRONIZED AUGMENTATION (image + mask together, train set only)

dsTrain = transform(dsTrain, @augmentImageAndMask);

%% 10. TRAINING OPTIONS

options = trainingOptions( ...
    'adam', ...
    'InitialLearnRate', 1e-4, ...      % reduced from 1e-3 to prevent overshoot with weighted loss
    'GradientThreshold', 1, ...        % clips exploding gradients (fixes NaN loss)
    'MaxEpochs', 30, ...
    'MiniBatchSize', 2, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', dsValidation, ...
    'ValidationFrequency', 5, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

%% ============================================
% 11. TRAIN THE NETWORK (class-weighted loss)
% ============================================

fprintf('\n============================================\n');
fprintf(' STARTING VESSEL SEGMENTATION TRAINING (V2)\n');
fprintf('============================================\n\n');

% --- DIAGNOSTIC STEP: plain crossentropy (no weights) to isolate the NaN cause.
% If this trains fine, the problem is the weighted-loss syntax below, not augmentation.
% Once confirmed, we'll switch back to the weighted version with corrected syntax.
trainedVesselNet = trainnet( ...
    dsTrain, ...
    lgraph, ...
    "crossentropy", ...
    options);

% --- WEIGHTED VERSION (currently disabled for diagnostics — do not delete):
% classWeightsDL = dlarray(classWeights, 'C');
% lossFcn = @(Y,T) crossentropy(Y, T, classWeightsDL, 'WeightsFormat', 'C');
% trainedVesselNet = trainnet(dsTrain, lgraph, lossFcn, options);

%% ============================================
% 12. SAVE MODEL
% ============================================

modelPath = fullfile( ...
    modelFolder, ...
    'trained_vessel_segmentation_v2.mat');

save( ...
    modelPath, ...
    'trainedVesselNet', ...
    'imageSize', ...
    'classNames', ...
    'classWeights');

fprintf('\n============================================\n');
fprintf(' MODEL SAVED SUCCESSFULLY\n');
fprintf('============================================\n');

fprintf('Model location:\n%s\n', modelPath);

%% ============================================
% 13. TEST THE MODEL
% ============================================

fprintf('\n============================================\n');
fprintf(' TESTING VESSEL SEGMENTATION MODEL (V2)\n');
fprintf('============================================\n');

predictedLabels = semanticseg( ...
    imdsTest, ...
    trainedVesselNet, ...
    'MiniBatchSize', 2);

%% ============================================
% 13b. PER-IMAGE METRICS (Pixel Acc, Precision, Recall, Dice, IoU)
% ============================================

testImages = imdsTest.Files;
testMasks = pxdsTest.Files;

numTestImages = numel(testImages);

pixelAccuracyList = zeros(numTestImages,1);
iouList = zeros(numTestImages,1);
diceList = zeros(numTestImages,1);
precisionList = zeros(numTestImages,1);
recallList = zeros(numTestImages,1);

for i = 1:numTestImages

    fprintf('\nProcessing test image %d of %d...\n', ...
        i, numTestImages);

    %% Read test fundus image
    img = imread(testImages{i});

    if size(img,3) == 1
        img = repmat(img,[1 1 3]);
    end

    imgResized = imresize(img, imageSize(1:2));

    %% Predict vessel segmentation
    predictedMask = semanticseg(imgResized, trainedVesselNet);

    %% Read ground truth mask
    gtMask = imread(testMasks{i});

    if size(gtMask,3) == 3
        gtMask = rgb2gray(gtMask);
    end

    gtMask = imresize(gtMask, imageSize(1:2), 'nearest');

    %% Convert masks to binary
    gtBinary = gtMask > 0;

    predictedBinary = strcmp(string(predictedMask), "vessel");

    if ~any(predictedBinary(:))
        predictedBinary = strcmpi(string(predictedMask), "Vessel");
    end

    %% Calculate confusion values

    TP = sum(predictedBinary(:) & gtBinary(:));
    TN = sum(~predictedBinary(:) & ~gtBinary(:));
    FP = sum(predictedBinary(:) & ~gtBinary(:));
    FN = sum(~predictedBinary(:) & gtBinary(:));

    pixelAccuracy = (TP + TN) / (TP + TN + FP + FN);
    precision = TP / (TP + FP + eps);
    recall = TP / (TP + FN + eps);
    diceScore = (2 * TP) / (2 * TP + FP + FN + eps);
    iouScore = TP / (TP + FP + FN + eps);

    pixelAccuracyList(i) = pixelAccuracy;
    precisionList(i) = precision;
    recallList(i) = recall;
    diceList(i) = diceScore;
    iouList(i) = iouScore;

    fprintf('Pixel Accuracy : %.4f\n', pixelAccuracy);
    fprintf('Precision      : %.4f\n', precision);
    fprintf('Recall         : %.4f\n', recall);
    fprintf('Dice Score     : %.4f\n', diceScore);
    fprintf('IoU Score      : %.4f\n', iouScore);

end

%% ============================================
% FINAL TEST RESULTS
% ============================================

meanPixelAccuracy = mean(pixelAccuracyList);
meanPrecision = mean(precisionList);
meanRecall = mean(recallList);
meanDice = mean(diceList);
meanIoU = mean(iouList);

fprintf('\n============================================\n');
fprintf(' FINAL VESSEL SEGMENTATION TEST RESULTS (V2)\n');
fprintf('============================================\n');

fprintf('Number of Test Images : %d\n\n', numTestImages);

fprintf('Mean Pixel Accuracy   : %.2f %%\n', meanPixelAccuracy * 100);
fprintf('Mean Precision        : %.2f %%\n', meanPrecision * 100);
fprintf('Mean Recall           : %.2f %%\n', meanRecall * 100);
fprintf('Mean Dice Score       : %.2f %%\n', meanDice * 100);
fprintf('Mean IoU Score        : %.2f %%\n', meanIoU * 100);

fprintf('============================================\n');

%% ============================================
% SAVE TEST METRICS
% ============================================

results.pixelAccuracy = pixelAccuracyList;
results.precision = precisionList;
results.recall = recallList;
results.dice = diceList;
results.iou = iouList;

results.meanPixelAccuracy = meanPixelAccuracy;
results.meanPrecision = meanPrecision;
results.meanRecall = meanRecall;
results.meanDice = meanDice;
results.meanIoU = meanIoU;

save(fullfile(modelFolder, ...
    'vessel_segmentation_results_v2.mat'), ...
    'results');

fprintf('\nResults saved successfully!\n');

%% ============================================
% VISUALIZE ONE TEST RESULT
% ============================================

testIndex = 1;

img = imread(testImages{testIndex});

if size(img,3) == 1
    img = repmat(img,[1 1 3]);
end

imgResized = imresize(img, imageSize(1:2));

predictedMask = semanticseg(imgResized, trainedVesselNet);

gtMask = imread(testMasks{testIndex});

if size(gtMask,3) == 3
    gtMask = rgb2gray(gtMask);
end

gtMask = imresize(gtMask, imageSize(1:2), 'nearest');

gtBinary = gtMask > 0;

predictedBinary = strcmp(string(predictedMask), "vessel");

if ~any(predictedBinary(:))
    predictedBinary = strcmpi(string(predictedMask), "Vessel");
end

figure('Name', 'TRUST-DR Vessel Segmentation Result (V2)', ...
       'Color', 'white');

subplot(1,3,1);
imshow(img);
title('Original DRIVE Fundus Image');

subplot(1,3,2);
imshow(gtBinary);
title('Ground Truth Vessel Mask');

subplot(1,3,3);
imshow(predictedBinary);
title(sprintf('Predicted Vessel Mask (V2)\nDice: %.2f%% | IoU: %.2f%%', ...
    diceList(testIndex)*100, ...
    iouList(testIndex)*100));

sgtitle('TRUST-DR: Retinal Blood Vessel Segmentation (V2 - Augmented + Weighted)');

fprintf('\n============================================\n');
fprintf(' VESSEL SEGMENTATION V2 COMPLETE\n');
fprintf('============================================\n');

fprintf('\nTRUST-DR Vessel Segmentation V2 module is ready.\n');


%% ============================================
% HELPER FUNCTIONS (must stay at end of file)
% ============================================

function dataOut = augmentImageAndMask(data)
    img = data{1};
    mask = data{2};

    maskCats = categories(mask);
    maskNum = uint8(mask);

    % Random rotation (-10 to +10 degrees), same angle for image + mask
    angle = rand*20 - 10;
    img = imrotate(img, angle, 'bilinear', 'crop');
    maskNum = imrotate(maskNum, angle, 'nearest', 'crop');

    % Random horizontal flip, same decision for both
    if rand > 0.5
        img = fliplr(img);
        maskNum = fliplr(maskNum);
    end

    % Random vertical flip, same decision for both
    if rand > 0.5
        img = flipud(img);
        maskNum = flipud(maskNum);
    end

    mask = categorical(maskNum, 1:numel(maskCats), maskCats);
    dataOut = {img, mask};
end