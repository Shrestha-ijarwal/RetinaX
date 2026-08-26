%% IDRiD_Evaluate_UNet.m
% Evaluate trained U-Net on IDRiD validation dataset
% Corrected version for MATLAB R2026a trainnet workflow

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

modelPath = fullfile( ...
    basePath, ...
    'IDRiD_UNet_Segmentation_256.mat');


%% ============================================================
% 2. LOAD TRAINED MODEL
% =============================================================

fprintf('\n============================================\n');
fprintf('IDRiD U-NET EVALUATION\n');
fprintf('============================================\n');

fprintf('\nLoading trained model...\n');

load(modelPath, ...
    'net', ...
    'classNames', ...
    'labelIDs', ...
    'inputSize');

numClasses = numel(classNames);

fprintf('Model loaded successfully.\n');
fprintf('Input size: %d x %d\n', ...
    inputSize(1), inputSize(2));

fprintf('Number of classes: %d\n', numClasses);


%% ============================================================
% 3. CREATE IMAGE DATASTORE
% =============================================================

imds = imageDatastore( ...
    trainImagePath, ...
    'FileExtensions', {'.jpg'});

fprintf('\nTotal images found: %d\n', numel(imds.Files));


%% ============================================================
% 4. CREATE PIXEL LABEL DATASTORE
% =============================================================

pxds = pixelLabelDatastore( ...
    maskPath, ...
    classNames, ...
    labelIDs, ...
    'FileExtensions', {'.png'});

fprintf('Total masks found: %d\n', numel(pxds.Files));


%% ============================================================
% 5. RECREATE SAME TRAIN / VALIDATION SPLIT
% =============================================================

rng(42);

numImages = numel(imds.Files);

indices = randperm(numImages);

numTrain = round(0.8 * numImages);

trainIdx = indices(1:numTrain);
valIdx = indices(numTrain+1:end);

imdsVal = subset(imds, valIdx);
pxdsVal = subset(pxds, valIdx);

fprintf('\n============================================\n');
fprintf('VALIDATION DATA\n');
fprintf('============================================\n');

fprintf('Validation images: %d\n', ...
    numel(imdsVal.Files));


%% ============================================================
% 6. COMBINE AND RESIZE VALIDATION DATA
% =============================================================

dsVal = combine(imdsVal, pxdsVal);

dsValResized = transform( ...
    dsVal, ...
    @(data) resizeData(data, inputSize));

fprintf('Validation data resized to: %d x %d\n', ...
    inputSize(1), inputSize(2));


%% ============================================================
% 7. RUN PREDICTIONS
% =============================================================

fprintf('\n============================================\n');
fprintf('RUNNING PREDICTIONS\n');
fprintf('============================================\n');

reset(dsValResized);

numVal = numel(imdsVal.Files);

predictedMasks = cell(numVal, 1);
groundTruthMasks = cell(numVal, 1);
validationImages = cell(numVal, 1);

for i = 1:numVal

    % Read image and ground truth
    data = read(dsValResized);

    image = data{1};
    groundTruth = data{2};

    % Store image and ground truth
    validationImages{i} = image;
    groundTruthMasks{i} = groundTruth;


    
    % --------------------------------------------------------
    % PREPARE IMAGE FOR dlnetwork
    % --------------------------------------------------------

    % Convert uint8 image to single in range [0,1]
    imageInput = im2single(image);

    % Convert H x W x C image to dlarray
    % SSCB = Spatial, Spatial, Channel, Batch
    imageInput = dlarray(imageInput, 'SSCB');


    % --------------------------------------------------------
    % PREDICT NETWORK SCORES
    % --------------------------------------------------------

    scores = predict(net, imageInput);

    % Convert dlarray to normal numeric array
    scores = extractdata(scores);


    % --------------------------------------------------------
    % REMOVE SINGLE BATCH DIMENSION
    % --------------------------------------------------------

    scores = squeeze(scores);


    % --------------------------------------------------------
    % FIND CLASS WITH HIGHEST SCORE
    % --------------------------------------------------------

    [~, predictedClassIndex] = max(scores, [], 3);


    % --------------------------------------------------------
    % CONVERT CLASS INDICES TO CATEGORICAL LABELS
    % --------------------------------------------------------

    predictedMask = categorical( ...
        predictedClassIndex, ...
        1:numClasses, ...
        classNames);


    % --------------------------------------------------------
    % FIND CLASS WITH HIGHEST SCORE
    % --------------------------------------------------------

    [~, predictedClassIndex] = max(scores, [], 3);


    % --------------------------------------------------------
    % CONVERT NUMERIC CLASS INDICES TO OUR CLASS NAMES
    %
    % 1 = Background
    % 2 = MA
    % 3 = HE
    % 4 = EX
    % 5 = SE
    % 6 = OD
    % --------------------------------------------------------

    predictedMask = categorical( ...
        predictedClassIndex, ...
        1:numClasses, ...
        classNames);


    % Store prediction
    predictedMasks{i} = predictedMask;

    fprintf('Processed validation image %d/%d\n', ...
        i, numVal);

end

fprintf('All validation predictions completed.\n');


%% ============================================================
% 8. VERIFY LABEL FORMATS
% =============================================================

fprintf('\n============================================\n');
fprintf('VERIFYING LABELS\n');
fprintf('============================================\n');

fprintf('\nGround truth classes in first image:\n');
disp(categories(groundTruthMasks{1}));

fprintf('\nPredicted classes in first image:\n');
disp(categories(predictedMasks{1}));

fprintf('\nClasses actually present in first ground truth:\n');
disp(unique(groundTruthMasks{1}));

fprintf('\nClasses actually present in first prediction:\n');
disp(unique(predictedMasks{1}));


%% ============================================================
% 9. CALCULATE OVERALL PIXEL ACCURACY
% =============================================================

fprintf('\n============================================\n');
fprintf('CALCULATING METRICS\n');
fprintf('============================================\n');

totalCorrect = 0;
totalPixels = 0;

for i = 1:numVal

    gt = groundTruthMasks{i};
    pred = predictedMasks{i};

    totalCorrect = totalCorrect + ...
        sum(gt(:) == pred(:));

    totalPixels = totalPixels + numel(gt);

end

pixelAccuracy = totalCorrect / totalPixels;

fprintf('\nOverall Pixel Accuracy: %.4f (%.2f%%)\n', ...
    pixelAccuracy, pixelAccuracy * 100);


%% ============================================================
% 10. CALCULATE PER-CLASS DICE AND IoU
% =============================================================

diceScores = zeros(numClasses, 1);
iouScores = zeros(numClasses, 1);

for c = 1:numClasses

    currentClass = classNames(c);

    totalIntersection = 0;
    totalPrediction = 0;
    totalGroundTruth = 0;
    totalUnion = 0;

    for i = 1:numVal

        gt = groundTruthMasks{i};
        pred = predictedMasks{i};

        % Convert to binary masks for current class
        gtBinary = (gt == currentClass);
        predBinary = (pred == currentClass);

        % Intersection
        intersection = sum(gtBinary(:) & predBinary(:));

        % Pixel counts
        predictionCount = sum(predBinary(:));
        groundTruthCount = sum(gtBinary(:));

        % Union
        unionCount = sum(gtBinary(:) | predBinary(:));

        totalIntersection = ...
            totalIntersection + intersection;

        totalPrediction = ...
            totalPrediction + predictionCount;

        totalGroundTruth = ...
            totalGroundTruth + groundTruthCount;

        totalUnion = ...
            totalUnion + unionCount;

    end


    % --------------------------------------------------------
    % DICE SCORE
    % --------------------------------------------------------

    if (totalPrediction + totalGroundTruth) == 0

        diceScores(c) = NaN;

    else

        diceScores(c) = ...
            (2 * totalIntersection) / ...
            (totalPrediction + totalGroundTruth);

    end


    % --------------------------------------------------------
    % IoU SCORE
    % --------------------------------------------------------

    if totalUnion == 0

        iouScores(c) = NaN;

    else

        iouScores(c) = ...
            totalIntersection / totalUnion;

    end

end


%% ============================================================
% 11. DISPLAY CLASS-WISE RESULTS
% =============================================================

fprintf('\n============================================\n');
fprintf('CLASS-WISE SEGMENTATION RESULTS\n');
fprintf('============================================\n');

fprintf('\n%-15s %-12s %-12s\n', ...
    'Class', 'Dice', 'IoU');

fprintf('------------------------------------------\n');

for c = 1:numClasses

    fprintf('%-15s %-12.4f %-12.4f\n', ...
        char(classNames(c)), ...
        diceScores(c), ...
        iouScores(c));

end


%% ============================================================
% 12. MEAN FOREGROUND DICE AND IoU
% Background is excluded
% =============================================================

foregroundDice = diceScores(2:end);
foregroundIoU = iouScores(2:end);

meanDice = mean(foregroundDice, 'omitnan');
meanIoU = mean(foregroundIoU, 'omitnan');


fprintf('\n============================================\n');
fprintf('OVERALL FOREGROUND RESULTS\n');
fprintf('============================================\n');

fprintf('Mean Dice (excluding background): %.4f\n', ...
    meanDice);

fprintf('Mean IoU  (excluding background): %.4f\n', ...
    meanIoU);


%% ============================================================
% 13. CREATE RESULTS TABLE
% =============================================================

resultsTable = table( ...
    classNames(:), ...
    diceScores, ...
    iouScores, ...
    'VariableNames', {'Class', 'Dice', 'IoU'});

disp(resultsTable);


%% ============================================================
% 14. VISUALIZE FIRST VALIDATION RESULT
% =============================================================

exampleIndex = 1;

exampleImage = validationImages{exampleIndex};
exampleGT = groundTruthMasks{exampleIndex};
examplePrediction = predictedMasks{exampleIndex};

figure;

subplot(1,3,1);

imshow(exampleImage);

title('Original Retinal Image');


subplot(1,3,2);

imshow(labeloverlay( ...
    exampleImage, ...
    exampleGT));

title('Ground Truth');


subplot(1,3,3);

imshow(labeloverlay( ...
    exampleImage, ...
    examplePrediction));

title('U-Net Prediction');


%% ============================================================
% 15. SAVE EVALUATION RESULTS
% =============================================================

resultsPath = fullfile( ...
    basePath, ...
    'IDRiD_UNet_Evaluation_Results.mat');

save(resultsPath, ...
    'resultsTable', ...
    'pixelAccuracy', ...
    'meanDice', ...
    'meanIoU', ...
    'diceScores', ...
    'iouScores', ...
    'predictedMasks', ...
    'groundTruthMasks');

fprintf('\n============================================\n');
fprintf('EVALUATION COMPLETE\n');
fprintf('============================================\n');

fprintf('Pixel Accuracy : %.4f (%.2f%%)\n', ...
    pixelAccuracy, ...
    pixelAccuracy * 100);

fprintf('Mean Dice      : %.4f\n', ...
    meanDice);

fprintf('Mean IoU       : %.4f\n', ...
    meanIoU);

fprintf('\nResults saved to:\n%s\n', ...
    resultsPath);


%% ============================================================
% LOCAL FUNCTION
% =============================================================

function data = resizeData(data, inputSize)

    % Resize RGB retinal image
    data{1} = imresize(data{1}, inputSize);

    % Resize categorical segmentation mask
    % Nearest-neighbor preserves class labels
    data{2} = imresize( ...
        data{2}, ...
        inputSize, ...
        'nearest');

end