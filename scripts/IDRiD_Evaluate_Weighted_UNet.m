%% IDRiD_Evaluate_Weighted_UNet.m
% Evaluation of Weighted U-Net for IDRiD Multiclass Segmentation
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
% 1. DATASET AND MODEL PATHS
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
    'IDRiD_UNet_Weighted_256.mat');


fprintf('\n============================================\n');
fprintf('IDRiD WEIGHTED U-NET EVALUATION\n');
fprintf('============================================\n');


%% ============================================================
% 2. LOAD TRAINED MODEL
% =============================================================

fprintf('\nLoading trained weighted model...\n');

load(modelPath, ...
    'net', ...
    'classNames', ...
    'labelIDs', ...
    'inputSize');

numClasses = numel(classNames);

fprintf('Model loaded successfully.\n');

fprintf('Input size: %d x %d\n', ...
    inputSize(1), ...
    inputSize(2));

fprintf('Number of classes: %d\n', numClasses);


%% ============================================================
% 3. CREATE DATASTORES
% =============================================================

imds = imageDatastore( ...
    trainImagePath, ...
    'FileExtensions', {'.jpg'});

pxds = pixelLabelDatastore( ...
    maskPath, ...
    classNames, ...
    labelIDs, ...
    'FileExtensions', {'.png'});


fprintf('\nTotal images found: %d\n', ...
    numel(imds.Files));

fprintf('Total masks found: %d\n', ...
    numel(pxds.Files));


%% ============================================================
% 4. VERIFY IMAGE-MASK PAIRING
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

fprintf('\nValidation images: %d\n', ...
    numel(imdsVal.Files));

fprintf('Validation data resized to: %d x %d\n', ...
    inputSize(1), ...
    inputSize(2));


%% ============================================================
% 6. INITIALIZE METRIC VARIABLES
% =============================================================

% Intersection for each class
intersection = zeros(numClasses, 1);

% Total predicted pixels for each class
predictedCount = zeros(numClasses, 1);

% Total ground-truth pixels for each class
groundTruthCount = zeros(numClasses, 1);

% Overall correct pixels
correctPixels = 0;

% Total evaluated pixels
totalPixels = 0;


% Store first prediction and ground truth for visualization
firstImage = [];
firstPrediction = [];
firstGroundTruth = [];


%% ============================================================
% 7. RUN PREDICTIONS
% =============================================================

fprintf('\n============================================\n');
fprintf('RUNNING PREDICTIONS\n');
fprintf('============================================\n\n');


for i = 1:numel(imdsVal.Files)

    %% --------------------------------------------------------
    % Read validation image
    %% --------------------------------------------------------

    image = readimage(imdsVal, i);

    image = imresize(image, inputSize);


    %% --------------------------------------------------------
    % Read ground truth
    %% --------------------------------------------------------

    groundTruth = readimage(pxdsVal, i);

    groundTruth = imresize( ...
        groundTruth, ...
        inputSize, ...
        'nearest');


    %% --------------------------------------------------------
    % Convert image for dlnetwork prediction
    %
    % H = Height
    % W = Width
    % C = Channels
    % B = Batch
    %% --------------------------------------------------------

    image = single(image);

    dlX = dlarray(image, 'SSCB');


    %% --------------------------------------------------------
    % Run network prediction
    %% --------------------------------------------------------

    scores = predict(net, dlX);


    % Convert dlarray to normal numeric array
    scores = extractdata(scores);


    %% --------------------------------------------------------
    % Find predicted class
    %
    % Network output:
    % Height x Width x Classes x Batch
    %% --------------------------------------------------------

    [~, predictionIDs] = max(scores, [], 3);


    % Remove singleton batch dimension
    predictionIDs = squeeze(predictionIDs);


    %% --------------------------------------------------------
    % Convert categorical ground truth to numeric class IDs
    %% --------------------------------------------------------

    groundTruthIDs = zeros( ...
        size(groundTruth), ...
        'uint8');


    for c = 1:numClasses

        groundTruthIDs( ...
            groundTruth == classNames(c)) = labelIDs(c);

    end


    %% --------------------------------------------------------
    % Ensure prediction is numeric
    %% --------------------------------------------------------

    predictionIDs = uint8(predictionIDs - 1);


    %% --------------------------------------------------------
    % Calculate overall pixel accuracy
    %% --------------------------------------------------------

    correctPixels = correctPixels + ...
        sum(predictionIDs(:) == groundTruthIDs(:));

    totalPixels = totalPixels + ...
        numel(groundTruthIDs);


    %% --------------------------------------------------------
    % Calculate class statistics
    %% --------------------------------------------------------

    for c = 1:numClasses

        classID = labelIDs(c);

        predictedPixels = ...
            predictionIDs == classID;

        groundTruthPixels = ...
            groundTruthIDs == classID;


        intersection(c) = intersection(c) + ...
            sum(predictedPixels(:) & groundTruthPixels(:));


        predictedCount(c) = predictedCount(c) + ...
            sum(predictedPixels(:));


        groundTruthCount(c) = groundTruthCount(c) + ...
            sum(groundTruthPixels(:));

    end


    %% --------------------------------------------------------
    % Save first example
    %% --------------------------------------------------------

    if i == 1

        firstImage = image;

        firstPrediction = predictionIDs;

        firstGroundTruth = groundTruthIDs;

    end


    fprintf('Processed validation image %d/%d\n', ...
        i, ...
        numel(imdsVal.Files));

end


fprintf('\nAll validation predictions completed.\n');


%% ============================================================
% 8. VERIFY PREDICTED CLASSES
% =============================================================

fprintf('\n============================================\n');
fprintf('VERIFYING LABELS\n');
fprintf('============================================\n');


fprintf('\nClasses actually present in first ground truth:\n');

gtClasses = unique(firstGroundTruth);

for i = 1:numel(gtClasses)

    classIndex = gtClasses(i) + 1;

    fprintf('%s\n', ...
        classNames(classIndex));

end


fprintf('\nClasses actually present in first prediction:\n');

predClasses = unique(firstPrediction);

for i = 1:numel(predClasses)

    classIndex = predClasses(i) + 1;

    fprintf('%s\n', ...
        classNames(classIndex));

end


%% ============================================================
% 9. CALCULATE PIXEL ACCURACY
% =============================================================

pixelAccuracy = correctPixels / totalPixels;


%% ============================================================
% 10. CALCULATE DICE AND IoU
% =============================================================

diceScores = zeros(numClasses, 1);

iouScores = zeros(numClasses, 1);


for c = 1:numClasses

    %% Dice coefficient
    denominatorDice = ...
        predictedCount(c) + groundTruthCount(c);

    if denominatorDice > 0

        diceScores(c) = ...
            (2 * intersection(c)) / denominatorDice;

    else

        diceScores(c) = NaN;

    end


    %% Intersection over Union
    unionCount = ...
        predictedCount(c) + ...
        groundTruthCount(c) - ...
        intersection(c);

    if unionCount > 0

        iouScores(c) = ...
            intersection(c) / unionCount;

    else

        iouScores(c) = NaN;

    end

end


%% ============================================================
% 11. CALCULATE FOREGROUND MEAN SCORES
% =============================================================

foregroundDice = diceScores(2:end);

foregroundIoU = iouScores(2:end);


% Ignore NaN values if any class is absent
meanDice = mean( ...
    foregroundDice, ...
    'omitnan');

meanIoU = mean( ...
    foregroundIoU, ...
    'omitnan');


%% ============================================================
% 12. DISPLAY RESULTS
% =============================================================

fprintf('\n============================================\n');
fprintf('CALCULATING METRICS\n');
fprintf('============================================\n');

fprintf('\nOverall Pixel Accuracy: %.4f (%.2f%%)\n', ...
    pixelAccuracy, ...
    pixelAccuracy * 100);


fprintf('\n============================================\n');
fprintf('CLASS-WISE SEGMENTATION RESULTS\n');
fprintf('============================================\n\n');


fprintf('%-15s %-12s %-12s\n', ...
    'Class', ...
    'Dice', ...
    'IoU');

fprintf('------------------------------------------\n');


for c = 1:numClasses

    fprintf('%-15s %-12.4f %-12.4f\n', ...
        classNames(c), ...
        diceScores(c), ...
        iouScores(c));

end


fprintf('\n============================================\n');
fprintf('OVERALL FOREGROUND RESULTS\n');
fprintf('============================================\n');


fprintf('\nMean Dice (excluding background): %.4f\n', ...
    meanDice);

fprintf('Mean IoU  (excluding background): %.4f\n', ...
    meanIoU);


%% ============================================================
% 13. CREATE RESULTS TABLE
% =============================================================

resultsTable = table( ...
    classNames, ...
    diceScores, ...
    iouScores, ...
    predictedCount, ...
    groundTruthCount, ...
    'VariableNames', ...
    {'Class', ...
     'Dice', ...
     'IoU', ...
     'PredictedPixels', ...
     'GroundTruthPixels'});


fprintf('\n');

disp(resultsTable);


%% ============================================================
% 14. VISUALIZE FIRST VALIDATION RESULT
% =============================================================

figure( ...
    'Name', 'IDRiD Weighted U-Net Evaluation', ...
    'NumberTitle', 'off');


subplot(1,3,1);

imshow(uint8(firstImage));

title('Original Image');


subplot(1,3,2);

imagesc(firstGroundTruth);

axis image off;

title('Ground Truth');

colormap(gca, 'parula');

colorbar;

caxis([0 numClasses-1]);


subplot(1,3,3);

imagesc(firstPrediction);

axis image off;

title('Prediction');

colormap(gca, 'parula');

colorbar;

caxis([0 numClasses-1]);


%% ============================================================
% 15. SAVE EVALUATION RESULTS
% =============================================================

resultsPath = fullfile( ...
    basePath, ...
    'IDRiD_UNet_Weighted_Evaluation_Results.mat');


save(resultsPath, ...
    'resultsTable', ...
    'pixelAccuracy', ...
    'diceScores', ...
    'iouScores', ...
    'meanDice', ...
    'meanIoU', ...
    'intersection', ...
    'predictedCount', ...
    'groundTruthCount', ...
    'classNames', ...
    'labelIDs');


%% ============================================================
% 16. FINAL OUTPUT
% =============================================================

fprintf('\n============================================\n');
fprintf('EVALUATION COMPLETE\n');
fprintf('============================================\n');


fprintf('\nPixel Accuracy : %.4f (%.2f%%)\n', ...
    pixelAccuracy, ...
    pixelAccuracy * 100);

fprintf('Mean Dice      : %.4f\n', ...
    meanDice);

fprintf('Mean IoU       : %.4f\n', ...
    meanIoU);


fprintf('\nResults saved to:\n%s\n', ...
    resultsPath);


fprintf('\n============================================\n');
fprintf('CLASS SUMMARY\n');
fprintf('============================================\n');


for c = 1:numClasses

    fprintf( ...
        '%-12s Dice: %.4f   IoU: %.4f\n', ...
        classNames(c), ...
        diceScores(c), ...
        iouScores(c));

end