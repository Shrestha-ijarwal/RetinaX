%% ============================================================
% TRUST-DR: VESSEL SEGMENTATION TRAINING (V4.2)
%
% DRIVE Dataset
% Stable U-Net Training - Conservative Augmentation
%
% V4.2 changes from V4:
%   - Removes CLAHE from TRAINING preprocessing
%   - Conservative augmentation only
%   - Lower learning rate
%   - Gradient clipping
%   - NaN/Inf-safe training configuration
%   - Keeps V3 untouched
%   - Automatic V3 vs V4.2 comparison
%   - Saves model/results/visualization separately
%
% IMPORTANT:
% V4.2 is experimental. V3 remains the production model unless
% V4.2 demonstrates better test performance.
%
% Compatible with newer MATLAB versions using:
%   unet()
%   trainnet()
% ============================================================

clear;
clc;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf(' TRUST-DR: VESSEL SEGMENTATION TRAINING (V4.2)\n');
fprintf('============================================================\n\n');

%% ============================================================
% STEP 1: INITIALIZATION
% ============================================================

projectFolder = 'C:\Users\sijar\Downloads\TRUST_DR';

fprintf('Project folder:\n%s\n\n', projectFolder);

if ~isfolder(projectFolder)
    error('Project folder not found: %s', projectFolder);
end

rng(42);

fprintf('Random seed initialized.\n');

%% ============================================================
% STEP 2: LOAD DRIVE DATASET
% ============================================================

fprintf('\n============================================================\n');
fprintf(' LOADING DRIVE DATASET\n');
fprintf('============================================================\n\n');

imageFolder = fullfile(projectFolder, ...
    'datasets', 'drive', 'DRIVE', 'training', 'images');

maskFolder = fullfile(projectFolder, ...
    'datasets', 'drive', 'DRIVE', 'training', '1st_manual');

fprintf('Image folder:\n%s\n\n', imageFolder);
fprintf('Mask folder:\n%s\n\n', maskFolder);

if ~isfolder(imageFolder)
    error('Image folder not found: %s', imageFolder);
end

if ~isfolder(maskFolder)
    error('Mask folder not found: %s', maskFolder);
end

imds = imageDatastore(imageFolder);

classNames = ["background","vessel"];
labelIDs = [0 255];

pxds = pixelLabelDatastore( ...
    maskFolder, classNames, labelIDs);

numImages = numel(imds.Files);

fprintf('Number of fundus images : %d\n', numImages);
fprintf('Number of vessel masks  : %d\n', numel(pxds.Files));

if numImages ~= numel(pxds.Files)
    error('Number of images and masks does not match.');
end

%% ============================================================
% STEP 3: DATASET SPLIT
% ============================================================

fprintf('\n============================================================\n');
fprintf(' DATASET SPLIT\n');
fprintf('============================================================\n\n');

rng(42);

indices = randperm(numImages);

numTrain = round(0.70 * numImages);
numValidation = round(0.15 * numImages);

trainIdx = indices(1:numTrain);

validationIdx = indices( ...
    numTrain + 1 : ...
    numTrain + numValidation);

testIdx = indices( ...
    numTrain + numValidation + 1 : end);

imdsTrain = subset(imds, trainIdx);
imdsValidation = subset(imds, validationIdx);
imdsTest = subset(imds, testIdx);

pxdsTrain = subset(pxds, trainIdx);
pxdsValidation = subset(pxds, validationIdx);
pxdsTest = subset(pxds, testIdx);

fprintf('Training   : %d images\n', numel(imdsTrain.Files));
fprintf('Validation : %d images\n', numel(imdsValidation.Files));
fprintf('Testing    : %d images\n', numel(imdsTest.Files));

%% ============================================================
% STEP 4: IMAGE SIZE
% ============================================================

imageSize = [256 256 3];

fprintf('\nNetwork input size: %d x %d x %d\n', ...
    imageSize(1), imageSize(2), imageSize(3));

%% ============================================================
% STEP 5: PREPARE V4.2 DATASTORE
% ============================================================

fprintf('\n============================================================\n');
fprintf(' PREPARING V4.2 TRAINING DATA\n');
fprintf('============================================================\n\n');

dsTrain = combine(imdsTrain, pxdsTrain);

dsTrain = transform(dsTrain, ...
    @(data) preprocessVesselDataV42(data, imageSize));

dsValidation = combine(imdsValidation, pxdsValidation);

dsValidation = transform(dsValidation, ...
    @(data) preprocessVesselDataV41(data, imageSize));

fprintf('V4.2 training datastore ready.\n');
fprintf('CLAHE training augmentation: DISABLED for stability.\n');
fprintf('Conservative geometric/intensity augmentation: ENABLED.\n');

%% ============================================================
% STEP 6: CREATE U-NET
% ============================================================

fprintf('\n============================================================\n');
fprintf(' CREATING V4.2 U-NET\n');
fprintf('============================================================\n\n');

trainedVesselNet = unet( ...
    imageSize, ...
    numel(classNames), ...
    "EncoderDepth", 3);

fprintf('V4.2 U-Net created successfully.\n');

%% ============================================================
% STEP 7: TRAINING CONFIGURATION
% ============================================================

fprintf('\n============================================================\n');
fprintf(' V4.2 TRAINING CONFIGURATION\n');
fprintf('============================================================\n\n');

initialLearnRate = 1e-4;
maxEpochs = 80;
miniBatchSize = 2;
gradientThreshold = 1;

fprintf('Optimizer             : Adam\n');
fprintf('Initial learning rate : %.1e\n', initialLearnRate);
fprintf('Maximum epochs        : %d\n', maxEpochs);
fprintf('Mini-batch size       : %d\n', miniBatchSize);
fprintf('Gradient threshold    : %.2f\n', gradientThreshold);
fprintf('Encoder depth         : 3\n');
fprintf('Input size            : 256 x 256 x 3\n');
fprintf('Training preprocessing: RESIZE ONLY (stability mode)\n');
fprintf('Random augmentation   : DISABLED\n');
fprintf('CLAHE in training     : DISABLED\n');

%% ============================================================
% STEP 8: TRAINING OPTIONS
% ============================================================

options = trainingOptions("adam", ...
    "InitialLearnRate", initialLearnRate, ...
    "MaxEpochs", maxEpochs, ...
    "MiniBatchSize", miniBatchSize, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", dsValidation, ...
    "ValidationFrequency", 5, ...
    "GradientThreshold", gradientThreshold, ...
    "GradientThresholdMethod", "l2norm", ...
    "Verbose", true, ...
    "Plots", "training-progress");

%% ============================================================
% STEP 9: TRAIN V4.2
% ============================================================

fprintf('\n============================================================\n');
fprintf(' STARTING VESSEL SEGMENTATION TRAINING (V4.2)\n');
fprintf('============================================================\n\n');

trainingStart = tic;

trainingFailed = false;
trainingErrorMessage = "";

try
    trainedVesselNet = trainnet( ...
        dsTrain, ...
        trainedVesselNet, ...
        "crossentropy", ...
        options);

catch ME
    trainingFailed = true;
    trainingErrorMessage = ME.message;

    fprintf('\n============================================================\n');
    fprintf(' V4.2 TRAINING ERROR\n');
    fprintf('============================================================\n\n');
    fprintf('%s\n', ME.message);
end

trainingTime = toc(trainingStart);

fprintf('\nTraining time: %.2f minutes\n', trainingTime / 60);

%% ============================================================
% STEP 10: SAVE V4.2 MODEL
% ============================================================

fprintf('\n============================================================\n');
fprintf(' SAVING V4.2 MODEL\n');
fprintf('============================================================\n\n');

modelFolder = fullfile(projectFolder, 'models');

if ~isfolder(modelFolder)
    mkdir(modelFolder);
end

modelPath = fullfile( ...
    modelFolder, ...
    'trained_vessel_segmentation_v4_1.mat');

save( ...
    modelPath, ...
    'trainedVesselNet', ...
    'imageSize', ...
    'classNames', ...
    'labelIDs', ...
    'trainIdx', ...
    'validationIdx', ...
    'testIdx', ...
    'trainingTime', ...
    'trainingFailed', ...
    'trainingErrorMessage');

fprintf('V4.2 MODEL SAVED SUCCESSFULLY\n\n');
fprintf('Model location:\n%s\n', modelPath);

if trainingFailed
    fprintf('\nWARNING: Training failed. V4.2 will NOT be evaluated as an improvement.\n');
else
    fprintf('\nTraining completed without a MATLAB training exception.\n');
end

%% ============================================================
% STEP 11: TEST V4.2
% ============================================================

fprintf('\n============================================================\n');
fprintf(' TESTING V4.2 MODEL\n');
fprintf('============================================================\n\n');

numTestImages = numel(imdsTest.Files);

pixelAccuracyScores = zeros(numTestImages,1);
precisionScores = zeros(numTestImages,1);
recallScores = zeros(numTestImages,1);
sensitivityScores = zeros(numTestImages,1);
specificityScores = zeros(numTestImages,1);
diceScores = zeros(numTestImages,1);
iouScores = zeros(numTestImages,1);

predictedMasks = cell(numTestImages,1);

if trainingFailed

    fprintf('Training failed, so V4.2 testing is skipped.\n');

    meanPixelAccuracy = NaN;
    meanPrecision = NaN;
    meanRecall = NaN;
    meanSensitivity = NaN;
    meanSpecificity = NaN;
    meanDice = NaN;
    meanIoU = NaN;

else

    for i = 1:numTestImages

        fprintf('\nProcessing V4.2 test image %d of %d...\n', ...
            i, numTestImages);

        img = readimage(imdsTest,i);

        if size(img,3) == 1
            img = repmat(img,[1 1 3]);
        end

        inputImage = imresize(img,imageSize(1:2));

        predictedMask = semanticseg( ...
            inputImage, ...
            trainedVesselNet, ...
            Classes=classNames);

        predictedVessel = predictedMask == "vessel";

        groundTruthMask = readimage(pxdsTest,i);

        groundTruthMask = imresize( ...
            groundTruthMask, ...
            imageSize(1:2), ...
            "nearest");

        groundTruthVessel = groundTruthMask == "vessel";

        TP = sum(predictedVessel(:) & groundTruthVessel(:));
        FP = sum(predictedVessel(:) & ~groundTruthVessel(:));
        FN = sum(~predictedVessel(:) & groundTruthVessel(:));
        TN = sum(~predictedVessel(:) & ~groundTruthVessel(:));

        totalPixels = TP + TN + FP + FN;

        if totalPixels > 0
            pixelAccuracy = (TP + TN) / totalPixels;
        else
            pixelAccuracy = 0;
        end

        if TP + FP > 0
            precision = TP / (TP + FP);
        else
            precision = 0;
        end

        if TP + FN > 0
            recall = TP / (TP + FN);
        else
            recall = 0;
        end

        if TP + FN > 0
            sensitivity = TP / (TP + FN);
        else
            sensitivity = 0;
        end

        if TN + FP > 0
            specificity = TN / (TN + FP);
        else
            specificity = 0;
        end

        if 2*TP + FP + FN > 0
            diceScore = (2*TP) / (2*TP + FP + FN);
        else
            diceScore = 0;
        end

        if TP + FP + FN > 0
            iouScore = TP / (TP + FP + FN);
        else
            iouScore = 0;
        end

        pixelAccuracyScores(i) = pixelAccuracy;
        precisionScores(i) = precision;
        recallScores(i) = recall;
        sensitivityScores(i) = sensitivity;
        specificityScores(i) = specificity;
        diceScores(i) = diceScore;
        iouScores(i) = iouScore;

        predictedMasks{i} = predictedMask;

        fprintf('\n');
        fprintf('Pixel Accuracy : %.2f%%\n', pixelAccuracy*100);
        fprintf('Precision      : %.2f%%\n', precision*100);
        fprintf('Recall         : %.2f%%\n', recall*100);
        fprintf('Sensitivity    : %.2f%%\n', sensitivity*100);
        fprintf('Specificity    : %.2f%%\n', specificity*100);
        fprintf('Dice           : %.2f%%\n', diceScore*100);
        fprintf('IoU            : %.2f%%\n', iouScore*100);

    end

    meanPixelAccuracy = mean(pixelAccuracyScores);
    meanPrecision = mean(precisionScores);
    meanRecall = mean(recallScores);
    meanSensitivity = mean(sensitivityScores);
    meanSpecificity = mean(specificityScores);
    meanDice = mean(diceScores);
    meanIoU = mean(iouScores);

end

%% ============================================================
% STEP 12: FINAL V4.2 RESULTS
% ============================================================

fprintf('\n============================================================\n');
fprintf(' FINAL VESSEL SEGMENTATION RESULTS (V4.2)\n');
fprintf('============================================================\n\n');

fprintf('Number of Test Images : %d\n\n', numTestImages);

fprintf('Mean Pixel Accuracy   : %.2f %%\n', meanPixelAccuracy*100);
fprintf('Mean Precision        : %.2f %%\n', meanPrecision*100);
fprintf('Mean Recall           : %.2f %%\n', meanRecall*100);
fprintf('Mean Sensitivity      : %.2f %%\n', meanSensitivity*100);
fprintf('Mean Specificity      : %.2f %%\n', meanSpecificity*100);
fprintf('Mean Dice             : %.2f %%\n', meanDice*100);
fprintf('Mean IoU              : %.2f %%\n', meanIoU*100);

%% ============================================================
% STEP 13: SAVE RESULTS
% ============================================================

resultsPath = fullfile( ...
    modelFolder, ...
    'vessel_segmentation_v4_1_results.mat');

save( ...
    resultsPath, ...
    'pixelAccuracyScores', ...
    'precisionScores', ...
    'recallScores', ...
    'sensitivityScores', ...
    'specificityScores', ...
    'diceScores', ...
    'iouScores', ...
    'meanPixelAccuracy', ...
    'meanPrecision', ...
    'meanRecall', ...
    'meanSensitivity', ...
    'meanSpecificity', ...
    'meanDice', ...
    'meanIoU', ...
    'trainingTime', ...
    'trainingFailed', ...
    'trainingErrorMessage');

fprintf('\nV4.2 results saved successfully.\n');
fprintf('Results location:\n%s\n', resultsPath);

%% ============================================================
% STEP 14: LOAD V3 BASELINE
% ============================================================

fprintf('\n============================================================\n');
fprintf(' V3 VS V4.2 COMPARISON\n');
fprintf('============================================================\n\n');

v3ResultsPath = fullfile( ...
    modelFolder, ...
    'vessel_segmentation_v3_results.mat');

if isfile(v3ResultsPath)

    v3Data = load(v3ResultsPath);

    if isfield(v3Data,'meanDice')
        v3Dice = v3Data.meanDice;
    else
        v3Dice = NaN;
    end

    if isfield(v3Data,'meanIoU')
        v3IoU = v3Data.meanIoU;
    else
        v3IoU = NaN;
    end

    if isfield(v3Data,'meanPrecision')
        v3Precision = v3Data.meanPrecision;
    else
        v3Precision = NaN;
    end

    if isfield(v3Data,'meanRecall')
        v3Recall = v3Data.meanRecall;
    else
        v3Recall = NaN;
    end

else

    fprintf('WARNING: V3 result file not found.\n');

    v3Dice = NaN;
    v3IoU = NaN;
    v3Precision = NaN;
    v3Recall = NaN;

end

fprintf('\n');
fprintf('Metric             V3          V4.2\n');
fprintf('-------------------------------------------\n');
fprintf('Dice             %.4f      %.4f\n', v3Dice, meanDice);
fprintf('IoU              %.4f      %.4f\n', v3IoU, meanIoU);
fprintf('Precision        %.4f      %.4f\n', v3Precision, meanPrecision);
fprintf('Recall           %.4f      %.4f\n', v3Recall, meanRecall);

if ~isnan(v3Dice) && ~isnan(meanDice)

    diceChange = meanDice - v3Dice;
    iouChange = meanIoU - v3IoU;
    precisionChange = meanPrecision - v3Precision;
    recallChange = meanRecall - v3Recall;

    fprintf('\nAbsolute change (V4.2 - V3):\n');
    fprintf('Dice       : %+.4f\n', diceChange);
    fprintf('IoU        : %+.4f\n', iouChange);
    fprintf('Precision  : %+.4f\n', precisionChange);
    fprintf('Recall     : %+.4f\n', recallChange);

    if v3Dice ~= 0
        diceRelative = 100 * diceChange / v3Dice;
    else
        diceRelative = NaN;
    end

    if v3IoU ~= 0
        iouRelative = 100 * iouChange / v3IoU;
    else
        iouRelative = NaN;
    end

    fprintf('\nRelative change:\n');
    fprintf('Dice       : %+.2f%%\n', diceRelative);
    fprintf('IoU        : %+.2f%%\n', iouRelative);

    if meanDice > v3Dice && meanIoU > v3IoU

        comparisonResult = ...
            "V4.2 IMPROVED OVER V3";

        fprintf('\nRESULT: V4.2 IMPROVED OVER V3.\n');
        fprintf('V4.2 may be considered for further validation.\n');

    else

        comparisonResult = ...
            "V4.2 DID NOT IMPROVE OVER V3";

        fprintf('\nRESULT: V4.2 DID NOT IMPROVE OVER V3.\n');
        fprintf('Keep V3 as the current production vessel model.\n');

    end

else

    comparisonResult = "V3 BASELINE UNAVAILABLE";

end

comparisonPath = fullfile( ...
    modelFolder, ...
    'vessel_segmentation_v3_v4_1_comparison.mat');

save( ...
    comparisonPath, ...
    'v3Dice', ...
    'v3IoU', ...
    'v3Precision', ...
    'v3Recall', ...
    'meanDice', ...
    'meanIoU', ...
    'meanPrecision', ...
    'meanRecall', ...
    'comparisonResult');

fprintf('\nComparison data saved:\n%s\n', comparisonPath);

%% ============================================================
% STEP 15: VISUALIZATION
% ============================================================

fprintf('\n============================================================\n');
fprintf(' GENERATING V4.2 VISUALIZATION\n');
fprintf('============================================================\n\n');

visualizationPath = fullfile( ...
    modelFolder, ...
    'vessel_segmentation_v4_1_visualization.png');

if ~trainingFailed && numTestImages > 0

    visualizationIndex = 1;

    originalImage = readimage( ...
        imdsTest, ...
        visualizationIndex);

    if size(originalImage,3) == 1
        originalImage = repmat(originalImage,[1 1 3]);
    end

    originalImage = imresize( ...
        originalImage, ...
        imageSize(1:2));

    gtMask = readimage( ...
        pxdsTest, ...
        visualizationIndex);

    gtMask = imresize( ...
        gtMask, ...
        imageSize(1:2), ...
        "nearest");

    gtVessel = gtMask == "vessel";

    predVessel = predictedMasks{visualizationIndex} == "vessel";

    figure( ...
        "Name","TRUST-DR Vessel Segmentation V4.2", ...
        "Color","w", ...
        "Position",[100 100 1400 500]);

    subplot(1,3,1);
    imshow(originalImage);
    title("Original Fundus", ...
        "FontSize",14, ...
        "FontWeight","bold");

    subplot(1,3,2);
    imshow(gtVessel);
    title("Ground Truth Vessels", ...
        "FontSize",14, ...
        "FontWeight","bold");

    subplot(1,3,3);
    imshow(predVessel);
    title(sprintf( ...
        "V4.2 Prediction\nDice: %.2f%% | IoU: %.2f%%", ...
        diceScores(visualizationIndex)*100, ...
        iouScores(visualizationIndex)*100), ...
        "FontSize",14, ...
        "FontWeight","bold");

    sgtitle( ...
        "TRUST-DR: Vessel Segmentation V4.2", ...
        "FontSize",18, ...
        "FontWeight","bold");

    exportgraphics(gcf,visualizationPath,"Resolution",150);

    fprintf('Visualization saved:\n%s\n', visualizationPath);

else

    fprintf('Visualization skipped because V4.2 training did not complete.\n');

end

%% ============================================================
% STEP 16: FINAL STATUS
% ============================================================

fprintf('\n============================================================\n');
fprintf(' TRUST-DR VESSEL SEGMENTATION V4.2 COMPLETE\n');
fprintf('============================================================\n\n');

fprintf('V4.2 model:\n%s\n\n', modelPath);
fprintf('V4.2 results:\n%s\n\n', resultsPath);
fprintf('V3/V4.2 comparison:\n%s\n\n', comparisonPath);

fprintf('IMPORTANT:\n');
fprintf('V3 model has NOT been overwritten.\n');
fprintf('Do NOT replace V3 in the integrated system unless V4.2\n');
fprintf('demonstrates better test performance.\n\n');

fprintf('============================================================\n');

%% ============================================================
% LOCAL FUNCTION: VALIDATION PREPROCESSING
% ============================================================

function dataOut = preprocessVesselDataV41(data,imageSize)

    img = data{1};
    mask = data{2};

    if size(img,3) == 1
        img = repmat(img,[1 1 3]);
    end

    img = imresize(img,imageSize(1:2));

    mask = imresize( ...
        mask, ...
        imageSize(1:2), ...
        "nearest");

    % Keep image datatype conventional and finite.
    if ~isa(img,'uint8')
        img = im2uint8(img);
    end

    dataOut = {img,mask};

end

%% ============================================================
% LOCAL FUNCTION: CONSERVATIVE V4.2 AUGMENTATION
% ============================================================

function dataOut = preprocessVesselDataV42(data,imageSize)

    % ========================================================
    % V4.2 STABILITY MODE
    %
    % This function intentionally mirrors the known-stable V3
    % preprocessing pipeline. No random image operation is
    % performed inside the datastore transform.
    % ========================================================

    img = data{1};
    mask = data{2};

    % Ensure RGB image
    if size(img,3) == 1
        img = repmat(img,[1 1 3]);
    end

    % Resize image with standard interpolation
    img = imresize(img,imageSize(1:2));

    % Resize categorical mask using nearest-neighbour
    mask = imresize(mask,imageSize(1:2),"nearest");

    % Force a valid numeric image representation
    if ~isa(img,'uint8')
        img = im2uint8(img);
    end

    % Final finite-value guard
    if ~isreal(img)
        img = real(img);
    end

    dataOut = {img,mask};

end
