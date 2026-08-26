%% IDRiD_Evaluate_Balanced_Weighted_UNet.m
% ============================================================
% IDRiD BALANCED WEIGHTED U-NET EVALUATION
%
% Classes:
% 0 = Background
% 1 = Microaneurysms (MA)
% 2 = Haemorrhages (HE)
% 3 = Hard Exudates (EX)
% 4 = Soft Exudates (SE)
% 5 = Optic Disc (OD)
%
% This script:
% 1. Loads the balanced weighted U-Net
% 2. Recreates the SAME validation split used during training
% 3. Runs predictions
% 4. Calculates Pixel Accuracy
% 5. Calculates Dice for every class
% 6. Calculates IoU for every class
% 7. Calculates mean foreground Dice and IoU
% 8. Saves all evaluation results
% ============================================================

clear;
clc;
close all;

%% ============================================================
% 1. DISPLAY HEADER
% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('IDRiD BALANCED WEIGHTED U-NET EVALUATION\n');
fprintf('============================================\n\n');


%% ============================================================
% 2. DATASET PATHS
% ============================================================

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
% 3. MODEL PATH
% ============================================================

% IMPORTANT:
% Change this filename ONLY if your balanced training script
% saved the model using a different name.

modelPath = fullfile( ...
    basePath, ...
    'IDRiD_UNet_BalancedWeighted_256.mat');


%% ============================================================
% 4. LOAD TRAINED MODEL
% ============================================================

fprintf('Loading trained balanced weighted model...\n\n');

if ~isfile(modelPath)

    error(['Model file not found:\n' modelPath ...
        '\n\nPlease check the filename used in your training script.']);

end

loadedModel = load(modelPath);

fprintf('Model file loaded successfully.\n\n');


%% ============================================================
% 5. EXTRACT VARIABLES FROM MODEL FILE
% ============================================================

if isfield(loadedModel, 'net')
    net = loadedModel.net;
else
    error('The saved model file does not contain a variable named "net".');
end


if isfield(loadedModel, 'classNames')
    classNames = loadedModel.classNames;
else

    classNames = [
        "Background"
        "MA"
        "HE"
        "EX"
        "SE"
        "OD"
    ];

end


if isfield(loadedModel, 'labelIDs')
    labelIDs = loadedModel.labelIDs;
else
    labelIDs = [0 1 2 3 4 5];
end


if isfield(loadedModel, 'inputSize')
    inputSize = loadedModel.inputSize;
else
    inputSize = [256 256];
end


numClasses = numel(classNames);


fprintf('Model loaded successfully.\n');
fprintf('Input size: %d x %d\n', ...
    inputSize(1), inputSize(2));

fprintf('Number of classes: %d\n\n', numClasses);


%% ============================================================
% 6. CREATE IMAGE DATASTORE
% ============================================================

imds = imageDatastore( ...
    trainImagePath, ...
    'FileExtensions', {'.jpg'});


%% ============================================================
% 7. CREATE PIXEL LABEL DATASTORE
% ============================================================

pxds = pixelLabelDatastore( ...
    maskPath, ...
    classNames, ...
    labelIDs, ...
    'FileExtensions', {'.png'});


fprintf('Total images found: %d\n', numel(imds.Files));
fprintf('Total masks found: %d\n\n', numel(pxds.Files));


%% ============================================================
% 8. VERIFY IMAGE-MASK PAIRING
% ============================================================

fprintf('Verifying image-mask pairing...\n');

for i = 1:numel(imds.Files)

    [~, imageName, ~] = fileparts(imds.Files{i});
    [~, maskName, ~] = fileparts(pxds.Files{i});

    expectedMaskName = [imageName '_mask'];

    if ~strcmp(maskName, expectedMaskName)

        error( ...
            'Image-mask mismatch:\nImage: %s\nMask: %s', ...
            imageName, ...
            maskName);

    end

end

fprintf('SUCCESS: Image-mask pairing verified.\n\n');


%% ============================================================
% 9. RECREATE SAME TRAIN / VALIDATION SPLIT
% ============================================================
%
% IMPORTANT:
% This must use the SAME random seed and split method
% used in the training script.
%
% Training script:
% rng(42)
% indices = randperm(numImages)
% numTrain = round(0.8 * numImages)
%
% ============================================================

rng(42);

numImages = numel(imds.Files);

indices = randperm(numImages);

numTrain = round(0.8 * numImages);

trainIdx = indices(1:numTrain);
valIdx = indices(numTrain+1:end);


imdsVal = subset(imds, valIdx);
pxdsVal = subset(pxds, valIdx);


fprintf('\n');
fprintf('============================================\n');
fprintf('VALIDATION DATA\n');
fprintf('============================================\n\n');

fprintf('Validation images: %d\n', ...
    numel(imdsVal.Files));

fprintf('Validation data resized to: %d x %d\n\n', ...
    inputSize(1), inputSize(2));


%% ============================================================
% 10. INITIALIZE METRIC VARIABLES
% ============================================================

totalCorrectPixels = 0;
totalPixels = 0;

intersection = zeros(numClasses, 1);
union = zeros(numClasses, 1);

predictedPixels = zeros(numClasses, 1);
groundTruthPixels = zeros(numClasses, 1);


%% ============================================================
% 11. STORE FIRST IMAGE FOR VISUALIZATION
% ============================================================

firstImage = [];
firstGroundTruth = [];
firstPrediction = [];


%% ============================================================
% 12. RUN PREDICTIONS
% ============================================================

fprintf('============================================\n');
fprintf('RUNNING PREDICTIONS\n');
fprintf('============================================\n\n');


for i = 1:numel(imdsVal.Files)

    %% --------------------------------------------------------
    % Read original image
    % ---------------------------------------------------------

    image = imread(imdsVal.Files{i});


    %% --------------------------------------------------------
    % Read ground truth mask
    % ---------------------------------------------------------

    groundTruthCategorical = readimage(pxdsVal, i);


    %% --------------------------------------------------------
    % Convert categorical ground truth to numeric labels
    % ---------------------------------------------------------

    groundTruth = zeros(size(groundTruthCategorical), 'uint8');

    for c = 1:numClasses

        groundTruth( ...
            groundTruthCategorical == classNames(c)) = ...
            uint8(labelIDs(c));

    end


    %% --------------------------------------------------------
    % Resize image
    % ---------------------------------------------------------

    resizedImage = imresize(image, inputSize);


    %% --------------------------------------------------------
    % Resize ground truth
    % ---------------------------------------------------------

    resizedGroundTruth = imresize( ...
        groundTruth, ...
        inputSize, ...
        'nearest');


    %% --------------------------------------------------------
    % Convert image to single
    % ---------------------------------------------------------

    resizedImage = single(resizedImage);


    %% --------------------------------------------------------
    % Run network prediction
    % ---------------------------------------------------------
    %
    % trainnet returns a dlnetwork.
    % predict requires numeric data.
    %
    % Network output expected as:
    % H x W x Classes
    %
    % ---------------------------------------------------------

    scores = predict(net, resizedImage);


    %% --------------------------------------------------------
    % Convert dlarray output if necessary
    % ---------------------------------------------------------

    if isa(scores, 'dlarray')

        scores = extractdata(scores);

    end


    %% --------------------------------------------------------
    % Move from GPU if necessary
    % ---------------------------------------------------------

    if isa(scores, 'gpuArray')

        scores = gather(scores);

    end


    %% --------------------------------------------------------
    % Ensure numeric type
    % ---------------------------------------------------------

    scores = double(scores);


    %% --------------------------------------------------------
    % Remove unnecessary singleton dimensions
    % ---------------------------------------------------------

    scores = squeeze(scores);


    %% --------------------------------------------------------
    % Convert network scores to predicted class
    % ---------------------------------------------------------

    [~, predictionIndex] = max(scores, [], 3);


    %% --------------------------------------------------------
    % Convert class indices to original label IDs
    % ---------------------------------------------------------

    prediction = zeros(size(predictionIndex), 'uint8');

    for c = 1:numClasses

        prediction(predictionIndex == c) = ...
            uint8(labelIDs(c));

    end


    %% ========================================================
    % CALCULATE PIXEL ACCURACY
    % ========================================================

    correctPixels = sum( ...
        prediction(:) == resizedGroundTruth(:));

    totalCorrectPixels = ...
        totalCorrectPixels + correctPixels;

    totalPixels = ...
        totalPixels + numel(resizedGroundTruth);


    %% ========================================================
    % CALCULATE CLASS-WISE INTERSECTION AND UNION
    % ========================================================

    for c = 1:numClasses

        classID = labelIDs(c);


        %% Ground truth pixels for this class

        gtClass = resizedGroundTruth == classID;


        %% Predicted pixels for this class

        predClass = prediction == classID;


        %% Intersection

        classIntersection = sum( ...
            gtClass(:) & predClass(:));


        %% Union

        classUnion = sum( ...
            gtClass(:) | predClass(:));


        %% Accumulate

        intersection(c) = ...
            intersection(c) + classIntersection;

        union(c) = ...
            union(c) + classUnion;


        %% Total predicted pixels

        predictedPixels(c) = ...
            predictedPixels(c) + sum(predClass(:));


        %% Total ground truth pixels

        groundTruthPixels(c) = ...
            groundTruthPixels(c) + sum(gtClass(:));

    end


    %% ========================================================
    % SAVE FIRST IMAGE FOR VISUALIZATION
    % ========================================================

    if i == 1

        firstImage = resizedImage;

        firstGroundTruth = resizedGroundTruth;

        firstPrediction = prediction;

    end


    %% ========================================================
    % DISPLAY PROGRESS
    % ========================================================

    fprintf( ...
        'Processed validation image %d/%d\n', ...
        i, ...
        numel(imdsVal.Files));

end


fprintf('\n');
fprintf('All validation predictions completed.\n\n');


%% ============================================================
% 13. VERIFY LABELS
% ============================================================

fprintf('============================================\n');
fprintf('VERIFYING LABELS\n');
fprintf('============================================\n\n');


groundTruthClasses = unique(firstGroundTruth);
predictionClasses = unique(firstPrediction);


fprintf('Classes actually present in first ground truth:\n');

for i = 1:numel(groundTruthClasses)

    classIndex = find( ...
        labelIDs == groundTruthClasses(i), ...
        1);

    fprintf('%s\n', classNames(classIndex));

end


fprintf('\n');


fprintf('Classes actually present in first prediction:\n');

for i = 1:numel(predictionClasses)

    classIndex = find( ...
        labelIDs == predictionClasses(i), ...
        1);

    fprintf('%s\n', classNames(classIndex));

end


fprintf('\n');


%% ============================================================
% 14. CALCULATE OVERALL PIXEL ACCURACY
% ============================================================

pixelAccuracy = ...
    totalCorrectPixels / totalPixels;


%% ============================================================
% 15. CALCULATE DICE AND IoU
% ============================================================

diceScores = zeros(numClasses, 1);

iouScores = zeros(numClasses, 1);


for c = 1:numClasses

    %% --------------------------------------------------------
    % Dice
    %
    % Dice = 2 * Intersection /
    %        (Predicted Pixels + Ground Truth Pixels)
    % --------------------------------------------------------

    denominatorDice = ...
        predictedPixels(c) + groundTruthPixels(c);


    if denominatorDice == 0

        diceScores(c) = NaN;

    else

        diceScores(c) = ...
            (2 * intersection(c)) / denominatorDice;

    end


    %% --------------------------------------------------------
    % IoU
    %
    % IoU = Intersection / Union
    % --------------------------------------------------------

    if union(c) == 0

        iouScores(c) = NaN;

    else

        iouScores(c) = ...
            intersection(c) / union(c);

    end

end


%% ============================================================
% 16. CALCULATE FOREGROUND MEAN METRICS
% ============================================================
%
% Excludes Background
%
% Classes:
% 2 = MA
% 3 = HE
% 4 = EX
% 5 = SE
% 6 = OD
%
% ============================================================

foregroundDice = diceScores(2:end);

foregroundIoU = iouScores(2:end);


% Ignore NaN values

meanDice = mean( ...
    foregroundDice(~isnan(foregroundDice)));

meanIoU = mean( ...
    foregroundIoU(~isnan(foregroundIoU)));


%% ============================================================
% 17. DISPLAY RESULTS
% ============================================================

fprintf('============================================\n');
fprintf('CALCULATING METRICS\n');
fprintf('============================================\n\n');


fprintf( ...
    'Overall Pixel Accuracy: %.4f (%.2f%%)\n\n', ...
    pixelAccuracy, ...
    pixelAccuracy * 100);


fprintf('============================================\n');
fprintf('CLASS-WISE SEGMENTATION RESULTS\n');
fprintf('============================================\n\n');


fprintf('%-15s %-12s %-12s\n', ...
    'Class', ...
    'Dice', ...
    'IoU');

fprintf('------------------------------------------\n');


for c = 1:numClasses

    fprintf('%-15s %-12.4f %-12.4f\n', ...
        char(classNames(c)), ...
        diceScores(c), ...
        iouScores(c));

end


fprintf('\n');


fprintf('============================================\n');
fprintf('OVERALL FOREGROUND RESULTS\n');
fprintf('============================================\n\n');


fprintf( ...
    'Mean Dice (excluding background): %.4f\n', ...
    meanDice);

fprintf( ...
    'Mean IoU  (excluding background): %.4f\n\n', ...
    meanIoU);


%% ============================================================
% 18. CREATE RESULTS TABLE
% ============================================================

resultsTable = table( ...
    classNames, ...
    diceScores, ...
    iouScores, ...
    predictedPixels, ...
    groundTruthPixels, ...
    'VariableNames', { ...
        'Class', ...
        'Dice', ...
        'IoU', ...
        'PredictedPixels', ...
        'GroundTruthPixels'});


disp(resultsTable);


%% ============================================================
% 19. VISUALIZE FIRST VALIDATION RESULT
% ============================================================

fprintf('\n');
fprintf('Creating visualization...\n');


figure( ...
    'Name', ...
    'IDRiD Balanced Weighted U-Net Evaluation');


%% Original Image

subplot(1, 3, 1);

imshow(uint8(firstImage));

title('Original Image');


%% Ground Truth

subplot(1, 3, 2);

imagesc(double(firstGroundTruth));

axis image;

axis off;

title('Ground Truth');

colormap(parula(numClasses));

caxis([0 numClasses - 1]);

colorbar;


%% Prediction

subplot(1, 3, 3);

imagesc(double(firstPrediction));

axis image;

axis off;

title('Balanced U-Net Prediction');

colormap(parula(numClasses));

caxis([0 numClasses - 1]);

colorbar;


%% ============================================================
% 20. SAVE RESULTS
% ============================================================

resultsPath = fullfile( ...
    basePath, ...
    'IDRiD_Balanced_Weighted_UNet_Evaluation_Results.mat');


save( ...
    resultsPath, ...
    'pixelAccuracy', ...
    'diceScores', ...
    'iouScores', ...
    'meanDice', ...
    'meanIoU', ...
    'resultsTable', ...
    'classNames', ...
    'labelIDs', ...
    'predictedPixels', ...
    'groundTruthPixels', ...
    'valIdx');


%% ============================================================
% 21. FINAL SUMMARY
% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('EVALUATION COMPLETE\n');
fprintf('============================================\n\n');


fprintf( ...
    'Pixel Accuracy : %.4f (%.2f%%)\n', ...
    pixelAccuracy, ...
    pixelAccuracy * 100);

fprintf( ...
    'Mean Dice      : %.4f\n', ...
    meanDice);

fprintf( ...
    'Mean IoU       : %.4f\n\n', ...
    meanIoU);


fprintf('Results saved to:\n\n');

fprintf('%s\n\n', resultsPath);


fprintf('============================================\n');
fprintf('CLASS SUMMARY\n');
fprintf('============================================\n');


for c = 1:numClasses

    fprintf( ...
        '%-12s Dice: %.4f   IoU: %.4f\n', ...
        char(classNames(c)), ...
        diceScores(c), ...
        iouScores(c));

end


fprintf('\n');
fprintf('============================================\n');
fprintf('BALANCED WEIGHTED U-NET EVALUATION FINISHED\n');
fprintf('============================================\n');