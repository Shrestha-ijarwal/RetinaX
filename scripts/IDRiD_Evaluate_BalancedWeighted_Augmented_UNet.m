%% ============================================================
% IDRiD_Evaluate_BalancedWeighted_Augmented_UNet.m
%
% FINAL EVALUATION SCRIPT
%
% Balanced Weighted + Augmented U-Net
%
% Classes:
% 0 = Background
% 1 = MA
% 2 = HE
% 3 = EX
% 4 = SE
% 5 = OD
%
% IMPORTANT:
% - Recreates the SAME rng(42) train/validation split
% - Explicitly maps prediction class names to numeric IDs
% - Calculates metrics using numeric class IDs
% - Avoids categorical class-order mismatch
% =============================================================

clear;
clc;
close all;

fprintf('\n');
fprintf('============================================\n');
fprintf('IDRiD U-NET EVALUATION\n');
fprintf('BALANCED WEIGHTED + AUGMENTED MODEL\n');
fprintf('============================================\n\n');


%% ============================================================
% 1. DATASET PATHS
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
% 2. MODEL PATH
% ============================================================
%
% Your actual balanced-weighted model filename was:
%
% IDRiD_UNet_BalancedWeighted_256.mat
%
% If your AUGMENTED training script saved a different .mat file,
% change ONLY the filename below.
%
% First try the augmented model filename if you know it.
% ============================================================

modelCandidates = { ...
    'IDRiD_UNet_BalancedWeighted_Augmented_256.mat', ...
    'IDRiD_UNet_BalancedWeighted_Augmented.mat', ...
    'IDRiD_UNet_BalancedWeighted_256.mat' ...
    };

modelPath = '';

for k = 1:numel(modelCandidates)

    candidate = fullfile(basePath, modelCandidates{k});

    if isfile(candidate)
        modelPath = candidate;
        break;
    end

end

if isempty(modelPath)

    error(['No balanced weighted augmented model was found.\n\n' ...
           'Please check the .mat filename in:\n%s\n\n' ...
           'Available .mat files can be checked with:\n' ...
           'dir(fullfile(basePath,''*.mat''))'], ...
           basePath);

end


%% ============================================================
% 3. LOAD MODEL
% =============================================================

fprintf('Loading trained model...\n');

modelData = load(modelPath);

fprintf('Model file:\n%s\n\n', modelPath);


%% ============================================================
% 4. FIND NETWORK VARIABLE
% =============================================================

if isfield(modelData,'net')

    net = modelData.net;

elseif isfield(modelData,'trainedNet')

    net = modelData.trainedNet;

elseif isfield(modelData,'dlnet')

    net = modelData.dlnet;

else

    error(['Could not find the trained network variable.\n' ...
           'Expected variable name: net']);

end


%% ============================================================
% 5. DEFINE CLASSES EXPLICITLY
% =============================================================

classNames = [ ...
    "Background"
    "MA"
    "HE"
    "EX"
    "SE"
    "OD"
    ];

labelIDs = 0:5;

numClasses = 6;

inputSize = [256 256];


fprintf('Model loaded successfully.\n');

fprintf('\nModel information:\n');
fprintf('  Input size       : %d x %d\n', ...
    inputSize(1), inputSize(2));

fprintf('  Number of classes: %d\n', ...
    numClasses);

fprintf('\nClasses:\n');

for c = 1:numClasses

    fprintf('  %d = %s\n', ...
        labelIDs(c), ...
        classNames(c));

end


%% ============================================================
% 6. CREATE IMAGE DATASTORE
% =============================================================

fprintf('\nCreating image datastore...\n');

imds = imageDatastore( ...
    trainImagePath, ...
    'FileExtensions', {'.jpg'});


%% ============================================================
% 7. CREATE PIXEL LABEL DATASTORE
% =============================================================

fprintf('Creating pixel label datastore...\n');

pxds = pixelLabelDatastore( ...
    maskPath, ...
    classNames, ...
    labelIDs, ...
    'FileExtensions', {'.png'});


fprintf('\nDataset information:\n');

fprintf('  Images: %d\n', ...
    numel(imds.Files));

fprintf('  Masks : %d\n', ...
    numel(pxds.Files));


%% ============================================================
% 8. VERIFY IMAGE-MASK PAIRING
% =============================================================

fprintf('\nVerifying image-mask pairing...\n');

for i = 1:numel(imds.Files)

    [~, imageName, ~] = fileparts(imds.Files{i});

    [~, maskName, ~] = fileparts(pxds.Files{i});

    expectedMaskName = [imageName '_mask'];

    if ~strcmp(maskName, expectedMaskName)

        error(['Image-mask mismatch:\n' ...
               'Image: %s\n' ...
               'Mask : %s\n' ...
               'Expected: %s'], ...
               imageName, ...
               maskName, ...
               expectedMaskName);

    end

end

fprintf('SUCCESS: Image-mask pairing verified.\n');


%% ============================================================
% 9. RECREATE EXACT TRAIN / VALIDATION SPLIT
% =============================================================
%
% This MUST match the training scripts:
%
% rng(42)
% randperm(numImages)
% numTrain = round(0.8*numImages)
%
% 54 images -> 43 training + 11 validation
% =============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('RECREATING VALIDATION SPLIT\n');
fprintf('============================================\n');

rng(42);

numImages = numel(imds.Files);

indices = randperm(numImages);

numTrain = round(0.8 * numImages);

trainIdx = indices(1:numTrain);

valIdx = indices(numTrain+1:end);

fprintf('\nTotal images      : %d\n', numImages);
fprintf('Training images   : %d\n', numel(trainIdx));
fprintf('Validation images : %d\n', numel(valIdx));


%% ============================================================
% 10. CREATE VALIDATION DATASTORES
% =============================================================

imdsVal = subset(imds, valIdx);

pxdsVal = subset(pxds, valIdx);


%% ============================================================
% 11. DISPLAY VALIDATION FILES
% =============================================================

fprintf('\nValidation image IDs:\n');

for i = 1:numel(imdsVal.Files)

    [~, name, ~] = fileparts(imdsVal.Files{i});

    fprintf('%02d: %s\n', i, name);

end


%% ============================================================
% 12. STORAGE FOR METRICS
% =============================================================

confusionMatrix = zeros(numClasses, numClasses, 'double');


% Number of validation images
numVal = numel(imdsVal.Files);


% Store predictions and ground truth for optional visualization
allPredictions = cell(numVal,1);
allGroundTruth = cell(numVal,1);


%% ============================================================
% 13. RUN VALIDATION PREDICTIONS
% =============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('EVALUATING VALIDATION SET\n');
fprintf('============================================\n\n');


for i = 1:numVal

    fprintf('Evaluating image %d/%d...\n', ...
        i, numVal);


    %% --------------------------------------------------------
    % Read RGB image
    % ---------------------------------------------------------

    originalImage = imread(imdsVal.Files{i});


    % Convert grayscale image to RGB if necessary
    if ndims(originalImage) == 2

        originalImage = repmat( ...
            originalImage, ...
            [1 1 3]);

    end


    %% --------------------------------------------------------
    % Resize image
    % ---------------------------------------------------------

    inputImage = imresize( ...
        originalImage, ...
        inputSize);


    %% --------------------------------------------------------
    % Convert image to single
    % ---------------------------------------------------------

    inputImage = single(inputImage);


    %% --------------------------------------------------------
    % Normalize image
    %
    % trainnet normally receives image values from the datastore.
    % Keep the same 0-255 convention here.
    % ---------------------------------------------------------

    % Do NOT divide by 255 unless the training script did so.
    % The training script supplied RGB images directly.

    
    %% --------------------------------------------------------
    % Read ground-truth mask
    % ---------------------------------------------------------

    gtMask = imread(pxdsVal.Files{i});


    %% --------------------------------------------------------
    % Convert GT mask to numeric IDs
    % ---------------------------------------------------------

    gtNumeric = convertMaskToIDs( ...
        gtMask, ...
        classNames, ...
        labelIDs);


    %% --------------------------------------------------------
    % Resize GT mask
    % ---------------------------------------------------------

    gtNumeric = imresize( ...
        gtNumeric, ...
        inputSize, ...
        'nearest');


    %% --------------------------------------------------------
    % PREDICTION
    % ---------------------------------------------------------

    scores = predict(net, inputImage);


    %% --------------------------------------------------------
    % Convert prediction scores to class IDs
    % ---------------------------------------------------------

    predNumeric = convertScoresToIDs( ...
        scores, ...
        classNames, ...
        labelIDs);


    %% --------------------------------------------------------
    % Make sure prediction is 2-D
    % ---------------------------------------------------------

    predNumeric = squeeze(predNumeric);

    gtNumeric = squeeze(gtNumeric);


    %% --------------------------------------------------------
    % Safety check
    % ---------------------------------------------------------

    if ~isequal(size(predNumeric), size(gtNumeric))

        error(['Prediction/ground-truth size mismatch on image %d.\n' ...
               'Prediction: %s\n' ...
               'Ground truth: %s'], ...
               i, ...
               mat2str(size(predNumeric)), ...
               mat2str(size(gtNumeric)));

    end


    %% --------------------------------------------------------
    % Store
    % ---------------------------------------------------------

    allPredictions{i} = predNumeric;

    allGroundTruth{i} = gtNumeric;


    %% --------------------------------------------------------
    % Update confusion matrix
    %
    % ROW    = Ground Truth
    % COLUMN = Prediction
    % ---------------------------------------------------------

    for gtClass = 0:5

        for predClass = 0:5

            confusionMatrix( ...
                gtClass+1, ...
                predClass+1) = ...
                confusionMatrix( ...
                gtClass+1, ...
                predClass+1) + ...
                sum( ...
                (gtNumeric(:) == gtClass) & ...
                (predNumeric(:) == predClass));

        end

    end

end


fprintf('\nAll validation predictions completed.\n');


%% ============================================================
% 14. VERIFY LABELS
% =============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('VERIFYING LABELS\n');
fprintf('============================================\n');


firstGT = allGroundTruth{1};

firstPrediction = allPredictions{1};


gtPresent = unique(firstGT(:));

predPresent = unique(firstPrediction(:));


fprintf('\nClasses actually present in first ground truth:\n');

for i = 1:numel(gtPresent)

    id = gtPresent(i);

    fprintf('  %d = %s\n', ...
        id, ...
        classNames(id+1));

end


fprintf('\nClasses actually present in first prediction:\n');

for i = 1:numel(predPresent)

    id = predPresent(i);

    fprintf('  %d = %s\n', ...
        id, ...
        classNames(id+1));

end


%% ============================================================
% 15. CALCULATE METRICS
% =============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('CALCULATING METRICS\n');
fprintf('============================================\n');


dice = zeros(numClasses,1);

iou = zeros(numClasses,1);

precision = zeros(numClasses,1);

recall = zeros(numClasses,1);

predictedPixels = zeros(numClasses,1);

groundTruthPixels = zeros(numClasses,1);


for c = 1:numClasses

    TP = confusionMatrix(c,c);

    FP = sum(confusionMatrix(:,c)) - TP;

    FN = sum(confusionMatrix(c,:)) - TP;


    predictedPixels(c) = ...
        sum(confusionMatrix(:,c));

    groundTruthPixels(c) = ...
        sum(confusionMatrix(c,:));


    %% Precision

    if (TP + FP) > 0

        precision(c) = ...
            TP / (TP + FP);

    else

        precision(c) = 0;

    end


    %% Recall

    if (TP + FN) > 0

        recall(c) = ...
            TP / (TP + FN);

    else

        recall(c) = 0;

    end


    %% Dice

    if (2*TP + FP + FN) > 0

        dice(c) = ...
            (2*TP) / ...
            (2*TP + FP + FN);

    else

        dice(c) = 0;

    end


    %% IoU

    if (TP + FP + FN) > 0

        iou(c) = ...
            TP / ...
            (TP + FP + FN);

    else

        iou(c) = 0;

    end

end


%% ============================================================
% 16. OVERALL PIXEL ACCURACY
% =============================================================

totalCorrect = trace(confusionMatrix);

totalPixels = sum(confusionMatrix(:));

pixelAccuracy = ...
    totalCorrect / totalPixels;


%% ============================================================
% 17. MEAN METRICS
% =============================================================

meanIoUAll = mean(iou);

meanDiceAll = mean(dice);

meanPrecisionAll = mean(precision);

meanRecallAll = mean(recall);


% Lesion classes = classes 1-5
meanIoULesion = mean(iou(2:end));

meanDiceLesion = mean(dice(2:end));

meanPrecisionLesion = mean(precision(2:end));

meanRecallLesion = mean(recall(2:end));


%% ============================================================
% 18. DISPLAY FINAL RESULTS
% =============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('FINAL EVALUATION RESULTS\n');
fprintf('============================================\n\n');


fprintf('Overall Pixel Accuracy : %.4f (%.2f%%)\n', ...
    pixelAccuracy, ...
    pixelAccuracy*100);


fprintf('\nAll classes including Background:\n');

fprintf('  Mean IoU       : %.4f (%.2f%%)\n', ...
    meanIoUAll, ...
    meanIoUAll*100);

fprintf('  Mean Dice      : %.4f (%.2f%%)\n', ...
    meanDiceAll, ...
    meanDiceAll*100);

fprintf('  Mean Precision : %.4f (%.2f%%)\n', ...
    meanPrecisionAll, ...
    meanPrecisionAll*100);

fprintf('  Mean Recall    : %.4f (%.2f%%)\n', ...
    meanRecallAll, ...
    meanRecallAll*100);


fprintf('\nLesion classes only:\n');

fprintf('  Mean IoU       : %.4f (%.2f%%)\n', ...
    meanIoULesion, ...
    meanIoULesion*100);

fprintf('  Mean Dice      : %.4f (%.2f%%)\n', ...
    meanDiceLesion, ...
    meanDiceLesion*100);

fprintf('  Mean Precision : %.4f (%.2f%%)\n', ...
    meanPrecisionLesion, ...
    meanPrecisionLesion*100);

fprintf('  Mean Recall    : %.4f (%.2f%%)\n', ...
    meanRecallLesion, ...
    meanRecallLesion*100);


%% ============================================================
% 19. PER-CLASS RESULTS
% =============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('PER-CLASS RESULTS\n');
fprintf('============================================\n\n');


fprintf('%-15s %-12s %-12s %-12s %-12s\n', ...
    'Class', ...
    'IoU', ...
    'Dice', ...
    'Precision', ...
    'Recall');

fprintf('%s\n', ...
    repmat('-',1,70));


for c = 1:numClasses

    fprintf('%-15s %-12.4f %-12.4f %-12.4f %-12.4f\n', ...
        classNames(c), ...
        iou(c), ...
        dice(c), ...
        precision(c), ...
        recall(c));

end


%% ============================================================
% 20. CONFUSION MATRIX
% =============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('CONFUSION MATRIX\n');
fprintf('============================================\n\n');


disp(array2table( ...
    confusionMatrix, ...
    'VariableNames', cellstr(classNames), ...
    'RowNames', cellstr(classNames)));


%% ============================================================
% 21. PREDICTED / GROUND-TRUTH PIXELS
% =============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('PIXEL COUNTS\n');
fprintf('============================================\n\n');


fprintf('%-15s %-18s %-18s\n', ...
    'Class', ...
    'PredictedPixels', ...
    'GroundTruthPixels');

fprintf('%s\n', ...
    repmat('-',1,55));


for c = 1:numClasses

    fprintf('%-15s %-18d %-18d\n', ...
        classNames(c), ...
        predictedPixels(c), ...
        groundTruthPixels(c));

end


%% ============================================================
% 22. QUALITATIVE VISUALIZATION
% =============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('GENERATING QUALITATIVE RESULTS\n');
fprintf('============================================\n');


figure('Name', ...
    'IDRiD Augmented U-Net Evaluation', ...
    'NumberTitle', ...
    'off');


% Display first validation prediction

originalImage = imread(imdsVal.Files{1});

subplot(1,3,1);

imshow(originalImage);

title('Original Image');


subplot(1,3,2);

imagesc(allGroundTruth{1});

axis image off;

title('Ground Truth');

colorbar;


subplot(1,3,3);

imagesc(allPredictions{1});

axis image off;

title('Prediction');

colorbar;


%% ============================================================
% 23. SAVE RESULTS
% =============================================================

resultsFolder = fullfile( ...
    basePath, ...
    'IDRiD_Augmented_Evaluation');


if ~exist(resultsFolder,'dir')

    mkdir(resultsFolder);

end


%% ============================================================
% 24. SAVE MAT FILE
% =============================================================

resultsMatPath = fullfile( ...
    resultsFolder, ...
    'IDRiD_BalancedWeighted_Augmented_Evaluation_Results.mat');


save(resultsMatPath, ...
    'pixelAccuracy', ...
    'meanIoUAll', ...
    'meanDiceAll', ...
    'meanPrecisionAll', ...
    'meanRecallAll', ...
    'meanIoULesion', ...
    'meanDiceLesion', ...
    'meanPrecisionLesion', ...
    'meanRecallLesion', ...
    'iou', ...
    'dice', ...
    'precision', ...
    'recall', ...
    'predictedPixels', ...
    'groundTruthPixels', ...
    'confusionMatrix', ...
    'classNames', ...
    'labelIDs', ...
    'inputSize', ...
    'valIdx', ...
    'modelPath');


%% ============================================================
% 25. SAVE CSV
% =============================================================

csvPath = fullfile( ...
    resultsFolder, ...
    'IDRiD_BalancedWeighted_Augmented_PerClass_Metrics.csv');


metricsTable = table( ...
    classNames, ...
    iou, ...
    dice, ...
    precision, ...
    recall, ...
    predictedPixels, ...
    groundTruthPixels, ...
    'VariableNames', { ...
    'Class', ...
    'IoU', ...
    'Dice', ...
    'Precision', ...
    'Recall', ...
    'PredictedPixels', ...
    'GroundTruthPixels'});


writetable(metricsTable, csvPath);


%% ============================================================
% 26. SAVE CONFUSION MATRIX CSV
% =============================================================

confusionCSVPath = fullfile( ...
    resultsFolder, ...
    'IDRiD_BalancedWeighted_Augmented_ConfusionMatrix.csv');


confusionTable = array2table( ...
    confusionMatrix, ...
    'VariableNames', cellstr(classNames), ...
    'RowNames', cellstr(classNames));


writetable( ...
    confusionTable, ...
    confusionCSVPath, ...
    'WriteRowNames', true);


%% ============================================================
% 27. SAVE QUALITATIVE FIGURE
% =============================================================

figurePath = fullfile( ...
    resultsFolder, ...
    'IDRiD_Augmented_Qualitative_Result.png');


saveas(gcf, figurePath);


%% ============================================================
% 28. FINAL OUTPUT
% =============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('EVALUATION COMPLETE\n');
fprintf('============================================\n\n');


fprintf('Validation images : %d\n\n', numVal);


fprintf('Overall metrics:\n');

fprintf('  Pixel Accuracy : %.4f\n', ...
    pixelAccuracy);

fprintf('  Mean IoU       : %.4f\n', ...
    meanIoUAll);

fprintf('  Mean Dice      : %.4f\n', ...
    meanDiceAll);

fprintf('  Mean Precision : %.4f\n', ...
    meanPrecisionAll);

fprintf('  Mean Recall    : %.4f\n', ...
    meanRecallAll);


fprintf('\nLesion-only metrics:\n');

fprintf('  Mean IoU       : %.4f\n', ...
    meanIoULesion);

fprintf('  Mean Dice      : %.4f\n', ...
    meanDiceLesion);

fprintf('  Mean Precision : %.4f\n', ...
    meanPrecisionLesion);

fprintf('  Mean Recall    : %.4f\n', ...
    meanRecallLesion);


fprintf('\nResults saved to:\n%s\n', ...
    resultsMatPath);

fprintf('\nCSV results saved to:\n%s\n', ...
    csvPath);

fprintf('\nConfusion matrix saved to:\n%s\n', ...
    confusionCSVPath);

fprintf('\nVisualization saved to:\n%s\n', ...
    figurePath);


fprintf('\n');
fprintf('============================================\n');
fprintf('CLASS SUMMARY\n');
fprintf('============================================\n');


for c = 1:numClasses

    fprintf('%-12s IoU: %.4f   Dice: %.4f   Precision: %.4f   Recall: %.4f\n', ...
        classNames(c), ...
        iou(c), ...
        dice(c), ...
        precision(c), ...
        recall(c));

end


fprintf('\n');
fprintf('============================================\n');
fprintf('BALANCED WEIGHTED + AUGMENTED EVALUATION FINISHED\n');
fprintf('============================================\n');


%% ============================================================
% LOCAL FUNCTION 1
% Convert pixel-label image to numeric IDs
% =============================================================

function numericMask = convertMaskToIDs(mask, classNames, labelIDs)

    % If categorical, explicitly use category names
    if iscategorical(mask)

        numericMask = zeros(size(mask), 'uint8');

        for k = 1:numel(classNames)

            numericMask( ...
                mask == classNames(k)) = ...
                uint8(labelIDs(k));

        end

        return;

    end


    % If numeric mask
    numericMask = uint8(mask);

    % Handle logical masks
    if islogical(mask)

        numericMask = uint8(mask);

    end

end


%% ============================================================
% LOCAL FUNCTION 2
% Convert network prediction scores to numeric class IDs
% =============================================================

function numericPrediction = convertScoresToIDs( ...
    scores, classNames, labelIDs)

    % ---------------------------------------------------------
    % Case 1:
    % Prediction returned as numeric array
    % ---------------------------------------------------------

    if isnumeric(scores)

        scoresSize = size(scores);


        % Expected format:
        % H x W x C
        if numel(scoresSize) == 3 && ...
                scoresSize(3) == numel(classNames)

            [~, classIndex] = max(scores, [], 3);

            numericPrediction = ...
                uint8(classIndex - 1);

            return;

        end


        % If already H x W
        if ismatrix(scores)

            numericPrediction = ...
                uint8(scores);

            return;

        end

    end


    % ---------------------------------------------------------
    % Case 2:
    % Categorical prediction
    % ---------------------------------------------------------

    if iscategorical(scores)

        numericPrediction = ...
            zeros(size(scores), 'uint8');

        categoriesPresent = categories(scores);

        for k = 1:numel(classNames)

            targetName = classNames(k);

            % Find matching category
            match = strcmp( ...
                string(categoriesPresent), ...
                string(targetName));

            if any(match)

                numericPrediction( ...
                    scores == targetName) = ...
                    uint8(labelIDs(k));

            end

        end

        return;

    end


    % ---------------------------------------------------------
    % Case 3:
    % Cell output
    % ---------------------------------------------------------

    if iscell(scores)

        scores = scores{1};

        numericPrediction = ...
            convertScoresToIDs( ...
            scores, ...
            classNames, ...
            labelIDs);

        return;

    end


    % ---------------------------------------------------------
    % Unknown output
    % ---------------------------------------------------------

    error(['Unable to interpret network prediction output.\n' ...
           'Prediction class: %s\n' ...
           'Prediction size: %s'], ...
           class(scores), ...
           mat2str(size(scores)));

end