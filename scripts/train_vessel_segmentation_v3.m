%% ============================================================
%  TRUST-DR: VESSEL SEGMENTATION TRAINING (V3)
%
%  DRIVE Dataset
%  Stable U-Net Training + Data Augmentation
%
%  Improvements over V1:
%    - Training data augmentation
%    - Lower learning rate
%    - Gradient clipping
%    - More training epochs
%    - Manual Dice / IoU evaluation
%    - Visualization
%
%  Compatible with newer MATLAB versions using:
%    unet()
% ============================================================

clear;
clc;
close all;

fprintf('\n');
fprintf('============================================\n');
fprintf(' TRUST-DR: VESSEL SEGMENTATION TRAINING (V3)\n');
fprintf('============================================\n\n');

%% ============================================================
% STEP 1: INITIALIZE TRUST-DR VESSEL SEGMENTATION V3
% ============================================================

clc;
clear;
close all;

fprintf('\n');
fprintf('============================================\n');
fprintf(' TRUST-DR: VESSEL SEGMENTATION TRAINING (V3)\n');
fprintf('============================================\n');

% ------------------------------------------------------------
% Define project folder
% ------------------------------------------------------------

projectFolder = 'C:\Users\sijar\Downloads\TRUST_DR';

fprintf('\nProject folder:\n');
fprintf('%s\n', projectFolder);

% ------------------------------------------------------------
% Verify project folder exists
% ------------------------------------------------------------

if ~isfolder(projectFolder)
    error('Project folder not found: %s', projectFolder);
end

fprintf('\nProject folder verified successfully.\n');

% ------------------------------------------------------------
% Set random seed for reproducibility
% ------------------------------------------------------------

rng(42);

fprintf('Random seed initialized.\n');

fprintf('\n============================================\n');
fprintf(' INITIALIZATION COMPLETE\n');
fprintf('============================================\n');


%% ============================================================
% STEP 2: LOAD DRIVE DATASET
% ============================================================

fprintf('\nLoading DRIVE dataset...\n');

% Define dataset paths
imageFolder = fullfile(projectFolder, ...
    'datasets', 'drive', 'DRIVE', 'training', 'images');

maskFolder = fullfile(projectFolder, ...
    'datasets', 'drive', 'DRIVE', 'training', '1st_manual');

% Display paths
fprintf('\nImage folder:\n%s\n', imageFolder);
fprintf('\nMask folder:\n%s\n', maskFolder);

% Check folders exist
if ~isfolder(imageFolder)
    error('Image folder not found: %s', imageFolder);
end

if ~isfolder(maskFolder)
    error('Mask folder not found: %s', maskFolder);
end

% Create image datastore
imds = imageDatastore(imageFolder);

% Define segmentation classes
classNames = ["background", "vessel"];
labelIDs = [0 255];

% Create pixel label datastore
pxds = pixelLabelDatastore( ...
    maskFolder, classNames, labelIDs);

% Display dataset information
fprintf('\nNumber of fundus images: %d\n', numel(imds.Files));
fprintf('Number of vessel masks: %d\n', numel(pxds.Files));

% Verify counts
if numel(imds.Files) ~= numel(pxds.Files)
    error('Number of images and masks does not match.');
end

fprintf('\n============================================\n');
fprintf(' DRIVE DATASET LOADED SUCCESSFULLY\n');
fprintf('============================================\n');

% Display dataset information
fprintf('\nNumber of fundus images: %d\n', numel(imds.Files));
fprintf('Number of vessel masks: %d\n', numel(pxds.Files));

% Verify image and mask counts
if numel(imds.Files) ~= numel(pxds.Files)
    error('Number of images and masks does not match.');
end

% Total number of images
numImages = numel(imds.Files);

fprintf('\n============================================\n');
fprintf(' DRIVE DATASET LOADED SUCCESSFULLY\n');
fprintf('============================================\n');


%% ============================================================
% STEP 3: SHUFFLE AND SPLIT DATASET
% ============================================================

rng(42);


% Random indices
indices = randperm(numImages);


% 70% Training
% 15% Validation
% 15% Testing

numTrain = round(0.70 * numImages);
numValidation = round(0.15 * numImages);


trainIdx = indices(1:numTrain);

validationIdx = indices( ...
    numTrain + 1 : ...
    numTrain + numValidation);


testIdx = indices( ...
    numTrain + numValidation + 1 : end);


% Split image datastore
imdsTrain = subset(imds, trainIdx);
imdsValidation = subset(imds, validationIdx);
imdsTest = subset(imds, testIdx);


% Split mask datastore
pxdsTrain = subset(pxds, trainIdx);
pxdsValidation = subset(pxds, validationIdx);
pxdsTest = subset(pxds, testIdx);


fprintf('Dataset split:\n');

fprintf('Training   : %d images\n', ...
    numel(imdsTrain.Files));

fprintf('Validation : %d images\n', ...
    numel(imdsValidation.Files));

fprintf('Testing    : %d images\n\n', ...
    numel(imdsTest.Files));


%% ============================================================
% STEP 4: IMAGE SIZE
% ============================================================

% Moderate resolution for CPU training
imageSize = [256 256 3];


fprintf('Network input size: %d x %d x %d\n\n', ...
    imageSize(1), ...
    imageSize(2), ...
    imageSize(3));


%% ============================================================
% STEP 5: CREATE TRAINING DATASTORE
% ============================================================

fprintf('Preparing training datastore...\n');


% Combine training images and masks
dsTrain = combine(imdsTrain, pxdsTrain);


% Apply preprocessing and augmentation
dsTrain = transform(dsTrain, ...
    @(data) augmentVesselData(data, imageSize));


% Validation datastore
dsValidation = combine(imdsValidation, pxdsValidation);


% Apply preprocessing only
dsValidation = transform(dsValidation, ...
    @(data) preprocessVesselData(data, imageSize));


fprintf('Training datastore ready.\n\n');


%% ============================================================
% STEP 6: CREATE U-NET NETWORK
% ============================================================

fprintf('Creating U-Net segmentation network...\n\n');


% Build U-Net
% "EncoderDepth" 3 keeps model size reasonable
trainedVesselNet = unet( ...
    imageSize, ...
    numel(classNames), ...
    "EncoderDepth", 3);


% Display network information
fprintf('Network created successfully.\n\n');

disp(trainedVesselNet);


% ============================================
% STEP 7: TRAINING OPTIONS
% ============================================

fprintf('\n');
fprintf('============================================\n');
fprintf(' STARTING VESSEL SEGMENTATION TRAINING (V3)\n');
fprintf('============================================\n');

options = trainingOptions("adam", ...
    "InitialLearnRate", 1e-4, ...
    "MaxEpochs", 50, ...
    "MiniBatchSize", 2, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", dsValidation, ...
    "ValidationFrequency", 5, ...
    "Verbose", true, ...
    "Plots", "training-progress");


%% ============================================================
% STEP 8: TRAIN NETWORK
% ============================================================

trainedVesselNet = trainnet( ...
    dsTrain, ...
    trainedVesselNet, ...
    "crossentropy", ...
    options);


%% ============================================================
% STEP 9: SAVE TRAINED MODEL
% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf(' SAVING V3 MODEL\n');
fprintf('============================================\n\n');

% Project root
projectFolder = 'C:\Users\sijar\Downloads\TRUST_DR';

% Model output folder
modelFolder = fullfile(projectFolder, 'models');

% Create folder if it does not exist
if ~exist(modelFolder, 'dir')
    mkdir(modelFolder);
end

% Model file path
modelPath = fullfile( ...
    modelFolder, ...
    'trained_vessel_segmentation_v3.mat');

% Save trained model and required metadata
save( ...
    modelPath, ...
    'trainedVesselNet', ...
    'imageSize', ...
    'classNames', ...
    'labelIDs', ...
    'trainIdx', ...
    'validationIdx', ...
    'testIdx');

fprintf('MODEL SAVED SUCCESSFULLY\n\n');
fprintf('Model location:\n%s\n\n', modelPath);


%% ============================================================
% STEP 10: TEST THE TRAINED MODEL
% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf(' TESTING VESSEL SEGMENTATION MODEL (V3)\n');
fprintf('============================================\n\n');


numTestImages = numel(imdsTest.Files);


pixelAccuracyScores = zeros(numTestImages, 1);

precisionScores = zeros(numTestImages, 1);

recallScores = zeros(numTestImages, 1);

diceScores = zeros(numTestImages, 1);

iouScores = zeros(numTestImages, 1);


% Store predictions
predictedMasks = cell(numTestImages, 1);


for i = 1:numTestImages

    fprintf('\n');
    fprintf('Processing test image %d of %d...\n', ...
        i, numTestImages);


    %% Load original image

    img = readimage(imdsTest, i);


    % Convert grayscale to RGB if needed
    if size(img, 3) == 1

        img = repmat(img, [1 1 3]);

    end


    % Resize image
    inputImage = imresize( ...
        img, ...
        imageSize(1:2));


    %% Predict vessel mask

    predictedMask = semanticseg( ...
        inputImage, ...
        trainedVesselNet, ...
        Classes = classNames);


    % Convert categorical prediction to logical vessel mask
    predictedVessel = ...
        predictedMask == "vessel";


    %% Load ground truth

    groundTruthMask = readimage(pxdsTest, i);


    % Resize ground truth
    groundTruthMask = imresize( ...
        groundTruthMask, ...
        imageSize(1:2), ...
        "nearest");


    % Convert categorical ground truth to logical vessel mask
    groundTruthVessel = ...
        groundTruthMask == "vessel";


    %% Calculate confusion values

    TP = sum( ...
        predictedVessel(:) & ...
        groundTruthVessel(:));


    FP = sum( ...
        predictedVessel(:) & ...
        ~groundTruthVessel(:));


    FN = sum( ...
        ~predictedVessel(:) & ...
        groundTruthVessel(:));


    TN = sum( ...
        ~predictedVessel(:) & ...
        ~groundTruthVessel(:));


    %% Pixel Accuracy

    pixelAccuracy = ...
        (TP + TN) / ...
        (TP + TN + FP + FN);


    %% Precision

    if (TP + FP) == 0

        precision = 0;

    else

        precision = TP / (TP + FP);

    end


    %% Recall

    if (TP + FN) == 0

        recall = 0;

    else

        recall = TP / (TP + FN);

    end


    %% Dice Score

    if (2 * TP + FP + FN) == 0

        diceScore = 0;

    else

        diceScore = ...
            (2 * TP) / ...
            (2 * TP + FP + FN);

    end


    %% IoU Score

    if (TP + FP + FN) == 0

        iouScore = 0;

    else

        iouScore = ...
            TP / ...
            (TP + FP + FN);

    end


    %% Store metrics

    pixelAccuracyScores(i) = pixelAccuracy;

    precisionScores(i) = precision;

    recallScores(i) = recall;

    diceScores(i) = diceScore;

    iouScores(i) = iouScore;


    %% Store prediction

    predictedMasks{i} = predictedMask;


    %% Display metrics

    fprintf('\n');

    fprintf('Pixel Accuracy : %.2f%%\n', ...
        pixelAccuracy * 100);

    fprintf('Precision      : %.2f%%\n', ...
        precision * 100);

    fprintf('Recall         : %.2f%%\n', ...
        recall * 100);

    fprintf('Dice Score     : %.2f%%\n', ...
        diceScore * 100);

    fprintf('IoU Score      : %.2f%%\n', ...
        iouScore * 100);

end


%% ============================================================
% STEP 11: FINAL RESULTS
% ============================================================

meanPixelAccuracy = mean(pixelAccuracyScores);

meanPrecision = mean(precisionScores);

meanRecall = mean(recallScores);

meanDice = mean(diceScores);

meanIoU = mean(iouScores);


fprintf('\n');
fprintf('============================================\n');
fprintf(' FINAL VESSEL SEGMENTATION RESULTS (V3)\n');
fprintf('============================================\n\n');


fprintf('Number of Test Images : %d\n\n', ...
    numTestImages);


fprintf('Mean Pixel Accuracy   : %.2f %%\n', ...
    meanPixelAccuracy * 100);

fprintf('Mean Precision        : %.2f %%\n', ...
    meanPrecision * 100);

fprintf('Mean Recall           : %.2f %%\n', ...
    meanRecall * 100);

fprintf('Mean Dice Score       : %.2f %%\n', ...
    meanDice * 100);

fprintf('Mean IoU Score        : %.2f %%\n', ...
    meanIoU * 100);


fprintf('============================================\n\n');


%% ============================================================
% STEP 12: SAVE RESULTS
% ============================================================

resultsPath = fullfile( ...
    modelFolder, ...
    'vessel_segmentation_v3_results.mat');


save( ...
    resultsPath, ...
    'pixelAccuracyScores', ...
    'precisionScores', ...
    'recallScores', ...
    'diceScores', ...
    'iouScores', ...
    'meanPixelAccuracy', ...
    'meanPrecision', ...
    'meanRecall', ...
    'meanDice', ...
    'meanIoU');


fprintf('Results saved successfully!\n\n');

fprintf('Results location:\n%s\n\n', ...
    resultsPath);


%% ============================================================
% STEP 13: VISUALIZE RESULTS
% ============================================================

fprintf('Generating vessel segmentation visualization...\n');


% Select a test image
visualizationIndex = 1;


% Load original image
originalImage = readimage( ...
    imdsTest, ...
    visualizationIndex);


% Load ground truth
groundTruthMask = readimage( ...
    pxdsTest, ...
    visualizationIndex);


% Resize images
displayOriginal = imresize( ...
    originalImage, ...
    imageSize(1:2));


if size(displayOriginal, 3) == 1

    displayOriginal = repmat( ...
        displayOriginal, ...
        [1 1 3]);

end


groundTruthMask = imresize( ...
    groundTruthMask, ...
    imageSize(1:2), ...
    "nearest");


% Convert masks to logical
groundTruthVessel = ...
    groundTruthMask == "vessel";


predictedMask = predictedMasks{visualizationIndex};

predictedVessel = ...
    predictedMask == "vessel";


%% Create figure

figure( ...
    "Name", ...
    "TRUST-DR: Vessel Segmentation V3", ...
    "Color", ...
    "w", ...
    "Position", ...
    [100 100 1500 500]);


%% Original Image

subplot(1, 3, 1);

imshow(displayOriginal);

title( ...
    "Original Fundus", ...
    "FontSize", 16, ...
    "FontWeight", "bold");


%% Ground Truth

subplot(1, 3, 2);

imshow(groundTruthVessel);

title( ...
    "Ground Truth Vessels", ...
    "FontSize", 16, ...
    "FontWeight", "bold");


%% Prediction

subplot(1, 3, 3);

imshow(predictedVessel);

title( ...
    sprintf( ...
        "Predicted Vessels\nDice: %.2f%% | IoU: %.2f%%", ...
        diceScores(visualizationIndex) * 100, ...
        iouScores(visualizationIndex) * 100), ...
    "FontSize", 16, ...
    "FontWeight", "bold");


sgtitle( ...
    "TRUST-DR: Retinal Blood Vessel Segmentation (V3)", ...
    "FontSize", 20, ...
    "FontWeight", "bold");


fprintf('\n');
fprintf('============================================\n');
fprintf(' VESSEL SEGMENTATION V3 COMPLETE\n');
fprintf('============================================\n\n');

fprintf('TRUST-DR Vessel Segmentation V3 module is ready.\n\n');


%% ============================================================
% LOCAL FUNCTION:
% PREPROCESS IMAGE + MASK
% ============================================================

function dataOut = preprocessVesselData(data, imageSize)

    img = data{1};

    mask = data{2};


    % Ensure RGB image
    if size(img, 3) == 1

        img = repmat(img, [1 1 3]);

    end


    % Resize image
    img = imresize( ...
        img, ...
        imageSize(1:2));


    % Resize categorical mask
    mask = imresize( ...
        mask, ...
        imageSize(1:2), ...
        "nearest");


    % Return processed data
    dataOut = {img, mask};

end


%% ============================================================
% LOCAL FUNCTION:
% DATA AUGMENTATION
% ============================================================

function dataOut = augmentVesselData(data, imageSize)

    img = data{1};

    mask = data{2};


    % Ensure RGB image
    if size(img, 3) == 1

        img = repmat(img, [1 1 3]);

    end


    % Resize first
    img = imresize( ...
        img, ...
        imageSize(1:2));


    mask = imresize( ...
        mask, ...
        imageSize(1:2), ...
        "nearest");


    %% Random horizontal flip

    if rand > 0.5

        img = fliplr(img);

        mask = fliplr(mask);

    end


    %% Random vertical flip

    if rand > 0.5

        img = flipud(img);

        mask = flipud(mask);

    end


    %% Random brightness adjustment

    if rand > 0.5

        % Convert to double
        imgDouble = im2double(img);


        % Random brightness factor
        brightnessFactor = ...
            0.90 + 0.20 * rand;


        imgDouble = ...
            imgDouble * brightnessFactor;


        % Limit values
        imgDouble = ...
            min(max(imgDouble, 0), 1);


        % Convert back
        img = im2uint8(imgDouble);

    end


    % Return augmented image and mask
    dataOut = {img, mask};

end
