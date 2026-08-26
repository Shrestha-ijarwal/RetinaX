%% ============================================================
% TRUST-DR: VESSEL SEGMENTATION TRAINING (V4)
%
% DRIVE Dataset
%
% V4 is based directly on the TRUST-DR V3 pipeline.
%
% V3 baseline:
%   - U-Net
%   - EncoderDepth = 3
%   - Input = 256 x 256 x 3
%   - Adam
%   - Learning rate = 1e-4
%   - 50 epochs
%   - MiniBatchSize = 2
%   - Horizontal/vertical flips
%   - Brightness augmentation
%
% V4 improvements:
%   - CLAHE preprocessing
%   - Stronger vessel-oriented augmentation
%   - Horizontal / vertical flips
%   - Small random rotations
%   - Brightness variation
%   - Contrast variation
%   - Same U-Net architecture for fair comparison
%   - Lower learning rate
%   - More epochs
%   - Validation monitoring
%   - Manual Dice / IoU / Precision / Recall
%   - Sensitivity / Specificity
%   - V3 vs V4 comparison
%   - V4 visualization
%
% IMPORTANT:
%   V3 model is NOT overwritten.
%
% Compatible with newer MATLAB versions using:
%   unet()
%   trainnet()
%
%% ============================================================

clear;
clc;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf(' TRUST-DR: VESSEL SEGMENTATION TRAINING (V4)\n');
fprintf('============================================================\n\n');


%% ============================================================
% STEP 1: INITIALIZATION
% ============================================================

projectFolder = 'C:\Users\sijar\Downloads\TRUST_DR';

fprintf('Project folder:\n');
fprintf('%s\n\n', projectFolder);

if ~isfolder(projectFolder)
    error('Project folder not found: %s', projectFolder);
end

rng(42);

fprintf('Random seed initialized.\n');


%% ============================================================
% STEP 2: LOAD DRIVE DATASET
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' LOADING DRIVE DATASET\n');
fprintf('============================================================\n');

imageFolder = fullfile( ...
    projectFolder, ...
    'datasets', ...
    'drive', ...
    'DRIVE', ...
    'training', ...
    'images');

maskFolder = fullfile( ...
    projectFolder, ...
    'datasets', ...
    'drive', ...
    'DRIVE', ...
    'training', ...
    '1st_manual');

fprintf('\nImage folder:\n%s\n', imageFolder);
fprintf('\nMask folder:\n%s\n', maskFolder);


% Verify folders

if ~isfolder(imageFolder)
    error('Image folder not found: %s', imageFolder);
end

if ~isfolder(maskFolder)
    error('Mask folder not found: %s', maskFolder);
end


% Image datastore

imds = imageDatastore(imageFolder);


% Segmentation classes

classNames = ["background", "vessel"];

labelIDs = [0 255];


% Pixel label datastore

pxds = pixelLabelDatastore( ...
    maskFolder, ...
    classNames, ...
    labelIDs);


% Dataset size

numImages = numel(imds.Files);

fprintf('\nNumber of fundus images : %d\n', ...
    numImages);

fprintf('Number of vessel masks  : %d\n', ...
    numel(pxds.Files));


if numImages ~= numel(pxds.Files)

    error( ...
        'Number of images and masks does not match.');

end


fprintf('\nDRIVE dataset loaded successfully.\n');


%% ============================================================
% STEP 3: SHUFFLE AND SPLIT DATASET
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' DATASET SPLIT\n');
fprintf('============================================================\n');

rng(42);

indices = randperm(numImages);


% 70% Training
% 15% Validation
% 15% Testing

numTrain = round(0.70 * numImages);

numValidation = round(0.15 * numImages);


trainIdx = ...
    indices(1:numTrain);


validationIdx = ...
    indices( ...
        numTrain + 1 : ...
        numTrain + numValidation);


testIdx = ...
    indices( ...
        numTrain + numValidation + 1 : ...
        end);


% Image datastores

imdsTrain = ...
    subset(imds,trainIdx);

imdsValidation = ...
    subset(imds,validationIdx);

imdsTest = ...
    subset(imds,testIdx);


% Mask datastores

pxdsTrain = ...
    subset(pxds,trainIdx);

pxdsValidation = ...
    subset(pxds,validationIdx);

pxdsTest = ...
    subset(pxds,testIdx);


fprintf('\nTraining   : %d images\n', ...
    numel(imdsTrain.Files));

fprintf('Validation : %d images\n', ...
    numel(imdsValidation.Files));

fprintf('Testing    : %d images\n', ...
    numel(imdsTest.Files));


%% ============================================================
% STEP 4: NETWORK INPUT SIZE
% ============================================================

imageSize = [256 256 3];

fprintf('\n');
fprintf('Network input size: %d x %d x %d\n', ...
    imageSize(1), ...
    imageSize(2), ...
    imageSize(3));


%% ============================================================
% STEP 5: CREATE TRAINING DATASTORE
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' PREPARING V4 TRAINING DATA\n');
fprintf('============================================================\n');


% Combine image and mask

dsTrain = combine( ...
    imdsTrain, ...
    pxdsTrain);


% V4 preprocessing + augmentation

dsTrain = transform( ...
    dsTrain, ...
    @(data) augmentVesselDataV4( ...
        data, ...
        imageSize));


% Validation datastore

dsValidation = combine( ...
    imdsValidation, ...
    pxdsValidation);


% Validation preprocessing only

dsValidation = transform( ...
    dsValidation, ...
    @(data) preprocessVesselDataV4( ...
        data, ...
        imageSize));


fprintf('\nV4 training datastore ready.\n');


%% ============================================================
% STEP 6: CREATE U-NET
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' CREATING V4 U-NET\n');
fprintf('============================================================\n');


% Keep the same architecture as V3.
%
% This is intentional.
%
% It allows us to determine whether the V4 preprocessing and
% augmentation pipeline actually improves performance rather
% than simply changing the network architecture.

trainedVesselNetV4 = unet( ...
    imageSize, ...
    numel(classNames), ...
    "EncoderDepth",3);


fprintf('\nV4 U-Net created successfully.\n');


%% ============================================================
% STEP 7: TRAINING OPTIONS
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' V4 TRAINING CONFIGURATION\n');
fprintf('============================================================\n');


options = trainingOptions( ...
    "adam", ...
    "InitialLearnRate",5e-5, ...
    "MaxEpochs",80, ...
    "MiniBatchSize",2, ...
    "Shuffle","every-epoch", ...
    "ValidationData",dsValidation, ...
    "ValidationFrequency",5, ...
    "Verbose",true, ...
    "Plots","training-progress");


fprintf('\n');
fprintf('Optimizer             : Adam\n');
fprintf('Initial learning rate : 5e-5\n');
fprintf('Maximum epochs        : 80\n');
fprintf('Mini-batch size       : 2\n');
fprintf('Encoder depth         : 3\n');
fprintf('Input size            : 256 x 256 x 3\n');
fprintf('CLAHE                  : ENABLED\n');
fprintf('Augmentation           : ENABLED\n');
fprintf('Rotation               : ENABLED\n');
fprintf('Brightness             : ENABLED\n');
fprintf('Contrast               : ENABLED\n');


%% ============================================================
% STEP 8: TRAIN V4
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' STARTING VESSEL SEGMENTATION TRAINING (V4)\n');
fprintf('============================================================\n\n');


trainingStartTime = tic;


trainedVesselNetV4 = trainnet( ...
    dsTrain, ...
    trainedVesselNetV4, ...
    "crossentropy", ...
    options);


trainingTime = toc(trainingStartTime);


fprintf('\n');
fprintf('============================================================\n');
fprintf(' V4 TRAINING COMPLETE\n');
fprintf('============================================================\n');

fprintf('\nTraining time: %.2f minutes\n', ...
    trainingTime / 60);


%% ============================================================
% STEP 9: SAVE V4 MODEL
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' SAVING V4 MODEL\n');
fprintf('============================================================\n');


modelFolder = ...
    fullfile( ...
        projectFolder, ...
        'models');


if ~exist(modelFolder,'dir')
    mkdir(modelFolder);
end


v4ModelPath = ...
    fullfile( ...
        modelFolder, ...
        'trained_vessel_segmentation_v4.mat');


save( ...
    v4ModelPath, ...
    'trainedVesselNetV4', ...
    'imageSize', ...
    'classNames', ...
    'labelIDs', ...
    'trainIdx', ...
    'validationIdx', ...
    'testIdx', ...
    'trainingTime');


fprintf('\nV4 MODEL SAVED SUCCESSFULLY\n');

fprintf('\nModel location:\n%s\n', ...
    v4ModelPath);


%% ============================================================
% STEP 10: TEST V4 MODEL
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' TESTING V4 MODEL\n');
fprintf('============================================================\n');


numTestImages = ...
    numel(imdsTest.Files);


% Metric arrays

pixelAccuracyScoresV4 = ...
    zeros(numTestImages,1);

precisionScoresV4 = ...
    zeros(numTestImages,1);

recallScoresV4 = ...
    zeros(numTestImages,1);

diceScoresV4 = ...
    zeros(numTestImages,1);

iouScoresV4 = ...
    zeros(numTestImages,1);

sensitivityScoresV4 = ...
    zeros(numTestImages,1);

specificityScoresV4 = ...
    zeros(numTestImages,1);


% Store predictions

predictedMasksV4 = ...
    cell(numTestImages,1);


for i = 1:numTestImages


    fprintf('\n');
    fprintf('Processing V4 test image %d of %d...\n', ...
        i, ...
        numTestImages);


    %% --------------------------------------------------------
    % LOAD IMAGE
    % ---------------------------------------------------------

    img = ...
        readimage(imdsTest,i);


    % Ensure RGB

    if size(img,3) == 1

        img = ...
            repmat( ...
                img, ...
                [1 1 3]);

    end


    % Resize

    inputImage = ...
        imresize( ...
            img, ...
            imageSize(1:2));


    %% --------------------------------------------------------
    % APPLY CLAHE
    % ---------------------------------------------------------

    inputImage = ...
        applyFundusCLAHE(inputImage);


    %% --------------------------------------------------------
    % PREDICTION
    % ---------------------------------------------------------

    predictedMask = ...
        semanticseg( ...
            inputImage, ...
            trainedVesselNetV4, ...
            Classes = classNames);


    predictedVessel = ...
        predictedMask == "vessel";


    %% --------------------------------------------------------
    % GROUND TRUTH
    % ---------------------------------------------------------

    groundTruthMask = ...
        readimage( ...
            pxdsTest, ...
            i);


    groundTruthMask = ...
        imresize( ...
            groundTruthMask, ...
            imageSize(1:2), ...
            "nearest");


    groundTruthVessel = ...
        groundTruthMask == "vessel";


    %% --------------------------------------------------------
    % CONFUSION VALUES
    % ---------------------------------------------------------

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


    %% --------------------------------------------------------
    % PIXEL ACCURACY
    % ---------------------------------------------------------

    denominator = ...
        TP + TN + FP + FN;


    if denominator == 0

        pixelAccuracy = 0;

    else

        pixelAccuracy = ...
            (TP + TN) / denominator;

    end


    %% --------------------------------------------------------
    % PRECISION
    % ---------------------------------------------------------

    if (TP + FP) == 0

        precision = 0;

    else

        precision = ...
            TP / (TP + FP);

    end


    %% --------------------------------------------------------
    % RECALL
    % ---------------------------------------------------------

    if (TP + FN) == 0

        recall = 0;

    else

        recall = ...
            TP / (TP + FN);

    end


    %% --------------------------------------------------------
    % DICE
    % ---------------------------------------------------------

    if (2*TP + FP + FN) == 0

        diceScore = 0;

    else

        diceScore = ...
            (2*TP) / ...
            (2*TP + FP + FN);

    end


    %% --------------------------------------------------------
    % IOU
    % ---------------------------------------------------------

    if (TP + FP + FN) == 0

        iouScore = 0;

    else

        iouScore = ...
            TP / ...
            (TP + FP + FN);

    end


    %% --------------------------------------------------------
    % SENSITIVITY
    % ---------------------------------------------------------

    if (TP + FN) == 0

        sensitivity = 0;

    else

        sensitivity = ...
            TP / (TP + FN);

    end


    %% --------------------------------------------------------
    % SPECIFICITY
    % ---------------------------------------------------------

    if (TN + FP) == 0

        specificity = 0;

    else

        specificity = ...
            TN / (TN + FP);

    end


    %% --------------------------------------------------------
    % STORE METRICS
    % ---------------------------------------------------------

    pixelAccuracyScoresV4(i) = ...
        pixelAccuracy;

    precisionScoresV4(i) = ...
        precision;

    recallScoresV4(i) = ...
        recall;

    diceScoresV4(i) = ...
        diceScore;

    iouScoresV4(i) = ...
        iouScore;

    sensitivityScoresV4(i) = ...
        sensitivity;

    specificityScoresV4(i) = ...
        specificity;


    predictedMasksV4{i} = ...
        predictedMask;


    %% --------------------------------------------------------
    % DISPLAY METRICS
    % ---------------------------------------------------------

    fprintf('\n');

    fprintf('Pixel Accuracy : %.2f%%\n', ...
        pixelAccuracy * 100);

    fprintf('Precision      : %.2f%%\n', ...
        precision * 100);

    fprintf('Recall         : %.2f%%\n', ...
        recall * 100);

    fprintf('Sensitivity    : %.2f%%\n', ...
        sensitivity * 100);

    fprintf('Specificity    : %.2f%%\n', ...
        specificity * 100);

    fprintf('Dice           : %.2f%%\n', ...
        diceScore * 100);

    fprintf('IoU            : %.2f%%\n', ...
        iouScore * 100);

end


%% ============================================================
% STEP 11: FINAL V4 RESULTS
% ============================================================

meanPixelAccuracyV4 = ...
    mean(pixelAccuracyScoresV4);

meanPrecisionV4 = ...
    mean(precisionScoresV4);

meanRecallV4 = ...
    mean(recallScoresV4);

meanSensitivityV4 = ...
    mean(sensitivityScoresV4);

meanSpecificityV4 = ...
    mean(specificityScoresV4);

meanDiceV4 = ...
    mean(diceScoresV4);

meanIoUV4 = ...
    mean(iouScoresV4);


fprintf('\n');
fprintf('============================================================\n');
fprintf(' FINAL VESSEL SEGMENTATION RESULTS (V4)\n');
fprintf('============================================================\n\n');


fprintf('Number of Test Images : %d\n\n', ...
    numTestImages);


fprintf('Mean Pixel Accuracy   : %.2f%%\n', ...
    meanPixelAccuracyV4 * 100);

fprintf('Mean Precision        : %.2f%%\n', ...
    meanPrecisionV4 * 100);

fprintf('Mean Recall           : %.2f%%\n', ...
    meanRecallV4 * 100);

fprintf('Mean Sensitivity      : %.2f%%\n', ...
    meanSensitivityV4 * 100);

fprintf('Mean Specificity      : %.2f%%\n', ...
    meanSpecificityV4 * 100);

fprintf('Mean Dice             : %.2f%%\n', ...
    meanDiceV4 * 100);

fprintf('Mean IoU              : %.2f%%\n', ...
    meanIoUV4 * 100);


%% ============================================================
% STEP 12: SAVE V4 RESULTS
% ============================================================

resultsPathV4 = ...
    fullfile( ...
        modelFolder, ...
        'vessel_segmentation_v4_results.mat');


save( ...
    resultsPathV4, ...
    'pixelAccuracyScoresV4', ...
    'precisionScoresV4', ...
    'recallScoresV4', ...
    'sensitivityScoresV4', ...
    'specificityScoresV4', ...
    'diceScoresV4', ...
    'iouScoresV4', ...
    'meanPixelAccuracyV4', ...
    'meanPrecisionV4', ...
    'meanRecallV4', ...
    'meanSensitivityV4', ...
    'meanSpecificityV4', ...
    'meanDiceV4', ...
    'meanIoUV4');


fprintf('\n');
fprintf('V4 results saved successfully.\n');

fprintf('\nResults location:\n%s\n', ...
    resultsPathV4);


%% ============================================================
% STEP 13: V3 VS V4 COMPARISON
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' V3 vs V4 COMPARISON\n');
fprintf('============================================================\n');


v3ResultsPath = ...
    fullfile( ...
        modelFolder, ...
        'vessel_segmentation_v3_results.mat');


if exist(v3ResultsPath,'file')


    v3 = load(v3ResultsPath);


    %% V3 Dice

    if isfield(v3,'meanDice')

        meanDiceV3 = ...
            v3.meanDice;

    else

        meanDiceV3 = NaN;

    end


    %% V3 IoU

    if isfield(v3,'meanIoU')

        meanIoUV3 = ...
            v3.meanIoU;

    else

        meanIoUV3 = NaN;

    end


    %% V3 Precision

    if isfield(v3,'meanPrecision')

        meanPrecisionV3 = ...
            v3.meanPrecision;

    else

        meanPrecisionV3 = NaN;

    end


    %% V3 Recall

    if isfield(v3,'meanRecall')

        meanRecallV3 = ...
            v3.meanRecall;

    else

        meanRecallV3 = NaN;

    end


    fprintf('\n');

    fprintf('Metric             V3          V4\n');

    fprintf('-------------------------------------------\n');

    fprintf('Dice             %.4f      %.4f\n', ...
        meanDiceV3, ...
        meanDiceV4);

    fprintf('IoU              %.4f      %.4f\n', ...
        meanIoUV3, ...
        meanIoUV4);

    fprintf('Precision        %.4f      %.4f\n', ...
        meanPrecisionV3, ...
        meanPrecisionV4);

    fprintf('Recall           %.4f      %.4f\n', ...
        meanRecallV3, ...
        meanRecallV4);


    %% Absolute differences

    diceDifference = ...
        meanDiceV4 - meanDiceV3;

    iouDifference = ...
        meanIoUV4 - meanIoUV3;

    precisionDifference = ...
        meanPrecisionV4 - meanPrecisionV3;

    recallDifference = ...
        meanRecallV4 - meanRecallV3;


    fprintf('\n');

    fprintf('Absolute change (V4 - V3):\n');

    fprintf('Dice       : %+0.4f\n', ...
        diceDifference);

    fprintf('IoU        : %+0.4f\n', ...
        iouDifference);

    fprintf('Precision  : %+0.4f\n', ...
        precisionDifference);

    fprintf('Recall     : %+0.4f\n', ...
        recallDifference);


    %% Percentage improvement

    if meanDiceV3 ~= 0

        dicePercentChange = ...
            ((meanDiceV4 - meanDiceV3) / ...
            meanDiceV3) * 100;

    else

        dicePercentChange = NaN;

    end


    if meanIoUV3 ~= 0

        iouPercentChange = ...
            ((meanIoUV4 - meanIoUV3) / ...
            meanIoUV3) * 100;

    else

        iouPercentChange = NaN;

    end


    fprintf('\n');

    fprintf('Relative change:\n');

    fprintf('Dice       : %+0.2f%%\n', ...
        dicePercentChange);

    fprintf('IoU        : %+0.2f%%\n', ...
        iouPercentChange);


    %% Determine result

    fprintf('\n');

    if meanDiceV4 > meanDiceV3 && ...
            meanIoUV4 > meanIoUV3

        fprintf('RESULT: V4 IMPROVED OVER V3.\n');

        fprintf(['V4 achieved higher Dice and IoU ' ...
            'on the test set.\n']);

    elseif meanDiceV4 < meanDiceV3 && ...
            meanIoUV4 < meanIoUV3

        fprintf('RESULT: V4 DID NOT IMPROVE OVER V3.\n');

        fprintf(['Keep V3 as the current production vessel ' ...
            'segmentation model.\n']);

    else

        fprintf('RESULT: MIXED V3/V4 PERFORMANCE.\n');

        fprintf(['Review Dice, IoU, precision and recall ' ...
            'before replacing V3.\n']);

    end


else


    fprintf('\n');

    fprintf('V3 result file not found:\n');

    fprintf('%s\n', ...
        v3ResultsPath);

    fprintf('\n');

    fprintf('V4 training and evaluation completed.\n');

    fprintf(['Run the V3 training/evaluation first if you ' ...
        'want an automatic comparison.\n']);


end


%% ============================================================
% STEP 14: SAVE COMPARISON DATA
% ============================================================

comparisonPath = ...
    fullfile( ...
        modelFolder, ...
        'vessel_segmentation_v3_v4_comparison.mat');


save( ...
    comparisonPath, ...
    'meanDiceV4', ...
    'meanIoUV4', ...
    'meanPrecisionV4', ...
    'meanRecallV4', ...
    'meanSensitivityV4', ...
    'meanSpecificityV4');


if exist('meanDiceV3','var')

    save( ...
        comparisonPath, ...
        'meanDiceV3', ...
        'meanIoUV3', ...
        'meanPrecisionV3', ...
        'meanRecallV3', ...
        '-append');

end


fprintf('\nComparison data saved:\n%s\n', ...
    comparisonPath);


%% ============================================================
% STEP 15: VISUALIZATION
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' GENERATING V4 VISUALIZATION\n');
fprintf('============================================================\n');


visualizationIndex = 1;


%% Original

originalImage = ...
    readimage( ...
        imdsTest, ...
        visualizationIndex);


%% Ground truth

groundTruthMask = ...
    readimage( ...
        pxdsTest, ...
        visualizationIndex);


%% Resize

displayOriginal = ...
    imresize( ...
        originalImage, ...
        imageSize(1:2));


if size(displayOriginal,3) == 1

    displayOriginal = ...
        repmat( ...
            displayOriginal, ...
            [1 1 3]);

end


%% CLAHE

displayCLAHE = ...
    applyFundusCLAHE( ...
        displayOriginal);


%% Ground truth resize

groundTruthMask = ...
    imresize( ...
        groundTruthMask, ...
        imageSize(1:2), ...
        "nearest");


groundTruthVessel = ...
    groundTruthMask == "vessel";


%% Prediction

predictedMask = ...
    predictedMasksV4{visualizationIndex};


predictedVessel = ...
    predictedMask == "vessel";


%% Create overlay

vesselOverlay = ...
    createVesselOverlay( ...
        displayOriginal, ...
        predictedVessel);


%% Figure

figure( ...
    "Name", ...
    "TRUST-DR: Vessel Segmentation V4", ...
    "Color", ...
    "w", ...
    "Position", ...
    [50 100 1800 500]);


%% Original

subplot(1,5,1);

imshow(displayOriginal);

title( ...
    "Original Fundus", ...
    "FontSize",14, ...
    "FontWeight","bold");


%% CLAHE

subplot(1,5,2);

imshow(displayCLAHE);

title( ...
    "CLAHE", ...
    "FontSize",14, ...
    "FontWeight","bold");


%% Ground truth

subplot(1,5,3);

imshow(groundTruthVessel);

title( ...
    "Ground Truth", ...
    "FontSize",14, ...
    "FontWeight","bold");


%% Prediction

subplot(1,5,4);

imshow(predictedVessel);

title( ...
    sprintf( ...
        "V4 Prediction\nDice %.2f%% | IoU %.2f%%", ...
        diceScoresV4(visualizationIndex)*100, ...
        iouScoresV4(visualizationIndex)*100), ...
    "FontSize",14, ...
    "FontWeight","bold");


%% Overlay

subplot(1,5,5);

imshow(vesselOverlay);

title( ...
    "V4 Vessel Overlay", ...
    "FontSize",14, ...
    "FontWeight","bold");


sgtitle( ...
    "TRUST-DR: Retinal Blood Vessel Segmentation V4", ...
    "FontSize",18, ...
    "FontWeight","bold");


%% ============================================================
% STEP 16: SAVE VISUALIZATION
% ============================================================

visualizationPath = ...
    fullfile( ...
        modelFolder, ...
        'vessel_segmentation_v4_visualization.png');


exportgraphics( ...
    gcf, ...
    visualizationPath, ...
    "Resolution",150);


fprintf('\nVisualization saved:\n%s\n', ...
    visualizationPath);


%% ============================================================
% FINAL
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' TRUST-DR VESSEL SEGMENTATION V4 COMPLETE\n');
fprintf('============================================================\n\n');


fprintf('V4 model:\n');
fprintf('%s\n\n', ...
    v4ModelPath);


fprintf('V4 results:\n');
fprintf('%s\n\n', ...
    resultsPathV4);


fprintf('V3/V4 comparison:\n');
fprintf('%s\n\n', ...
    comparisonPath);


fprintf('Visualization:\n');
fprintf('%s\n\n', ...
    visualizationPath);


fprintf('IMPORTANT:\n');
fprintf('V3 model has NOT been overwritten.\n');
fprintf('Do not replace V3 in the integrated system until\n');
fprintf('V4 demonstrates better test performance.\n\n');

fprintf('============================================================\n');


%% ============================================================
% LOCAL FUNCTION 1
% VALIDATION PREPROCESSING
% ============================================================

function dataOut = preprocessVesselDataV4( ...
    data, ...
    imageSize)


img = data{1};

mask = data{2};


%% Ensure RGB

if size(img,3) == 1

    img = ...
        repmat( ...
            img, ...
            [1 1 3]);

end


%% Resize

img = ...
    imresize( ...
        img, ...
        imageSize(1:2));


mask = ...
    imresize( ...
        mask, ...
        imageSize(1:2), ...
        "nearest");


%% CLAHE

img = ...
    applyFundusCLAHE(img);


dataOut = ...
    {img,mask};

end


%% ============================================================
% LOCAL FUNCTION 2
% V4 AUGMENTATION
% ============================================================

function dataOut = augmentVesselDataV4( ...
    data, ...
    imageSize)


img = data{1};

mask = data{2};


%% Ensure RGB

if size(img,3) == 1

    img = ...
        repmat( ...
            img, ...
            [1 1 3]);

end


%% Resize

img = ...
    imresize( ...
        img, ...
        imageSize(1:2));


mask = ...
    imresize( ...
        mask, ...
        imageSize(1:2), ...
        "nearest");


%% ============================================================
% CLAHE
% ============================================================

if rand > 0.25

    img = ...
        applyFundusCLAHE(img);

end


%% ============================================================
% RANDOM HORIZONTAL FLIP
% ============================================================

if rand > 0.5

    img = fliplr(img);

    mask = fliplr(mask);

end


%% ============================================================
% RANDOM VERTICAL FLIP
% ============================================================

if rand > 0.5

    img = flipud(img);

    mask = flipud(mask);

end


%% ============================================================
% RANDOM ROTATION
% ============================================================

if rand > 0.5

    angle = ...
        -10 + 20*rand;


    img = ...
        imrotate( ...
            img, ...
            angle, ...
            "bilinear", ...
            "crop");


    mask = ...
        imrotate( ...
            mask, ...
            angle, ...
            "nearest", ...
            "crop");

end


%% ============================================================
% RANDOM BRIGHTNESS
% ============================================================

if rand > 0.5

    imgDouble = ...
        im2double(img);


    brightnessFactor = ...
        0.85 + 0.30*rand;


    imgDouble = ...
        imgDouble * brightnessFactor;


    imgDouble = ...
        min( ...
            max(imgDouble,0), ...
            1);


    img = ...
        im2uint8(imgDouble);

end


%% ============================================================
% RANDOM CONTRAST
% ============================================================

if rand > 0.5

    imgDouble = ...
        im2double(img);


    contrastFactor = ...
        0.80 + 0.40*rand;


    meanValue = ...
        mean(imgDouble(:));


    imgDouble = ...
        (imgDouble - meanValue) * ...
        contrastFactor + ...
        meanValue;


    imgDouble = ...
        min( ...
            max(imgDouble,0), ...
            1);


    img = ...
        im2uint8(imgDouble);

end


%% ============================================================
% FINAL SIZE CHECK
% ============================================================

img = ...
    imresize( ...
        img, ...
        imageSize(1:2));


mask = ...
    imresize( ...
        mask, ...
        imageSize(1:2), ...
        "nearest");


dataOut = ...
    {img,mask};

end


%% ============================================================
% LOCAL FUNCTION 3
% FUNDUS CLAHE
% ============================================================

function out = applyFundusCLAHE(img)


%% Ensure uint8

if ~isa(img,"uint8")

    img = ...
        im2uint8(img);

end


%% Ensure RGB

if size(img,3) == 1

    img = ...
        repmat( ...
            img, ...
            [1 1 3]);

end


%% RGB → LAB

lab = ...
    rgb2lab(img);


%% Luminance channel

L = ...
    lab(:,:,1);


%% Normalize L

Lnormalized = ...
    mat2gray(L);


%% CLAHE

Lclahe = ...
    adapthisteq( ...
        Lnormalized, ...
        "ClipLimit",0.01, ...
        "NumTiles",[8 8]);


%% Restore L channel

lab(:,:,1) = ...
    Lclahe * 100;


%% LAB → RGB
%
% IMPORTANT:
% MATLAB R2026a does NOT accept:
%
%   lab2rgb(lab,'ColorSpace','lab')
%
% Therefore we use the standard call below.

out = ...
    lab2rgb(lab);


%% Convert to uint8

out = ...
    im2uint8(out);

end


%% ============================================================
% LOCAL FUNCTION 4
% CREATE VESSEL OVERLAY
% ============================================================

function overlay = createVesselOverlay( ...
    img, ...
    vesselMask)


%% Ensure RGB

if size(img,3) == 1

    img = ...
        repmat( ...
            img, ...
            [1 1 3]);

end


%% Convert to double

base = ...
    im2double(img);


%% Create vessel emphasis

overlay = ...
    base;


% Increase vessel visibility by brightening vessel pixels.
%
% We deliberately keep this simple so the generated overlay
% can be used later by the integrated TRUST-DR pipeline.

for channel = 1:3

    channelImage = ...
        overlay(:,:,channel);

    channelImage(vesselMask) = ...
        min( ...
            channelImage(vesselMask) * 0.35 + 0.65, ...
            1);

    overlay(:,:,channel) = ...
        channelImage;

end


overlay = ...
    im2uint8(overlay);

end