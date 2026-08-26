%% ============================================================
% IDRiD_Train_BalancedWeighted_Augmented_UNet.m
%
% IDRiD Multiclass Retinal Lesion Segmentation
%
% MATLAB R2026a
%
% Classes:
% 0 = Background
% 1 = MA
% 2 = HE
% 3 = EX
% 4 = SE
% 5 = OD
%
% Loss:
% Weighted Index Cross-Entropy
%
% IMPORTANT:
% - Integer categorical targets
% - Singleton target channel
% - Explicit target format
% - Class weights are 1 x 6
% - No manual softmax
% - No manual one-hot encoding
%% ============================================================

clear;
clc;
close all;


%% ============================================================
% 1. DISPLAY HEADER
%% ============================================================

fprintf('\n============================================\n');
fprintf('IDRiD BALANCED WEIGHTED AUGMENTED U-NET\n');
fprintf('============================================\n\n');

fprintf('Segmentation classes:\n');
fprintf('0 = Background\n');
fprintf('1 = MA\n');
fprintf('2 = HE\n');
fprintf('3 = EX\n');
fprintf('4 = SE\n');
fprintf('5 = OD\n\n');


%% ============================================================
% 2. DATASET PATHS
%% ============================================================

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
% 3. MODEL SAVE PATH
%% ============================================================

modelPath = fullfile( ...
    basePath, ...
    'IDRiD_UNet_BalancedWeighted_Augmented_256.mat');


%% ============================================================
% 4. CREATE IMAGE DATASTORE
%% ============================================================

fprintf('Creating image datastore...\n');

imds = imageDatastore( ...
    trainImagePath, ...
    'FileExtensions', {'.jpg'});


%% ============================================================
% 5. DEFINE CLASSES
%% ============================================================

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
% 6. CREATE PIXEL LABEL DATASTORE
%% ============================================================

fprintf('Creating pixel label datastore...\n');

pxds = pixelLabelDatastore( ...
    maskPath, ...
    classNames, ...
    labelIDs, ...
    'FileExtensions', {'.png'});


%% ============================================================
% 7. DATASET INFORMATION
%% ============================================================

fprintf('\n============================================\n');
fprintf('DATASET INFORMATION\n');
fprintf('============================================\n\n');

fprintf('Number of images: %d\n', ...
    numel(imds.Files));

fprintf('Number of masks : %d\n', ...
    numel(pxds.Files));


%% ============================================================
% 8. VERIFY IMAGE / MASK COUNT
%% ============================================================

if numel(imds.Files) ~= numel(pxds.Files)

    error( ...
        ['Number of images (%d) does not match ' ...
         'number of masks (%d).'], ...
        numel(imds.Files), ...
        numel(pxds.Files));

end


%% ============================================================
% 9. VERIFY IMAGE / MASK PAIRING
%% ============================================================

fprintf('\nVerifying image-mask pairing...\n');

for i = 1:numel(imds.Files)

    [~, imageName, ~] = ...
        fileparts(imds.Files{i});

    [~, maskName, ~] = ...
        fileparts(pxds.Files{i});

    expectedMaskName = ...
        [imageName '_mask'];

    if ~strcmp(maskName, expectedMaskName)

        error( ...
            ['Image-mask mismatch at index %d.\n' ...
             'Image       : %s\n' ...
             'Mask        : %s\n' ...
             'Expected    : %s'], ...
             i, ...
             imageName, ...
             maskName, ...
             expectedMaskName);

    end

end

fprintf('SUCCESS: Image-mask pairing verified.\n');


%% ============================================================
% 10. TRAIN / VALIDATION SPLIT
%% ============================================================

fprintf('\n============================================\n');
fprintf('TRAIN / VALIDATION SPLIT\n');
fprintf('============================================\n');

rng(42);

numImages = numel(imds.Files);

indices = randperm(numImages);

numTrain = round(0.8 * numImages);

trainIdx = indices(1:numTrain);

valIdx = indices(numTrain+1:end);


imdsTrain = subset( ...
    imds, ...
    trainIdx);

pxdsTrain = subset( ...
    pxds, ...
    trainIdx);


imdsVal = subset( ...
    imds, ...
    valIdx);

pxdsVal = subset( ...
    pxds, ...
    valIdx);


fprintf('Total images      : %d\n', ...
    numImages);

fprintf('Training images   : %d\n', ...
    numel(imdsTrain.Files));

fprintf('Validation images : %d\n', ...
    numel(imdsVal.Files));


%% ============================================================
% 11. NETWORK INPUT SIZE
%% ============================================================

inputSize = [256 256];

fprintf('\nNetwork input size: %d x %d\n', ...
    inputSize(1), ...
    inputSize(2));


%% ============================================================
% 12. CALCULATE CLASS PIXEL FREQUENCIES
%% ============================================================

fprintf('\n============================================\n');
fprintf('CALCULATING BALANCED CLASS FREQUENCIES\n');
fprintf('============================================\n\n');

classPixelCount = ...
    zeros(numClasses, 1);


for i = 1:numel(pxdsTrain.Files)

    mask = ...
        imread(pxdsTrain.Files{i});


    % Use first channel if necessary.

    if size(mask,3) > 1

        mask = ...
            mask(:,:,1);

    end


    % Resize segmentation mask.

    mask = ...
        imresize( ...
            mask, ...
            inputSize, ...
            'nearest');


    % Count pixels.

    for c = 1:numClasses

        classID = ...
            labelIDs(c);

        classPixelCount(c) = ...
            classPixelCount(c) + ...
            sum(mask(:) == classID);

    end


    fprintf( ...
        'Processed mask %d/%d\n', ...
        i, ...
        numel(pxdsTrain.Files));

end


%% ============================================================
% 13. CALCULATE CLASS FREQUENCIES
%% ============================================================

totalPixels = ...
    sum(classPixelCount);

classFrequency = ...
    classPixelCount ./ totalPixels;


%% ============================================================
% 14. CALCULATE BALANCED CLASS WEIGHTS
%% ============================================================

% Square-root inverse-frequency weighting.

classWeights = ...
    1 ./ sqrt(classFrequency + eps);


% Normalize.

classWeights = ...
    classWeights ./ mean(classWeights);


% Protect background.

classWeights(1) = ...
    max(classWeights(1), 0.1);


% Cap rare classes.

classWeights = ...
    min(classWeights, 5);


% IMPORTANT:
% classWeights must be a ROW vector:
%
% 1 x 6
%
% because WeightsFormat="UC" specifies:
% U = singleton/unspecified dimension
% C = class dimension

classWeights = ...
    single(classWeights(:)');


%% ============================================================
% 15. DISPLAY CLASS STATISTICS
%% ============================================================

fprintf('\n============================================\n');
fprintf('BALANCED CLASS STATISTICS AND WEIGHTS\n');
fprintf('============================================\n\n');

fprintf( ...
    '%-15s %-15s %-15s %-15s\n', ...
    'Class', ...
    'Pixel Count', ...
    'Frequency', ...
    'Weight');

fprintf( ...
    '---------------------------------------------------------------\n');


for c = 1:numClasses

    fprintf( ...
        '%-15s %-15d %-15.8f %-15.4f\n', ...
        char(classNames(c)), ...
        classPixelCount(c), ...
        classFrequency(c), ...
        classWeights(c));

end


fprintf('\nWeighting strategy:\n');

fprintf('- Square-root inverse-frequency weighting\n');

fprintf('- Background weight protected with minimum floor\n');

fprintf('- Rare-class weights capped for stability\n');

fprintf('- Weight vector size: %d x %d\n', ...
    size(classWeights,1), ...
    size(classWeights,2));


%% ============================================================
% 16. COMBINE DATASTORES
%% ============================================================

fprintf('\nCreating training and validation datastores...\n');

dsTrain = combine( ...
    imdsTrain, ...
    pxdsTrain);

dsVal = combine( ...
    imdsVal, ...
    pxdsVal);


%% ============================================================
% 17. RESIZE + AUGMENT TRAINING DATA
%% ============================================================

dsTrainResized = transform( ...
    dsTrain, ...
    @(data) resizeAndAugmentData( ...
        data, ...
        inputSize));


%% ============================================================
% 18. RESIZE VALIDATION DATA
%% ============================================================

dsValResized = transform( ...
    dsVal, ...
    @(data) resizeData( ...
        data, ...
        inputSize));


%% ============================================================
% 19. BUILD U-NET
%% ============================================================

fprintf('\n============================================\n');
fprintf('BUILDING U-NET\n');
fprintf('============================================\n\n');

net = unet( ...
    [inputSize 3], ...
    numClasses);


fprintf('U-Net created successfully.\n');

fprintf('Number of classes: %d\n', ...
    numClasses);


%% ============================================================
% 20. CONFIGURE WEIGHTED INDEX CROSS-ENTROPY
%% ============================================================

fprintf('\nConfiguring weighted index cross-entropy...\n');


lossFcn = @(Y,T) indexcrossentropy( ...
    Y, ...
    T, ...
    classWeights, ...
    WeightsFormat="UC");


fprintf('Weighted index cross-entropy configured.\n');

fprintf('No manual softmax will be applied.\n');

fprintf('No manual one-hot encoding will be applied.\n');

fprintf('Categorical targets will be integer encoded.\n');


%% ============================================================
% 21. TRAINING OPTIONS
%% ============================================================

fprintf('\nConfiguring training options...\n');


options = trainingOptions( ...
    'adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 2, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', dsValResized, ...
    'ValidationFrequency', 10, ...
    'CategoricalTargetEncoding', 'integer', ...
    'TargetDataFormats', 'SSCB', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'cpu');


%% ============================================================
% 22. START TRAINING
%% ============================================================

fprintf('\n============================================\n');
fprintf('STARTING BALANCED WEIGHTED AUGMENTED U-NET\n');
fprintf('============================================\n\n');

fprintf('Training configuration:\n');

fprintf('  Input size       : %d x %d\n', ...
    inputSize(1), ...
    inputSize(2));

fprintf('  Number of classes: %d\n', ...
    numClasses);

fprintf('  Training images  : %d\n', ...
    numel(imdsTrain.Files));

fprintf('  Validation images: %d\n', ...
    numel(imdsVal.Files));

fprintf('  Mini-batch size  : %d\n', ...
    2);

fprintf('  Maximum epochs   : %d\n', ...
    30);

fprintf('  Learning rate    : %.6f\n', ...
    1e-4);

fprintf('  Execution        : CPU\n');

fprintf('  Target encoding  : integer\n');

fprintf('  Target format    : SSCB\n');

fprintf('\n');


%% ============================================================
% 23. TRAIN NETWORK
%% ============================================================

try

    net = trainnet( ...
        dsTrainResized, ...
        net, ...
        lossFcn, ...
        options);


catch ME

    fprintf('\n============================================\n');
    fprintf('TRAINING FAILED\n');
    fprintf('============================================\n\n');

    fprintf('Error message:\n');

    fprintf('%s\n\n', ...
        ME.message);


    fprintf('Full error report:\n');

    fprintf( ...
        '%s\n', ...
        getReport( ...
            ME, ...
            'extended', ...
            'hyperlinks', ...
            'off'));


    rethrow(ME);

end


%% ============================================================
% 24. SAVE MODEL
%% ============================================================

fprintf('\n============================================\n');
fprintf('SAVING TRAINED MODEL\n');
fprintf('============================================\n\n');


save( ...
    modelPath, ...
    'net', ...
    'classNames', ...
    'labelIDs', ...
    'inputSize', ...
    'classWeights', ...
    'classPixelCount', ...
    'classFrequency');


fprintf('Model successfully saved.\n');

fprintf('\nModel saved to:\n');

fprintf('%s\n', ...
    modelPath);


%% ============================================================
% 25. TRAINING COMPLETE
%% ============================================================

fprintf('\n============================================\n');
fprintf('TRAINING COMPLETE\n');
fprintf('============================================\n\n');

fprintf('Model file:\n');

fprintf('%s\n\n', ...
    modelPath);

fprintf('Classes:\n');


for c = 1:numClasses

    fprintf( ...
        '  %d = %s\n', ...
        labelIDs(c), ...
        classNames(c));

end


fprintf('\n');


%% ============================================================
% LOCAL FUNCTION 1
% RESIZE VALIDATION DATA
%% ============================================================

function data = resizeData( ...
    data, ...
    inputSize)

    image = ...
        data{1};

    mask = ...
        data{2};


    %% --------------------------------------------------------
    % IMAGE
    %% --------------------------------------------------------

    image = ...
        imresize( ...
            image, ...
            inputSize);


    %% --------------------------------------------------------
    % MASK
    %% --------------------------------------------------------

    mask = ...
        imresize( ...
            mask, ...
            inputSize, ...
            'nearest');


    %% --------------------------------------------------------
    % ENSURE MASK IS CATEGORICAL
    %
    % pixelLabelDatastore normally supplies categorical masks.
    % This check guarantees that the target remains categorical
    % after transformation.
    %% --------------------------------------------------------

    if ~iscategorical(mask)

        mask = categorical( ...
            mask, ...
            labelIDs, ...
            classNames);

    end


    %% --------------------------------------------------------
    % CONVERT IMAGE TO SINGLE
    %% --------------------------------------------------------

    image = ...
        im2single(image);


    %% --------------------------------------------------------
    % RETURN
    %% --------------------------------------------------------

    data{1} = ...
        image;

    data{2} = ...
        mask;

end


%% ============================================================
% LOCAL FUNCTION 2
% RESIZE + AUGMENT TRAINING DATA
%% ============================================================

function data = resizeAndAugmentData( ...
    data, ...
    inputSize)

    image = ...
        data{1};

    mask = ...
        data{2};


    %% --------------------------------------------------------
    % RESIZE
    %% --------------------------------------------------------

    image = ...
        imresize( ...
            image, ...
            inputSize);


    mask = ...
        imresize( ...
            mask, ...
            inputSize, ...
            'nearest');


    %% --------------------------------------------------------
    % ENSURE CATEGORICAL MASK
    %% --------------------------------------------------------

    if ~iscategorical(mask)

        mask = categorical( ...
            mask, ...
            labelIDs, ...
            classNames);

    end


    %% --------------------------------------------------------
    % RANDOM HORIZONTAL FLIP
    %% --------------------------------------------------------

    if rand > 0.5

        image = ...
            fliplr(image);

        mask = ...
            fliplr(mask);

    end


    %% --------------------------------------------------------
    % SMALL RANDOM ROTATION
    %
    % Range:
    % -15 to +15 degrees
    %% --------------------------------------------------------

    rotationAngle = ...
        -15 + 30 * rand;


    image = ...
        imrotate( ...
            image, ...
            rotationAngle, ...
            'bilinear', ...
            'crop');


    mask = ...
        imrotate( ...
            mask, ...
            rotationAngle, ...
            'nearest', ...
            'crop');


    %% --------------------------------------------------------
    % RANDOM BRIGHTNESS
    %% --------------------------------------------------------

    brightnessFactor = ...
        0.90 + 0.20 * rand;


    image = ...
        im2single(image);


    image = ...
        image .* brightnessFactor;


    %% --------------------------------------------------------
    % CLAMP
    %% --------------------------------------------------------

    image = ...
        min(max(image, 0), 1);


    %% --------------------------------------------------------
    % RETURN
    %% --------------------------------------------------------

    data{1} = ...
        image;

    data{2} = ...
        mask;

end