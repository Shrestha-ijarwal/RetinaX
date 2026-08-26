%% ============================================================
% TRUST-DR MESSIDOR V4 FINE-TUNING
% CPU-FRIENDLY GREEN-CHANNEL CLAHE
% DAG-SAFE VERSION
%
% MATLAB R2026a
%
% IMPORTANT
% ------------------------------------------------------------
% 1. Original V3 DAGNetwork is preserved.
% 2. ResNet skip connections are NOT manually reconstructed.
% 3. No rgb2lab().
% 4. No lab2rgb().
% 5. No LAB -> XYZ conversion.
% 6. CLAHE is applied to the green channel only.
% 7. Existing Messidor train/validation/test splits are used.
% 8. Layer names are accessed through individual layer objects.
%% ============================================================

clear;
clc;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf('TRUST-DR MESSIDOR V4 FINE-TUNING\n');
fprintf('CPU-FRIENDLY GREEN CLAHE + DAG-SAFE\n');
fprintf('============================================================\n\n');


%% ============================================================
% 1. PATHS
% ============================================================

baseFolder = ...
    'C:\Users\sijar\Downloads\TRUST_DR';

modelPath = fullfile( ...
    baseFolder, ...
    'models', ...
    'trained_DR_model_v3.mat');

imageFolder = fullfile( ...
    baseFolder, ...
    'datasets', ...
    'archive', ...
    'messidor-2', ...
    'messidor-2', ...
    'preprocess');

labelsPath = fullfile( ...
    baseFolder, ...
    'datasets', ...
    'archive', ...
    'messidor_data.csv');

splitFolder = fullfile( ...
    baseFolder, ...
    'results', ...
    'messidor_training');

trainCSV = fullfile( ...
    splitFolder, ...
    'train.csv');

valCSV = fullfile( ...
    splitFolder, ...
    'validation.csv');

testCSV = fullfile( ...
    splitFolder, ...
    'test.csv');

resultsFolder = fullfile( ...
    baseFolder, ...
    'results', ...
    'messidor_v4');

modelOutputPath = fullfile( ...
    resultsFolder, ...
    'trained_DR_model_v4_MESSIDOR.mat');

trainingInfoPath = fullfile( ...
    resultsFolder, ...
    'MESSIDOR_V4_training_info.mat');


%% ============================================================
% 2. VERIFY FILES
% ============================================================

fprintf('VERIFYING FILES\n');
fprintf('------------------------------------------------------------\n');

if ~isfile(modelPath)
    error('V3 model not found:\n%s',modelPath);
end

if ~isfolder(imageFolder)
    error('Messidor image folder not found:\n%s',imageFolder);
end

if ~isfile(labelsPath)
    error('Messidor label CSV not found:\n%s',labelsPath);
end

if ~isfile(trainCSV)
    error('Training CSV not found:\n%s',trainCSV);
end

if ~isfile(valCSV)
    error('Validation CSV not found:\n%s',valCSV);
end

if ~isfile(testCSV)
    error('Test CSV not found:\n%s',testCSV);
end

if ~isfolder(resultsFolder)
    mkdir(resultsFolder);
end

fprintf('V3 model       : OK\n');
fprintf('Image folder   : OK\n');
fprintf('Labels CSV     : OK\n');
fprintf('Training CSV   : OK\n');
fprintf('Validation CSV : OK\n');
fprintf('Test CSV       : OK\n\n');


%% ============================================================
% 3. LOAD V3 MODEL
% ============================================================

fprintf('LOADING V3 MODEL\n');
fprintf('------------------------------------------------------------\n');

D = load(modelPath);

modelVariables = fieldnames(D);

fprintf('Variables inside model:\n');

for k = 1:numel(modelVariables)
    fprintf('  %s\n',modelVariables{k});
end

fprintf('\n');


%% ============================================================
% 4. FIND DAGNETWORK
% ============================================================

trainedNetV3 = [];

for k = 1:numel(modelVariables)

    candidate = D.(modelVariables{k});

    if isa(candidate,'DAGNetwork')

        trainedNetV3 = candidate;

        fprintf( ...
            'Found DAGNetwork in variable: %s\n', ...
            modelVariables{k});

        break;

    end

end

if isempty(trainedNetV3)

    error([ ...
        'No DAGNetwork was found inside the V3 MAT file. ' ...
        'Training stopped because the original DAG must be preserved.']);

end

fprintf('Network type: %s\n',class(trainedNetV3));


%% ============================================================
% 5. VERIFY ORIGINAL DAG
% ============================================================

fprintf('\n');
fprintf('VERIFYING ORIGINAL DAG\n');
fprintf('------------------------------------------------------------\n');

originalLayers = trainedNetV3.Layers;
originalConnections = trainedNetV3.Connections;

originalLayerCount = numel(originalLayers);
originalConnectionCount = height(originalConnections);

fprintf('Layers      : %d\n', ...
    originalLayerCount);

fprintf('Connections : %d\n', ...
    originalConnectionCount);

if originalLayerCount ~= 177

    warning( ...
        'Expected 177 layers, found %d.', ...
        originalLayerCount);

end

if originalConnectionCount ~= 192

    warning( ...
        'Expected 192 connections, found %d.', ...
        originalConnectionCount);

end


%% ============================================================
% 6. INPUT SIZE
% ============================================================

firstLayer = originalLayers(1);

if ~isprop(firstLayer,'InputSize')

    error('First layer does not contain InputSize.');

end

inputSize = firstLayer.InputSize;

fprintf('Input size: [%d %d %d]\n', ...
    inputSize(1), ...
    inputSize(2), ...
    inputSize(3));


%% ============================================================
% 7. DISPLAY FINAL ORIGINAL LAYERS
% ============================================================

fprintf('\n');
fprintf('ORIGINAL FINAL LAYERS\n');
fprintf('------------------------------------------------------------\n');

firstDisplayLayer = max(1,originalLayerCount-4);

for k = firstDisplayLayer:originalLayerCount

    currentLayer = originalLayers(k);

    fprintf( ...
        '%d: %s | %s\n', ...
        k, ...
        currentLayer.Name, ...
        class(currentLayer));

end


%% ============================================================
% 8. LOAD MESSIDOR SPLITS
% ============================================================

fprintf('\n');
fprintf('LOADING MESSIDOR SPLITS\n');
fprintf('------------------------------------------------------------\n');

Ttrain = readtable(trainCSV);
Tval   = readtable(valCSV);
Ttest  = readtable(testCSV);

fprintf('Training images   : %d\n',height(Ttrain));
fprintf('Validation images : %d\n',height(Tval));
fprintf('Final test images : %d\n\n',height(Ttest));


%% ============================================================
% 9. CLEAN SPLIT TABLES
% ============================================================

Ttrain = prepareSplitTable(Ttrain);
Tval   = prepareSplitTable(Tval);
Ttest  = prepareSplitTable(Ttest);


%% ============================================================
% 10. CLASS NAMES
% ============================================================

classNames = { ...
    'Mild', ...
    'Moderate', ...
    'No_DR', ...
    'Proliferative', ...
    'Severe'};

fprintf('MODEL CLASS ORDER\n');
fprintf('------------------------------------------------------------\n');

for k = 1:numel(classNames)

    fprintf( ...
        '%d -> %s\n', ...
        k-1, ...
        classNames{k});

end


%% ============================================================
% 11. CREATE CATEGORICAL LABELS
% ============================================================

Ttrain.Label = categorical( ...
    Ttrain.diagnosis, ...
    0:4, ...
    classNames);

Tval.Label = categorical( ...
    Tval.diagnosis, ...
    0:4, ...
    classNames);

Ttest.Label = categorical( ...
    Ttest.diagnosis, ...
    0:4, ...
    classNames);


%% ============================================================
% 12. ATTACH IMAGE PATHS
% ============================================================

fprintf('\n');
fprintf('VERIFYING IMAGE MATCHES\n');
fprintf('------------------------------------------------------------\n');

Ttrain = attachImagePaths( ...
    Ttrain, ...
    imageFolder);

Tval = attachImagePaths( ...
    Tval, ...
    imageFolder);

Ttest = attachImagePaths( ...
    Ttest, ...
    imageFolder);

fprintf( ...
    'Training valid images   : %d\n', ...
    height(Ttrain));

fprintf( ...
    'Validation valid images : %d\n', ...
    height(Tval));

fprintf( ...
    'Test valid images       : %d\n\n', ...
    height(Ttest));


%% ============================================================
% 13. CLASS DISTRIBUTION
% ============================================================

fprintf('TRAIN CLASS DISTRIBUTION\n');
disp(tabulate(Ttrain.diagnosis));

fprintf('VALIDATION CLASS DISTRIBUTION\n');
disp(tabulate(Tval.diagnosis));

fprintf('TEST CLASS DISTRIBUTION\n');
disp(tabulate(Ttest.diagnosis));


%% ============================================================
% 14. BALANCE TRAINING DATA
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('BUILDING CLASS-BALANCED TRAINING SET\n');
fprintf('============================================================\n');

classCounts = zeros(5,1);

for c = 0:4

    classCounts(c+1) = ...
        sum(Ttrain.diagnosis == c);

end

targetCount = max(classCounts);

fprintf( ...
    'Original training samples : %d\n', ...
    height(Ttrain));

fprintf( ...
    'Target samples per class  : %d\n', ...
    targetCount);

rng(42);

balancedRows = cell(5,1);

for c = 0:4

    idx = find(Ttrain.diagnosis == c);

    if isempty(idx)

        error( ...
            'Training set contains zero images for class %d.', ...
            c);

    end

    selectedIdx = idx( ...
        randi(numel(idx),targetCount,1));

    balancedRows{c+1} = ...
        Ttrain(selectedIdx,:);

end

Tbalanced = vertcat(balancedRows{:});

fprintf( ...
    '\nBalanced training samples : %d\n', ...
    height(Tbalanced));

fprintf('\nBALANCED DISTRIBUTION\n');
disp(tabulate(Tbalanced.diagnosis));


%% ============================================================
% 15. DATASTORES
% ============================================================

fprintf('\n');
fprintf('CREATING DATASTORES\n');
fprintf('------------------------------------------------------------\n');

trainFiles = cellstr(Tbalanced.imagePath);
trainLabels = Tbalanced.Label;

valFiles = cellstr(Tval.imagePath);
valLabels = Tval.Label;

testFiles = cellstr(Ttest.imagePath);
testLabels = Ttest.Label;

imdsTrain = imageDatastore( ...
    trainFiles, ...
    'Labels',trainLabels);

imdsVal = imageDatastore( ...
    valFiles, ...
    'Labels',valLabels);

imdsTest = imageDatastore( ...
    testFiles, ...
    'Labels',testLabels);


%% ============================================================
% 16. GREEN CHANNEL CLAHE
% ============================================================

fprintf('\n');
fprintf('BUILDING PREPROCESSING PIPELINE\n');
fprintf('------------------------------------------------------------\n');

fprintf('CLAHE method:\n');
fprintf('  Resize first\n');
fprintf('  Green-channel CLAHE\n');
fprintf('  No rgb2lab()\n');
fprintf('  No lab2rgb()\n');
fprintf('  No LAB -> XYZ conversion\n\n');

imdsTrain.ReadFcn = @(filename) ...
    readFundusGreenCLAHE( ...
        filename, ...
        inputSize);

imdsVal.ReadFcn = @(filename) ...
    readFundusGreenCLAHE( ...
        filename, ...
        inputSize);

imdsTest.ReadFcn = @(filename) ...
    readFundusGreenCLAHE( ...
        filename, ...
        inputSize);


%% ============================================================
% 17. DATA AUGMENTATION
% ============================================================

fprintf('CREATING DATA AUGMENTATION\n');
fprintf('------------------------------------------------------------\n');

augmenter = imageDataAugmenter( ...
    'RandRotation',[-12 12], ...
    'RandXReflection',true, ...
    'RandYReflection',false, ...
    'RandXTranslation',[-8 8], ...
    'RandYTranslation',[-8 8], ...
    'RandScale',[0.90 1.10]);

augTrain = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsTrain, ...
    'DataAugmentation',augmenter, ...
    'ColorPreprocessing','none');

augValidation = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsVal, ...
    'ColorPreprocessing','none');

augTest = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsTest, ...
    'ColorPreprocessing','none');

fprintf('CLAHE preprocessing enabled.\n');
fprintf('Augmentation enabled for training.\n');
fprintf('Validation augmentation disabled.\n');
fprintf('Test augmentation disabled.\n');


%% ============================================================
% 18. CREATE LAYER GRAPH
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('PREPARING V4 NETWORK\n');
fprintf('============================================================\n');

fprintf('Converting existing DAGNetwork to layerGraph...\n');

lgraph = layerGraph(trainedNetV3);


%% ============================================================
% 19. VERIFY DAG IMMEDIATELY AFTER CONVERSION
% ============================================================

fprintf('\n');
fprintf('DAG BEFORE MODIFICATION\n');
fprintf('------------------------------------------------------------\n');

layersAfterConversion = lgraph.Layers;
connectionsAfterConversion = lgraph.Connections;

fprintf( ...
    'Layers      : %d\n', ...
    numel(layersAfterConversion));

fprintf( ...
    'Connections : %d\n', ...
    height(connectionsAfterConversion));

if numel(layersAfterConversion) ~= ...
        originalLayerCount

    error( ...
        'Layer count changed during layerGraph conversion.');

end

if height(connectionsAfterConversion) ~= ...
        originalConnectionCount

    error( ...
        'Connection count changed during layerGraph conversion.');

end


%% ============================================================
% 20. FIND CLASSIFICATION OUTPUT LAYER SAFELY
% ============================================================

outputLayerName = '';

layersNow = lgraph.Layers;

for k = 1:numel(layersNow)

    currentLayer = layersNow(k);

    if isa( ...
            currentLayer, ...
            'nnet.cnn.layer.ClassificationOutputLayer')

        outputLayerName = currentLayer.Name;

        break;

    end

end

if isempty(outputLayerName)

    error( ...
        'ClassificationOutputLayer was not found.');

end

fprintf( ...
    'Classification output layer: %s\n', ...
    outputLayerName);


%% ============================================================
% 21. FIND OUTPUT LAYER INDEX SAFELY
% ============================================================

outputLayerIdx = [];

layersNow = lgraph.Layers;

for k = 1:numel(layersNow)

    currentLayer = layersNow(k);

    if strcmp( ...
            currentLayer.Name, ...
            outputLayerName)

        outputLayerIdx = k;

        break;

    end

end

if isempty(outputLayerIdx)

    error( ...
        'Could not determine output layer index.');

end

fprintf( ...
    'Output layer index: %d\n', ...
    outputLayerIdx);


%% ============================================================
% 22. FREEZE EARLY FEATURE EXTRACTOR
% ============================================================

fprintf('\n');
fprintf('FREEZING EARLY NETWORK LAYERS\n');
fprintf('------------------------------------------------------------\n');

freezeUntil = min(123,numel(lgraph.Layers));

fprintf( ...
    'Frozen layers: 1 to %d\n', ...
    freezeUntil);

fprintf( ...
    'Upper layers remain trainable.\n');


for k = 1:freezeUntil

    % IMPORTANT:
    % Get the individual layer first.
    % DO NOT use lgraph.Layers.Name{k}.

    currentLayer = lgraph.Layers(k);

    layerName = currentLayer.Name;

    modified = false;


    if isprop( ...
            currentLayer, ...
            'WeightLearnRateFactor')

        currentLayer.WeightLearnRateFactor = 0;

        modified = true;

    end


    if isprop( ...
            currentLayer, ...
            'BiasLearnRateFactor')

        currentLayer.BiasLearnRateFactor = 0;

        modified = true;

    end


    if isprop( ...
            currentLayer, ...
            'ScaleLearnRateFactor')

        currentLayer.ScaleLearnRateFactor = 0;

        modified = true;

    end


    if isprop( ...
            currentLayer, ...
            'OffsetLearnRateFactor')

        currentLayer.OffsetLearnRateFactor = 0;

        modified = true;

    end


    if modified

        lgraph = replaceLayer( ...
            lgraph, ...
            layerName, ...
            currentLayer);

    end

end


%% ============================================================
% 23. MAKE UPPER LAYERS TRAINABLE
% ============================================================

fprintf('\n');
fprintf('CONFIGURING TRAINABLE UPPER LAYERS\n');
fprintf('------------------------------------------------------------\n');

for k = freezeUntil+1:numel(lgraph.Layers)

    currentLayer = lgraph.Layers(k);

    layerName = currentLayer.Name;

    modified = false;


    if isprop( ...
            currentLayer, ...
            'WeightLearnRateFactor')

        currentLayer.WeightLearnRateFactor = 1;

        modified = true;

    end


    if isprop( ...
            currentLayer, ...
            'BiasLearnRateFactor')

        currentLayer.BiasLearnRateFactor = 1;

        modified = true;

    end


    if isprop( ...
            currentLayer, ...
            'ScaleLearnRateFactor')

        currentLayer.ScaleLearnRateFactor = 1;

        modified = true;

    end


    if isprop( ...
            currentLayer, ...
            'OffsetLearnRateFactor')

        currentLayer.OffsetLearnRateFactor = 1;

        modified = true;

    end


    if modified

        lgraph = replaceLayer( ...
            lgraph, ...
            layerName, ...
            currentLayer);

    end

end


%% ============================================================
% 24. VERIFY DAG AFTER FREEZING
% ============================================================

fprintf('\n');
fprintf('VERIFYING DAG AFTER FREEZING\n');
fprintf('------------------------------------------------------------\n');

finalLayersBeforeTraining = lgraph.Layers;
finalConnectionsBeforeTraining = lgraph.Connections;

fprintf( ...
    'Layers      : %d\n', ...
    numel(finalLayersBeforeTraining));

fprintf( ...
    'Connections : %d\n', ...
    height(finalConnectionsBeforeTraining));

if numel(finalLayersBeforeTraining) ~= ...
        originalLayerCount

    error( ...
        'Layer count changed after freezing.');

end

if height(finalConnectionsBeforeTraining) ~= ...
        originalConnectionCount

    error( ...
        'Connection count changed after freezing.');

end

fprintf('DAG CONNECTIONS PRESERVED.\n');


%% ============================================================
% 25. FINAL NETWORK VALIDATION
% ============================================================

fprintf('\n');
fprintf('VALIDATING NETWORK STRUCTURE\n');
fprintf('------------------------------------------------------------\n');

% Use analyzeNetwork only if available.
% It is not required for training.

try

    analyzeNetwork(lgraph);

catch ME

    warning( ...
        'analyzeNetwork could not be completed: %s', ...
        ME.message);

end


%% ============================================================
% 26. TRAINING CONFIGURATION
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('CONFIGURING V4 TRAINING\n');
fprintf('============================================================\n');

initialLR = 1e-4;

maxEpochs = 15;

miniBatchSize = 16;

validationFrequency = ...
    max(1, ...
    floor( ...
    height(Tbalanced) / ...
    miniBatchSize));


fprintf('Optimizer          : Adam\n');
fprintf('Initial LR         : %.1e\n',initialLR);
fprintf('Max epochs         : %d\n',maxEpochs);
fprintf('Mini-batch size    : %d\n',miniBatchSize);
fprintf('Validation freq.   : %d iterations\n', ...
    validationFrequency);
fprintf('Execution          : CPU\n');


%% ============================================================
% 27. TRAINING OPTIONS
% ============================================================

options = trainingOptions( ...
    'adam', ...
    'InitialLearnRate',initialLR, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augValidation, ...
    'ValidationFrequency',validationFrequency, ...
    'ValidationPatience',4, ...
    'ExecutionEnvironment','cpu', ...
    'Verbose',true, ...
    'Plots','training-progress');


%% ============================================================
% 28. START TRAINING
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('STARTING V4 FINE-TUNING\n');
fprintf('============================================================\n');

fprintf('\n');
fprintf('CPU training will now begin.\n');
fprintf('Green-channel CLAHE only.\n');
fprintf('No LAB conversion.\n');
fprintf('Original DAG connections preserved.\n\n');

tic;

[trainedNetV4,trainInfo] = trainNetwork( ...
    augTrain, ...
    lgraph, ...
    options);

trainingTime = toc;


%% ============================================================
% 29. TRAINING COMPLETE
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('V4 TRAINING COMPLETE\n');
fprintf('============================================================\n');

fprintf( ...
    'Training time: %.2f minutes\n', ...
    trainingTime/60);


%% ============================================================
% 30. VERIFY TRAINED NETWORK
% ============================================================

fprintf('\n');
fprintf('VERIFYING TRAINED NETWORK\n');
fprintf('------------------------------------------------------------\n');

fprintf( ...
    'Network type : %s\n', ...
    class(trainedNetV4));

fprintf( ...
    'Layers       : %d\n', ...
    numel(trainedNetV4.Layers));

fprintf( ...
    'Connections  : %d\n', ...
    height(trainedNetV4.Connections));


if ~isa(trainedNetV4,'DAGNetwork')

    error( ...
        'V4 result is not a DAGNetwork.');

end


if numel(trainedNetV4.Layers) ~= ...
        originalLayerCount

    error( ...
        'V4 layer count differs from V3.');

end


if height(trainedNetV4.Connections) ~= ...
        originalConnectionCount

    error( ...
        'V4 connection count differs from V3.');

end

fprintf('DAG verification PASSED.\n');


%% ============================================================
% 31. VALIDATION
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('RUNNING V4 VALIDATION\n');
fprintf('============================================================\n');

YPredVal = classify( ...
    trainedNetV4, ...
    augValidation, ...
    'ExecutionEnvironment','cpu');

YTrueVal = valLabels;

validationAccuracy = mean( ...
    YPredVal == YTrueVal);

fprintf( ...
    'Validation samples : %d\n', ...
    numel(YTrueVal));

fprintf( ...
    'Validation accuracy: %.2f%%\n', ...
    100*validationAccuracy);


%% ============================================================
% 32. VALIDATION CONFUSION MATRIX
% ============================================================

Cval = confusionmat( ...
    YTrueVal, ...
    YPredVal, ...
    'Order',categorical( ...
        classNames, ...
        classNames));


fprintf('\n');
fprintf('VALIDATION CONFUSION MATRIX\n');
fprintf('------------------------------------------------------------\n');

disp(Cval);

valRecall = zeros(5,1);

for c = 1:5

    denominator = sum(Cval(c,:));

    if denominator > 0

        valRecall(c) = ...
            Cval(c,c) / denominator;

    end

end

balancedValidationAccuracy = ...
    mean(valRecall);


fprintf('\n');
fprintf('VALIDATION PER-CLASS RECALL\n');

for c = 1:5

    fprintf( ...
        '%s | Samples: %d | Recall: %.2f%%\n', ...
        classNames{c}, ...
        sum(Cval(c,:)), ...
        100*valRecall(c));

end


fprintf('\n');
fprintf( ...
    'Validation Accuracy        : %.2f%%\n', ...
    100*validationAccuracy);

fprintf( ...
    'Validation Balanced Acc.   : %.2f%%\n', ...
    100*balancedValidationAccuracy);


%% ============================================================
% 33. FINAL TEST
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('RUNNING FINAL MESSIDOR TEST\n');
fprintf('============================================================\n');

YPredTest = classify( ...
    trainedNetV4, ...
    augTest, ...
    'ExecutionEnvironment','cpu');

YTrueTest = testLabels;

testAccuracy = mean( ...
    YPredTest == YTrueTest);


fprintf('\n');
fprintf('FINAL TEST RESULTS\n');
fprintf('------------------------------------------------------------\n');

fprintf( ...
    'Test samples : %d\n', ...
    numel(YTrueTest));

fprintf( ...
    'Test accuracy: %.2f%%\n', ...
    100*testAccuracy);


%% ============================================================
% 34. TEST CONFUSION MATRIX
% ============================================================

Ctest = confusionmat( ...
    YTrueTest, ...
    YPredTest, ...
    'Order',categorical( ...
        classNames, ...
        classNames));


fprintf('\n');
fprintf('FINAL TEST CONFUSION MATRIX\n');
fprintf('------------------------------------------------------------\n');

disp(Ctest);

testRecall = zeros(5,1);

for c = 1:5

    denominator = sum(Ctest(c,:));

    if denominator > 0

        testRecall(c) = ...
            Ctest(c,c) / denominator;

    end

end

balancedTestAccuracy = ...
    mean(testRecall);


fprintf('\n');
fprintf('FINAL TEST PER-CLASS RECALL\n');

for c = 1:5

    fprintf( ...
        '%s | Samples: %d | Recall: %.2f%%\n', ...
        classNames{c}, ...
        sum(Ctest(c,:)), ...
        100*testRecall(c));

end


fprintf('\n');
fprintf( ...
    'Final Test Accuracy      : %.2f%%\n', ...
    100*testAccuracy);

fprintf( ...
    'Final Test Balanced Acc. : %.2f%%\n', ...
    100*balancedTestAccuracy);


%% ============================================================
% 35. SAVE V4 MODEL
% ============================================================

fprintf('\n');
fprintf('SAVING V4 MODEL\n');
fprintf('------------------------------------------------------------\n');

save( ...
    modelOutputPath, ...
    'trainedNetV4', ...
    'trainInfo', ...
    'classNames', ...
    'inputSize', ...
    'validationAccuracy', ...
    'balancedValidationAccuracy', ...
    'testAccuracy', ...
    'balancedTestAccuracy', ...
    'Cval', ...
    'Ctest', ...
    'valRecall', ...
    'testRecall', ...
    'trainingTime', ...
    '-v7.3');


fprintf( ...
    'Model saved:\n%s\n', ...
    modelOutputPath);


%% ============================================================
% 36. SAVE TRAINING INFORMATION
% ============================================================

save( ...
    trainingInfoPath, ...
    'trainInfo', ...
    'validationAccuracy', ...
    'balancedValidationAccuracy', ...
    'testAccuracy', ...
    'balancedTestAccuracy', ...
    'Cval', ...
    'Ctest', ...
    'valRecall', ...
    'testRecall', ...
    'trainingTime', ...
    'classNames', ...
    '-v7.3');


fprintf( ...
    'Training information saved:\n%s\n', ...
    trainingInfoPath);


%% ============================================================
% 37. FINAL SUMMARY
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('TRUST-DR MESSIDOR V4 COMPLETE\n');
fprintf('============================================================\n');

fprintf( ...
    'V3 network layers      : %d\n', ...
    originalLayerCount);

fprintf( ...
    'V3 network connections : %d\n', ...
    originalConnectionCount);

fprintf( ...
    'V4 network layers      : %d\n', ...
    numel(trainedNetV4.Layers));

fprintf( ...
    'V4 network connections : %d\n', ...
    height(trainedNetV4.Connections));

fprintf('\n');

fprintf( ...
    'Validation Accuracy     : %.2f%%\n', ...
    100*validationAccuracy);

fprintf( ...
    'Validation Balanced Acc : %.2f%%\n', ...
    100*balancedValidationAccuracy);

fprintf( ...
    'Final Test Accuracy     : %.2f%%\n', ...
    100*testAccuracy);

fprintf( ...
    'Final Test Balanced Acc : %.2f%%\n', ...
    100*balancedTestAccuracy);

fprintf('\n');

fprintf( ...
    'V4 model:\n%s\n', ...
    modelOutputPath);

fprintf('\n');
fprintf('============================================================\n');
fprintf('DONE\n');
fprintf('============================================================\n');


%% ============================================================
% LOCAL FUNCTION 1
% PREPARE SPLIT TABLE
%% ============================================================

function T = prepareSplitTable(T_original)

    varNames = T_original.Properties.VariableNames;


    % ---------------------------------------------------------
    % Find filename column
    % ---------------------------------------------------------

    filenameCandidates = { ...
        'id_code', ...
        'filename', ...
        'image', ...
        'imageName', ...
        'name'};

    filenameVar = '';

    for k = 1:numel(filenameCandidates)

        idx = strcmpi( ...
            varNames, ...
            filenameCandidates{k});

        if any(idx)

            foundIndex = find(idx,1);

            filenameVar = ...
                varNames{foundIndex};

            break;

        end

    end


    if isempty(filenameVar)

        error([ ...
            'Could not find image filename column. ' ...
            'Expected id_code.']);

    end


    % ---------------------------------------------------------
    % Find diagnosis column
    % ---------------------------------------------------------

    diagnosisCandidates = { ...
        'diagnosis', ...
        'label', ...
        'grade', ...
        'dr_grade'};

    diagnosisVar = '';

    for k = 1:numel(diagnosisCandidates)

        idx = strcmpi( ...
            varNames, ...
            diagnosisCandidates{k});

        if any(idx)

            foundIndex = find(idx,1);

            diagnosisVar = ...
                varNames{foundIndex};

            break;

        end

    end


    if isempty(diagnosisVar)

        error( ...
            'Could not find diagnosis column.');

    end


    % ---------------------------------------------------------
    % Extract columns safely
    % ---------------------------------------------------------

    filenames = ...
        T_original.(filenameVar);

    diagnosis = ...
        T_original.(diagnosisVar);


    % ---------------------------------------------------------
    % Normalize filenames
    % ---------------------------------------------------------

    if iscell(filenames)

        filenames = string(filenames);

    elseif ischar(filenames)

        filenames = string(cellstr(filenames));

    else

        filenames = string(filenames);

    end

    filenames = strip(filenames);


    % ---------------------------------------------------------
    % Normalize diagnosis
    % ---------------------------------------------------------

    if iscategorical(diagnosis)

        diagnosis = double(diagnosis);

    elseif iscell(diagnosis)

        diagnosis = ...
            str2double(string(diagnosis));

    elseif isstring(diagnosis)

        diagnosis = ...
            str2double(diagnosis);

    else

        diagnosis = double(diagnosis);

    end


    diagnosis = diagnosis(:);
    filenames = filenames(:);


    % ---------------------------------------------------------
    % Remove invalid rows
    % ---------------------------------------------------------

    valid = ...
        ~ismissing(filenames) & ...
        ~isnan(diagnosis) & ...
        diagnosis >= 0 & ...
        diagnosis <= 4;

    filenames = filenames(valid);
    diagnosis = diagnosis(valid);


    % ---------------------------------------------------------
    % Build clean table
    % ---------------------------------------------------------

    T = table( ...
        filenames, ...
        diagnosis, ...
        'VariableNames', ...
        {'id_code','diagnosis'});

end


%% ============================================================
% LOCAL FUNCTION 2
% ATTACH IMAGE PATHS
%% ============================================================

function T = attachImagePaths(T,imageFolder)

    n = height(T);

    imagePath = strings(n,1);

    keep = false(n,1);


    for k = 1:n

        filename = char(T.id_code(k));

        candidate = ...
            fullfile(imageFolder,filename);

        if isfile(candidate)

            imagePath(k) = ...
                string(candidate);

            keep(k) = true;

        end

    end


    missingCount = ...
        sum(~keep);


    if missingCount > 0

        fprintf( ...
            'Removing %d missing image references.\n', ...
            missingCount);

    end


    T.imagePath = imagePath;

    T = T(keep,:);

end


%% ============================================================
% LOCAL FUNCTION 3
% GREEN CHANNEL CLAHE
%% ============================================================

function Iout = readFundusGreenCLAHE( ...
        filename,inputSize)

    % ---------------------------------------------------------
    % Read image
    % ---------------------------------------------------------

    I = imread(filename);


    % ---------------------------------------------------------
    % Ensure RGB
    % ---------------------------------------------------------

    if ndims(I) == 2

        I = repmat( ...
            I, ...
            [1 1 3]);

    end


    if size(I,3) > 3

        I = I(:,:,1:3);

    end


    % ---------------------------------------------------------
    % Convert safely to uint8
    % ---------------------------------------------------------

    if ~isa(I,'uint8')

        I = im2uint8(I);

    end


    % ---------------------------------------------------------
    % Resize BEFORE CLAHE
    % ---------------------------------------------------------

    I = imresize( ...
        I, ...
        inputSize(1:2));


    % ---------------------------------------------------------
    % GREEN CHANNEL
    % ---------------------------------------------------------

    G = I(:,:,2);


    % ---------------------------------------------------------
    % CLAHE
    %
    % No rgb2lab()
    % No lab2rgb()
    % No XYZ conversion
    % ---------------------------------------------------------

    Gdouble = im2double(G);

    Gclahe = adapthisteq( ...
        Gdouble, ...
        'NumTiles',[8 8], ...
        'ClipLimit',0.01);


    % ---------------------------------------------------------
    % Replace green channel
    % ---------------------------------------------------------

    I(:,:,2) = ...
        im2uint8(Gclahe);


    % ---------------------------------------------------------
    % Return
    % ---------------------------------------------------------

    Iout = I;

end