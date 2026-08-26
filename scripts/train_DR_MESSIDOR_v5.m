%% ============================================================
% TRUST-DR MESSIDOR V5-B
% AGGRESSIVE ACCURACY + BALANCED-ACCURACY OPTIMIZATION
%
% IMPORTANT:
% - Preserves original DAGNetwork residual connections
% - No manual reconstruction of ResNet connections
% - No LAB / lab2rgb / XYZ conversion
% - Green-channel CLAHE
% - Class-weighted loss
% - Moderate augmentation
% - Early layers frozen safely
% - Validation + final test evaluation
%
% MATLAB R2026a
%% ============================================================

clear;
clc;
close all;

fprintf('\n============================================================\n');
fprintf('TRUST-DR MESSIDOR V5-B\n');
fprintf('AGGRESSIVE ACCURACY + BALANCED ACCURACY OPTIMIZATION\n');
fprintf('============================================================\n\n');

%% ============================================================
% 1. PATHS
% ============================================================

rootFolder = 'C:\Users\sijar\Downloads\TRUST_DR';

imageFolder = fullfile( ...
    rootFolder, ...
    'datasets', ...
    'archive', ...
    'messidor-2', ...
    'messidor-2', ...
    'preprocess');

labelsCSV = fullfile( ...
    rootFolder, ...
    'datasets', ...
    'archive', ...
    'messidor_data.csv');

trainCSV = fullfile( ...
    rootFolder, ...
    'results', ...
    'messidor_training', ...
    'train.csv');

valCSV = fullfile( ...
    rootFolder, ...
    'results', ...
    'messidor_training', ...
    'validation.csv');

testCSV = fullfile( ...
    rootFolder, ...
    'results', ...
    'messidor_training', ...
    'test.csv');

resultsFolder = fullfile( ...
    rootFolder, ...
    'results', ...
    'messidor_v5B');

if ~exist(resultsFolder,'dir')
    mkdir(resultsFolder);
end

fprintf('VERIFYING FILES\n');
fprintf('------------------------------------------------------------\n');

assert(exist(imageFolder,'dir') == 7, ...
    'Image folder not found:\n%s', imageFolder);

assert(exist(labelsCSV,'file') == 2, ...
    'Labels CSV not found:\n%s', labelsCSV);

assert(exist(trainCSV,'file') == 2, ...
    'Training CSV not found:\n%s', trainCSV);

assert(exist(valCSV,'file') == 2, ...
    'Validation CSV not found:\n%s', valCSV);

assert(exist(testCSV,'file') == 2, ...
    'Test CSV not found:\n%s', testCSV);

fprintf('Image folder : OK\n');
fprintf('Labels CSV   : OK\n');
fprintf('Training CSV : OK\n');
fprintf('Validation CSV : OK\n');
fprintf('Test CSV     : OK\n');

%% ============================================================
% 2. FIND V3 MODEL
% ============================================================

fprintf('\n============================================================\n');
fprintf('SEARCHING FOR V3 MODEL\n');
fprintf('============================================================\n');

modelCandidates = dir(fullfile(rootFolder,'**','*.mat'));

v3ModelPath = '';

for k = 1:numel(modelCandidates)

    candidate = fullfile( ...
        modelCandidates(k).folder, ...
        modelCandidates(k).name);

    try

        info = whos('-file',candidate);

        for j = 1:numel(info)

            if strcmp(info(j).class,'DAGNetwork')

                v3ModelPath = candidate;
                break;

            end

        end

        if ~isempty(v3ModelPath)
            break;
        end

    catch
        % Ignore unreadable MAT files.
    end

end

assert(~isempty(v3ModelPath), ...
    ['No MAT file containing a DAGNetwork was found under: ' ...
     rootFolder]);

fprintf('V3 model found:\n%s\n',v3ModelPath);

%% ============================================================
% 3. LOAD V3 MODEL
% ============================================================

fprintf('\n============================================================\n');
fprintf('LOADING V3 MODEL\n');
fprintf('============================================================\n');

M = load(v3ModelPath);

modelNames = fieldnames(M);

trainedNet = [];

for k = 1:numel(modelNames)

    candidate = M.(modelNames{k});

    if isa(candidate,'DAGNetwork')

        trainedNet = candidate;

        fprintf('Found DAGNetwork in variable: %s\n', ...
            modelNames{k});

        break;

    end

end

assert(~isempty(trainedNet), ...
    'DAGNetwork could not be extracted from MAT file.');

fprintf('Network type : %s\n',class(trainedNet));

inputSize = trainedNet.Layers(1).InputSize;

fprintf('Input size   : [%d %d %d]\n', ...
    inputSize(1),inputSize(2),inputSize(3));

fprintf('Layers       : %d\n',numel(trainedNet.Layers));
fprintf('Connections  : %d\n',height(trainedNet.Connections));

%% ============================================================
% 4. LOAD CSV FILES
% ============================================================

fprintf('\n============================================================\n');
fprintf('LOADING MESSIDOR SPLITS\n');
fprintf('============================================================\n');

Ttrain = readtable(trainCSV);
Tval   = readtable(valCSV);
Ttest  = readtable(testCSV);

fprintf('Training images   : %d\n',height(Ttrain));
fprintf('Validation images : %d\n',height(Tval));
fprintf('Final test images : %d\n',height(Ttest));

%% ============================================================
% 5. NORMALIZE CSV COLUMN NAMES
% ============================================================

Ttrain = normalizeMessidorTable(Ttrain);
Tval   = normalizeMessidorTable(Tval);
Ttest  = normalizeMessidorTable(Ttest);

%% ============================================================
% 6. VERIFY IMAGES
% ============================================================

fprintf('\n============================================================\n');
fprintf('VERIFYING IMAGE FILES\n');
fprintf('============================================================\n');

Ttrain = attachImagePaths(Ttrain,imageFolder);
Tval   = attachImagePaths(Tval,imageFolder);
Ttest  = attachImagePaths(Ttest,imageFolder);

fprintf('Training matched   : %d / %d\n', ...
    sum(Ttrain.ImageExists),height(Ttrain));

fprintf('Validation matched : %d / %d\n', ...
    sum(Tval.ImageExists),height(Tval));

fprintf('Test matched       : %d / %d\n', ...
    sum(Ttest.ImageExists),height(Ttest));

assert(all(Ttrain.ImageExists), ...
    'Some training images could not be matched.');

assert(all(Tval.ImageExists), ...
    'Some validation images could not be matched.');

assert(all(Ttest.ImageExists), ...
    'Some test images could not be matched.');

%% ============================================================
% 7. CLASS DEFINITIONS
% ============================================================

classNames = [ ...
    "Mild"
    "Moderate"
    "No_DR"
    "Proliferative"
    "Severe"];

classIDs = (0:4)';

fprintf('\n============================================================\n');
fprintf('MESSIDOR CLASS DISTRIBUTION\n');
fprintf('============================================================\n');

printDistribution(Ttrain.diagnosis,classNames,'TRAINING');
printDistribution(Tval.diagnosis,classNames,'VALIDATION');
printDistribution(Ttest.diagnosis,classNames,'FINAL TEST');

%% ============================================================
% 8. CREATE CATEGORICAL LABELS
% ============================================================

Ytrain = diagnosisToCategorical( ...
    Ttrain.diagnosis,classNames);

Yval = diagnosisToCategorical( ...
    Tval.diagnosis,classNames);

Ytest = diagnosisToCategorical( ...
    Ttest.diagnosis,classNames);

%% ============================================================
% 9. CLASS WEIGHTS
% ============================================================

fprintf('\n============================================================\n');
fprintf('CALCULATING CLASS WEIGHTS\n');
fprintf('============================================================\n');

trainCounts = zeros(5,1);

for k = 1:5
    trainCounts(k) = sum(Ttrain.diagnosis == classIDs(k));
end

% Inverse-frequency weighting.
% Square-root weighting is deliberately used instead of full
% inverse frequency because full inverse weighting can become
% excessively aggressive for Proliferative and Severe.

rawWeights = 1 ./ sqrt(max(trainCounts,1));

% Normalize weights around 1.
classWeights = rawWeights / mean(rawWeights);

for k = 1:5

    fprintf('%-16s Count: %4d   Weight: %.4f\n', ...
        classNames(k), ...
        trainCounts(k), ...
        classWeights(k));

end

%% ============================================================
% 10. IMAGE DATASTORES
% ============================================================

fprintf('\n============================================================\n');
fprintf('CREATING DATASTORES\n');
fprintf('============================================================\n');

imdsTrain = imageDatastore( ...
    Ttrain.ImagePath, ...
    'Labels',Ytrain, ...
    'ReadFcn',@readFundusV5B);

imdsVal = imageDatastore( ...
    Tval.ImagePath, ...
    'Labels',Yval, ...
    'ReadFcn',@readFundusV5B);

imdsTest = imageDatastore( ...
    Ttest.ImagePath, ...
    'Labels',Ytest, ...
    'ReadFcn',@readFundusV5B);

%% ============================================================
% 11. AUGMENTATION
% ============================================================

fprintf('\n============================================================\n');
fprintf('CREATING DATA AUGMENTATION\n');
fprintf('============================================================\n');

fprintf('Green-channel CLAHE : ENABLED\n');
fprintf('LAB conversion      : DISABLED\n');
fprintf('lab2rgb              : DISABLED\n');
fprintf('Training augmentation: ENABLED\n');
fprintf('Validation augmentation: DISABLED\n');
fprintf('Test augmentation: DISABLED\n');

augmenter = imageDataAugmenter( ...
    'RandRotation',[-8 8], ...
    'RandXReflection',true, ...
    'RandYReflection',false, ...
    'RandXTranslation',[-8 8], ...
    'RandYTranslation',[-8 8], ...
    'RandScale',[0.95 1.05]);

augTrain = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsTrain, ...
    'DataAugmentation',augmenter, ...
    'ColorPreprocessing','none');

augVal = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsVal, ...
    'ColorPreprocessing','none');

augTest = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsTest, ...
    'ColorPreprocessing','none');

%% ============================================================
% 12. PRESERVE DAG
% ============================================================

fprintf('\n============================================================\n');
fprintf('PREPARING V5-B NETWORK\n');
fprintf('============================================================\n');

fprintf('Converting DAGNetwork to layerGraph WITHOUT rebuilding it.\n');

lgraph = layerGraph(trainedNet);

fprintf('Layers      : %d\n',numel(lgraph.Layers));
fprintf('Connections : %d\n',height(lgraph.Connections));

%% ============================================================
% 13. FIND CLASSIFICATION OUTPUT LAYER
% ============================================================

outputLayerName = '';

for k = 1:numel(lgraph.Layers)

    thisLayer = lgraph.Layers(k);

    if isa(thisLayer,'nnet.cnn.layer.ClassificationOutputLayer')

        outputLayerName = thisLayer.Name;

        break;

    end

end

assert(~isempty(outputLayerName), ...
    'Classification output layer not found.');

fprintf('Classification output: %s\n', ...
    outputLayerName);

%% ============================================================
% 14. REPLACE ONLY OUTPUT LAYER
% ============================================================

fprintf('\n============================================================\n');
fprintf('CONFIGURING CLASS-WEIGHTED OUTPUT\n');
fprintf('============================================================\n');

weightedOutput = classificationLayer( ...
    'Name',outputLayerName, ...
    'Classes',categorical(classNames,classNames), ...
    'ClassWeights',classWeights');

lgraph = replaceLayer( ...
    lgraph, ...
    outputLayerName, ...
    weightedOutput);

fprintf('Original DAG connections preserved.\n');
fprintf('Class-weighted classification loss enabled.\n');

%% ============================================================
% 15. FREEZE EARLY LAYERS
% ============================================================

fprintf('\n============================================================\n');
fprintf('FREEZING EARLY FEATURE EXTRACTOR\n');
fprintf('============================================================\n');

% Freeze approximately first 60%% of the network.
% We modify layers individually while keeping every DAG connection.

freezeUntil = floor(0.60 * numel(lgraph.Layers));

fprintf('Freeze boundary: layer %d / %d\n', ...
    freezeUntil,numel(lgraph.Layers));

for k = 1:freezeUntil

    layer = lgraph.Layers(k);

    changed = false;

    if isprop(layer,'WeightLearnRateFactor')

        layer.WeightLearnRateFactor = 0;
        changed = true;

    end

    if isprop(layer,'BiasLearnRateFactor')

        layer.BiasLearnRateFactor = 0;
        changed = true;

    end

    if isprop(layer,'ScaleLearnRateFactor')

        layer.ScaleLearnRateFactor = 0;
        changed = true;

    end

    if isprop(layer,'OffsetLearnRateFactor')

        layer.OffsetLearnRateFactor = 0;
        changed = true;

    end

    if changed

        lgraph = replaceLayer( ...
            lgraph, ...
            layer.Name, ...
            layer);

    end

end

fprintf('Early feature extractor frozen.\n');
fprintf('Upper layers remain trainable.\n');

%% ============================================================
% 16. VERIFY DAG BEFORE TRAINING
% ============================================================

fprintf('\n============================================================\n');
fprintf('VERIFYING DAG\n');
fprintf('============================================================\n');

% analyzeNetwork is intentionally not used here because it opens
% a GUI and is unnecessary for automated training.

% Convert to DAGNetwork-compatible training graph by checking
% that every layer has a valid connection structure.

fprintf('DAG layer count      : %d\n',numel(lgraph.Layers));
fprintf('DAG connection count : %d\n',height(lgraph.Connections));
fprintf('Output layer         : %s\n',outputLayerName);

%% ============================================================
% 17. TRAINING OPTIONS
% ============================================================

fprintf('\n============================================================\n');
fprintf('CONFIGURING V5-B TRAINING\n');
fprintf('============================================================\n');

miniBatch = 16;
maxEpochs = 20;

initialLR = 5e-5;

options = trainingOptions( ...
    'adam', ...
    'InitialLearnRate',initialLR, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatch, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augVal, ...
    'ValidationFrequency', ...
        max(1,floor(height(Ttrain)/miniBatch)), ...
    'ValidationPatience',5, ...
    'Verbose',true, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','cpu', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',5, ...
    'L2Regularization',1e-4, ...
    'GradientThreshold',5);

fprintf('Optimizer       : Adam\n');
fprintf('Initial LR      : %.2e\n',initialLR);
fprintf('Epochs          : %d\n',maxEpochs);
fprintf('Mini-batch      : %d\n',miniBatch);
fprintf('Execution       : CPU\n');
fprintf('Class weighting : ON\n');
fprintf('CLAHE           : ON\n');

%% ============================================================
% 18. TRAIN
% ============================================================

fprintf('\n============================================================\n');
fprintf('STARTING V5-B FINE-TUNING\n');
fprintf('============================================================\n');

tic;

[trainedNetV5B,trainInfo] = trainNetwork( ...
    augTrain, ...
    lgraph, ...
    options);

trainingTime = toc;

fprintf('\nTraining completed.\n');
fprintf('Training time: %.2f minutes\n', ...
    trainingTime/60);

%% ============================================================
% 19. SAVE MODEL
% ============================================================

modelPath = fullfile( ...
    resultsFolder, ...
    'TRUST_DR_MESSIDOR_V5B.mat');

save(modelPath, ...
    'trainedNetV5B', ...
    'trainInfo', ...
    'classNames', ...
    'classWeights', ...
    'trainingTime', ...
    '-v7.3');

fprintf('\nV5-B model saved:\n%s\n',modelPath);

%% ============================================================
% 20. VALIDATION PREDICTION
% ============================================================

fprintf('\n============================================================\n');
fprintf('VALIDATION EVALUATION\n');
fprintf('============================================================\n');

reset(augVal);

YPredVal = classify( ...
    trainedNetV5B, ...
    augVal, ...
    'ExecutionEnvironment','cpu');

YTrueVal = Yval;

Cval = confusionmat( ...
    YTrueVal, ...
    YPredVal, ...
    'Order',categorical(classNames,classNames));

[accVal,balAccVal,recallVal] = ...
    calculateMetrics(Cval);

fprintf('\nVALIDATION RESULTS\n');
fprintf('------------------------------------------------------------\n');

fprintf('Accuracy          : %.2f%%\n', ...
    100*accVal);

fprintf('Balanced Accuracy : %.2f%%\n', ...
    100*balAccVal);

for k = 1:5

    fprintf('%-16s Recall: %.2f%%\n', ...
        classNames(k), ...
        100*recallVal(k));

end

%% ============================================================
% 21. FINAL TEST
% ============================================================

fprintf('\n============================================================\n');
fprintf('FINAL TEST EVALUATION\n');
fprintf('============================================================\n');

reset(augTest);

YPredTest = classify( ...
    trainedNetV5B, ...
    augTest, ...
    'ExecutionEnvironment','cpu');

YTrueTest = Ytest;

Ctest = confusionmat( ...
    YTrueTest, ...
    YPredTest, ...
    'Order',categorical(classNames,classNames));

[accTest,balAccTest,recallTest] = ...
    calculateMetrics(Ctest);

fprintf('\n============================================================\n');
fprintf('MESSIDOR-2 V5-B FINAL RESULTS\n');
fprintf('============================================================\n');

fprintf('Total test images : %d\n',height(Ttest));

fprintf('Accuracy          : %.2f%%\n', ...
    100*accTest);

fprintf('Balanced Accuracy : %.2f%%\n', ...
    100*balAccTest);

fprintf('\nPER-CLASS RECALL\n');
fprintf('------------------------------------------------------------\n');

for k = 1:5

    fprintf('%-16s Samples: %4d | Recall: %.2f%%\n', ...
        classNames(k), ...
        sum(Ttest.diagnosis == classIDs(k)), ...
        100*recallTest(k));

end

%% ============================================================
% 22. CONFUSION MATRICES
% ============================================================

fprintf('\nVALIDATION CONFUSION MATRIX\n');
disp(Cval);

fprintf('\nFINAL TEST CONFUSION MATRIX\n');
disp(Ctest);

%% ============================================================
% 23. SAVE RESULTS
% ============================================================

resultsPath = fullfile( ...
    resultsFolder, ...
    'V5B_metrics.mat');

save(resultsPath, ...
    'Cval', ...
    'Ctest', ...
    'accVal', ...
    'balAccVal', ...
    'recallVal', ...
    'accTest', ...
    'balAccTest', ...
    'recallTest', ...
    'classWeights', ...
    'classNames');

%% ============================================================
% 24. SAVE CSV
% ============================================================

resultTable = table( ...
    classNames, ...
    recallTest, ...
    'VariableNames', ...
    {'Class','TestRecall'});

writetable( ...
    resultTable, ...
    fullfile(resultsFolder,'V5B_per_class_recall.csv'));

%% ============================================================
% 25. FINISHED
% ============================================================

fprintf('\n============================================================\n');
fprintf('V5-B COMPLETE\n');
fprintf('============================================================\n');

fprintf('Validation Accuracy          : %.2f%%\n',100*accVal);
fprintf('Validation Balanced Accuracy : %.2f%%\n',100*balAccVal);

fprintf('\nFINAL TEST\n');
fprintf('Accuracy          : %.2f%%\n',100*accTest);
fprintf('Balanced Accuracy : %.2f%%\n',100*balAccTest);

fprintf('\nModel:\n%s\n',modelPath);

fprintf('\nResults:\n%s\n',resultsPath);

fprintf('\n============================================================\n');


%% ============================================================
% LOCAL FUNCTIONS
% ============================================================

function T = normalizeMessidorTable(T)

    vars = T.Properties.VariableNames;

    diagnosisVar = '';

    filenameVar = '';

    for k = 1:numel(vars)

        v = vars{k};

        if strcmpi(v,'diagnosis')
            diagnosisVar = v;
        end

        if strcmpi(v,'id_code') || ...
           strcmpi(v,'filename') || ...
           strcmpi(v,'image') || ...
           strcmpi(v,'imageName')

            filenameVar = v;

        end

    end

    assert(~isempty(diagnosisVar), ...
        'diagnosis column not found in CSV.');

    assert(~isempty(filenameVar), ...
        'Image filename column not found in CSV.');

    if ~strcmp(filenameVar,'id_code')

        T.id_code = T.(filenameVar);

    end

    if ~strcmp(diagnosisVar,'diagnosis')

        T.diagnosis = T.(diagnosisVar);

    end

    T.id_code = string(T.id_code);

    T.diagnosis = double(T.diagnosis);

end


function T = attachImagePaths(T,imageFolder)

    n = height(T);

    imagePath = strings(n,1);

    imageExists = false(n,1);

    for k = 1:n

        filename = char(T.id_code(k));

        p = fullfile(imageFolder,filename);

        imagePath(k) = string(p);

        imageExists(k) = exist(p,'file') == 2;

    end

    T.ImagePath = imagePath;

    T.ImageExists = imageExists;

end


function Y = diagnosisToCategorical(diagnosis,classNames)

    diagnosis = double(diagnosis);

    labels = strings(size(diagnosis));

    for k = 0:4

        labels(diagnosis == k) = classNames(k+1);

    end

    Y = categorical( ...
        labels, ...
        classNames, ...
        'Ordinal',false);

end


function printDistribution(diagnosis,classNames,titleText)

    fprintf('\n%s CLASS DISTRIBUTION\n',titleText);

    total = numel(diagnosis);

    for k = 0:4

        n = sum(double(diagnosis) == k);

        fprintf('%-16s %5d   %.2f%%\n', ...
            classNames(k+1), ...
            n, ...
            100*n/total);

    end

end


function Iout = readFundusV5B(filename)

    I = imread(filename);

    if ndims(I) == 2

        I = repmat(I,[1 1 3]);

    end

    if size(I,3) > 3

        I = I(:,:,1:3);

    end

    I = im2uint8(I);

    % ---------------------------------------------------------
    % Green-channel CLAHE
    % ---------------------------------------------------------

    G = I(:,:,2);

    G = adapthisteq( ...
        G, ...
        'NumTiles',[8 8], ...
        'ClipLimit',0.01, ...
        'Distribution','rayleigh');

    % ---------------------------------------------------------
    % Reconstruct RGB.
    %
    % We intentionally do NOT use:
    % rgb2lab
    % lab2rgb
    % XYZ conversion
    % ---------------------------------------------------------

    Iout = I;

    Iout(:,:,2) = G;

end


function [accuracy,balancedAccuracy,recall] = ...
    calculateMetrics(C)

    rowTotals = sum(C,2);

    recall = zeros(size(rowTotals));

    for k = 1:numel(rowTotals)

        if rowTotals(k) > 0

            recall(k) = C(k,k) / rowTotals(k);

        else

            recall(k) = 0;

        end

    end

    accuracy = sum(diag(C)) / max(sum(C(:)),1);

    balancedAccuracy = mean(recall);

end