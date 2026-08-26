%% ============================================================
% TRUST-DR MESSIDOR V5-B
% FINAL TEST EVALUATION
% =============================================================
% Evaluates the trained V5-B model on the untouched final test set.
%
% Output:
%   Overall Accuracy
%   Balanced Accuracy
%   Per-Class Recall
%   Per-Class Precision
%   Per-Class F1 Score
%   Confusion Matrix
%   CSV result files
%
% =============================================================

clear;
clc;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf('TRUST-DR MESSIDOR V5-B\n');
fprintf('FINAL TEST EVALUATION\n');
fprintf('============================================================\n');

%% =============================================================
% PATH CONFIGURATION
% =============================================================

projectRoot = 'C:\Users\sijar\Downloads\TRUST_DR';

modelFile = fullfile( ...
    projectRoot, ...
    'models', ...
    'trained_DR_model_v5B.mat');

testCSV = fullfile( ...
    projectRoot, ...
    'results', ...
    'messidor_training', ...
    'test.csv');

resultsFolder = fullfile( ...
    projectRoot, ...
    'results', ...
    'messidor_v5B_evaluation');

imageRoot = fullfile( ...
    projectRoot, ...
    'datasets');

fprintf('\nVERIFYING FILES\n');
fprintf('------------------------------------------------------------\n');

assert(isfile(modelFile), ...
    'V5-B model not found:\n%s', modelFile);

assert(isfile(testCSV), ...
    'Test CSV not found:\n%s', testCSV);

assert(isfolder(imageRoot), ...
    'Image root folder not found:\n%s', imageRoot);

if ~exist(resultsFolder,'dir')
    mkdir(resultsFolder);
end

fprintf('V5-B model  : OK\n');
fprintf('Test CSV    : OK\n');
fprintf('Image root  : OK\n');


%% =============================================================
% LOAD V5-B MODEL
% =============================================================

fprintf('\nLOADING V5-B MODEL\n');
fprintf('------------------------------------------------------------\n');

S = load(modelFile);

modelVariableNames = fieldnames(S);

trainedNetV5B = [];

for k = 1:numel(modelVariableNames)

    candidate = S.(modelVariableNames{k});

    if isa(candidate,'DAGNetwork') || isa(candidate,'SeriesNetwork')

        trainedNetV5B = candidate;

        fprintf('Found network variable: %s\n', ...
            modelVariableNames{k});

        break;
    end

end

assert(~isempty(trainedNetV5B), ...
    'No DAGNetwork or SeriesNetwork found in model MAT file.');

fprintf('Network type : %s\n', class(trainedNetV5B));
fprintf('Input size   : [%d %d %d]\n', ...
    trainedNetV5B.Layers(1).InputSize);


%% =============================================================
% DETERMINE INPUT SIZE
% =============================================================

inputSize = [];

for k = 1:numel(trainedNetV5B.Layers)

    if isa(trainedNetV5B.Layers(k), ...
            'nnet.cnn.layer.ImageInputLayer')

        inputSize = trainedNetV5B.Layers(k).InputSize;
        break;

    end

end

assert(~isempty(inputSize), ...
    'Could not determine network input size.');

targetSize = inputSize(1:2);

fprintf('Evaluation size: [%d %d]\n', ...
    targetSize(1),targetSize(2));


%% =============================================================
% DETERMINE MODEL CLASS ORDER
% =============================================================

fprintf('\nMODEL CLASS ORDER\n');
fprintf('------------------------------------------------------------\n');

classNames = categorical.empty;

for k = 1:numel(trainedNetV5B.Layers)

    if isa(trainedNetV5B.Layers(k), ...
            'nnet.cnn.layer.ClassificationOutputLayer')

        classNames = trainedNetV5B.Layers(k).Classes;
        break;

    end

end

if isempty(classNames)

    error(['Could not read class names from the ', ...
           'classification output layer.']);

end

classNames = categorical(classNames);

for k = 1:numel(classNames)

    fprintf('%d -> %s\n', ...
        k-1, char(string(classNames(k))));

end


%% =============================================================
% LOAD FINAL TEST CSV
% =============================================================

fprintf('\nLOADING FINAL TEST SET\n');
fprintf('------------------------------------------------------------\n');

Ttest = readtable(testCSV, ...
    'VariableNamingRule','preserve');

fprintf('CSV rows: %d\n', height(Ttest));

fprintf('\nAvailable CSV variables:\n');

disp(Ttest.Properties.VariableNames');


%% =============================================================
% FIND DIAGNOSIS COLUMN
% =============================================================

variableNames = Ttest.Properties.VariableNames;

diagnosisVar = '';

possibleDiagnosisNames = { ...
    'diagnosis', ...
    'Diagnosis', ...
    'label', ...
    'Label', ...
    'grade', ...
    'Grade', ...
    'dr_grade', ...
    'DR_grade'};

for k = 1:numel(possibleDiagnosisNames)

    idx = find(strcmpi( ...
        variableNames, ...
        possibleDiagnosisNames{k}), ...
        1);

    if ~isempty(idx)

        diagnosisVar = variableNames{idx};
        break;

    end

end

assert(~isempty(diagnosisVar), ...
    'Could not automatically find the diagnosis column.');

fprintf('Diagnosis column found: %s\n', diagnosisVar);


%% =============================================================
% FIND IMAGE IDENTIFIER COLUMN
% =============================================================

filenameVar = '';

possibleFilenameNames = { ...
    'id_code', ...
    'image', ...
    'image_id', ...
    'filename', ...
    'file_name', ...
    'path', ...
    'filepath', ...
    'Image', ...
    'Filename'};

for k = 1:numel(possibleFilenameNames)

    idx = find(strcmpi( ...
        variableNames, ...
        possibleFilenameNames{k}), ...
        1);

    if ~isempty(idx)

        filenameVar = variableNames{idx};
        break;

    end

end

if isempty(filenameVar)

    % Use the first column that is not the diagnosis column
    for k = 1:numel(variableNames)

        if ~strcmp(variableNames{k},diagnosisVar)

            filenameVar = variableNames{k};
            break;

        end

    end

end

assert(~isempty(filenameVar), ...
    'Could not determine the image filename column.');

fprintf('Image ID column found: %s\n', filenameVar);


%% =============================================================
% CONVERT LABELS TO MODEL CLASS NAMES
% =============================================================

fprintf('\nPREPARING TEST LABELS\n');
fprintf('------------------------------------------------------------\n');

rawDiagnosis = Ttest.(diagnosisVar);

if isnumeric(rawDiagnosis)

    diagnosisNumeric = double(rawDiagnosis);

elseif iscategorical(rawDiagnosis)

    diagnosisNumeric = str2double(string(rawDiagnosis));

elseif isstring(rawDiagnosis) || iscellstr(rawDiagnosis)

    diagnosisNumeric = str2double(string(rawDiagnosis));

else

    diagnosisNumeric = double(rawDiagnosis);

end

assert(all(~isnan(diagnosisNumeric)), ...
    'Diagnosis column could not be converted to numeric grades.');

trueLabelsText = strings(height(Ttest),1);

for k = 1:height(Ttest)

    gradeValue = diagnosisNumeric(k);

    % Original TRUST-DR model mapping
    switch gradeValue

        case 0
            trueLabelsText(k) = "Mild";

        case 1
            trueLabelsText(k) = "Moderate";

        case 2
            trueLabelsText(k) = "No_DR";

        case 3
            trueLabelsText(k) = "Proliferative";

        case 4
            trueLabelsText(k) = "Severe";

        otherwise

            error( ...
                'Unexpected diagnosis value %g at row %d.', ...
                gradeValue,k);

    end

end

YTrue = categorical( ...
    trueLabelsText, ...
    string(classNames));

fprintf('Total test samples: %d\n', numel(YTrue));


%% =============================================================
% PRINT TEST DISTRIBUTION
% =============================================================

fprintf('\nFINAL TEST CLASS DISTRIBUTION\n');
fprintf('------------------------------------------------------------\n');

for k = 1:numel(classNames)

    currentClass = classNames(k);

    count = sum(YTrue == currentClass);

    percent = 100 * count / numel(YTrue);

    fprintf('%-16s : %4d  (%.2f%%)\n', ...
        char(string(currentClass)), ...
        count, ...
        percent);

end


%% =============================================================
% RECURSIVELY FIND IMAGE FILES
% =============================================================

fprintf('\nMATCHING TEST IMAGES\n');
fprintf('------------------------------------------------------------\n');

fprintf('Searching image folders. This may take a moment...\n');

imageFiles = [ ...
    dir(fullfile(imageRoot,'**','*.jpg')); ...
    dir(fullfile(imageRoot,'**','*.jpeg')); ...
    dir(fullfile(imageRoot,'**','*.png')); ...
    dir(fullfile(imageRoot,'**','*.JPG')); ...
    dir(fullfile(imageRoot,'**','*.JPEG')); ...
    dir(fullfile(imageRoot,'**','*.PNG'))];

assert(~isempty(imageFiles), ...
    'No image files found under:\n%s', imageRoot);

fprintf('Images discovered: %d\n', numel(imageFiles));

allBaseNames = strings(numel(imageFiles),1);
allFullPaths = strings(numel(imageFiles),1);

for k = 1:numel(imageFiles)

    [~,baseName,~] = fileparts(imageFiles(k).name);

    allBaseNames(k) = string(baseName);

    allFullPaths(k) = string( ...
        fullfile( ...
            imageFiles(k).folder, ...
            imageFiles(k).name));

end


%% =============================================================
% MATCH CSV IDS TO IMAGE FILES
% =============================================================

rawImageIDs = Ttest.(filenameVar);

imageIDs = string(rawImageIDs);

matchedPaths = strings(height(Ttest),1);

matchedMask = false(height(Ttest),1);

for k = 1:height(Ttest)

    currentID = imageIDs(k);

    % Remove extension if one exists
    [~,baseID,~] = fileparts(char(currentID));

    if isempty(baseID)
        baseID = char(currentID);
    end

    matchIndex = find( ...
        strcmpi(allBaseNames,string(baseID)), ...
        1);

    if isempty(matchIndex)

        % Try direct comparison as fallback
        matchIndex = find( ...
            contains(lower(allFullPaths), ...
            lower(string(currentID))), ...
            1);

    end

    if ~isempty(matchIndex)

        matchedPaths(k) = allFullPaths(matchIndex);

        matchedMask(k) = true;

    end

end

fprintf('Images matched: %d / %d\n', ...
    sum(matchedMask), ...
    height(Ttest));

assert(sum(matchedMask) > 0, ...
    'No test images were matched.');

% Keep only successfully matched images
Ttest = Ttest(matchedMask,:);
YTrue = YTrue(matchedMask);
matchedPaths = matchedPaths(matchedMask);

fprintf('Usable test images: %d\n', numel(YTrue));


%% =============================================================
% CREATE IMAGE DATASTORE
% =============================================================

fprintf('\nCREATING TEST DATASTORE\n');
fprintf('------------------------------------------------------------\n');

imdsTest = imageDatastore( ...
    cellstr(matchedPaths), ...
    'Labels',YTrue, ...
    'ReadFcn',@readFundusGreenCLAHE);

augTest = augmentedImageDatastore( ...
    targetSize, ...
    imdsTest, ...
    'ColorPreprocessing','gray2rgb');


%% =============================================================
% RUN V5-B PREDICTIONS
% =============================================================

fprintf('\nRUNNING FINAL TEST PREDICTIONS\n');
fprintf('------------------------------------------------------------\n');

miniBatchSize = 16;

YPred = classify( ...
    trainedNetV5B, ...
    augTest, ...
    'MiniBatchSize',miniBatchSize, ...
    'ExecutionEnvironment','cpu');

fprintf('Predictions completed: %d\n', numel(YPred));


%% =============================================================
% CONFUSION MATRIX
% =============================================================

fprintf('\n============================================================\n');
fprintf('FINAL TEST RESULTS - V5-B\n');
fprintf('============================================================\n');

order = classNames;

C = confusionmat( ...
    YTrue, ...
    YPred, ...
    'Order',order);


%% =============================================================
% OVERALL ACCURACY
% =============================================================

overallAccuracy = ...
    100 * sum(diag(C)) / sum(C(:));


%% =============================================================
% PER-CLASS METRICS
% =============================================================

numClasses = numel(order);

recall = zeros(numClasses,1);
precision = zeros(numClasses,1);
f1Score = zeros(numClasses,1);
specificity = zeros(numClasses,1);

for k = 1:numClasses

    TP = C(k,k);

    FN = sum(C(k,:)) - TP;

    FP = sum(C(:,k)) - TP;

    TN = sum(C(:)) - TP - FN - FP;

    if TP + FN > 0
        recall(k) = TP / (TP + FN);
    else
        recall(k) = 0;
    end

    if TP + FP > 0
        precision(k) = TP / (TP + FP);
    else
        precision(k) = 0;
    end

    if precision(k) + recall(k) > 0

        f1Score(k) = ...
            2 * precision(k) * recall(k) / ...
            (precision(k) + recall(k));

    else
        f1Score(k) = 0;
    end

    if TN + FP > 0
        specificity(k) = TN / (TN + FP);
    else
        specificity(k) = 0;
    end

end


%% =============================================================
% BALANCED ACCURACY
% =============================================================

balancedAccuracy = 100 * mean(recall);


%% =============================================================
% MACRO METRICS
% =============================================================

macroPrecision = 100 * mean(precision);
macroRecall = 100 * mean(recall);
macroF1 = 100 * mean(f1Score);


%% =============================================================
% PRINT CONFUSION MATRIX
% =============================================================

fprintf('\nCONFUSION MATRIX\n');
fprintf('------------------------------------------------------------\n');

disp(array2table(C, ...
    'VariableNames',cellstr(string(order)), ...
    'RowNames',cellstr(string(order))));


%% =============================================================
% PRINT PER-CLASS RESULTS
% =============================================================

fprintf('\nPER-CLASS METRICS\n');
fprintf('------------------------------------------------------------\n');

for k = 1:numClasses

    classSamples = sum(C(k,:));

    fprintf( ...
        '%-16s | Samples: %3d | Recall: %6.2f%% | Precision: %6.2f%% | F1: %6.2f%%\n', ...
        char(string(order(k))), ...
        classSamples, ...
        100*recall(k), ...
        100*precision(k), ...
        100*f1Score(k));

end


%% =============================================================
% PRINT FINAL SUMMARY
% =============================================================

fprintf('\n============================================================\n');
fprintf('V5-B FINAL PERFORMANCE SUMMARY\n');
fprintf('============================================================\n');

fprintf('Test Samples        : %d\n', numel(YTrue));

fprintf('Overall Accuracy    : %.2f%%\n', ...
    overallAccuracy);

fprintf('Balanced Accuracy   : %.2f%%\n', ...
    balancedAccuracy);

fprintf('Macro Precision     : %.2f%%\n', ...
    macroPrecision);

fprintf('Macro Recall        : %.2f%%\n', ...
    macroRecall);

fprintf('Macro F1 Score      : %.2f%%\n', ...
    macroF1);

fprintf('============================================================\n');


%% =============================================================
% CREATE RESULTS TABLE
% =============================================================

perClassTable = table( ...
    string(order), ...
    diag(C), ...
    sum(C,2), ...
    recall*100, ...
    precision*100, ...
    f1Score*100, ...
    specificity*100, ...
    'VariableNames',{ ...
        'Class', ...
        'TruePositive', ...
        'Samples', ...
        'Recall_Percent', ...
        'Precision_Percent', ...
        'F1_Percent', ...
        'Specificity_Percent'});


%% =============================================================
% SAVE RESULTS
% =============================================================

fprintf('\nSAVING RESULTS\n');
fprintf('------------------------------------------------------------\n');

% Per-class metrics
writetable( ...
    perClassTable, ...
    fullfile(resultsFolder, ...
    'V5B_PerClass_Metrics.csv'));

% Confusion matrix
confusionTable = array2table( ...
    C, ...
    'VariableNames',cellstr(string(order)), ...
    'RowNames',cellstr(string(order)));

writetable( ...
    confusionTable, ...
    fullfile(resultsFolder, ...
    'V5B_ConfusionMatrix.csv'), ...
    'WriteRowNames',true);


% Individual predictions
predictionTable = table( ...
    matchedPaths, ...
    string(YTrue), ...
    string(YPred), ...
    'VariableNames',{ ...
        'ImagePath', ...
        'Actual', ...
        'Predicted'});

writetable( ...
    predictionTable, ...
    fullfile(resultsFolder, ...
    'V5B_Test_Predictions.csv'));


% Summary
summaryTable = table( ...
    numel(YTrue), ...
    overallAccuracy, ...
    balancedAccuracy, ...
    macroPrecision, ...
    macroRecall, ...
    macroF1, ...
    'VariableNames',{ ...
        'TestSamples', ...
        'OverallAccuracy', ...
        'BalancedAccuracy', ...
        'MacroPrecision', ...
        'MacroRecall', ...
        'MacroF1'});

writetable( ...
    summaryTable, ...
    fullfile(resultsFolder, ...
    'V5B_Final_Summary.csv'));


% MAT results
save( ...
    fullfile(resultsFolder, ...
    'V5B_Final_Evaluation.mat'), ...
    'C', ...
    'YTrue', ...
    'YPred', ...
    'order', ...
    'overallAccuracy', ...
    'balancedAccuracy', ...
    'recall', ...
    'precision', ...
    'f1Score', ...
    'specificity', ...
    'macroPrecision', ...
    'macroRecall', ...
    'macroF1', ...
    '-v7.3');


%% =============================================================
% CREATE CONFUSION MATRIX FIGURE
% =============================================================

try

    fig = figure( ...
        'Name','TRUST-DR V5-B Final Test Confusion Matrix', ...
        'NumberTitle','off');

    confusionchart( ...
        YTrue, ...
        YPred, ...
        'Order',order);

    title(sprintf( ...
        'TRUST-DR V5-B Final Test\nAccuracy = %.2f%% | Balanced Accuracy = %.2f%%', ...
        overallAccuracy, ...
        balancedAccuracy));

    exportgraphics( ...
        fig, ...
        fullfile(resultsFolder, ...
        'V5B_ConfusionMatrix.png'), ...
        'Resolution',300);

catch ME

    warning( ...
        'Could not create confusion matrix image: %s', ...
        ME.message);

end


fprintf('Results folder:\n%s\n', resultsFolder);

fprintf('\n============================================================\n');
fprintf('V5-B FINAL TEST EVALUATION COMPLETE\n');
fprintf('============================================================\n');


%% =============================================================
% LOCAL FUNCTION
% GREEN-CHANNEL CLAHE PREPROCESSING
% NO rgb2lab()
% NO lab2rgb()
% =============================================================

function I = readFundusGreenCLAHE(filename)

    % Read image
    I = imread(filename);

    % Convert grayscale to RGB
    if ndims(I) == 2
        I = repmat(I,1,1,3);
    end

    % Remove alpha channel if present
    if size(I,3) > 3
        I = I(:,:,1:3);
    end

    % Convert to uint8
    if ~isa(I,'uint8')
        I = im2uint8(I);
    end

    % Extract green channel
    greenChannel = I(:,:,2);

    % CLAHE
    greenEnhanced = adapthisteq( ...
        greenChannel, ...
        'NumTiles',[8 8], ...
        'ClipLimit',0.01, ...
        'NBins',256);

    % Blend enhanced green channel
    I(:,:,2) = greenEnhanced;

end