%% ============================================================
% TRUST-DR - MESSIDOR-2 EXTERNAL VALIDATION
% ============================================================

clear;
clc;
close all;

fprintf('\n============================================================\n');
fprintf('TRUST-DR MESSIDOR-2 EXTERNAL VALIDATION\n');
fprintf('============================================================\n\n');

%% ------------------------------------------------------------
% 1. PATHS
% ------------------------------------------------------------

projectFolder = 'C:\Users\sijar\Downloads\TRUST_DR';

modelPath = fullfile(projectFolder, ...
    'models', ...
    'trained_DR_model_v3.mat');

imageFolder = fullfile(projectFolder, ...
    'datasets', ...
    'archive', ...
    'messidor-2', ...
    'messidor-2', ...
    'preprocess');

labelsPath = fullfile(projectFolder, ...
    'datasets', ...
    'archive', ...
    'messidor_data.csv');

resultsFolder = fullfile(projectFolder, ...
    'results', ...
    'messidor_validation');

if ~isfolder(resultsFolder)
    mkdir(resultsFolder);
end

%% ------------------------------------------------------------
% 2. LOAD DR MODEL
% ------------------------------------------------------------

fprintf('LOADING DR MODEL...\n');

D = load(modelPath);

% Your model file may store the network under a different name.
names = fieldnames(D);

drNet = [];

for i = 1:numel(names)

    candidate = D.(names{i});

    if isa(candidate,'DAGNetwork') || ...
            isa(candidate,'SeriesNetwork') || ...
            isa(candidate,'dlnetwork')

        drNet = candidate;
        break;

    end

end

if isempty(drNet)
    error('No neural network found in DR model file.');
end

fprintf('Model loaded: %s\n',class(drNet));

%% ------------------------------------------------------------
% 3. GET MODEL INPUT SIZE
% ------------------------------------------------------------

drInputSize = [224 224 3];

try

    if isa(drNet,'DAGNetwork') || isa(drNet,'SeriesNetwork')

        drInputSize = drNet.Layers(1).InputSize;

    elseif isa(drNet,'dlnetwork')

        layers = drNet.Layers;

        for i = 1:numel(layers)

            if isa(layers(i), ...
                    'nnet.cnn.layer.ImageInputLayer')

                drInputSize = layers(i).InputSize;
                break;

            end

        end

    end

catch
end

fprintf('Model input size: [%d %d %d]\n\n', ...
    drInputSize(1), ...
    drInputSize(2), ...
    drInputSize(3));

%% ------------------------------------------------------------
% 4. LOAD MESSIDOR LABELS
% ------------------------------------------------------------

fprintf('LOADING MESSIDOR LABELS...\n');

T = readtable(labelsPath);

% Keep only gradable images
T = T(T.adjudicated_gradable == 1,:);

fprintf('Gradable images in CSV: %d\n',height(T));

%% ------------------------------------------------------------
% 5. MATCH IMAGE FILES WITH LABELS
% ------------------------------------------------------------

validRows = false(height(T),1);

for i = 1:height(T)

    imagePath = fullfile(imageFolder,T.id_code{i});

    validRows(i) = isfile(imagePath);

end

T = T(validRows,:);

fprintf('Images successfully matched: %d\n\n',height(T));

if height(T) == 0
    error('No MESSIDOR images matched the CSV labels.');
end

%% ------------------------------------------------------------
% 6. CLASS NAME ORDER
% ------------------------------------------------------------

% APTOS / TRUST-DR model class mapping
modelClasses = ...
    ["Mild","Moderate","No_DR","Proliferative","Severe"];

fprintf('MODEL CLASS ORDER:\n');

for k = 1:numel(modelClasses)
    fprintf('%d -> %s\n',k-1,modelClasses(k));
end

fprintf('\n');

%% ------------------------------------------------------------
% 7. MESSIDOR GROUND-TRUTH LABEL MAPPING
% ------------------------------------------------------------

% MESSIDOR diagnosis mapping:
% 0 = No_DR
% 1 = Mild
% 2 = Moderate
% 3 = Severe
% 4 = Proliferative

trueDiagnosis = T.diagnosis;

trueLabels = strings(height(T),1);

for i = 1:height(T)

    switch trueDiagnosis(i)

        case 0
            trueLabels(i) = "No_DR";

        case 1
            trueLabels(i) = "Mild";

        case 2
            trueLabels(i) = "Moderate";

        case 3
            trueLabels(i) = "Severe";

        case 4
            trueLabels(i) = "Proliferative";

        otherwise
            trueLabels(i) = "Unknown";

    end

end

%% ------------------------------------------------------------
% 8. RUN VALIDATION
% ------------------------------------------------------------

numImages = height(T);

predictedLabels = strings(numImages,1);
confidence = zeros(numImages,1);

fprintf('RUNNING MESSIDOR VALIDATION...\n');
fprintf('Total images: %d\n\n',numImages);

for i = 1:numImages

    imagePath = fullfile(imageFolder,T.id_code{i});

    I = imread(imagePath);

    % Ensure RGB
    if ndims(I) == 2
        I = repmat(I,[1 1 3]);
    end

    if size(I,3) > 3
        I = I(:,:,1:3);
    end

    I = im2uint8(I);

    % ------------------------------------------------------------
    % APPLY SAME CLAHE PREPROCESSING USED DURING TRAINING
    % ------------------------------------------------------------

    I = applyFundusCLAHE(I);

    % Resize for DR model
    inputImage = imresize(I,drInputSize(1:2));

    % Classification
    [~,scoresRaw] = classify(drNet,inputImage);

    scores = double(squeeze(scoresRaw));
    scores = scores(:);

    % Normalize if necessary
    scores(~isfinite(scores)) = 0;

    if any(scores < 0) || ...
            sum(scores) <= 0 || ...
            abs(sum(scores)-1) > 1e-3

        ex = exp(scores-max(scores));
        scores = ex./sum(ex);

    else

        scores = scores./sum(scores);

    end

    [confidence(i),idx] = max(scores);

    predictedLabels(i) = modelClasses(idx);

    if mod(i,50) == 0 || i == numImages

        fprintf('Processed %d / %d images\n', ...
            i,numImages);

    end

end

%% ------------------------------------------------------------
% 9. CONVERT MODEL PREDICTIONS TO MESSIDOR SCALE
% ------------------------------------------------------------

predDiagnosis = nan(numImages,1);

for i = 1:numImages

    switch predictedLabels(i)

        case "No_DR"
            predDiagnosis(i) = 0;

        case "Mild"
            predDiagnosis(i) = 1;

        case "Moderate"
            predDiagnosis(i) = 2;

        case "Severe"
            predDiagnosis(i) = 3;

        case "Proliferative"
            predDiagnosis(i) = 4;

    end

end

%% ------------------------------------------------------------
% 10. ACCURACY
% ------------------------------------------------------------

validPrediction = ~isnan(predDiagnosis);

accuracy = ...
    mean(predDiagnosis(validPrediction) == ...
         trueDiagnosis(validPrediction));

fprintf('\n============================================================\n');
fprintf('MESSIDOR-2 VALIDATION RESULTS\n');
fprintf('============================================================\n');

fprintf('Total images tested : %d\n',numImages);
fprintf('Valid predictions   : %d\n',sum(validPrediction));
fprintf('Accuracy            : %.2f%%\n',100*accuracy);

%% ------------------------------------------------------------
% 11. CONFUSION MATRIX
% ------------------------------------------------------------

figure('Color','w');

confusionchart( ...
    categorical(trueDiagnosis), ...
    categorical(predDiagnosis), ...
    'Title','TRUST-DR External Validation on MESSIDOR-2');

confusionPath = fullfile(resultsFolder, ...
    'messidor_confusion_matrix.png');

exportgraphics(gcf,confusionPath,'Resolution',200);

%% ------------------------------------------------------------
% 12. SAVE RESULTS
% ------------------------------------------------------------

resultsTable = table( ...
    string(T.id_code), ...
    trueDiagnosis, ...
    trueLabels, ...
    predictedLabels, ...
    predDiagnosis, ...
    confidence, ...
    'VariableNames', ...
    {'Image','TrueDiagnosis','TrueLabel', ...
     'PredictedLabel','PredictedDiagnosis','Confidence'});

csvPath = fullfile(resultsFolder, ...
    'messidor_validation_results.csv');

writetable(resultsTable,csvPath);

matPath = fullfile(resultsFolder, ...
    'messidor_validation_results.mat');

save(matPath, ...
    'resultsTable', ...
    'accuracy', ...
    'trueDiagnosis', ...
    'predDiagnosis', ...
    'predictedLabels', ...
    'confidence');

fprintf('\nResults saved to:\n%s\n',resultsFolder);
fprintf('============================================================\n');
fprintf('MESSIDOR VALIDATION COMPLETE\n');
fprintf('============================================================\n');

%% ============================================================
% LOCAL FUNCTION: FUNDUS CLAHE
%% ============================================================

function out = applyFundusCLAHE(img)

img = im2uint8(img);

% Convert RGB to CIE L*a*b*
lab = rgb2lab(img);

% Extract luminance channel
L = lab(:,:,1) / 100;

% Apply CLAHE
L = adapthisteq( ...
    L, ...
    'NumTiles',[8 8], ...
    'ClipLimit',0.01);

% Restore luminance
lab(:,:,1) = 100 * L;

% Convert back to RGB
out = lab2rgb(lab);

% Convert safely to uint8
out = im2uint8(min(max(out,0),1));

end