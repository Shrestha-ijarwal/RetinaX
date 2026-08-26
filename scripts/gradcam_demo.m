clc;
clear;
close all;

%% ============================================================
% TRUST-DR - GRAD-CAM EXPLAINABILITY
% Using trained DR Model V2
%% ============================================================

%% 1. LOAD TRAINED MODEL

modelPath = ...
    'C:\Users\sijar\Downloads\TRUST_DR\models\trained_DR_model_v2.mat';

load(modelPath, 'trainedNet');

disp('V2 MODEL LOADED SUCCESSFULLY');


%% 2. SELECT A FUNDUS IMAGE

% We will use an image from the Moderate folder for testing

imageFolder = ...
    'C:\Users\sijar\Downloads\TRUST_DR\organized_dataset\Moderate';

% Get image files
imageFiles = dir(fullfile(imageFolder, '*.png'));

% If images are JPG instead, uncomment these lines:
% imageFiles = dir(fullfile(imageFolder, '*.jpg'));

% Select first image
selectedImage = fullfile( ...
    imageFiles(1).folder, ...
    imageFiles(1).name);

fprintf('Selected image:\n%s\n', selectedImage);


%% 3. READ ORIGINAL IMAGE

img = imread(selectedImage);

figure;
imshow(img);
title('Original Fundus Image');


%% 4. APPLY SAME CLAHE PREPROCESSING

imgCLAHE = applyCLAHE(img);

figure;
imshow(imgCLAHE);
title('CLAHE Enhanced Image');


%% 5. RESIZE TO RESNET INPUT

inputSize = trainedNet.Layers(1).InputSize;

imgResized = imresize( ...
    imgCLAHE, ...
    inputSize(1:2));


%% 6. CLASSIFY IMAGE

[predictedLabel, scores] = classify( ...
    trainedNet, ...
    imgResized);

confidence = max(scores) * 100;

fprintf('\n====================================\n');

fprintf('PREDICTION: %s\n', ...
    string(predictedLabel));

fprintf('CONFIDENCE: %.2f%%\n', ...
    confidence);

fprintf('====================================\n');


%% 7. GRAD-CAM

% Last convolutional activation layer in ResNet-50

featureLayer = 'activation_49_relu';

scoreMap = gradCAM( ...
    trainedNet, ...
    imgResized, ...
    predictedLabel, ...
    'FeatureLayer', featureLayer);


%% 8. RESIZE AND NORMALIZE HEATMAP

scoreMap = imresize( ...
    scoreMap, ...
    inputSize(1:2));

scoreMap = mat2gray(scoreMap);


%% 9. CREATE HEATMAP

heatmapRGB = ind2rgb( ...
    gray2ind(scoreMap, 256), ...
    jet(256));


%% 10. CREATE OVERLAY

overlay = 0.60 * im2double(imgResized) + ...
          0.40 * heatmapRGB;

overlay = im2uint8(overlay);


%% 11. DISPLAY FINAL RESULT

figure('Name', 'TRUST-DR Grad-CAM Explainability');

subplot(1,3,1);

imshow(img);

title('Original Fundus Image');


subplot(1,3,2);

imshow(imgResized);

title(sprintf( ...
    'Prediction: %s\nConfidence: %.1f%%', ...
    string(predictedLabel), ...
    confidence));


subplot(1,3,3);

imshow(overlay);

title('Grad-CAM Attention Map');


%% 12. SAVE RESULT

outputFolder = ...
    'C:\Users\sijar\Downloads\TRUST_DR\results';

if ~exist(outputFolder, 'dir')

    mkdir(outputFolder);

end


outputFile = fullfile( ...
    outputFolder, ...
    'gradcam_result.png');

imwrite(overlay, outputFile);


fprintf('\nGRAD-CAM RESULT SAVED:\n');

fprintf('%s\n', outputFile);


%% ============================================================
% HELPER FUNCTION
%% ============================================================

function imgOut = applyCLAHE(img)

    if size(img,3) == 3

        labImg = rgb2lab(img);

        L = labImg(:,:,1) / 100;

        L = adapthisteq(L);

        labImg(:,:,1) = L * 100;

        imgOut = lab2rgb(labImg);

        imgOut = im2uint8(imgOut);

    else

        imgOut = adapthisteq(img);

    end

end