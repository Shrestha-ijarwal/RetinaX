clc;
clear;
close all;

%% ============================================================
% TRUST-DR
% Explainable AI for Diabetic Retinopathy Screening
%
% COMPLETE MVP PIPELINE
% WITH TRANSPARENT GRAD-CAM OVERLAY
%% ============================================================

fprintf('\n');
fprintf('===========================================\n');
fprintf('        TRUST-DR SCREENING SYSTEM\n');
fprintf('===========================================\n\n');


%% ============================================================
% 1. LOAD TRAINED MODEL
%% ============================================================

fprintf('Loading DR classification model...\n\n');

modelPath = ...
    'C:\Users\sijar\Downloads\TRUST_DR\models\trained_DR_model_v2.mat';

if ~isfile(modelPath)

    error(['Model not found at:\n' modelPath]);

end

load(modelPath, 'trainedNet');

fprintf('Model loaded successfully!\n\n');


%% ============================================================
% 2. SELECT FUNDUS IMAGE
%% ============================================================

fprintf('Select a fundus image...\n\n');

[fileName, filePath] = uigetfile( ...
    {'*.png;*.jpg;*.jpeg;*.tif', ...
    'Fundus Images (*.png, *.jpg, *.jpeg, *.tif)'}, ...
    'Select Patient Fundus Image');

if isequal(fileName, 0)

    fprintf('No image selected. Program stopped.\n');
    return;

end

imagePath = fullfile(filePath, fileName);

fprintf('Selected Image:\n%s\n\n', imagePath);


%% ============================================================
% 3. READ IMAGE
%% ============================================================

originalImage = imread(imagePath);

% Ensure RGB image

if size(originalImage,3) == 1

    originalImage = repmat(originalImage, [1 1 3]);

end


%% ============================================================
% STEP 1: IMAGE QUALITY ASSESSMENT
%% ============================================================

fprintf('\n');
fprintf('====================================\n');
fprintf('STEP 1: IMAGE QUALITY ASSESSMENT\n');
fprintf('====================================\n');


% Convert to grayscale

grayImage = rgb2gray(originalImage);

grayImageDouble = im2double(grayImage);


%% Focus Score

laplacianFilter = fspecial('laplacian', 0.2);

lapImage = imfilter( ...
    grayImageDouble, ...
    laplacianFilter, ...
    'replicate');

focusScore = var(lapImage(:));


%% Brightness Score

brightnessScore = mean(grayImageDouble(:));


%% Contrast Score

contrastScore = std(grayImageDouble(:));


fprintf('Focus Score      : %.6f\n', focusScore);
fprintf('Brightness Score : %.4f\n', brightnessScore);
fprintf('Contrast Score   : %.4f\n\n', contrastScore);


%% ============================================================
% QUALITY DECISION
%% ============================================================

FOCUS_BAD = 0.00003;

BRIGHTNESS_LOW = 0.05;

BRIGHTNESS_HIGH = 0.90;

CONTRAST_BAD = 0.03;


if focusScore < FOCUS_BAD || ...
        brightnessScore < BRIGHTNESS_LOW || ...
        brightnessScore > BRIGHTNESS_HIGH || ...
        contrastScore < CONTRAST_BAD

    qualityResult = "UNGRADABLE";

    qualityAction = ...
        "Image quality insufficient. Please recapture the fundus image.";

    fprintf('QUALITY RESULT: UNGRADABLE\n');
    fprintf('ACTION: %s\n', qualityAction);


elseif focusScore < 0.00020 || ...
        brightnessScore < 0.10 || ...
        brightnessScore > 0.75 || ...
        contrastScore < 0.10

    qualityResult = "BORDERLINE";

    qualityAction = ...
        "Borderline quality. Applying CLAHE enhancement.";

    fprintf('QUALITY RESULT: BORDERLINE\n');
    fprintf('ACTION: %s\n', qualityAction);


else

    qualityResult = "GOOD";

    qualityAction = ...
        "Image accepted for DR analysis.";

    fprintf('QUALITY RESULT: GOOD\n');
    fprintf('ACTION: %s\n', qualityAction);

end


%% ============================================================
% STOP IF IMAGE IS UNGRADABLE
%% ============================================================

if qualityResult == "UNGRADABLE"

    figure( ...
        'Name', 'TRUST-DR Quality Result', ...
        'Color', [0.12 0.12 0.12]);

    imshow(originalImage);

    title( ...
        {'UNGRADABLE FUNDUS IMAGE', ...
        'Please Recapture Image'}, ...
        'Color', 'w', ...
        'FontSize', 18, ...
        'FontWeight', 'bold');

    fprintf('\n');
    fprintf('====================================\n');
    fprintf('SCREENING STOPPED\n');
    fprintf('Please capture a better quality image.\n');
    fprintf('====================================\n');

    return;

end


%% ============================================================
% STEP 2: IMAGE PREPROCESSING
%% ============================================================

fprintf('\n');
fprintf('====================================\n');
fprintf('STEP 2: IMAGE PREPROCESSING\n');
fprintf('====================================\n');


if qualityResult == "BORDERLINE"

    processedImage = applyCLAHE(originalImage);

else

    processedImage = originalImage;

end


%% Resize for ResNet-50

inputSize = trainedNet.Layers(1).InputSize;

networkImage = imresize( ...
    processedImage, ...
    inputSize(1:2));


%% ============================================================
% STEP 3: DR SEVERITY CLASSIFICATION
%% ============================================================

fprintf('\n');
fprintf('====================================\n');
fprintf('STEP 3: DR SEVERITY CLASSIFICATION\n');
fprintf('====================================\n');


[predictedLabel, scores] = classify( ...
    trainedNet, ...
    networkImage);


confidence = max(scores) * 100;

predictedLabel = string(predictedLabel);


fprintf('PREDICTION: %s\n', predictedLabel);
fprintf('CONFIDENCE: %.2f%%\n', confidence);


%% ============================================================
% STEP 4: REFERABLE DR DECISION
%% ============================================================

fprintf('\n');
fprintf('====================================\n');
fprintf('STEP 4: REFERABLE DR DECISION\n');
fprintf('====================================\n');


referableClasses = [ ...
    "Moderate", ...
    "Severe", ...
    "Proliferative"];


if ismember(predictedLabel, referableClasses)

    referralResult = "REFERABLE DR";

    referralAction = ...
        "Priority ophthalmologist evaluation recommended.";

    fprintf('RESULT: REFERABLE DR\n');
    fprintf('ACTION: %s\n', referralAction);

else

    referralResult = "NON-REFERABLE DR";

    referralAction = ...
        "Routine screening follow-up recommended.";

    fprintf('RESULT: NON-REFERABLE DR\n');
    fprintf('ACTION: %s\n', referralAction);

end


%% ============================================================
% STEP 5: HUMAN-IN-THE-LOOP DECISION
%% ============================================================

fprintf('\n');
fprintf('====================================\n');
fprintf('STEP 5: HUMAN-IN-THE-LOOP DECISION\n');
fprintf('====================================\n');


HIGH_CONFIDENCE = 85;

LOW_CONFIDENCE = 60;


if referralResult == "REFERABLE DR"

    reviewStatus = "PRIORITY HUMAN REVIEW";

    reviewAction = ...
        "Refer case to ophthalmologist for priority validation.";


elseif confidence < LOW_CONFIDENCE

    reviewStatus = "MANDATORY HUMAN REVIEW";

    reviewAction = ...
        "AI confidence is low. Ophthalmologist validation required.";


elseif qualityResult == "BORDERLINE"

    reviewStatus = "HUMAN REVIEW RECOMMENDED";

    reviewAction = ...
        "Borderline image quality detected. Please validate AI result.";


elseif confidence < HIGH_CONFIDENCE

    reviewStatus = "HUMAN REVIEW RECOMMENDED";

    reviewAction = ...
        "Moderate AI confidence. Human validation recommended.";


else

    reviewStatus = "AI SCREENING COMPLETE";

    reviewAction = ...
        "High-confidence AI screening result. Routine follow-up recommended.";

end


fprintf('REVIEW STATUS: %s\n', reviewStatus);
fprintf('ACTION: %s\n', reviewAction);


%% ============================================================
% STEP 6: GENERATE GRAD-CAM
%% ============================================================

fprintf('\n');
fprintf('====================================\n');
fprintf('STEP 6: GENERATING GRAD-CAM\n');
fprintf('====================================\n');


try

    %% Generate Grad-CAM

    gradcamMap = gradCAM( ...
        trainedNet, ...
        networkImage, ...
        predictedLabel);


    %% Normalize Grad-CAM

    gradcamMap = mat2gray(gradcamMap);


    %% Resize Grad-CAM to match network image

    gradcamMap = imresize( ...
        gradcamMap, ...
        [size(networkImage,1), ...
         size(networkImage,2)]);


    %% Create coloured heatmap

    heatmapImage = ind2rgb( ...
        gray2ind(gradcamMap, 256), ...
        jet(256));


    %% Convert to uint8

    heatmapImage = im2uint8(heatmapImage);


    %% ========================================================
    % TRANSPARENT GRAD-CAM OVERLAY
    %% ========================================================

    % Original image

    baseImage = im2double(networkImage);


    % Heatmap

    heatmapDouble = im2double(heatmapImage);


    % Transparency strength
    % Increase = stronger heatmap
    % Recommended range: 0.35 - 0.60

    alpha = 0.50;


    % Use Grad-CAM itself as an attention-dependent alpha mask

    attentionMask = gradcamMap;


    % Expand mask from HxW to HxWx3

    attentionMask3 = repmat( ...
        attentionMask, ...
        [1 1 3]);


    %% Blend original fundus with heatmap

    gradcamOverlayDouble = ...
        baseImage .* ...
        (1 - alpha .* attentionMask3) + ...
        heatmapDouble .* ...
        (alpha .* attentionMask3);


    %% Convert to uint8

    gradcamOverlay = im2uint8(gradcamOverlayDouble);


    %% Standalone heatmap

    gradcamHeatmap = heatmapImage;


    fprintf('Grad-CAM generated successfully!\n');


catch ME

    warning('Grad-CAM generation failed.');

    fprintf('Grad-CAM Error:\n%s\n', ME.message);

    gradcamMap = [];

    gradcamHeatmap = networkImage;

    gradcamOverlay = networkImage;

end


%% ============================================================
% FINAL RESULT
%% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('          FINAL TRUST-DR RESULT\n');
fprintf('============================================\n');

fprintf('Image Quality : %s\n', qualityResult);

fprintf('DR Grade      : %s\n', predictedLabel);

fprintf('Confidence    : %.2f%%\n', confidence);

fprintf('Screening     : %s\n', referralResult);

fprintf('Recommendation: %s\n', referralAction);

fprintf('Human Review  : %s\n', reviewStatus);

fprintf('Review Action : %s\n', reviewAction);

fprintf('============================================\n\n');


%% ============================================================
% STEP 7: CREATE VISUAL DASHBOARD
%% ============================================================

figure( ...
    'Name', 'TRUST-DR AI Screening Result', ...
    'Color', [0.12 0.12 0.12], ...
    'Position', [100 100 1500 850]);


%% ------------------------------------------------------------
% PANEL 1: ORIGINAL FUNDUS
%% ------------------------------------------------------------

subplot(2,3,1);

imshow(originalImage);

title( ...
    '1. ORIGINAL FUNDUS IMAGE', ...
    'Color', 'w', ...
    'FontSize', 15, ...
    'FontWeight', 'bold');


%% ------------------------------------------------------------
% PANEL 2: PROCESSED IMAGE
%% ------------------------------------------------------------

subplot(2,3,2);

imshow(processedImage);

title( ...
    sprintf('2. PREPROCESSING: %s', qualityResult), ...
    'Color', 'w', ...
    'FontSize', 15, ...
    'FontWeight', 'bold');


%% ------------------------------------------------------------
% PANEL 3: CLASSIFICATION
%% ------------------------------------------------------------

subplot(2,3,3);

imshow(networkImage);

classificationTitle = sprintf( ...
    '3. DR GRADE: %s\nConfidence: %.2f%%\n%s', ...
    predictedLabel, ...
    confidence, ...
    referralResult);

title( ...
    classificationTitle, ...
    'Color', 'w', ...
    'FontSize', 15, ...
    'FontWeight', 'bold');


%% ------------------------------------------------------------
% PANEL 4: STANDALONE GRAD-CAM
%% ------------------------------------------------------------

subplot(2,3,4);

imshow(gradcamHeatmap);

title( ...
    '4. GRAD-CAM ATTENTION MAP', ...
    'Color', 'w', ...
    'FontSize', 15, ...
    'FontWeight', 'bold');


%% ------------------------------------------------------------
% PANEL 5: TRANSPARENT OVERLAY
%% ------------------------------------------------------------

subplot(2,3,5);

imshow(gradcamOverlay);

title( ...
    '5. GRAD-CAM OVERLAY ON FUNDUS', ...
    'Color', 'w', ...
    'FontSize', 15, ...
    'FontWeight', 'bold');


%% ------------------------------------------------------------
% PANEL 6: HUMAN-IN-THE-LOOP STATUS
%% ------------------------------------------------------------

subplot(2,3,6);

axis off;


text(0.05, 0.90, ...
    'HUMAN-IN-THE-LOOP', ...
    'Color', 'w', ...
    'FontSize', 19, ...
    'FontWeight', 'bold');


text(0.05, 0.68, ...
    sprintf('Status:\n%s', reviewStatus), ...
    'Color', [0.4 1 0.5], ...
    'FontSize', 15, ...
    'FontWeight', 'bold');


text(0.05, 0.40, ...
    sprintf('Action:\n%s', reviewAction), ...
    'Color', 'w', ...
    'FontSize', 13);


text(0.05, 0.12, ...
    'AI assists screening. Final validation remains with the clinician.', ...
    'Color', [1 0.8 0.3], ...
    'FontSize', 11);


sgtitle( ...
    'TRUST-DR: Explainable AI Screening Pipeline', ...
    'Color', 'w', ...
    'FontSize', 20, ...
    'FontWeight', 'bold');


%% ============================================================
% STEP 8: SAVE RESULTS
%% ============================================================

resultsFolder = ...
    'C:\Users\sijar\Downloads\TRUST_DR\results';


if ~exist(resultsFolder, 'dir')

    mkdir(resultsFolder);

end


timestamp = datestr(now, 'yyyymmdd_HHMMSS');


%% Save dashboard

dashboardPath = fullfile( ...
    resultsFolder, ...
    ['TRUST_DR_dashboard_' timestamp '.png']);


exportgraphics( ...
    gcf, ...
    dashboardPath, ...
    'Resolution', 200);


%% Save Grad-CAM overlay

gradcamOverlayPath = fullfile( ...
    resultsFolder, ...
    ['TRUST_DR_gradcam_overlay_' timestamp '.png']);


imwrite(gradcamOverlay, gradcamOverlayPath);


%% Save standalone heatmap

gradcamHeatmapPath = fullfile( ...
    resultsFolder, ...
    ['TRUST_DR_gradcam_heatmap_' timestamp '.png']);


imwrite(gradcamHeatmap, gradcamHeatmapPath);


%% Save text report

reportPath = fullfile( ...
    resultsFolder, ...
    ['TRUST_DR_report_' timestamp '.txt']);


fid = fopen(reportPath, 'w');


fprintf(fid, ...
    '===========================================\n');

fprintf(fid, ...
    '        TRUST-DR SCREENING REPORT\n');

fprintf(fid, ...
    '===========================================\n\n');


fprintf(fid, ...
    'Image File: %s\n\n', imagePath);


fprintf(fid, ...
    'IMAGE QUALITY ASSESSMENT\n');

fprintf(fid, ...
    'Focus Score      : %.6f\n', focusScore);

fprintf(fid, ...
    'Brightness Score : %.4f\n', brightnessScore);

fprintf(fid, ...
    'Contrast Score   : %.4f\n', contrastScore);

fprintf(fid, ...
    'Quality Result   : %s\n', qualityResult);

fprintf(fid, ...
    'Quality Action   : %s\n\n', qualityAction);


fprintf(fid, ...
    'DR SCREENING RESULT\n');

fprintf(fid, ...
    'DR Grade         : %s\n', predictedLabel);

fprintf(fid, ...
    'Confidence       : %.2f%%\n', confidence);

fprintf(fid, ...
    'Screening Result : %s\n', referralResult);

fprintf(fid, ...
    'Recommendation   : %s\n\n', referralAction);


fprintf(fid, ...
    'HUMAN-IN-THE-LOOP REVIEW\n');

fprintf(fid, ...
    'Review Status    : %s\n', reviewStatus);

fprintf(fid, ...
    'Review Action    : %s\n\n', reviewAction);


fprintf(fid, ...
    'EXPLAINABILITY\n');

fprintf(fid, ...
    'Grad-CAM attention map generated.\n');

fprintf(fid, ...
    'Transparent Grad-CAM overlay generated.\n\n');


fprintf(fid, ...
    '===========================================\n');

fprintf(fid, ...
    'AI screening output requires clinical validation.\n');

fprintf(fid, ...
    '===========================================\n');


fclose(fid);


fprintf('Results saved successfully!\n\n');

fprintf('Dashboard:\n%s\n\n', dashboardPath);

fprintf('Grad-CAM Overlay:\n%s\n\n', gradcamOverlayPath);

fprintf('Grad-CAM Heatmap:\n%s\n\n', gradcamHeatmapPath);

fprintf('Report:\n%s\n\n', reportPath);


%% ============================================================
% HELPER FUNCTION: CLAHE
%% ============================================================

function outputImage = applyCLAHE(inputImage)

    if size(inputImage,3) == 1

        outputImage = adapthisteq(inputImage);

        return;

    end


    %% Convert RGB to Lab

    labImage = rgb2lab(inputImage);


    %% Extract luminance channel

    L = labImage(:,:,1);


    %% Normalize

    Lnormalized = L / 100;


    %% Apply CLAHE

    Lenhanced = adapthisteq(Lnormalized);


    %% Replace luminance

    labImage(:,:,1) = Lenhanced * 100;


    %% Convert back to RGB

    outputImage = lab2rgb(labImage);


    %% Convert to uint8

    outputImage = im2uint8(outputImage);

end