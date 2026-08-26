clc;
clear;
close all;

%% ============================================================
% TRUST-DR - FUNDUS IMAGE QUALITY ASSESSMENT
%
% Checks:
% 1. Focus / Blur
% 2. Brightness / Illumination
% 3. Contrast
%
% Output:
% GOOD       -> Continue to DR analysis
% BORDERLINE -> Apply CLAHE enhancement
% UNGRADABLE -> Ask for image recapture
%
% NOTE:
% Thresholds are empirically tuned for this MVP dataset.
%% ============================================================


%% ============================================================
% 1. SELECT TEST IMAGE
%% ============================================================

imageFolder = ...
    'C:\Users\sijar\Downloads\TRUST_DR\organized_dataset\Moderate';

imageFiles = dir(fullfile(imageFolder, '*.png'));

if isempty(imageFiles)
    error('No PNG images found in the selected folder.');
end

% Select first image
selectedImage = fullfile( ...
    imageFiles(1).folder, ...
    imageFiles(1).name);

fprintf('\nSelected Image:\n%s\n', selectedImage);


%% ============================================================
% 2. READ ORIGINAL IMAGE
%% ============================================================

img = imread(selectedImage);

figure('Name','Original Fundus Image');
imshow(img);
title('Original Fundus Image');


%% ============================================================
% 3. CONVERT TO GRAYSCALE
%% ============================================================

if size(img,3) == 3

    grayImg = rgb2gray(img);

else

    grayImg = img;

end

grayImg = im2double(grayImg);


%% ============================================================
% 4. FOCUS / BLUR ASSESSMENT
%
% Variance of Laplacian
%
% Higher score = sharper image
% Lower score  = blurrier image
%% ============================================================

laplacianFilter = fspecial('laplacian', 0.2);

laplacianImage = imfilter( ...
    grayImg, ...
    laplacianFilter, ...
    'replicate');

focusScore = var(laplacianImage(:));


%% ============================================================
% 5. BRIGHTNESS / ILLUMINATION ASSESSMENT
%
% Mean intensity:
% Low  -> dark image
% High -> overexposed image
%% ============================================================

brightnessScore = mean(grayImg(:));


%% ============================================================
% 6. CONTRAST ASSESSMENT
%
% Standard deviation of pixel intensity
%
% Low contrast -> washed-out / poor visibility
%% ============================================================

contrastScore = std(grayImg(:));


%% ============================================================
% 7. DISPLAY QUALITY METRICS
%% ============================================================

fprintf('\n====================================\n');
fprintf('FUNDUS IMAGE QUALITY METRICS\n');
fprintf('====================================\n');

fprintf('Focus Score      : %.6f\n', focusScore);
fprintf('Brightness Score : %.4f\n', brightnessScore);
fprintf('Contrast Score   : %.4f\n', contrastScore);


%% ============================================================
% 8. QUALITY THRESHOLDS
%
% Empirically tuned for TRUST-DR MVP.
% These are not clinically validated thresholds.
%% ============================================================

% ----------------------------
% Focus thresholds
% ----------------------------

GOOD_FOCUS = 0.00005;

BAD_FOCUS = 0.00001;


% ----------------------------
% Brightness thresholds
% ----------------------------

MIN_BRIGHTNESS = 0.12;

MAX_BRIGHTNESS = 0.85;

SEVERE_DARK = 0.05;

SEVERE_BRIGHT = 0.95;


% ----------------------------
% Contrast threshold
% ----------------------------

MIN_CONTRAST = 0.06;


%% ============================================================
% 9. QUALITY DECISION LOGIC
%% ============================================================


% ----------------------------
% UNGRADABLE IMAGE
% ----------------------------

if focusScore < BAD_FOCUS

    qualityResult = "UNGRADABLE";

    action = ...
        "Image is severely blurred. Please recapture.";

elseif brightnessScore < SEVERE_DARK

    qualityResult = "UNGRADABLE";

    action = ...
        "Image is extremely dark. Please recapture with better illumination.";

elseif brightnessScore > SEVERE_BRIGHT

    qualityResult = "UNGRADABLE";

    action = ...
        "Image is severely overexposed. Please recapture.";


% ----------------------------
% BORDERLINE IMAGE
% ----------------------------

elseif focusScore < GOOD_FOCUS || ...
       brightnessScore < MIN_BRIGHTNESS || ...
       brightnessScore > MAX_BRIGHTNESS || ...
       contrastScore < MIN_CONTRAST

    qualityResult = "BORDERLINE";

    action = ...
        "Image quality is borderline. Adaptive enhancement will be applied before AI analysis.";


% ----------------------------
% GOOD IMAGE
% ----------------------------

else

    qualityResult = "GOOD";

    action = ...
        "Image accepted for DR analysis.";

end


%% ============================================================
% 10. DISPLAY FINAL QUALITY RESULT
%% ============================================================

fprintf('\n====================================\n');

fprintf('QUALITY RESULT: %s\n', qualityResult);

fprintf('ACTION: %s\n', action);

fprintf('====================================\n');


%% ============================================================
% 11. IMAGE PROCESSING BASED ON RESULT
%% ============================================================

if qualityResult == "BORDERLINE"

    fprintf('\nApplying CLAHE enhancement...\n');

    enhancedImg = applyCLAHE(img);

    figure('Name','TRUST-DR Borderline Image Enhancement');

    subplot(1,2,1);

    imshow(img);

    title('Original Borderline Image');


    subplot(1,2,2);

    imshow(enhancedImg);

    title('CLAHE Enhanced Image');


    fprintf('CLAHE enhancement completed.\n');


elseif qualityResult == "UNGRADABLE"

    figure('Name','TRUST-DR Recapture Required');

    imshow(img);

    title({ ...
        'UNGRADABLE FUNDUS IMAGE', ...
        'Please Recapture Image'});


else

    figure('Name','TRUST-DR Image Accepted');

    imshow(img);

    title({ ...
        'GOOD FUNDUS IMAGE', ...
        'Accepted for DR Analysis'});

end


%% ============================================================
% 12. SAVE QUALITY REPORT
%% ============================================================

resultsFolder = ...
    'C:\Users\sijar\Downloads\TRUST_DR\results';

if ~exist(resultsFolder, 'dir')

    mkdir(resultsFolder);

end


reportFile = fullfile( ...
    resultsFolder, ...
    'quality_report.txt');


fid = fopen(reportFile, 'w');


fprintf(fid, ...
    'TRUST-DR FUNDUS IMAGE QUALITY REPORT\n');

fprintf(fid, ...
    '====================================\n\n');


fprintf(fid, ...
    'Image Path:\n%s\n\n', ...
    selectedImage);


fprintf(fid, ...
    'Focus Score: %.6f\n', ...
    focusScore);

fprintf(fid, ...
    'Brightness Score: %.4f\n', ...
    brightnessScore);

fprintf(fid, ...
    'Contrast Score: %.4f\n\n', ...
    contrastScore);


fprintf(fid, ...
    'QUALITY RESULT: %s\n', ...
    qualityResult);

fprintf(fid, ...
    'ACTION: %s\n', ...
    action);


fclose(fid);


fprintf('\nQUALITY REPORT SAVED:\n');

fprintf('%s\n', reportFile);


%% ============================================================
% HELPER FUNCTION
% CLAHE CONTRAST ENHANCEMENT
%% ============================================================

function imgOut = applyCLAHE(img)

    if size(img,3) == 3

        % Convert RGB image to Lab color space
        labImg = rgb2lab(img);

        % Extract lightness channel
        L = labImg(:,:,1) / 100;

        % Apply adaptive histogram equalization
        L = adapthisteq(L);

        % Put enhanced channel back
        labImg(:,:,1) = L * 100;

        % Convert back to RGB
        imgOut = lab2rgb(labImg);

        % Convert to uint8
        imgOut = im2uint8(imgOut);

    else

        % For grayscale images
        imgOut = adapthisteq(img);

    end

end