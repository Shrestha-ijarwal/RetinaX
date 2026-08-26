clc;
clear;
close all;

%% ============================================
%      TRUST-DR: DRIVE DATA PREPARATION
%      Retinal Blood Vessel Segmentation
% ============================================

fprintf('\n============================================\n');
fprintf('   TRUST-DR: DRIVE DATA PREPARATION\n');
fprintf('============================================\n\n');

%% 1. DEFINE DATASET PATHS

driveFolder = ...
    'C:\Users\sijar\Downloads\TRUST_DR\datasets\drive\DRIVE';

trainImageFolder = fullfile( ...
    driveFolder, 'training', 'images');

trainMaskFolder = fullfile( ...
    driveFolder, 'training', '1st_manual');

testImageFolder = fullfile( ...
    driveFolder, 'test', 'images');

testMaskFolder = fullfile( ...
    driveFolder, 'test', '1st_manual');

%% 2. CHECK TRAINING FOLDERS

if ~isfolder(trainImageFolder)
    error('Training images folder not found!');
end

if ~isfolder(trainMaskFolder)
    error('Training masks folder not found!');
end

fprintf('Training image folder found.\n');
fprintf('Training vessel mask folder found.\n\n');

%% 3. GET TRAINING IMAGE FILES

trainImageFiles = dir( ...
    fullfile(trainImageFolder, '*.tif'));

numTrainImages = length(trainImageFiles);

fprintf('Number of training images found: %d\n', ...
    numTrainImages);

%% 4. CREATE OUTPUT FOLDER

outputFolder = ...
    'C:\Users\sijar\Downloads\TRUST_DR\prepared_drive';

if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

preparedImageFolder = fullfile( ...
    outputFolder, 'images');

preparedMaskFolder = fullfile( ...
    outputFolder, 'masks');

if ~isfolder(preparedImageFolder)
    mkdir(preparedImageFolder);
end

if ~isfolder(preparedMaskFolder)
    mkdir(preparedMaskFolder);
end

%% 5. DEFINE IMAGE SIZE

% We will resize all images to a consistent size.
% 256 x 256 is manageable for initial segmentation experiments.

targetSize = [256 256];

fprintf('Target image size: %d x %d\n\n', ...
    targetSize(1), targetSize(2));

%% 6. PROCESS ALL TRAINING IMAGES

fprintf('Preparing DRIVE training data...\n\n');

for i = 1:numTrainImages

    %% Get image information

    imageName = trainImageFiles(i).name;

    imagePath = fullfile( ...
        trainImageFiles(i).folder, ...
        imageName);

    %% Load original fundus image

    img = imread(imagePath);

    %% Resize image

    imgResized = imresize(img, targetSize);

    %% Extract image ID
    %
    % Example:
    % 21_training.tif
    %
    % Extract:
    % 21

    imageID = extractBefore(imageName, '_');

    %% Find corresponding manual vessel mask

    maskName = imageID + "_manual1.gif";

    maskPath = fullfile( ...
        trainMaskFolder, ...
        maskName);

    if ~isfile(maskPath)

        warning( ...
            'Mask not found for image: %s', ...
            imageName);

        continue;

    end

    %% Load vessel mask

    mask = imread(maskPath);

    %% Convert mask to grayscale if needed

    if size(mask,3) == 3
        mask = rgb2gray(mask);
    end

    %% Convert to binary

    mask = mask > 0;

    %% Resize mask
    %
    % Use nearest-neighbor interpolation so vessel labels
    % remain binary.

    maskResized = imresize( ...
        mask, ...
        targetSize, ...
        'nearest');

    %% Convert resized mask to uint8

    maskResized = uint8(maskResized) * 255;

    %% Create output filenames

    outputImageName = imageID + ".png";

    outputMaskName = imageID + "_mask.png";

    %% Save processed image

    imwrite( ...
        imgResized, ...
        fullfile( ...
            preparedImageFolder, ...
            outputImageName));

    %% Save processed mask

    imwrite( ...
        maskResized, ...
        fullfile( ...
            preparedMaskFolder, ...
            outputMaskName));

    %% Display progress

    fprintf( ...
        'Processed %d / %d : %s\n', ...
        i, ...
        numTrainImages, ...
        imageName);

end

%% ============================================
% 7. VERIFY PREPARED DATA
% ============================================

fprintf('\n============================================\n');
fprintf('VERIFYING PREPARED DATA\n');
fprintf('============================================\n');

preparedImages = dir( ...
    fullfile(preparedImageFolder, '*.png'));

preparedMasks = dir( ...
    fullfile(preparedMaskFolder, '*.png'));

fprintf('\nPrepared images: %d\n', ...
    length(preparedImages));

fprintf('Prepared masks : %d\n', ...
    length(preparedMasks));

%% ============================================
% 8. DISPLAY ONE EXAMPLE
% ============================================

if ~isempty(preparedImages)

    exampleImagePath = fullfile( ...
        preparedImages(1).folder, ...
        preparedImages(1).name);

    exampleID = erase( ...
        preparedImages(1).name, ...
        '.png');

    exampleMaskPath = fullfile( ...
        preparedMaskFolder, ...
        exampleID + "_mask.png");

    exampleImage = imread(exampleImagePath);

    exampleMask = imread(exampleMaskPath);

    figure( ...
        'Name', 'TRUST-DR Prepared DRIVE Data', ...
        'Color', 'w');

    subplot(1,2,1);

    imshow(exampleImage);

    title('Prepared Fundus Image');

    subplot(1,2,2);

    imshow(exampleMask);

    title('Prepared Ground Truth Vessel Mask');

end

%% ============================================
% FINAL MESSAGE
% ============================================

fprintf('\n============================================\n');
fprintf(' DRIVE DATA PREPARATION COMPLETED SUCCESSFULLY\n');
fprintf('============================================\n');

fprintf('\nPrepared dataset location:\n');

fprintf('%s\n', outputFolder);

fprintf('\nFolder structure:\n');

fprintf('prepared_drive\n');
fprintf('   ├── images\n');
fprintf('   └── masks\n');

fprintf('\nNext step: Train retinal vessel segmentation model.\n');