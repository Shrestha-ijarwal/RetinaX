%% IDRiD_Preprocessing.m
% IDRiD retinal lesion segmentation preprocessing
%
% Classes:
%   0 = Background
%   1 = Microaneurysms (MA)
%   2 = Haemorrhages (HE)
%   3 = Hard Exudates (EX)
%   4 = Soft Exudates (SE)
%   5 = Optic Disc (OD)

clear;
clc;
close all;

%% ============================================================
% 1. DATASET PATHS
% =============================================================

basePath = 'C:\Users\sijar\Downloads\TRUST_DR\datasets\A. Segmentation\A. Segmentation';

originalPath = fullfile(basePath, '1. Original Images');
gtPath       = fullfile(basePath, '2. All Segmentation Groundtruths');

trainImagePath = fullfile(originalPath, 'a. Training Set');
trainGTPath    = fullfile(gtPath, 'a. Training Set');

% Ground-truth folders
maPath = fullfile(trainGTPath, '1. Microaneurysms');
hePath = fullfile(trainGTPath, '2. Haemorrhages');
exPath = fullfile(trainGTPath, '3. Hard Exudates');
sePath = fullfile(trainGTPath, '4. Soft Exudates');
odPath = fullfile(trainGTPath, '5. Optic Disc');

%% ============================================================
% 2. OUTPUT FOLDER
% =============================================================

combinedTrainPath = fullfile(basePath, '3. Combined Training Masks');

if ~exist(combinedTrainPath, 'dir')
    mkdir(combinedTrainPath);
end

%% ============================================================
% 3. CLASS DEFINITIONS
% =============================================================

classNames = [
    "Background"
    "MA"
    "HE"
    "EX"
    "SE"
    "OD"
];

fprintf('\n============================================\n');
fprintf('IDRiD PREPROCESSING\n');
fprintf('============================================\n');

fprintf('\nClasses:\n');

for c = 1:length(classNames)
    fprintf('%d = %s\n', c-1, classNames(c));
end

%% ============================================================
% 4. FIND TRAINING IMAGES
% =============================================================

imageFiles = dir(fullfile(trainImagePath, '*.jpg'));

numImages = length(imageFiles);

fprintf('\nNumber of training images found: %d\n', numImages);

if numImages == 0
    error('No training images were found.');
end

%% ============================================================
% 5. CREATE COMBINED MASKS
% =============================================================

fprintf('\nCreating combined masks...\n\n');

for i = 1:numImages

    % ---------------------------------------------------------
    % Get image ID
    % Example: IDRiD_01
    % ---------------------------------------------------------

    imageName = imageFiles(i).name;
    [~, imageID, ~] = fileparts(imageName);

    fprintf('[%02d/%02d] Processing %s ... ', ...
        i, numImages, imageID);

    % ---------------------------------------------------------
    % Read original image
    % ---------------------------------------------------------

    imageFile = fullfile(trainImagePath, imageName);
    img = imread(imageFile);

    [H, W, ~] = size(img);

    % ---------------------------------------------------------
    % Initialize combined mask
    % ---------------------------------------------------------

    combinedMask = zeros(H, W, 'uint8');

    %% --------------------------------------------------------
    % MA = Class 1
    % ---------------------------------------------------------

    maFile = fullfile(maPath, [imageID '_MA.tif']);

    if exist(maFile, 'file')

        ma = imread(maFile);

        combinedMask(ma > 0) = 1;

    end

    %% --------------------------------------------------------
    % HE = Class 2
    % ---------------------------------------------------------

    heFile = fullfile(hePath, [imageID '_HE.tif']);

    if exist(heFile, 'file')

        he = imread(heFile);

        combinedMask(he > 0) = 2;

    end

    %% --------------------------------------------------------
    % EX = Class 3
    % ---------------------------------------------------------

    exFile = fullfile(exPath, [imageID '_EX.tif']);

    if exist(exFile, 'file')

        ex = imread(exFile);

        combinedMask(ex > 0) = 3;

    end

    %% --------------------------------------------------------
    % SE = Class 4
    % ---------------------------------------------------------

    seFile = fullfile(sePath, [imageID '_SE.tif']);

    if exist(seFile, 'file')

        se = imread(seFile);

        combinedMask(se > 0) = 4;

    end

    %% --------------------------------------------------------
    % OD = Class 5
    % ---------------------------------------------------------

    odFile = fullfile(odPath, [imageID '_OD.tif']);

    if exist(odFile, 'file')

        od = imread(odFile);

        combinedMask(od > 0) = 5;

    end

    %% --------------------------------------------------------
    % Save combined mask
    % ---------------------------------------------------------

    outputFile = fullfile( ...
        combinedTrainPath, ...
        [imageID '_mask.png']);

    imwrite(combinedMask, outputFile);

    fprintf('DONE\n');

end

%% ============================================================
% 6. VERIFY OUTPUT
% =============================================================

outputFiles = dir(fullfile(combinedTrainPath, '*.png'));

fprintf('\n============================================\n');
fprintf('PREPROCESSING COMPLETE\n');
fprintf('============================================\n');

fprintf('Input images  : %d\n', numImages);
fprintf('Output masks  : %d\n', length(outputFiles));

%% ============================================================
% 7. CHECK MASK LABELS
% =============================================================

fprintf('\nChecking generated masks...\n\n');

allLabelsFound = [];

for i = 1:length(outputFiles)

    maskFile = fullfile( ...
        combinedTrainPath, ...
        outputFiles(i).name);

    mask = imread(maskFile);

    labels = unique(mask);

    allLabelsFound = union(allLabelsFound, labels);

end

fprintf('Labels found across training masks:\n');

disp(allLabelsFound');

fprintf('\nExpected labels:\n');
fprintf('0 = Background\n');
fprintf('1 = MA\n');
fprintf('2 = HE\n');
fprintf('3 = EX\n');
fprintf('4 = SE\n');
fprintf('5 = OD\n');

%% ============================================================
% 8. VERIFY FIRST IMAGE
% =============================================================

firstMaskFile = fullfile( ...
    combinedTrainPath, ...
    'IDRiD_01_mask.png');

if exist(firstMaskFile, 'file')

    testMask = imread(firstMaskFile);

    fprintf('\nIDRiD_01 labels:\n');
    disp(unique(testMask)');

    figure;

    imagesc(testMask);
    axis image;
    colorbar;

    title('IDRiD\_01 Combined Ground Truth');

    colormap(jet(6));

end

%% ============================================================
% 9. FINAL MESSAGE
% =============================================================

fprintf('\n============================================\n');
fprintf('All combined training masks are ready.\n');
fprintf('============================================\n');

fprintf('\nSaved to:\n%s\n\n', combinedTrainPath);