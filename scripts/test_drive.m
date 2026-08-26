clc;
clear;
close all;

%% TRUST-DR
% DRIVE DATASET LOADING TEST

fprintf('\n====================================\n');
fprintf('       TRUST-DR: DRIVE TEST\n');
fprintf('====================================\n\n');

%% 1. DEFINE DATASET PATHS

driveFolder = ...
    'C:\Users\sijar\Downloads\TRUST_DR\datasets\drive\DRIVE';

imageFolder = fullfile(driveFolder, ...
    'training', 'images');

maskFolder = fullfile(driveFolder, ...
    'training', '1st_manual');

%% 2. CHECK FOLDERS

if ~isfolder(imageFolder)
    error('Training images folder not found!');
end

if ~isfolder(maskFolder)
    error('Manual mask folder not found!');
end

fprintf('Images folder found!\n');
fprintf('Manual vessel masks folder found!\n\n');

%% 3. GET IMAGE FILES

imageFiles = dir(fullfile(imageFolder, '*.tif'));

fprintf('Number of training images: %d\n', ...
    length(imageFiles));

%% 4. LOAD FIRST FUNDUS IMAGE

imagePath = fullfile( ...
    imageFiles(1).folder, ...
    imageFiles(1).name);

fundusImage = imread(imagePath);

fprintf('\nLoaded image:\n');
fprintf('%s\n', imageFiles(1).name);

%% 5. FIND CORRESPONDING VESSEL MASK

imageName = erase(imageFiles(1).name, '.tif');

% DRIVE mask naming example:
% 21_training.tif
% 21_manual1.gif

imageID = extractBefore(imageName, '_');

maskPattern = char(imageID + "_manual1.gif");

maskPath = fullfile(maskFolder, maskPattern);

if ~isfile(maskPath)
    error('Corresponding vessel mask not found: %s', maskPattern);
end

vesselMask = imread(maskPath);

%% Convert mask to binary

if size(vesselMask,3) == 3
    vesselMask = rgb2gray(vesselMask);
end

vesselMask = vesselMask > 0;

fprintf('Corresponding vessel mask loaded!\n');

%% 6. DISPLAY RESULTS

figure('Name', 'TRUST-DR DRIVE Dataset Test', ...
       'Color', 'w');

subplot(1,2,1);

imshow(fundusImage);

title('Original DRIVE Fundus Image');

subplot(1,2,2);

imshow(vesselMask);

title('Ground Truth Blood Vessel Mask');

fprintf('\n====================================\n');
fprintf('DRIVE DATASET LOADED SUCCESSFULLY!\n');
fprintf('====================================\n');

fprintf('\nNext module: Blood Vessel Segmentation\n');