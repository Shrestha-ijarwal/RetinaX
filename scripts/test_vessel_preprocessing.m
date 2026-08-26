clear; clc; close all;

projectFolder = 'C:\Users\sijar\Downloads\TRUST_DR';

%% Load V6 / V4.2 vessel model
vesselPath = fullfile(projectFolder, ...
    'models', ...
    'trained_vessel_segmentation_v4_1.mat');

V = load(vesselPath);
vesselNet = V.trainedVesselNet;

%% Select the SAME image used in integration
[fileName,filePath] = uigetfile( ...
    {'*.png;*.jpg;*.jpeg;*.tif;*.tiff','Image Files'}, ...
    'Select the same fundus image');

if isequal(fileName,0)
    return;
end

I = imread(fullfile(filePath,fileName));

if ndims(I) == 2
    I = repmat(I,[1 1 3]);
end

if size(I,3) > 3
    I = I(:,:,1:3);
end

I = im2uint8(I);

%% Resize
I256 = imresize(I,[256 256]);

%% 1. ORIGINAL
inputOriginal = I256;

%% 2. CLAHE
lab = rgb2lab(I256);
L = lab(:,:,1)/100;

L = adapthisteq(L, ...
    'NumTiles',[8 8], ...
    'ClipLimit',0.01);

lab(:,:,1) = L*100;

inputCLAHE = im2uint8(lab2rgb(lab));

%% 3. GREEN CHANNEL REPLICATED TO RGB
green = I256(:,:,2);

inputGreen = cat(3,green,green,green);

%% Run all three
inputs = {inputOriginal,inputCLAHE,inputGreen};

names = ["Original","CLAHE","Green Channel"];

figure('Color','w','Position',[100 100 1400 500]);

for k = 1:3

    predictedLabels = semanticseg( ...
        inputs{k}, ...
        vesselNet, ...
        Classes=["background","vessel"]);

    vesselMask = predictedLabels == "vessel";

    vesselPercent = ...
        100*nnz(vesselMask)/numel(vesselMask);

    fprintf('%s vessel area: %.4f%%\n', ...
        names(k),vesselPercent);

    subplot(2,3,k);
    imshow(inputs{k});
    title(names(k));

    subplot(2,3,k+3);
    imshow(vesselMask);
    title(sprintf('%s: %.4f%%', ...
        names(k),vesselPercent));

end