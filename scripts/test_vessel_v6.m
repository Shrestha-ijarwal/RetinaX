clear; clc; close all;

projectFolder = 'C:\Users\sijar\Downloads\TRUST_DR';

vesselPath = fullfile(projectFolder, ...
    'models', ...
    'trained_vessel_segmentation_v4_1.mat');

V = load(vesselPath);

% Show variables inside MAT file
disp('VARIABLES INSIDE MODEL FILE:');
disp(fieldnames(V));

% Find network
vesselNet = V.trainedVesselNet;

% Use one known DRIVE image
testImagePath = fullfile(projectFolder, ...
    'datasets','drive','DRIVE','test','images');

files = dir(fullfile(testImagePath,'*.*'));

files = files(~[files.isdir]);

testFile = fullfile(testImagePath,files(1).name);

fprintf('\nTesting image:\n%s\n\n',testFile);

I = imread(testFile);

if ndims(I) == 2
    I = repmat(I,[1 1 3]);
end

if size(I,3) > 3
    I = I(:,:,1:3);
end

I = im2uint8(I);

% EXACT network input size
I = imresize(I,[256 256]);

fprintf('Input datatype: %s\n',class(I));
fprintf('Input range: [%g %g]\n',min(I(:)),max(I(:)));

% Prediction
predictedLabels = semanticseg( ...
    I,vesselNet, ...
    Classes=["background","vessel"]);

vesselMask = predictedLabels == "vessel";

vesselPixels = nnz(vesselMask);
vesselPercent = 100*vesselPixels/numel(vesselMask);

fprintf('\nRESULT\n');
fprintf('Vessel pixels: %d\n',vesselPixels);
fprintf('Vessel percentage: %.4f%%\n',vesselPercent);

figure;

subplot(1,2,1);
imshow(I);
title('DRIVE Test Image');

subplot(1,2,2);
imshow(vesselMask);
title(sprintf('Vessel Mask: %.4f%%',vesselPercent));