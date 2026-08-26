clc;
clear;

% Project folder
baseFolder = 'C:\Users\sijar\Downloads\TRUST_DR';

% Read APTOS labels
data = readtable(fullfile(baseFolder, 'train.csv'));

% Folder containing extracted PNG images
imageFolder = fullfile(baseFolder, 'train_images');

% Output folder
outputFolder = fullfile(baseFolder, 'organized_dataset');

% Create output folder
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% DR class names
classNames = {'No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'};

% Create class folders
for i = 1:numel(classNames)
    folder = fullfile(outputFolder, classNames{i});

    if ~exist(folder, 'dir')
        mkdir(folder);
    end
end

% Copy images into correct class folders
for i = 1:height(data)

    imageName = char(data.id_code(i)) + ".png";
    diagnosis = data.diagnosis(i);

    sourceFile = fullfile(imageFolder, imageName);

    destinationFolder = fullfile( ...
        outputFolder, ...
        classNames{diagnosis + 1});

    if exist(sourceFile, 'file')
        copyfile(sourceFile, destinationFolder);
    else
        fprintf('Missing image: %s\n', imageName);
    end

    if mod(i, 100) == 0
        fprintf('Processed %d / %d images\n', i, height(data));
    end
end

disp('DONE! Dataset organized successfully.');