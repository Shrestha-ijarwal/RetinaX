%% MESSIDOR STRATIFIED SPLIT

clear;
clc;

projectFolder = 'C:\Users\sijar\Downloads\TRUST_DR';

labelsPath = fullfile(projectFolder, ...
    'datasets','archive','messidor_data.csv');

imageFolder = fullfile(projectFolder, ...
    'datasets','archive','messidor-2','messidor-2','preprocess');

T = readtable(labelsPath);

% Keep gradable images
T = T(T.adjudicated_gradable == 1,:);

% Confirm all images exist
exists = false(height(T),1);

for i = 1:height(T)
    exists(i) = isfile(fullfile(imageFolder,T.id_code{i}));
end

T = T(exists,:);

fprintf('Total usable images: %d\n',height(T));

%% Stratified 70/15/15 split

rng(42);

trainIdx = [];
valIdx   = [];
testIdx  = [];

for g = 0:4

    idx = find(T.diagnosis == g);

    idx = idx(randperm(numel(idx)));

    n = numel(idx);

    nTrain = floor(0.70*n);
    nVal   = floor(0.15*n);

    trainIdx = [trainIdx; idx(1:nTrain)];
    valIdx   = [valIdx; ...
        idx(nTrain+1:nTrain+nVal)];
    testIdx  = [testIdx; ...
        idx(nTrain+nVal+1:end)];

end

% Shuffle each split
trainIdx = trainIdx(randperm(numel(trainIdx)));
valIdx   = valIdx(randperm(numel(valIdx)));
testIdx  = testIdx(randperm(numel(testIdx)));

TTrain = T(trainIdx,:);
TVal   = T(valIdx,:);
TTest  = T(testIdx,:);

fprintf('\nTRAIN: %d\n',height(TTrain));
fprintf('VALIDATION: %d\n',height(TVal));
fprintf('FINAL TEST: %d\n',height(TTest));

fprintf('\nTRAIN CLASS DISTRIBUTION\n');
tabulate(TTrain.diagnosis)

fprintf('\nVALIDATION CLASS DISTRIBUTION\n');
tabulate(TVal.diagnosis)

fprintf('\nFINAL TEST CLASS DISTRIBUTION\n');
tabulate(TTest.diagnosis)

%% Save split

splitFolder = fullfile(projectFolder, ...
    'results','messidor_training');

if ~isfolder(splitFolder)
    mkdir(splitFolder);
end

save(fullfile(splitFolder,'messidor_stratified_split.mat'), ...
    'TTrain','TVal','TTest', ...
    'trainIdx','valIdx','testIdx');

writetable(TTrain,fullfile(splitFolder,'train.csv'));
writetable(TVal,fullfile(splitFolder,'validation.csv'));
writetable(TTest,fullfile(splitFolder,'test.csv'));

fprintf('\nSplit saved successfully.\n');