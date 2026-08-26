%% ============================================================
% TRUST-DR SIMULINK SYSTEM ARCHITECTURE
% XAI-CORRECTED VERSION
%
% TRUST-DR:
% AI-ASSISTED DIABETIC RETINOPATHY SCREENING
%
% Architecture:
%
% Image Input
%      |
%      v
% Safety Gateway ------------------> Reject Invalid Image
%      |
%      v
% Preprocessing
%      |
%      +--------------------+
%      |                    |
%      v                    v
% DR Severity V3 CNN       XAI Input 1
%      |                    ^
%      |                    |
%      +-----> XAI Input 2 |
%                           |
%                           v
%                     XAI Explanation
%                           |
%      +--------------------+
%      |
%      +----> Vessel Analysis
%      |
%      +----> Lesion Analysis
%      |
%      v
% Safety + Confidence
%      |
%      v
% TRUST-DR Report
%
% ============================================================

clear;
clc;

fprintf('\n');
fprintf('============================================================\n');
fprintf(' TRUST-DR SIMULINK ARCHITECTURE\n');
fprintf(' XAI-CORRECTED VERSION\n');
fprintf('============================================================\n\n');


%% ============================================================
% MODEL CONFIGURATION
% ============================================================

modelName = 'TRUST_DR_System_Architecture';

modelFolder = 'C:\Users\sijar\Downloads\TRUST_DR\models';

if ~exist(modelFolder,'dir')
    mkdir(modelFolder);
end

modelFile = fullfile( ...
    modelFolder, ...
    [modelName '.slx']);


%% ============================================================
% CLOSE OLD MODEL
% ============================================================

fprintf('Checking existing model...\n');

if bdIsLoaded(modelName)
    close_system(modelName,0);
end

if exist(modelFile,'file')
    delete(modelFile);
end


%% ============================================================
% CREATE MODEL
% ============================================================

fprintf('Creating new Simulink model...\n');

new_system(modelName);
open_system(modelName);

set_param(modelName, ...
    'Solver','FixedStepDiscrete', ...
    'StopTime','1');


%% ============================================================
% BLOCK POSITIONS
% ============================================================

% Image
posImage = [30 260 170 330];

% Safety
posSafety = [220 245 410 345];

% Reject
posReject = [450 420 640 500];

% Preprocessing
posPre = [450 250 650 340];

% CNN
posCNN = [730 80 940 160];

% Vessel
posVessel = [730 220 940 300];

% Lesion
posLesion = [730 360 940 440];

% XAI
posXAI = [1000 80 1210 180];

% Safety Confidence
posConfidence = [1000 270 1210 370];

% Final Report
posReport = [1280 205 1510 370];


%% ============================================================
% CREATE MAIN BLOCKS
% ============================================================

fprintf('\nCreating architecture blocks...\n');


% ------------------------------------------------------------
% IMAGE INPUT
% ------------------------------------------------------------

createArchitectureBlock( ...
    modelName, ...
    'Image Input', ...
    posImage, ...
    0, ...
    1);


% ------------------------------------------------------------
% SAFETY GATEWAY
% ------------------------------------------------------------

createArchitectureBlock( ...
    modelName, ...
    'Safety Gateway', ...
    posSafety, ...
    1, ...
    2);


% ------------------------------------------------------------
% REJECT INVALID IMAGE
% ------------------------------------------------------------

createArchitectureBlock( ...
    modelName, ...
    'Reject Invalid Image', ...
    posReject, ...
    1, ...
    0);


% ------------------------------------------------------------
% PREPROCESSING
% ------------------------------------------------------------

createArchitectureBlock( ...
    modelName, ...
    'Preprocessing', ...
    posPre, ...
    1, ...
    1);


% ------------------------------------------------------------
% DR SEVERITY V3 CNN
% ------------------------------------------------------------

createArchitectureBlock( ...
    modelName, ...
    'DR Severity V3 CNN', ...
    posCNN, ...
    1, ...
    1);


% ------------------------------------------------------------
% VESSEL ANALYSIS
% ------------------------------------------------------------

createArchitectureBlock( ...
    modelName, ...
    'Vessel Analysis', ...
    posVessel, ...
    1, ...
    1);


% ------------------------------------------------------------
% LESION ANALYSIS
% ------------------------------------------------------------

createArchitectureBlock( ...
    modelName, ...
    'Lesion Analysis', ...
    posLesion, ...
    1, ...
    1);


%% ============================================================
% XAI BLOCK
% ============================================================
%
% XAI HAS TWO INPUTS:
%
% Input 1 = Preprocessed retinal image
% Input 2 = V3 CNN prediction/features
%
% Output = Explanation
%
% ============================================================

fprintf('Creating XAI block with TWO inputs...\n');

createXAIBlock( ...
    modelName, ...
    'XAI Explanation', ...
    posXAI);


% ------------------------------------------------------------
% SAFETY + CONFIDENCE
% ------------------------------------------------------------

createArchitectureBlock( ...
    modelName, ...
    'Safety Confidence', ...
    posConfidence, ...
    3, ...
    1);


% ------------------------------------------------------------
% FINAL REPORT
% ------------------------------------------------------------

createArchitectureBlock( ...
    modelName, ...
    'TRUST-DR Report', ...
    posReport, ...
    5, ...
    0);


%% ============================================================
% CONNECT MAIN PIPELINE
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' CONNECTING MAIN PIPELINE\n');
fprintf('============================================================\n');


% ------------------------------------------------------------
% IMAGE -> SAFETY GATEWAY
% ------------------------------------------------------------

safeConnect( ...
    modelName, ...
    'Image Input',1, ...
    'Safety Gateway',1);


% ------------------------------------------------------------
% SAFETY -> PREPROCESSING
% ------------------------------------------------------------

safeConnect( ...
    modelName, ...
    'Safety Gateway',1, ...
    'Preprocessing',1);


% ------------------------------------------------------------
% SAFETY -> REJECT
% ------------------------------------------------------------

safeConnect( ...
    modelName, ...
    'Safety Gateway',2, ...
    'Reject Invalid Image',1);


% ------------------------------------------------------------
% PREPROCESSING -> CNN
% ------------------------------------------------------------

safeConnect( ...
    modelName, ...
    'Preprocessing',1, ...
    'DR Severity V3 CNN',1);


% ------------------------------------------------------------
% PREPROCESSING -> VESSEL
% ------------------------------------------------------------

safeConnect( ...
    modelName, ...
    'Preprocessing',1, ...
    'Vessel Analysis',1);


% ------------------------------------------------------------
% PREPROCESSING -> LESION
% ------------------------------------------------------------

safeConnect( ...
    modelName, ...
    'Preprocessing',1, ...
    'Lesion Analysis',1);


%% ============================================================
% CORRECT XAI CONNECTIONS
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' CONNECTING XAI PIPELINE\n');
fprintf('============================================================\n');

fprintf('XAI Input 1 = Preprocessed image\n');

safeConnect( ...
    modelName, ...
    'Preprocessing',1, ...
    'XAI Explanation',1);


fprintf('XAI Input 2 = V3 CNN output\n');

safeConnect( ...
    modelName, ...
    'DR Severity V3 CNN',1, ...
    'XAI Explanation',2);


%% ============================================================
% XAI -> REPORT
% ============================================================

fprintf('XAI -> TRUST-DR Report\n');

safeConnect( ...
    modelName, ...
    'XAI Explanation',1, ...
    'TRUST-DR Report',4);


%% ============================================================
% ANALYSIS -> SAFETY + CONFIDENCE
% ============================================================

fprintf('\n');
fprintf('Connecting safety-aware decision layer...\n');


% CNN -> Safety Confidence input 1

safeConnect( ...
    modelName, ...
    'DR Severity V3 CNN',1, ...
    'Safety Confidence',1);


% Vessel -> Safety Confidence input 2

safeConnect( ...
    modelName, ...
    'Vessel Analysis',1, ...
    'Safety Confidence',2);


% Lesion -> Safety Confidence input 3

safeConnect( ...
    modelName, ...
    'Lesion Analysis',1, ...
    'Safety Confidence',3);


%% ============================================================
% SAFETY CONFIDENCE -> REPORT
% ============================================================

safeConnect( ...
    modelName, ...
    'Safety Confidence',1, ...
    'TRUST-DR Report',5);


%% ============================================================
% OTHER REPORT CONNECTIONS
% ============================================================

fprintf('\nConnecting report inputs...\n');


% Report input 1 = DR severity

safeConnect( ...
    modelName, ...
    'DR Severity V3 CNN',1, ...
    'TRUST-DR Report',1);


% Report input 2 = Vessel analysis

safeConnect( ...
    modelName, ...
    'Vessel Analysis',1, ...
    'TRUST-DR Report',2);


% Report input 3 = Lesion analysis

safeConnect( ...
    modelName, ...
    'Lesion Analysis',1, ...
    'TRUST-DR Report',3);


%% ============================================================
% BLOCK COLORS
% ============================================================

fprintf('\nFormatting architecture...\n');


setBlockColor( ...
    modelName,'Image Input','lightBlue');

setBlockColor( ...
    modelName,'Safety Gateway','yellow');

setBlockColor( ...
    modelName,'Reject Invalid Image','red');

setBlockColor( ...
    modelName,'Preprocessing','lightBlue');

setBlockColor( ...
    modelName,'DR Severity V3 CNN','lightGreen');

setBlockColor( ...
    modelName,'Vessel Analysis','lightGreen');

setBlockColor( ...
    modelName,'Lesion Analysis','lightGreen');

setBlockColor( ...
    modelName,'XAI Explanation','orange');

setBlockColor( ...
    modelName,'Safety Confidence','yellow');

setBlockColor( ...
    modelName,'TRUST-DR Report','yellow');


%% ============================================================
% TITLE
% ============================================================

add_block( ...
    'built-in/Note', ...
    [modelName '/TITLE'], ...
    'Position',[480 20 1100 60]);

set_param( ...
    [modelName '/TITLE'], ...
    'Text','TRUST-DR | AI-ASSISTED DIABETIC RETINOPATHY SCREENING');


%% ============================================================
% ARCHITECTURE LABELS
% ============================================================

add_block( ...
    'built-in/Note', ...
    [modelName '/GATE_LABEL'], ...
    'Position',[25 205 430 230]);

set_param( ...
    [modelName '/GATE_LABEL'], ...
    'Text','1. IMAGE VALIDATION + SAFETY GATE');


add_block( ...
    'built-in/Note', ...
    [modelName '/ANALYSIS_LABEL'], ...
    'Position',[700 510 1100 540]);

set_param( ...
    [modelName '/ANALYSIS_LABEL'], ...
    'Text','2. MULTI-STAGE RETINAL ANALYSIS');


add_block( ...
    'built-in/Note', ...
    [modelName '/REPORT_LABEL'], ...
    'Position',[1160 420 1510 450]);

set_param( ...
    [modelName '/REPORT_LABEL'], ...
    'Text','3. SAFETY-AWARE DECISION + EXPLAINABLE REPORT');


%% ============================================================
% SAVE
% ============================================================

fprintf('\n');
fprintf('Saving Simulink model...\n');

save_system(modelName,modelFile);


%% ============================================================
% OPEN / FIT
% ============================================================

open_system(modelName);

try
    set_param(modelName,'ZoomFactor','FitSystem');
catch
end


%% ============================================================
% FINISHED
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' TRUST-DR SIMULINK MODEL CREATED SUCCESSFULLY\n');
fprintf('============================================================\n');
fprintf('\n');

fprintf('Model:\n');
fprintf('%s\n',modelFile);

fprintf('\n');
fprintf('XAI INPUTS:\n');
fprintf('  Input 1 -> Preprocessed retinal image\n');
fprintf('  Input 2 -> V3 CNN output\n');
fprintf('  Output  -> XAI explanation\n');

fprintf('\n');
fprintf('MAIN PIPELINE:\n');
fprintf('  Image Input\n');
fprintf('       -> Safety Gateway\n');
fprintf('       -> Preprocessing\n');
fprintf('       -> V3 CNN\n');
fprintf('       -> XAI\n');
fprintf('       -> TRUST-DR Report\n');

fprintf('\n');
fprintf('ANALYSIS BRANCHES:\n');
fprintf('  Preprocessing -> Vessel Analysis\n');
fprintf('  Preprocessing -> Lesion Analysis\n');

fprintf('\n');
fprintf('SAFETY BRANCH:\n');
fprintf('  CNN + Vessel + Lesion\n');
fprintf('       -> Safety + Confidence\n');
fprintf('       -> TRUST-DR Report\n');

fprintf('\n');
fprintf('REJECTION BRANCH:\n');
fprintf('  Safety Gateway -> Reject Invalid Image\n');

fprintf('\n');
fprintf('============================================================\n');


%% ============================================================
% LOCAL FUNCTION
% CREATE STANDARD ARCHITECTURE BLOCK
% ============================================================

function createArchitectureBlock( ...
    modelName, ...
    blockName, ...
    position, ...
    nInputs, ...
    nOutputs)

blockPath = [modelName '/' blockName];

add_block( ...
    'built-in/Subsystem', ...
    blockPath, ...
    'Position',position);

open_system(blockPath);


% ------------------------------------------------------------
% REMOVE DEFAULT INTERNAL BLOCKS
% ------------------------------------------------------------

internalBlocks = find_system( ...
    blockPath, ...
    'SearchDepth',1, ...
    'Type','Block');

for k = 2:numel(internalBlocks)

    try
        delete_block(internalBlocks{k});
    catch
    end

end


% ------------------------------------------------------------
% CREATE INPUT PORTS
% ------------------------------------------------------------

for p = 1:nInputs

    y = 30 + (p-1)*35;

    add_block( ...
        'simulink/Ports & Subsystems/In1', ...
        [blockPath '/In' num2str(p)], ...
        'Position',[20 y 50 y+20], ...
        'Port',num2str(p));

end


% ------------------------------------------------------------
% CREATE OUTPUT PORTS
% ------------------------------------------------------------

for p = 1:nOutputs

    y = 30 + (p-1)*35;

    add_block( ...
        'simulink/Ports & Subsystems/Out1', ...
        [blockPath '/Out' num2str(p)], ...
        'Position',[150 y 180 y+20], ...
        'Port',num2str(p));

end


% ------------------------------------------------------------
% INTERNAL CONNECTION
% ONLY FOR SINGLE INPUT / SINGLE OUTPUT BLOCKS
% ------------------------------------------------------------

if nInputs == 1 && nOutputs == 1

    add_line( ...
        blockPath, ...
        'In1/1', ...
        'Out1/1');

end


close_system(blockPath);

end


%% ============================================================
% LOCAL FUNCTION
% CREATE XAI BLOCK
% ============================================================
%
% XAI:
%
% In1 = image
% In2 = CNN result
%
% Bus Creator combines both inputs.
%
% This is an ARCHITECTURAL representation.
% Actual Grad-CAM implementation remains in MATLAB.
%
% ============================================================

function createXAIBlock( ...
    modelName, ...
    blockName, ...
    position)

blockPath = [modelName '/' blockName];

add_block( ...
    'built-in/Subsystem', ...
    blockPath, ...
    'Position',position);

open_system(blockPath);


% ------------------------------------------------------------
% REMOVE DEFAULT BLOCKS
% ------------------------------------------------------------

internalBlocks = find_system( ...
    blockPath, ...
    'SearchDepth',1, ...
    'Type','Block');

for k = 2:numel(internalBlocks)

    try
        delete_block(internalBlocks{k});
    catch
    end

end


% ------------------------------------------------------------
% INPUT 1
% ------------------------------------------------------------

add_block( ...
    'simulink/Ports & Subsystems/In1', ...
    [blockPath '/Image_Input'], ...
    'Position',[20 30 50 50], ...
    'Port','1');


% ------------------------------------------------------------
% INPUT 2
% ------------------------------------------------------------

add_block( ...
    'simulink/Ports & Subsystems/In1', ...
    [blockPath '/CNN_Output'], ...
    'Position',[20 100 50 120], ...
    'Port','2');


% ------------------------------------------------------------
% BUS CREATOR
% ------------------------------------------------------------

add_block( ...
    'simulink/Signal Routing/Bus Creator', ...
    [blockPath '/XAI_Evidence_Bus'], ...
    'Position',[90 50 150 110], ...
    'Inputs','2');


% ------------------------------------------------------------
% OUTPUT
% ------------------------------------------------------------

add_block( ...
    'simulink/Ports & Subsystems/Out1', ...
    [blockPath '/XAI_Output'], ...
    'Position',[210 70 240 90], ...
    'Port','1');


% ------------------------------------------------------------
% CONNECTIONS INSIDE XAI
% ------------------------------------------------------------

add_line( ...
    blockPath, ...
    'Image_Input/1', ...
    'XAI_Evidence_Bus/1');


add_line( ...
    blockPath, ...
    'CNN_Output/1', ...
    'XAI_Evidence_Bus/2');


add_line( ...
    blockPath, ...
    'XAI_Evidence_Bus/1', ...
    'XAI_Output/1');


close_system(blockPath);

end


%% ============================================================
% LOCAL FUNCTION
% SAFE CONNECTION
% ============================================================

function safeConnect( ...
    modelName, ...
    sourceBlock, ...
    sourcePort, ...
    destinationBlock, ...
    destinationPort)

sourcePath = ...
    [sourceBlock '/' num2str(sourcePort)];

destinationPath = ...
    [destinationBlock '/' num2str(destinationPort)];

try

    add_line( ...
        modelName, ...
        sourcePath, ...
        destinationPath, ...
        'autorouting','smart');

    fprintf( ...
        '  OK   %s -> %s\n', ...
        sourcePath, ...
        destinationPath);

catch ME

    fprintf( ...
        '  ERROR %s -> %s\n', ...
        sourcePath, ...
        destinationPath);

    fprintf( ...
        '         %s\n', ...
        ME.message);

end

end


%% ============================================================
% LOCAL FUNCTION
% SET BLOCK COLOR
% ============================================================

function setBlockColor( ...
    modelName, ...
    blockName, ...
    colorName)

try

    set_param( ...
        [modelName '/' blockName], ...
        'BackgroundColor',colorName);

catch

    % Cosmetic operation only.

end

end