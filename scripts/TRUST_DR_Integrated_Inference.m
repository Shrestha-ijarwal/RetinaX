%% ============================================================
% TRUST_DR_Integrated_Inference.m
% TRUST-DR INTEGRATED INFERENCE SYSTEM
%
% Includes:
%   1. DR classification (APTOS)
%   2. CLAHE preprocessing + original/CLAHE stability check
%   3. Image Quality Assessment
%   4. Fundus-domain Safety Gateway
%   5. Vessel segmentation (DRIVE V6)
%   6. Lesion segmentation (IDRiD)
%   7. Grad-CAM
%   8. HUMAN-IN-THE-LOOP review
%   9. Separate PNG / MAT / CSV outputs
%
% IMPORTANT:
% The current Safety Gateway is a FUNDUS-DOMAIN heuristic.
% It can reject obvious non-fundus images, but it cannot reliably
% distinguish diabetic retinopathy from hypertensive retinopathy,
% AMD, glaucoma, etc. A dedicated OOD/negative-class model is
% required for that stronger capability.
%
% MATLAB R2026a
%% ============================================================

clear;
clc;
close all;

fprintf('\n============================================================\n');
fprintf('TRUST_DR INTEGRATED INFERENCE SYSTEM\n');
fprintf('============================================================\n\n');

%% ============================================================
% 1. MODEL PATHS
% ============================================================

projectFolder = 'C:\Users\sijar\Downloads\TRUST_DR';

drPath = fullfile(projectFolder,'models','trained_DR_model_v3.mat');

vesselPath = fullfile(projectFolder,'models', ...
    'trained_vessel_segmentation_v4_1.mat');

lesionPath = fullfile(projectFolder,'datasets', ...
    'A. Segmentation','A. Segmentation', ...
    'IDRiD_UNet_BalancedWeighted_Augmented_256.mat');

resultsFolder = fullfile(projectFolder, ...
    'results','integrated_inference');

if ~isfolder(resultsFolder)
    mkdir(resultsFolder);
end

fprintf('VERIFYING MODEL FILES\n');
fprintf('------------------------------------------------------------\n');
fprintf('DR model:\n%s\n\n',drPath);
fprintf('Vessel model:\n%s\n\n',vesselPath);
fprintf('Lesion model:\n%s\n\n',lesionPath);

if ~isfile(drPath)
    error('DR model not found:\n%s',drPath);
end

if ~isfile(vesselPath)
    error('Vessel model not found:\n%s',vesselPath);
end

if ~isfile(lesionPath)
    error('Lesion model not found:\n%s',lesionPath);
end

fprintf('SUCCESS: All model files found.\n\n');

%% ============================================================
% 2. LOAD MODELS
% ============================================================

fprintf('LOADING DR CLASSIFICATION MODEL\n');
fprintf('------------------------------------------------------------\n');

D = load(drPath);
drNet = findNetworkVariable(D);
drInputSize = getInputSize(drNet,[224 224 3]);

fprintf('DR model loaded: %s\n',class(drNet));
fprintf('DR input size: %d x %d x %d\n\n', ...
    drInputSize(1),drInputSize(2),drInputSize(3));

fprintf('LOADING VESSEL SEGMENTATION MODEL\n');
fprintf('------------------------------------------------------------\n');

V = load(vesselPath);
vesselNet = findNetworkVariable(V);
vesselInputSize = getInputSize(vesselNet,[256 256 3]);

fprintf('Vessel model loaded: %s\n',class(vesselNet));
fprintf('Vessel input size: %d x %d x %d\n\n', ...
    vesselInputSize(1),vesselInputSize(2),vesselInputSize(3));

fprintf('LOADING LESION SEGMENTATION MODEL\n');
fprintf('------------------------------------------------------------\n');

L = load(lesionPath);
lesionNet = findNetworkVariable(L);
lesionInputSize = [256 256 3];

lesionNames = ["Background","MA","HE","EX","SE","OD"];

fprintf('Lesion model loaded: %s\n',class(lesionNet));
fprintf('Lesion input size: %d x %d x %d\n\n', ...
    lesionInputSize(1),lesionInputSize(2),lesionInputSize(3));

%% ============================================================
% 3. SELECT IMAGE
% ============================================================

fprintf('SELECT RETINAL FUNDUS IMAGE\n');
fprintf('------------------------------------------------------------\n');

[fileName,filePath] = uigetfile( ...
    {'*.png;*.jpg;*.jpeg;*.tif;*.tiff','Image Files'}, ...
    'Select retinal image');

if isequal(fileName,0)
    fprintf('No image selected. Program stopped.\n');
    return;
end

imagePath = fullfile(filePath,fileName);
[~,baseName,~] = fileparts(fileName);

fprintf('Selected image:\n%s\n\n',imagePath);

%% ============================================================
% 4. READ IMAGE
% ============================================================

originalImage = imread(imagePath);

if ndims(originalImage) == 2
    originalImage = repmat(originalImage,[1 1 3]);
end

if size(originalImage,3) > 3
    originalImage = originalImage(:,:,1:3);
end

originalImage = im2uint8(originalImage);

[H,W,~] = size(originalImage);

fprintf('Original image size:\n');
fprintf('  Height   : %d\n',H);
fprintf('  Width    : %d\n',W);
fprintf('  Channels : %d\n\n',size(originalImage,3));

%% ============================================================
% 5. CLAHE / FUNDUS PREPROCESSING
% ============================================================

fprintf('============================================================\n');
fprintf('CLAHE / FUNDUS PREPROCESSING\n');
fprintf('============================================================\n');

% R2026a-safe implementation:
% DO NOT use lab2rgb(lab,'ColorSpace','lab').
claheImage = applyFundusCLAHE(originalImage);

fprintf('CLAHE preprocessing completed successfully.\n');
fprintf('Original image preserved unchanged.\n\n');

clahePath = fullfile(resultsFolder, ...
    [baseName '_00_CLAHE.png']);

imwrite(claheImage,clahePath);

%% ============================================================
% 6. IMAGE QUALITY ASSESSMENT
% ============================================================

fprintf('============================================================\n');
fprintf('IMAGE QUALITY ASSESSMENT\n');
fprintf('============================================================\n');

grayImage = rgb2gray(originalImage);
grayDouble = im2double(grayImage);

laplacianFilter = fspecial('laplacian',0.2);
lapImage = imfilter(grayDouble,laplacianFilter,'replicate');

focusScore = var(lapImage(:));
brightnessScore = mean(grayDouble(:));
contrastScore = std(grayDouble(:));

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

elseif focusScore < 2*FOCUS_BAD || ...
        contrastScore < 1.5*CONTRAST_BAD

    qualityResult = "BORDERLINE";
    qualityAction = ...
        "Borderline image quality. Human validation recommended.";

else

    qualityResult = "ACCEPTABLE";
    qualityAction = ...
        "Image quality acceptable for AI screening.";

end

fprintf('Focus Score      : %.6f\n',focusScore);
fprintf('Brightness Score : %.4f\n',brightnessScore);
fprintf('Contrast Score   : %.4f\n',contrastScore);
fprintf('Quality Result   : %s\n',qualityResult);
fprintf('Quality Action   : %s\n\n',qualityAction);

%% ============================================================
% 7. FUNDUS-DOMAIN SAFETY GATEWAY
% ============================================================

fprintf('============================================================\n');
fprintf('DR APPLICABILITY / SAFETY GATEWAY\n');
fprintf('============================================================\n');

[gatewayPass,gatewayScore,gatewayReason,gatewayDetails] = ...
    runDRSafetyGateway(originalImage,claheImage,qualityResult);

fprintf('Fundus-likeness score : %.2f%%\n',100*gatewayScore);
fprintf('Gateway decision      : %s\n',gatewayDetails.Decision);
fprintf('Reason                : %s\n',gatewayReason);

if ~gatewayPass

    fprintf('\nSAFETY GATEWAY: BLOCKED - DR inference stopped.\n\n');

    gatewayReportPath = fullfile(resultsFolder, ...
        [baseName '_SAFETY_GATEWAY_BLOCKED.png']);

    createGatewayBlockedFigure( ...
        originalImage,qualityResult,gatewayScore, ...
        gatewayReason,gatewayReportPath);

    gatewayMatPath = fullfile(resultsFolder, ...
        [baseName '_SafetyGateway_Results.mat']);

    save(gatewayMatPath, ...
        'imagePath','gatewayPass','gatewayScore','gatewayReason', ...
        'gatewayDetails','qualityResult','qualityAction', ...
        'focusScore','brightnessScore','contrastScore', ...
        'originalImage','claheImage','-v7.3');

    fprintf('Blocked-case visualization saved:\n%s\n', ...
        gatewayReportPath);

    fprintf('No DR/vessel/lesion prediction was produced.\n');
    fprintf('============================================================\n');
    fprintf('TRUST_DR STOPPED BY SAFETY GATEWAY\n');
    fprintf('============================================================\n');

    return;
end

fprintf('\nSAFETY GATEWAY: PASSED - DR inference may proceed.\n\n');

%% ============================================================
% 8. DR CLASSIFICATION - ORIGINAL + CLAHE
% ============================================================

fprintf('============================================================\n');
fprintf('RUNNING DR CLASSIFICATION (APTOS)\n');
fprintf('============================================================\n');

fprintf('Running DR classifier on ORIGINAL image...\n');

drImage = imresize(originalImage,drInputSize(1:2));

[~,drScoresRaw] = classify(drNet,drImage);

drScoresOriginal = normalizeScores(drScoresRaw);

classNamesDR = getClassificationNames( ...
    drNet,numel(drScoresOriginal));

[drConfOriginal,drIdxOriginal] = max(drScoresOriginal);

drPredOriginal = string(classNamesDR(drIdxOriginal));

fprintf('\nORIGINAL DR RESULT\n');
fprintf('------------------------------------------------------------\n');
fprintf('Predicted Class : %s\n',drPredOriginal);
fprintf('Confidence      : %.4f (%.2f%%)\n', ...
    drConfOriginal,100*drConfOriginal);

fprintf('\nDR CLASS PROBABILITIES - ORIGINAL\n');
fprintf('------------------------------------------------------------\n');

for k = 1:numel(drScoresOriginal)
    fprintf('%-18s %.4f (%.2f%%)\n', ...
        char(classNamesDR(k)), ...
        drScoresOriginal(k), ...
        100*drScoresOriginal(k));
end

fprintf('\n============================================================\n');
fprintf('ORIGINAL vs CLAHE DR COMPARISON\n');
fprintf('============================================================\n');

fprintf('Running DR classifier on CLAHE image...\n');

claheDRImage = imresize(claheImage,drInputSize(1:2));

[~,drScoresCLAHEraw] = classify(drNet,claheDRImage);

drScoresCLAHE = normalizeScores(drScoresCLAHEraw);

[drConfCLAHE,drIdxCLAHE] = max(drScoresCLAHE);

drPredCLAHE = string(classNamesDR(drIdxCLAHE));

fprintf('\nORIGINAL IMAGE\n');
fprintf('Prediction : %s\n',drPredOriginal);
fprintf('Confidence : %.2f%%\n',100*drConfOriginal);

fprintf('\nCLAHE IMAGE\n');
fprintf('Prediction : %s\n',drPredCLAHE);
fprintf('Confidence : %.2f%%\n',100*drConfCLAHE);

if drPredOriginal == drPredCLAHE

    claheStability = "STABLE";

    fprintf('\nRESULT: DR prediction is STABLE after CLAHE.\n');

else

    claheStability = "UNSTABLE";

    fprintf('\nRESULT: DR prediction CHANGED after CLAHE.\n');

end

% Final DR result is based on the original image.
drPred = drPredOriginal;
drScores = drScoresOriginal;
drConf = drConfOriginal;

%% ============================================================
% 9. POST-CLASSIFICATION SAFETY CHECK
% ============================================================

fprintf('\n============================================================\n');
fprintf('POST-CLASSIFICATION SAFETY CHECK\n');
fprintf('============================================================\n');

POST_MIN_CONFIDENCE = 0.60;

if claheStability == "UNSTABLE"

    predictionSafety = "BLOCKED";

    predictionSafetyReason = ...
        "Original and CLAHE predictions disagree. Human review required; automated DR result blocked.";

elseif drConf < POST_MIN_CONFIDENCE

    predictionSafety = "BLOCKED";

    predictionSafetyReason = ...
        "DR classifier confidence is below the safety threshold.";

else

    predictionSafety = "PASSED";

    predictionSafetyReason = ...
        "Original/CLAHE prediction is stable and confidence meets the safety threshold.";

end

fprintf('Prediction safety : %s\n',predictionSafety);
fprintf('Reason            : %s\n',predictionSafetyReason);

%% ============================================================
% 10. REFERRAL
% ============================================================

referableClasses = ["Moderate","Severe","Proliferative"];

if ismember(drPred,referableClasses)

    referralResult = "REFERABLE DR";
    referralAction = ...
        "Priority ophthalmologist evaluation recommended.";

else

    referralResult = "NON-REFERABLE DR";
    referralAction = ...
        "Routine screening follow-up recommended.";

end

if predictionSafety == "BLOCKED"

    referralResult = "AUTOMATED RESULT BLOCKED";

    referralAction = ...
        "Do not use AI result for automated referral. Human assessment required.";

elseif qualityResult == "UNGRADABLE"

    referralResult = "AUTOMATED RESULT BLOCKED";

    referralAction = ...
        "Image quality insufficient. Human assessment required.";

end

fprintf('\nREFERRAL RESULT : %s\n',referralResult);
fprintf('ACTION          : %s\n',referralAction);

%% ============================================================
% 11. VESSEL SEGMENTATION
% ============================================================

fprintf('\n============================================================\n');
fprintf('RUNNING VESSEL SEGMENTATION (DRIVE V6)\n');
fprintf('============================================================\n');

vesselInput = imresize(claheImage,vesselInputSize(1:2));

fprintf('Vessel inference input: [%d %d %d]\n', ...
    size(vesselInput,1),size(vesselInput,2),size(vesselInput,3));

fprintf('Vessel datatype: %s\n',class(vesselInput));
fprintf('Vessel value range: [%g %g]\n', ...
    min(vesselInput(:)),max(vesselInput(:)));

predictedVesselLabels = semanticseg( ...
    vesselInput,vesselNet, ...
    Classes=["background","vessel"]);

vesselMask256 = predictedVesselLabels == "vessel";

vesselPixels256 = nnz(vesselMask256);

vesselPercent256 = ...
    100*vesselPixels256/numel(vesselMask256);

fprintf('Vessel pixels at network resolution: %d\n', ...
    vesselPixels256);

fprintf('Vessel percentage at network resolution: %.4f%%\n', ...
    vesselPercent256);

vesselMaskOriginal = imresize( ...
    vesselMask256,[H W],'nearest');

vesselPixels = nnz(vesselMaskOriginal);

vesselArea = ...
    100*vesselPixels/max(H*W,1);

fprintf('\nVESSEL CLASS STATISTICS\n');
fprintf('------------------------------------------------------------\n');
fprintf('Background pixels : %d\n',nnz(~vesselMaskOriginal));
fprintf('Vessel pixels     : %d\n',vesselPixels);
fprintf('Vessel area       : %.4f%%\n\n',vesselArea);

%% ============================================================
% 12. LESION SEGMENTATION
% ============================================================

fprintf('============================================================\n');
fprintf('RUNNING LESION SEGMENTATION (IDRiD)\n');
fprintf('============================================================\n');

lesionInput = imresize( ...
    originalImage,lesionInputSize(1:2));

lesionInput = im2single(lesionInput);

fprintf('Lesion network input: [%d %d %d]\n', ...
    size(lesionInput,1),size(lesionInput,2),size(lesionInput,3));

fprintf('Lesion input range: [%.4f %.4f]\n', ...
    min(lesionInput(:)),max(lesionInput(:)));

lesionLabels256 = semanticseg( ...
    lesionInput,lesionNet,Classes=lesionNames);

fprintf('Lesion inference method: semanticseg()\n');

lesionLabels256Numeric = ...
    zeros(size(lesionLabels256),'uint8');

for c = 1:numel(lesionNames)

    lesionLabels256Numeric( ...
        lesionLabels256 == lesionNames(c)) = uint8(c-1);

end

lesionLabelsOriginal = imresize( ...
    lesionLabels256Numeric,[H W],'nearest');

fprintf('\nLESION PIXEL STATISTICS\n');
fprintf('------------------------------------------------------------\n');

lesionCounts = zeros(6,1);
lesionAreas = zeros(6,1);

for c = 0:5

    lesionCounts(c+1) = ...
        nnz(lesionLabelsOriginal == c);

    lesionAreas(c+1) = ...
        100*lesionCounts(c+1)/max(H*W,1);

    fprintf('%-12s Pixels: %-10d Area: %.6f%%\n', ...
        char(lesionNames(c+1)), ...
        lesionCounts(c+1), ...
        lesionAreas(c+1));

end

lesionForegroundPixels = sum(lesionCounts(2:end));

lesionForegroundArea = ...
    100*lesionForegroundPixels/max(H*W,1);

fprintf('Total foreground lesion area: %.6f%%\n\n', ...
    lesionForegroundArea);

%% ============================================================
% 13. VISUAL OUTPUTS
% ============================================================

fprintf('============================================================\n');
fprintf('GENERATING DISTINCT VISUAL OUTPUTS\n');
fprintf('============================================================\n');

originalPath = fullfile(resultsFolder, ...
    [baseName '_01_Original.png']);

vesselMaskPath = fullfile(resultsFolder, ...
    [baseName '_02_VesselMask.png']);

lesionMaskPath = fullfile(resultsFolder, ...
    [baseName '_03_LesionMask.png']);

vesselOverlayPath = fullfile(resultsFolder, ...
    [baseName '_04_VesselOverlay.png']);

lesionOverlayPath = fullfile(resultsFolder, ...
    [baseName '_05_LesionOverlay.png']);

combinedOverlayPath = fullfile(resultsFolder, ...
    [baseName '_06_CombinedOverlay.png']);

analysisPath = fullfile(resultsFolder, ...
    [baseName '_07_TRUST_DR_Analysis.png']);

gradcamOverlayPath = fullfile(resultsFolder, ...
    [baseName '_08_GradCAM_Overlay.png']);

gradcamHeatmapPath = fullfile(resultsFolder, ...
    [baseName '_09_GradCAM_Heatmap.png']);

csvPath = fullfile(resultsFolder, ...
    [baseName '_LesionSummary.csv']);

matPath = fullfile(resultsFolder, ...
    [baseName '_TRUST_DR_Results.mat']);

imwrite(originalImage,originalPath);
imwrite(claheImage,clahePath);

vesselMaskRGB = ...
    uint8(repmat(vesselMaskOriginal,[1 1 3]))*255;

imwrite(vesselMaskRGB,vesselMaskPath);

lesionRGB = zeros(H,W,3,'uint8');

maMask = lesionLabelsOriginal == 1;
lesionRGB(:,:,1) = uint8(255*maMask);

heMask = lesionLabelsOriginal == 2;
lesionRGB(:,:,1) = max(lesionRGB(:,:,1), ...
    uint8(255*heMask));
lesionRGB(:,:,2) = max(lesionRGB(:,:,2), ...
    uint8(255*heMask));

exMask = lesionLabelsOriginal == 3;
lesionRGB(:,:,2) = max(lesionRGB(:,:,2), ...
    uint8(255*exMask));

seMask = lesionLabelsOriginal == 4;
lesionRGB(:,:,3) = max(lesionRGB(:,:,3), ...
    uint8(255*seMask));

odMask = lesionLabelsOriginal == 5;
lesionRGB(:,:,1) = max(lesionRGB(:,:,1), ...
    uint8(255*odMask));
lesionRGB(:,:,3) = max(lesionRGB(:,:,3), ...
    uint8(255*odMask));

imwrite(lesionRGB,lesionMaskPath);

vesselOverlay = makeOverlay( ...
    originalImage,vesselMaskOriginal,[0 1 0],0.45);

lesionOverlay = makeLesionOverlay( ...
    originalImage,lesionLabelsOriginal);

combinedOverlay = makeCombinedOverlay( ...
    originalImage,vesselMaskOriginal,lesionLabelsOriginal);

imwrite(vesselOverlay,vesselOverlayPath);
imwrite(lesionOverlay,lesionOverlayPath);
imwrite(combinedOverlay,combinedOverlayPath);

%% ============================================================
% 14. GRAD-CAM
% ============================================================

fprintf('Generating Grad-CAM...\n');

gradcamMap = [];
gradcamOverlay = drImage;
gradcamHeatmap = zeros(size(drImage),'uint8');

try

    gradcamMap = gradCAM( ...
        drNet,drImage,drPred);

    gradcamMap = double(squeeze(gradcamMap));

    gradcamMap = mat2gray(gradcamMap);

    gradcamMap = imresize( ...
        gradcamMap,[size(drImage,1),size(drImage,2)]);

    heatmapImage = ...
        ind2rgb(gray2ind(gradcamMap,256),jet(256));

    gradcamHeatmap = im2uint8(heatmapImage);

    baseD = im2double(drImage);
    heatD = im2double(gradcamHeatmap);

    alpha = 0.50;

    mask3 = repmat(gradcamMap,[1 1 3]);

    overlayD = ...
        baseD.*(1-alpha*mask3) + ...
        heatD.*(alpha*mask3);

    gradcamOverlay = ...
        im2uint8(min(max(overlayD,0),1));

    fprintf('Grad-CAM generated successfully.\n');

catch ME

    fprintf('Grad-CAM warning: %s\n',ME.message);
    fprintf('The rest of the integrated analysis will still be saved.\n');

end

imwrite(gradcamOverlay,gradcamOverlayPath);
imwrite(gradcamHeatmap,gradcamHeatmapPath);

%% ============================================================
% 15. HUMAN-IN-THE-LOOP REVIEW
% ============================================================
%
% This is the actual human review stage.
%
% The AI produces a recommendation, but the reviewer decides
% whether the result can be accepted or requires specialist review.
%
% For a real clinical deployment, this must be replaced by the
% approved clinical workflow and appropriate authentication,
% audit logging and medical governance.
%% ============================================================

fprintf('\n============================================================\n');
fprintf('HUMAN-IN-THE-LOOP REVIEW\n');
fprintf('============================================================\n');

if predictionSafety == "BLOCKED"

    suggestedReview = ...
        "AI result is blocked and requires human review.";

elseif qualityResult == "BORDERLINE"

    suggestedReview = ...
        "Image quality is borderline; human validation is recommended.";

elseif ismember(drPred,referableClasses)

    suggestedReview = ...
        "Referable DR detected; ophthalmologist validation is recommended.";

else

    suggestedReview = ...
        "AI result available; human validation remains recommended.";

end

fprintf('AI recommendation : %s\n',drPred);
fprintf('AI confidence     : %.2f%%\n',100*drConf);
fprintf('Review trigger    : %s\n',suggestedReview);

reviewPrompt = sprintf([ ...
    'TRUST-DR HUMAN REVIEW\n\n' ...
    'AI prediction: %s\n' ...
    'Confidence: %.2f%%\n' ...
    'CLAHE stability: %s\n' ...
    'Image quality: %s\n' ...
    'Gateway: %s\n\n' ...
    'Choose the human reviewer decision.'], ...
    drPred,100*drConf,claheStability, ...
    qualityResult,gatewayDetails.Decision);

reviewChoice = questdlg( ...
    reviewPrompt, ...
    'TRUST-DR Human-in-the-Loop Review', ...
    'Accept AI Result', ...
    'Override / Specialist Review', ...
    'Leave Pending', ...
    'Leave Pending');

switch reviewChoice

    case 'Accept AI Result'

        humanReviewDecision = "ACCEPTED_BY_HUMAN";
        humanReviewAction = ...
            "Human reviewer accepted the AI screening result.";

        reviewStatus = "HUMAN REVIEW COMPLETED";

    case 'Override / Specialist Review'

        humanReviewDecision = "OVERRIDDEN_BY_HUMAN";
        humanReviewAction = ...
            "Human reviewer rejected automated use of the AI result; specialist review required.";

        reviewStatus = "SPECIALIST REVIEW REQUIRED";

    otherwise

        humanReviewDecision = "PENDING_HUMAN_REVIEW";
        humanReviewAction = ...
            "Human reviewer did not finalize the case.";

        reviewStatus = "PENDING HUMAN REVIEW";

end

fprintf('\nHUMAN REVIEW DECISION : %s\n', ...
    humanReviewDecision);

fprintf('HUMAN REVIEW ACTION   : %s\n\n', ...
    humanReviewAction);

%% ============================================================
% 16. FINAL ANALYSIS FIGURE
% ============================================================

fprintf('Creating final analysis figure...\n');

fig = figure( ...
    'Name','TRUST-DR Integrated Analysis', ...
    'NumberTitle','off', ...
    'Color','w', ...
    'Units','normalized', ...
    'Position',[0.03 0.05 0.94 0.88], ...
    'Visible','on');

tiledlayout(fig,3,3, ...
    'Padding','compact', ...
    'TileSpacing','compact');

nexttile;
imshow(originalImage);
title('1. Original Fundus', ...
    'Color','black','Interpreter','none');

nexttile;
imshow(claheImage);
title('2. CLAHE Preprocessed', ...
    'Color','black','Interpreter','none');

nexttile;
imshow(vesselOverlay);
title(sprintf('3. Vessel Overlay (%.4f%%)',vesselArea), ...
    'Color','black');

nexttile;
imshow(lesionOverlay);
title(sprintf('4. Lesion Overlay (%.4f%%)', ...
    lesionForegroundArea), ...
    'Color','black');

nexttile;
imshow(combinedOverlay);
title('5. Combined Overlay', ...
    'Color','black');

nexttile;
imshow(gradcamOverlay);
title(sprintf('6. DR Grad-CAM: %s',drPred), ...
    'Color','black','Interpreter','none');

nexttile;
imshow(gradcamHeatmap);
title('7. Grad-CAM Heatmap', ...
    'Color','black');

nexttile;
axis off;

text(0.02,0.94,sprintf('DR: %s',drPred), ...
    'Color','black','FontSize',14,'FontWeight','bold');

text(0.02,0.84,sprintf('Confidence: %.2f%%',100*drConf), ...
    'Color','black','FontSize',12);

text(0.02,0.74,sprintf('CLAHE: %s',claheStability), ...
    'Color','black','FontSize',12);

text(0.02,0.64,sprintf('Quality: %s',qualityResult), ...
    'Color','black','FontSize',12);

text(0.02,0.54,sprintf('Gateway: %s',gatewayDetails.Decision), ...
    'Color','black','FontSize',12);

text(0.02,0.44,sprintf('Prediction Safety: %s',predictionSafety), ...
    'Color','black','FontSize',11);

text(0.02,0.34,sprintf('Vessel Area: %.4f%%',vesselArea), ...
    'Color','black','FontSize',11);

text(0.02,0.24,sprintf('Lesion Area: %.4f%%',lesionForegroundArea), ...
    'Color','black','FontSize',11);

text(0.02,0.14,sprintf('Human Review: %s',humanReviewDecision), ...
    'Color','black','FontSize',10, ...
    'Interpreter','none');

nexttile;
axis off;

text(0.02,0.94,'LESION COUNTS', ...
    'Color','black','FontSize',13, ...
    'FontWeight','bold');

y = 0.80;

for c = 1:6

    text(0.02,y, ...
        sprintf('%-12s %d', ...
        char(lesionNames(c)),lesionCounts(c)), ...
        'Color','black','FontSize',11, ...
        'Interpreter','none');

    y = y - 0.12;

end

drawnow;

try

    exportgraphics(fig,analysisPath,'Resolution',180);

catch

    saveas(fig,analysisPath);

end

%% ============================================================
% 17. LESION CSV
% ============================================================

lesionTable = table( ...
    lesionNames(:), ...
    lesionCounts(:), ...
    lesionAreas(:), ...
    'VariableNames', ...
    {'Class','Pixels','AreaPercent'});

writetable(lesionTable,csvPath);

%% ============================================================
% 18. SAVE ALL RESULTS
% ============================================================

save(matPath, ...
    'imagePath', ...
    'originalImage', ...
    'claheImage', ...
    'drPred', ...
    'drConf', ...
    'drScores', ...
    'classNamesDR', ...
    'drPredOriginal', ...
    'drConfOriginal', ...
    'drScoresOriginal', ...
    'drPredCLAHE', ...
    'drConfCLAHE', ...
    'drScoresCLAHE', ...
    'claheStability', ...
    'qualityResult', ...
    'qualityAction', ...
    'focusScore', ...
    'brightnessScore', ...
    'contrastScore', ...
    'gatewayPass', ...
    'gatewayScore', ...
    'gatewayReason', ...
    'gatewayDetails', ...
    'predictionSafety', ...
    'predictionSafetyReason', ...
    'referralResult', ...
    'referralAction', ...
    'reviewStatus', ...
    'reviewChoice', ...
    'humanReviewDecision', ...
    'humanReviewAction', ...
    'vesselMaskOriginal', ...
    'vesselPixels', ...
    'vesselArea', ...
    'lesionLabelsOriginal', ...
    'lesionNames', ...
    'lesionCounts', ...
    'lesionAreas', ...
    'lesionForegroundPixels', ...
    'lesionForegroundArea', ...
    'gradcamMap', ...
    'originalPath', ...
    'clahePath', ...
    'vesselMaskPath', ...
    'lesionMaskPath', ...
    'vesselOverlayPath', ...
    'lesionOverlayPath', ...
    'combinedOverlayPath', ...
    'analysisPath', ...
    'gradcamOverlayPath', ...
    'gradcamHeatmapPath', ...
    'csvPath', ...
    '-v7.3');

%% ============================================================
% 19. FINAL REPORT
% ============================================================

fprintf('\n============================================================\n');
fprintf('TRUST_DR FINAL INTEGRATED REPORT\n');
fprintf('============================================================\n');

fprintf('\nIMAGE QUALITY\n');
fprintf('------------------------------------------------------------\n');
fprintf('Focus Score      : %.6f\n',focusScore);
fprintf('Brightness Score : %.4f\n',brightnessScore);
fprintf('Contrast Score   : %.4f\n',contrastScore);
fprintf('Quality Result   : %s\n',qualityResult);

fprintf('\nDR SAFETY GATEWAY\n');
fprintf('------------------------------------------------------------\n');
fprintf('Gateway Score     : %.2f%%\n',100*gatewayScore);
fprintf('Gateway Decision  : %s\n',gatewayDetails.Decision);
fprintf('Gateway Reason    : %s\n',gatewayReason);

fprintf('\nDR CLASSIFICATION - APTOS\n');
fprintf('------------------------------------------------------------\n');
fprintf('Original Severity : %s\n',drPredOriginal);
fprintf('Original Conf.    : %.2f%%\n',100*drConfOriginal);
fprintf('CLAHE Severity    : %s\n',drPredCLAHE);
fprintf('CLAHE Conf.       : %.2f%%\n',100*drConfCLAHE);
fprintf('Stability         : %s\n',claheStability);
fprintf('Prediction Safety : %s\n',predictionSafety);
fprintf('Referral Result   : %s\n',referralResult);

fprintf('\nHUMAN-IN-THE-LOOP\n');
fprintf('------------------------------------------------------------\n');
fprintf('Review Decision   : %s\n',humanReviewDecision);
fprintf('Review Action     : %s\n',humanReviewAction);

fprintf('\nVESSEL SEGMENTATION - DRIVE V6\n');
fprintf('------------------------------------------------------------\n');
fprintf('Vessel Pixels     : %d\n',vesselPixels);
fprintf('Vessel Area       : %.4f%%\n',vesselArea);

fprintf('\nLESION SEGMENTATION - IDRiD\n');
fprintf('------------------------------------------------------------\n');

for c = 1:6

    fprintf('%-12s Pixels: %-10d Area: %.6f%%\n', ...
        char(lesionNames(c)), ...
        lesionCounts(c), ...
        lesionAreas(c));

end

fprintf('Total Foreground Lesion Area : %.6f%%\n', ...
    lesionForegroundArea);

fprintf('\nOUTPUT FILES\n');
fprintf('------------------------------------------------------------\n');
fprintf('Original       : %s\n',originalPath);
fprintf('CLAHE          : %s\n',clahePath);
fprintf('Vessel Mask    : %s\n',vesselMaskPath);
fprintf('Lesion Mask    : %s\n',lesionMaskPath);
fprintf('Vessel Overlay : %s\n',vesselOverlayPath);
fprintf('Lesion Overlay : %s\n',lesionOverlayPath);
fprintf('Combined       : %s\n',combinedOverlayPath);
fprintf('Analysis       : %s\n',analysisPath);
fprintf('Grad-CAM       : %s\n',gradcamOverlayPath);
fprintf('Grad-CAM Heat  : %s\n',gradcamHeatmapPath);
fprintf('MAT            : %s\n',matPath);
fprintf('CSV            : %s\n',csvPath);

fprintf('\n============================================================\n');
fprintf('TRUST_DR INTEGRATED INFERENCE COMPLETE\n');
fprintf('============================================================\n\n');

%% ============================================================
% LOCAL FUNCTIONS
%% ============================================================

function net = findNetworkVariable(S)

names = fieldnames(S);

preferred = { ...
    'trainedNet', ...
    'net', ...
    'trainedVesselNet', ...
    'trainedNetwork', ...
    'trainedLesionNet'};

for i = 1:numel(preferred)

    if isfield(S,preferred{i})

        candidate = S.(preferred{i});

        if isa(candidate,'DAGNetwork') || ...
                isa(candidate,'dlnetwork') || ...
                isa(candidate,'SeriesNetwork')

            net = candidate;
            return;

        end

    end

end

for i = 1:numel(names)

    candidate = S.(names{i});

    if isa(candidate,'DAGNetwork') || ...
            isa(candidate,'dlnetwork') || ...
            isa(candidate,'SeriesNetwork')

        net = candidate;
        return;

    end

end

error('No MATLAB neural network object was found in the MAT file.');

end

%% ============================================================

function sz = getInputSize(net,fallback)

sz = fallback;

try

    if isa(net,'DAGNetwork') || ...
            isa(net,'SeriesNetwork')

        s = net.Layers(1).InputSize;

        if numel(s) == 2
            sz = [s 1];
        else
            sz = s(1:3);
        end

    elseif isa(net,'dlnetwork')

        layers = net.Layers;

        for i = 1:numel(layers)

            if isa(layers(i),'nnet.cnn.layer.ImageInputLayer')

                s = layers(i).InputSize;

                if numel(s) == 2
                    sz = [s 1];
                else
                    sz = s(1:3);
                end

                return;

            end

        end

    end

catch

end

end

%% ============================================================

function names = getClassificationNames(net,n)

names = strings(n,1);

try

    if isa(net,'DAGNetwork') || ...
            isa(net,'SeriesNetwork')

        layers = net.Layers;

        for i = numel(layers):-1:1

            if isprop(layers(i),'Classes')

                cls = layers(i).Classes;

                if ~isempty(cls)

                    names = string(cls);
                    break;

                end

            end

        end

    end

catch

end

if numel(names) ~= n || all(strlength(names) == 0)

    defaultNames = ...
        ["Mild","Moderate","No_DR","Proliferative","Severe"];

    if n == 5

        names = defaultNames(:);

    else

        for i = 1:n
            names(i) = "Class_" + string(i-1);
        end

    end

end

names = names(:);

end

%% ============================================================

function scores = normalizeScores(rawScores)

scores = double(squeeze(rawScores));
scores = scores(:);

scores(~isfinite(scores)) = 0;

if isempty(scores)

    error('Classifier returned an empty score vector.');

end

if any(scores < 0) || ...
        sum(scores) <= 0 || ...
        abs(sum(scores)-1) > 1e-3

    ex = exp(scores-max(scores));
    scores = ex./max(sum(ex),eps);

else

    scores = scores./max(sum(scores),eps);

end

end

%% ============================================================
% R2026a-safe CLAHE
%% ============================================================

function out = applyFundusCLAHE(img)

img = im2uint8(img);

% rgb2lab returns CIE L*a*b*.
lab = rgb2lab(img);

% L* is in [0,100].
L = lab(:,:,1)/100;

% CLAHE on luminance only.
L = adapthisteq( ...
    L, ...
    'NumTiles',[8 8], ...
    'ClipLimit',0.01);

lab(:,:,1) = 100*L;

% IMPORTANT:
% In MATLAB R2026a the correct call is lab2rgb(lab).
% 'lab' is NOT a valid ColorSpace value.
out = lab2rgb(lab);

out = im2uint8( ...
    min(max(out,0),1));

end

%% ============================================================
% FUNDUS SAFETY GATEWAY
%% ============================================================

function [pass,score,reason,details] = ...
    runDRSafetyGateway(originalImage,claheImage,qualityResult)

I = im2double(originalImage);

G = rgb2gray(I);

R = I(:,:,1);
Gr = I(:,:,2);
B = I(:,:,3);

[H,W,~] = size(I);

[xx,yy] = meshgrid( ...
    linspace(-1,1,W), ...
    linspace(-1,1,H));

radius = sqrt(xx.^2 + yy.^2);

centerMask = radius < 0.72;
outerMask = radius > 0.88;

centerMean = mean(G(centerMask));
outerMean = mean(G(outerMask));

greenDominance = ...
    mean(Gr(:))-mean((R(:)+B(:))/2);

centerContrast = std(G(centerMask));

darkOuterFraction = ...
    mean(G(outerMask) < 0.12);

rimSeparation = centerMean-outerMean;

rgbRange = max(I(:))-min(I(:));

C = im2double(claheImage);

cG = rgb2gray(C);

claheCenterMean = ...
    mean(cG(centerMask));

claheChange = ...
    abs(claheCenterMean-centerMean);

s1 = clamp01((rimSeparation+0.02)/0.35);
s2 = clamp01(darkOuterFraction/0.45);
s3 = clamp01((centerContrast-0.04)/0.18);
s4 = clamp01((greenDominance+0.01)/0.12);
s5 = clamp01(1-claheChange/0.15);
s6 = clamp01(rgbRange/0.50);

score = ...
    0.25*s1 + ...
    0.20*s2 + ...
    0.20*s3 + ...
    0.15*s4 + ...
    0.10*s5 + ...
    0.10*s6;

% This gate is deliberately for obvious non-fundus inputs.
% It is NOT a disease/OOD classifier.
pass = ...
    (score >= 0.55) && ...
    qualityResult ~= "UNGRADABLE";

if qualityResult == "UNGRADABLE"

    reason = "Image quality gate failed.";
    decision = "BLOCKED";

elseif score < 0.55

    reason = ...
        "Input does not sufficiently resemble a retinal fundus photograph.";

    decision = "BLOCKED";

else

    reason = ...
        "Input passes available fundus-domain checks. " + ...
        "This does not exclude other retinal diseases.";

    decision = "PASSED";

end

details = struct;

details.Decision = decision;
details.CenterMean = centerMean;
details.OuterMean = outerMean;
details.RimSeparation = rimSeparation;
details.DarkOuterFraction = darkOuterFraction;
details.CenterContrast = centerContrast;
details.GreenDominance = greenDominance;
details.CLAHECenterChange = claheChange;
details.ComponentScores = [s1 s2 s3 s4 s5 s6];

end

%% ============================================================

function y = clamp01(x)

y = min(max(x,0),1);

end

%% ============================================================

function createGatewayBlockedFigure( ...
    img,qualityResult,score,reason,outPath)

f = figure( ...
    'Name','TRUST-DR Safety Gateway BLOCKED', ...
    'NumberTitle','off', ...
    'Color','w', ...
    'Position',[150 150 1100 650]);

imshow(img);

title( ...
    'TRUST-DR SAFETY GATEWAY: INFERENCE BLOCKED', ...
    'Color','black', ...
    'FontSize',18, ...
    'FontWeight','bold');

text( ...
    0.02,0.08, ...
    sprintf( ...
    'Fundus-likeness: %.2f%% | Quality: %s | %s', ...
    100*score,qualityResult,reason), ...
    'Units','normalized', ...
    'Color','black', ...
    'FontSize',13, ...
    'BackgroundColor','white', ...
    'Interpreter','none');

drawnow;

try
    exportgraphics(f,outPath,'Resolution',180);
catch
    saveas(f,outPath);
end

end

%% ============================================================

function out = makeOverlay(baseImage,mask,rgb,alpha)

base = im2double(baseImage);

color = zeros(size(base));

color(:,:,1) = rgb(1);
color(:,:,2) = rgb(2);
color(:,:,3) = rgb(3);

m = repmat(mask,[1 1 3]);

outD = ...
    base.*(1-alpha*m) + ...
    color.*(alpha*m);

out = im2uint8( ...
    min(max(outD,0),1));

end

%% ============================================================

function out = makeLesionOverlay(baseImage,labels)

base = im2double(baseImage);

color = zeros(size(base));

m = labels == 1;
color(:,:,1) = max(color(:,:,1),double(m));

m = labels == 2;
color(:,:,1) = max(color(:,:,1),double(m));
color(:,:,2) = max(color(:,:,2),double(m));

m = labels == 3;
color(:,:,2) = max(color(:,:,2),double(m));

m = labels == 4;
color(:,:,3) = max(color(:,:,3),double(m));

m = labels == 5;
color(:,:,1) = max(color(:,:,1),double(m));
color(:,:,3) = max(color(:,:,3),double(m));

mask = labels > 0;

m3 = repmat(mask,[1 1 3]);

alpha = 0.70;

outD = ...
    base.*(1-alpha*m3) + ...
    color.*(alpha*m3);

out = im2uint8( ...
    min(max(outD,0),1));

end

%% ============================================================

function out = makeCombinedOverlay( ...
    baseImage,vesselMask,lesionLabels)

base = im2double(baseImage);

% Vessel = cyan.
vesselColor = zeros(size(base));

vesselColor(:,:,2) = 1;
vesselColor(:,:,3) = 1;

vm = repmat(vesselMask,[1 1 3]);

outD = ...
    base.*(1-0.35*vm) + ...
    vesselColor.*(0.35*vm);

% Lesion colours.
lesionColor = zeros(size(base));

m = lesionLabels == 1;
lesionColor(:,:,1) = ...
    max(lesionColor(:,:,1),double(m));

m = lesionLabels == 2;
lesionColor(:,:,1) = ...
    max(lesionColor(:,:,1),double(m));
lesionColor(:,:,2) = ...
    max(lesionColor(:,:,2),double(m));

m = lesionLabels == 3;
lesionColor(:,:,2) = ...
    max(lesionColor(:,:,2),double(m));

m = lesionLabels == 4;
lesionColor(:,:,3) = ...
    max(lesionColor(:,:,3),double(m));

m = lesionLabels == 5;
lesionColor(:,:,1) = ...
    max(lesionColor(:,:,1),double(m));
lesionColor(:,:,3) = ...
    max(lesionColor(:,:,3),double(m));

lm = repmat(lesionLabels > 0,[1 1 3]);

outD = ...
    outD.*(1-0.70*lm) + ...
    lesionColor.*(0.70*lm);

out = im2uint8( ...
    min(max(outD,0),1));

end
