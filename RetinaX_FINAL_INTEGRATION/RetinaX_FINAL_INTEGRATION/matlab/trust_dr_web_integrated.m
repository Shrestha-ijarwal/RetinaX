function trust_dr_web_integrated(imagePath, outputDir)
% RETINAX_WEB_INTEGRATED
% Web-callable RetinaX inference wrapper.
% Derived from the supplied RetinaX_Integrated_Inference.m.
% No uigetfile, questdlg, or visible figures are used.

if ~isfolder(outputDir), mkdir(outputDir); end

projectFolder = 'C:\Users\sijar\Downloads\RetinaX';
drPath = fullfile(projectFolder,'models','trained_DR_model_v3.mat');
vesselPath = fullfile(projectFolder,'models','trained_vessel_segmentation_v4_1.mat');
lesionPath = fullfile(projectFolder,'datasets','A. Segmentation','A. Segmentation','IDRiD_UNet_BalancedWeighted_Augmented_256.mat');

assert(isfile(drPath),'DR model not found: %s',drPath);
assert(isfile(vesselPath),'Vessel model not found: %s',vesselPath);
assert(isfile(lesionPath),'Lesion model not found: %s',lesionPath);

D=load(drPath); drNet=findNetworkVariable(D); drInputSize=getInputSize(drNet,[224 224 3]);
V=load(vesselPath); vesselNet=findNetworkVariable(V); vesselInputSize=getInputSize(vesselNet,[256 256 3]);
L=load(lesionPath); lesionNet=findNetworkVariable(L); lesionInputSize=[256 256 3];
lesionNames=["Background","MA","HE","EX","SE","OD"];

originalImage=imread(imagePath);
if ndims(originalImage)==2, originalImage=repmat(originalImage,[1 1 3]); end
if size(originalImage,3)>3, originalImage=originalImage(:,:,1:3); end
originalImage=im2uint8(originalImage); [H,W,~]=size(originalImage);
claheImage=applyFundusCLAHE(originalImage);

% Image quality - exact thresholds from integrated pipeline.
grayDouble=im2double(rgb2gray(originalImage));
lapImage=imfilter(grayDouble,fspecial('laplacian',0.2),'replicate');
focusScore=var(lapImage(:)); brightnessScore=mean(grayDouble(:)); contrastScore=std(grayDouble(:));
FOCUS_BAD=0.00003; BRIGHTNESS_LOW=0.05; BRIGHTNESS_HIGH=0.90; CONTRAST_BAD=0.03;
if focusScore<FOCUS_BAD || brightnessScore<BRIGHTNESS_LOW || brightnessScore>BRIGHTNESS_HIGH || contrastScore<CONTRAST_BAD
 qualityResult="UNGRADABLE"; qualityAction="Image quality insufficient. Please recapture the fundus image.";
elseif focusScore<2*FOCUS_BAD || contrastScore<1.5*CONTRAST_BAD
 qualityResult="BORDERLINE"; qualityAction="Borderline image quality. Human validation recommended.";
else
 qualityResult="ACCEPTABLE"; qualityAction="Image quality acceptable for AI screening.";
end

[gatewayPass,gatewayScore,gatewayReason,gatewayDetails]=runDRSafetyGateway(originalImage,claheImage,qualityResult);

R=struct;
R.quality_result=char(qualityResult); R.quality_action=char(qualityAction);
R.focus_score=focusScore; R.brightness_score=brightnessScore; R.contrast_score=contrastScore;
R.gateway_pass=logical(gatewayPass); R.gateway_score=gatewayScore; R.gateway_reason=char(gatewayReason);
R.gateway_decision=char(gatewayDetails.Decision);

imwrite(originalImage,fullfile(outputDir,'original.png'));
imwrite(claheImage,fullfile(outputDir,'clahe.png'));
R.original_file='original.png'; R.clahe_file='clahe.png';

if ~gatewayPass
 R.dr_prediction='NOT_RUN'; R.dr_confidence=0; R.clahe_stability='NOT_RUN';
 R.prediction_safety='BLOCKED'; R.prediction_safety_reason='Fundus-domain safety gateway blocked inference.';
 R.referral_result='AUTOMATED RESULT BLOCKED'; R.referral_action=char(gatewayReason);
 R.vessel_area_percent=0; R.lesion_foreground_area_percent=0;
 R.vessel_overlay_file=''; R.lesion_overlay_file=''; R.combined_overlay_file=''; R.gradcam_overlay_file=''; R.gradcam_heatmap_file='';
 R.lesion_names=cellstr(lesionNames); R.lesion_counts=zeros(6,1); R.lesion_areas=zeros(6,1);
 writeJSON(R,fullfile(outputDir,'result.json')); return;
end

% DR classification: original + CLAHE stability.
drImage=imresize(originalImage,drInputSize(1:2)); [~,raw]=classify(drNet,drImage);
drScoresOriginal=normalizeScores(raw); classNamesDR=getClassificationNames(drNet,numel(drScoresOriginal));
[drConfOriginal,idx]=max(drScoresOriginal); drPredOriginal=string(classNamesDR(idx));
claheDRImage=imresize(claheImage,drInputSize(1:2)); [~,raw2]=classify(drNet,claheDRImage);
drScoresCLAHE=normalizeScores(raw2); [drConfCLAHE,idx2]=max(drScoresCLAHE); drPredCLAHE=string(classNamesDR(idx2));
if drPredOriginal==drPredCLAHE, claheStability="STABLE"; else, claheStability="UNSTABLE"; end
drPred=drPredOriginal; drConf=drConfOriginal;

POST_MIN_CONFIDENCE=0.60;
if claheStability=="UNSTABLE"
 predictionSafety="BLOCKED"; predictionSafetyReason="Original and CLAHE predictions disagree. Human review required; automated DR result blocked.";
elseif drConf<POST_MIN_CONFIDENCE
 predictionSafety="BLOCKED"; predictionSafetyReason="DR classifier confidence is below the safety threshold.";
else
 predictionSafety="PASSED"; predictionSafetyReason="Original/CLAHE prediction is stable and confidence meets the safety threshold.";
end

referableClasses=["Moderate","Severe","Proliferative"];
if ismember(drPred,referableClasses)
 referralResult="REFERABLE DR"; referralAction="Priority ophthalmologist evaluation recommended.";
else
 referralResult="NON-REFERABLE DR"; referralAction="Routine screening follow-up recommended.";
end
if predictionSafety=="BLOCKED"
 referralResult="AUTOMATED RESULT BLOCKED"; referralAction="Do not use AI result for automated referral. Human assessment required.";
end

% Vessel segmentation - DRIVE.
vesselInput=imresize(claheImage,vesselInputSize(1:2));
predictedVesselLabels=semanticseg(vesselInput,vesselNet,Classes=["background","vessel"]);
vesselMask256=predictedVesselLabels=="vessel";
vesselMaskOriginal=imresize(vesselMask256,[H W],'nearest');
vesselArea=100*nnz(vesselMaskOriginal)/max(H*W,1);

% Lesion segmentation - IDRiD.
lesionInput=im2single(imresize(originalImage,lesionInputSize(1:2)));
lesionLabels256=semanticseg(lesionInput,lesionNet,Classes=lesionNames);
lesionLabels256Numeric=zeros(size(lesionLabels256),'uint8');
for c=1:numel(lesionNames), lesionLabels256Numeric(lesionLabels256==lesionNames(c))=uint8(c-1); end
lesionLabelsOriginal=imresize(lesionLabels256Numeric,[H W],'nearest');
lesionCounts=zeros(6,1); lesionAreas=zeros(6,1);
for c=0:5
 lesionCounts(c+1)=nnz(lesionLabelsOriginal==c);
 lesionAreas(c+1)=100*lesionCounts(c+1)/max(H*W,1);
end
lesionForegroundArea=100*sum(lesionCounts(2:end))/max(H*W,1);

% Output overlays.
vesselOverlay=makeOverlay(originalImage,vesselMaskOriginal,[0 1 0],0.45);
lesionOverlay=makeLesionOverlay(originalImage,lesionLabelsOriginal);
combinedOverlay=makeCombinedOverlay(originalImage,vesselMaskOriginal,lesionLabelsOriginal);
imwrite(uint8(repmat(vesselMaskOriginal,[1 1 3]))*255,fullfile(outputDir,'vessel_mask.png'));
imwrite(vesselOverlay,fullfile(outputDir,'vessel_overlay.png'));
imwrite(lesionOverlay,fullfile(outputDir,'lesion_overlay.png'));
imwrite(combinedOverlay,fullfile(outputDir,'combined_overlay.png'));

% Grad-CAM.
gradcamOverlay=drImage; gradcamHeatmap=zeros(size(drImage),'uint8');
try
 gm=gradCAM(drNet,drImage,drPred); gm=mat2gray(double(squeeze(gm))); gm=imresize(gm,[size(drImage,1),size(drImage,2)]);
 gradcamHeatmap=im2uint8(ind2rgb(gray2ind(gm,256),jet(256)));
 baseD=im2double(drImage); heatD=im2double(gradcamHeatmap); m3=repmat(gm,[1 1 3]); alpha=0.50;
 gradcamOverlay=im2uint8(min(max(baseD.*(1-alpha*m3)+heatD.*(alpha*m3),0),1));
catch ME
 warning('Grad-CAM warning: %s',ME.message);
end
imwrite(gradcamOverlay,fullfile(outputDir,'gradcam_overlay.png'));
imwrite(gradcamHeatmap,fullfile(outputDir,'gradcam_heatmap.png'));

R.dr_prediction=char(drPred); R.dr_confidence=drConf;
R.dr_prediction_clahe=char(drPredCLAHE); R.dr_confidence_clahe=drConfCLAHE;
R.clahe_stability=char(claheStability); R.prediction_safety=char(predictionSafety);
R.prediction_safety_reason=char(predictionSafetyReason); R.referral_result=char(referralResult); R.referral_action=char(referralAction);
R.vessel_area_percent=vesselArea; R.lesion_foreground_area_percent=lesionForegroundArea;
R.lesion_names=cellstr(lesionNames); R.lesion_counts=lesionCounts; R.lesion_areas=lesionAreas;
R.vessel_overlay_file='vessel_overlay.png'; R.lesion_overlay_file='lesion_overlay.png';
R.combined_overlay_file='combined_overlay.png'; R.gradcam_overlay_file='gradcam_overlay.png'; R.gradcam_heatmap_file='gradcam_heatmap.png';
writeJSON(R,fullfile(outputDir,'result.json'));
end

function writeJSON(data,path)
txt=jsonencode(data,PrettyPrint=true); fid=fopen(path,'w'); assert(fid>=0,'Cannot write JSON: %s',path);
cleanup=onCleanup(@() fclose(fid)); fwrite(fid,txt,'char');
end


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
    'Name','RetinaX Safety Gateway BLOCKED', ...
    'NumberTitle','off', ...
    'Color','w', ...
    'Position',[150 150 1100 650]);

imshow(img);

title( ...
    'RetinaX SAFETY GATEWAY: INFERENCE BLOCKED', ...
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
