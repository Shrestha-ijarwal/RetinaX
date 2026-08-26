%% ============================================================
% TRUST-DR
% HUMAN-IN-THE-LOOP OPHTHALMOLOGIST REVIEW
%% ============================================================

clc;

fprintf('\n');
fprintf('===========================================\n');
fprintf('     TRUST-DR HUMAN-IN-THE-LOOP REVIEW\n');
fprintf('===========================================\n\n');


%% ------------------------------------------------------------
% CHECK REQUIRED VARIABLES
%% ------------------------------------------------------------

requiredVars = { ...
    'originalImage', ...
    'processedImage', ...
    'predictedLabel', ...
    'confidence', ...
    'referralResult', ...
    'reviewStatus', ...
    'gradcamOverlay', ...
    'qualityResult'};

for i = 1:length(requiredVars)

    if ~exist(requiredVars{i}, 'var')

        error( ...
            'Required variable "%s" not found. Run TRUST_DR_MAIN first.', ...
            requiredVars{i});

    end

end


%% ------------------------------------------------------------
% CREATE REVIEW WINDOW
%% ------------------------------------------------------------

reviewFigure = figure( ...
    'Name', 'TRUST-DR Ophthalmologist Review', ...
    'Color', [0.12 0.12 0.12], ...
    'Position', [100 100 1500 850]);


%% ============================================================
% PANEL 1 — ORIGINAL IMAGE
%% ============================================================

subplot(2,3,1);

imshow(originalImage);

title( ...
    'PATIENT FUNDUS IMAGE', ...
    'Color', 'w', ...
    'FontSize', 15, ...
    'FontWeight', 'bold');


%% ============================================================
% PANEL 2 — AI PROCESSED IMAGE
%% ============================================================

subplot(2,3,2);

imshow(processedImage);

title( ...
    sprintf('IMAGE QUALITY: %s', qualityResult), ...
    'Color', 'w', ...
    'FontSize', 15, ...
    'FontWeight', 'bold');


%% ============================================================
% PANEL 3 — GRAD-CAM
%% ============================================================

subplot(2,3,3);

imshow(gradcamOverlay);

title( ...
    'AI ATTENTION MAP — GRAD-CAM', ...
    'Color', 'w', ...
    'FontSize', 15, ...
    'FontWeight', 'bold');


%% ============================================================
% PANEL 4 — AI RESULT
%% ============================================================

subplot(2,3,4);

axis off;


text(0.05, 0.90, ...
    'AI SCREENING RESULT', ...
    'Color', 'w', ...
    'FontSize', 20, ...
    'FontWeight', 'bold');


text(0.05, 0.70, ...
    sprintf('DR Grade: %s', predictedLabel), ...
    'Color', 'w', ...
    'FontSize', 17);


text(0.05, 0.55, ...
    sprintf('Confidence: %.2f%%', confidence), ...
    'Color', 'w', ...
    'FontSize', 17);


text(0.05, 0.40, ...
    sprintf('Screening: %s', referralResult), ...
    'Color', 'w', ...
    'FontSize', 17, ...
    'FontWeight', 'bold');


text(0.05, 0.20, ...
    'AI recommendation is NOT the final diagnosis.', ...
    'Color', [1 0.8 0.3], ...
    'FontSize', 13);


%% ============================================================
% PANEL 5 — HUMAN REVIEW INSTRUCTIONS
%% ============================================================

subplot(2,3,5);

axis off;


text(0.05, 0.90, ...
    'OPHTHALMOLOGIST REVIEW', ...
    'Color', 'w', ...
    'FontSize', 20, ...
    'FontWeight', 'bold');


text(0.05, 0.68, ...
    '1. Review fundus image', ...
    'Color', 'w', ...
    'FontSize', 16);


text(0.05, 0.53, ...
    '2. Examine Grad-CAM evidence', ...
    'Color', 'w', ...
    'FontSize', 16);


text(0.05, 0.38, ...
    '3. Validate or override AI decision', ...
    'Color', 'w', ...
    'FontSize', 16);


text(0.05, 0.18, ...
    'Final decision remains with specialist.', ...
    'Color', [0.4 1 0.5], ...
    'FontSize', 14);


%% ============================================================
% PANEL 6 — HUMAN DECISION
%% ============================================================

subplot(2,3,6);

axis off;


text(0.05, 0.90, ...
    'FINAL HUMAN DECISION', ...
    'Color', 'w', ...
    'FontSize', 20, ...
    'FontWeight', 'bold');


%% ============================================================
% ASK OPHTHALMOLOGIST FOR DECISION
%% ============================================================

choices = { ...
    'ACCEPT AI RESULT', ...
    'MODIFY DIAGNOSIS', ...
    'REQUEST IMAGE RECAPTURE'};


[selectedIndex, okPressed] = listdlg( ...
    'PromptString', ...
    'Select final ophthalmologist decision:', ...
    'SelectionMode', ...
    'single', ...
    'ListString', ...
    choices, ...
    'ListSize', ...
    [300 180]);


%% ============================================================
% HANDLE DECISION
%% ============================================================

if okPressed == 0

    finalDecision = "REVIEW PENDING";

    finalGrade = predictedLabel;

    reviewComment = ...
        "Specialist review has not been completed.";

elseif selectedIndex == 1

    finalDecision = "AI RESULT ACCEPTED";

    finalGrade = predictedLabel;

    reviewComment = ...
        "Ophthalmologist validated the AI screening result.";

elseif selectedIndex == 2

    %% Ask for corrected DR grade

    grades = { ...
        'No_DR', ...
        'Mild', ...
        'Moderate', ...
        'Severe', ...
        'Proliferative'};


    [gradeIndex, gradeOK] = listdlg( ...
        'PromptString', ...
        'Select corrected DR grade:', ...
        'SelectionMode', ...
        'single', ...
        'ListString', ...
        grades, ...
        'ListSize', ...
        [300 220]);


    if gradeOK

        finalGrade = string(grades{gradeIndex});

        finalDecision = "AI RESULT MODIFIED";

        reviewComment = ...
            "Ophthalmologist overrode the AI prediction.";

    else

        finalGrade = predictedLabel;

        finalDecision = "REVIEW PENDING";

        reviewComment = ...
            "Diagnosis modification cancelled.";

    end


elseif selectedIndex == 3

    finalDecision = "IMAGE RECAPTURE REQUESTED";

    finalGrade = "NOT GRADED";

    reviewComment = ...
        "Specialist requested a new fundus image.";

end


%% ============================================================
% DISPLAY FINAL HUMAN DECISION
%% ============================================================

text(0.05, 0.65, ...
    sprintf('Status: %s', finalDecision), ...
    'Color', [0.4 1 0.5], ...
    'FontSize', 16, ...
    'FontWeight', 'bold');


text(0.05, 0.45, ...
    sprintf('Final Grade: %s', finalGrade), ...
    'Color', 'w', ...
    'FontSize', 17);


text(0.05, 0.20, ...
    reviewComment, ...
    'Color', [0.8 0.8 0.8], ...
    'FontSize', 13);


%% ============================================================
% SAVE HUMAN REVIEW
%% ============================================================

resultsFolder = ...
    'C:\Users\sijar\Downloads\TRUST_DR\results';


if ~exist(resultsFolder, 'dir')

    mkdir(resultsFolder);

end


timestamp = datestr(now, 'yyyymmdd_HHMMSS');


humanReviewPath = fullfile( ...
    resultsFolder, ...
    ['human_review_' timestamp '.txt']);


fid = fopen(humanReviewPath, 'w');


fprintf(fid, ...
    '===========================================\n');

fprintf(fid, ...
    'TRUST-DR HUMAN-IN-THE-LOOP REVIEW\n');

fprintf(fid, ...
    '===========================================\n\n');


fprintf(fid, ...
    'AI Prediction: %s\n', predictedLabel);

fprintf(fid, ...
    'AI Confidence: %.2f%%\n', confidence);

fprintf(fid, ...
    'Screening Result: %s\n', referralResult);

fprintf(fid, ...
    'Image Quality: %s\n\n', qualityResult);


fprintf(fid, ...
    'FINAL HUMAN DECISION: %s\n', finalDecision);

fprintf(fid, ...
    'FINAL DR GRADE: %s\n', finalGrade);

fprintf(fid, ...
    'COMMENT: %s\n', reviewComment);


fclose(fid);


%% ============================================================
% FINAL CONSOLE OUTPUT
%% ============================================================

fprintf('\n');
fprintf('===========================================\n');
fprintf('       HUMAN REVIEW COMPLETED\n');
fprintf('===========================================\n');

fprintf('AI Prediction   : %s\n', predictedLabel);

fprintf('Final Decision  : %s\n', finalDecision);

fprintf('Final DR Grade  : %s\n', finalGrade);

fprintf('Review Saved At :\n%s\n', humanReviewPath);

fprintf('===========================================\n');