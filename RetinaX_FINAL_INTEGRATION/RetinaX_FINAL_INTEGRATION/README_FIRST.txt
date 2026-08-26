RetinaX FINAL WEB + MATLAB INTEGRATION
======================================

This version connects the supplied integrated RetinaX logic to the website.

CONNECTED PIPELINE
------------------
Upload
→ Image quality assessment
→ Fundus-domain safety gateway
→ if BLOCKED: stop inference
→ if PASSED: original + CLAHE DR classification
→ stability + confidence safety check
→ DRIVE vessel segmentation
→ IDRiD lesion segmentation
→ Grad-CAM
→ referral logic
→ browser-based human review
→ report

IMPORTANT CLINICAL WORDING
--------------------------
The gateway is a FUNDUS-DOMAIN heuristic. It is not a DR-vs-other-retinal-disease classifier.
It rejects obvious non-fundus / unsuitable inputs but does not exclude AMD, glaucoma,
hypertensive retinopathy, etc.

SETUP
-----
1. Open matlab/trust_dr_web_integrated.m.
2. Confirm this line matches your project:
   projectFolder = 'C:\Users\sijar\Downloads\RetinaX';

The model paths are taken directly from your integrated script:
- models\trained_DR_model_v3.mat
- models\trained_vessel_segmentation_v4_1.mat
- datasets\A. Segmentation\A. Segmentation\IDRiD_UNet_BalancedWeighted_Augmented_256.mat

3. Install Flask:
   py -m pip install -r requirements.txt

4. Check MATLAB:
   matlab -batch "disp('MATLAB OK')"

If MATLAB is not on PATH in PowerShell:
   $env:MATLAB_EXE="C:\Program Files\MATLAB\R2026a\bin\matlab.exe"

5. Run:
   py server.py

6. Open:
   http://127.0.0.1:5000

Do NOT double-click index.html for the connected version.

HUMAN REVIEW
------------
The MATLAB questdlg has intentionally been removed from the web path.
Accept / Override / Recapture is handled in the browser and saved to:
runtime\outputs\<case_id>\human_review.json


VISIBLE CLAHE STEP
------------------
RetinaX now includes a dedicated preprocessing screen showing:
- Original fundus image
- CLAHE preprocessed image
- Original DR prediction
- CLAHE DR prediction
- Original confidence
- CLAHE confidence
- STABLE / UNSTABLE comparison

This uses the same original-vs-CLAHE safety logic from the MATLAB pipeline.
