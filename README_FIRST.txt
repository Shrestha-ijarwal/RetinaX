RETINAX — COMPLETE WEB INTEGRATION (frontend + landmark output)

Includes: server.py, requirements.txt, matlab/trust_dr_web_integrated.m,
frontend/index.html, frontend/app.js, frontend/styles.css.

IMPORTANT: This package does NOT contain the four landmark MATLAB functions,
trained model files or datasets. Keep your EXISTING files in RetinaX/scripts:
localizeOpticDisc.m, refineOpticDiscCenter.m, localizeFovea.m, runRetinalLandmarks.m.
The web wrapper uses the first three directly, matching the previously tested
standalone integrated MATLAB script. It does not assume an undocumented
runRetinalLandmarks return format.

INSTALL
1. Back up your existing RetinaX/server.py, matlab/trust_dr_web_integrated.m,
   and frontend/ folder.
2. Copy the included files into matching locations under your RetinaX root.
   Do not delete your models/, datasets/, scripts/ or runtime/ folders.
3. In matlab/trust_dr_web_integrated.m, verify projectFolder points to your
   real project path (currently C:\Users\sijar\Downloads\RetinaX).
4. Restart Flask, hard-refresh the browser, upload a fundus image and go to
   Step 05: Explainability & Evidence. Check the landmarks card.
5. If no landmark PNG appears, check MATLAB warnings in the Flask terminal
   and confirm all landmark functions are on the MATLAB path.

BEHAVIOR
- Image quality/fundus gateway runs first; blocked cases do not run landmarks.
- Landmark errors are caught: they do not change DR prediction/referral.
- A successful attempt writes runtime/outputs/<case>/landmarks.png and
  result.json landmarks_file/landmarks_status/optic_disc_status/fovea_status.
- Flask adds landmarks_url. Frontend shows image or an unavailable message.
- Language selector: English/Hindi/Tamil. Theme toggle: light/dark.
- Original upper-left T icon removed.
- Landmark positions are experimental and NOT clinically validated.
- No dedicated neovascularization detector is added or claimed.

TEST LIMITATION: MATLAB and trained models are unavailable in this environment;
actual end-to-end inference must be tested on your Windows installation.
