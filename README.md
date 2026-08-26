# RetinaX

### AI-Assisted Diabetic Retinopathy Screening with Explainability, Safety Checks, and Human-in-the-Loop Review

RetinaX is an AI-assisted diabetic retinopathy (DR) screening and decision-support system designed with rural and resource-constrained healthcare settings in mind.

Instead of providing only a disease prediction, RetinaX combines **image-quality assessment, DR severity classification, retinal vessel segmentation, lesion analysis, explainable AI, safety checks, referral logic, and human review** within a single screening workflow.

> **RetinaX is an academic/research prototype and is not intended to replace diagnosis or clinical judgment by a qualified ophthalmologist.**

---

## Problem Statement

Diabetic retinopathy is a major complication of diabetes and can lead to preventable vision loss when it is not detected early.

In rural and underserved regions, regular retinal screening can be difficult because of limited access to ophthalmologists, specialist centres, and screening infrastructure.

RetinaX explores how AI-assisted retinal image analysis can support the **early identification and referral of patients who may require specialist assessment**, while maintaining safety mechanisms and human oversight.

---

##  Proposed Solution

RetinaX accepts a retinal fundus image and processes it through a multi-stage AI pipeline.

The system first determines whether the image is suitable for automated analysis. Acceptable images undergo preprocessing and DR severity classification using a **ResNet-50 CNN with transfer learning**.

Additional **U-Net-based vessel and lesion segmentation** provides structural retinal evidence, while **Grad-CAM** visualizes regions that influenced the classifier's prediction.

Confidence checks, original-vs-CLAHE prediction stability, image-quality information, and referral logic are then used to support the final screening decision. Cases requiring additional attention can be escalated for human review.

---

##  System Workflow


Fundus Image Acquisition
          │
          ▼
Image Quality Assessment
+ Fundus-Domain Safety Gateway
          │
     ┌────┴────┐
     │         │
   PASS      BLOCK
     │         └──► Recapture / Human Assessment
     ▼
CLAHE Preprocessing
          │
          ▼
ResNet-50 DR Severity Classification
          │
          ▼
 ┌────────┼─────────┐
 │        │         │
 ▼        ▼         ▼
Vessel   Lesion   Grad-CAM
U-Net    U-Net      XAI
 │        │         │
 └────────┼─────────┘
          ▼
Safety + Referral Logic
          │
          ▼
Human-in-the-Loop Review
          │
          ▼
RetinaX Screening Report

##  AI Pipeline

### 1. Image Quality & Input Safety

Before running DR inference, RetinaX evaluates whether the input image is suitable for retinal analysis.

The current prototype considers:

- Focus / blur
- Brightness
- Contrast
- Gradability
- Fundus-domain validity

Images that fail the safety checks can be blocked from automated inference and directed toward recapture or human assessment.

The fundus-domain gateway is an **input-safety heuristic** and should not be interpreted as a classifier capable of excluding other retinal diseases.

---

### 2. Image Preprocessing

RetinaX uses **Contrast Limited Adaptive Histogram Equalization (CLAHE)** to improve retinal contrast.

The preprocessing pipeline includes operations such as:

- CLAHE enhancement
- Resizing
- Standardization / normalization
- Image augmentation during training

The system also compares predictions from the original and CLAHE-enhanced images as an additional stability check.

---

### 3. DR Severity Classification

The primary DR classifier uses:

**ResNet-50 CNN + Transfer Learning**

The network predicts five diabetic retinopathy severity classes:

| Class | DR Severity |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

The pretrained ResNet-50 classification head was modified for five-class retinal image classification.

Training includes:

- Transfer learning
- Data augmentation
- Class weighting
- Adam optimization
- Learning-rate decay
- Validation-based monitoring

---

### 4. Retinal Vessel Segmentation

RetinaX uses a **U-Net-based segmentation network** to identify retinal blood vessels.

The vessel module provides:

- Vessel segmentation mask
- Vessel overlay
- Vessel-area estimation
- Structural retinal evidence

**Dataset:** DRIVE

---

### 5. DR Lesion Analysis

A separate **U-Net-based segmentation model** is used to identify retinal lesion regions.

The lesion-analysis module provides visual evidence associated with structures such as:

- Microaneurysms (MA)
- Hemorrhages (HE)
- Hard Exudates (EX)
- Soft Exudates (SE)

The implementation may also represent additional annotated structures available in the segmentation model.

**Dataset:** IDRiD

---

##  Explainable AI

RetinaX incorporates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize regions that influenced the ResNet-50 DR classification.

The interface can present:

- Original fundus image
- Grad-CAM heatmap
- Grad-CAM overlay
- Vessel segmentation
- Lesion segmentation
- Combined structural evidence

Grad-CAM is treated as **model explainability**, not as independent clinical proof that a highlighted region represents a particular pathology.

---

##  Safety & Decision Support

RetinaX includes multiple checks around the core AI models.

These include:

- Image-quality assessment
- Fundus-domain input validation
- Prediction confidence
- Original vs. CLAHE prediction stability
- Low-confidence detection
- Referable-DR logic
- Inference blocking for safety-triggered cases
- Human-review escalation

In the current workflow, **Moderate, Severe, and Proliferative DR** predictions are treated as referable DR categories.

---

## Human-in-the-Loop

RetinaX is designed as a decision-support system rather than a replacement for ophthalmologists.

The human-review workflow allows a reviewer to:

- Review the original fundus image
- Inspect the AI prediction
- Inspect confidence information
- View Grad-CAM
- Review vessel and lesion evidence
- Accept the AI screening result
- Override the result
- Request image recapture



##  Rural Deployment Concept

RetinaX is designed around a possible rural tele-ophthalmology workflow:


Rural PHC / Screening Camp
          │
          ▼
     RetinaX Screening
          │
          ▼
Compact Screening Results
          │
          ▼
Low-Bandwidth Communication
          │
          ▼
District / Specialist Centre
          │
          ▼
Ophthalmologist Review
          │
          ▼
Referral / Follow-up


The goal is to perform initial screening closer to the patient while allowing cases requiring specialist attention to be escalated through tele-ophthalmology.

---

##  Tech Stack

### Programming Languages

- MATLAB
- Python
- HTML5
- CSS3
- JavaScript

### AI / Deep Learning

- ResNet-50
- Convolutional Neural Networks (CNN)
- Transfer Learning
- U-Net
- DAGNetwork / MATLAB layer graphs
- dlnetwork

### Image Processing

- CLAHE
- Image enhancement
- Image resizing
- Normalization
- Data augmentation

### Explainable AI

- Grad-CAM
- Attention heatmaps
- Visual evidence overlays

### MATLAB

- MATLAB R2026a
- Deep Learning Toolbox
- Image Processing Toolbox

### Web Integration

- Python
- Flask
- JSON
- HTML/CSS/JavaScript frontend

---

## 📊 Datasets

| Dataset | Purpose |
|---|---|
| **APTOS 2019 Blindness Detection** | DR severity classification |
| **DRIVE** | Retinal vessel segmentation |
| **IDRiD** | Retinal lesion segmentation |
| **Messidor-2*** | External validation |



## ⚙️ Running the Prototype

### Requirements

- MATLAB R2026a
- MATLAB Deep Learning Toolbox
- MATLAB Image Processing Toolbox
- Python 3.x
- Flask
- Required trained `.mat` models

### Install Python dependencies


pip install -r requirements.txt


### Start the RetinaX backend


python server.py


Then open:


http://127.0.0.1:5000


The Flask backend connects the web interface with the MATLAB inference pipeline.

---

## Evaluation

RetinaX evaluates different modules independently using appropriate metrics.

### DR Classification

- Accuracy
- Confusion Matrix
- Referable DR Sensitivity
- Referable DR Specificity

### Vessel / Lesion Segmentation

- Pixel Accuracy
- Precision
- Recall
- Dice Score
- Intersection over Union (IoU)

### Safety Evaluation

- Image-quality status
- Prediction confidence
- Original-vs-CLAHE stability
- Safety-gateway outcome

> Add the final experimentally obtained values here rather than estimated or illustrative results.

---

##  RetinaX Interface

The RetinaX web interface provides a visual workflow for:

1. Fundus image upload
2. Image-quality and safety analysis
3. Original vs. CLAHE visualization
4. DR severity prediction
5. Vessel and lesion evidence
6. Grad-CAM explainability
7. Safety/referral decision
8. Human review
9. Final screening report


## Key Features

- Five-class diabetic retinopathy severity classification
- ResNet-50 transfer learning
- Automated image-quality assessment
- Fundus-domain safety gateway
- CLAHE preprocessing
- Original-vs-CLAHE stability analysis
- Retinal vessel segmentation
- DR lesion segmentation
- Grad-CAM explainability
- Confidence-aware safety logic
- Referable DR assessment
- Human-in-the-loop review
- Tele-ophthalmology-oriented deployment concept
- Integrated web interface

---

##  Limitations

RetinaX is currently a **research/academic prototype**.

Important limitations include:

- It is not a certified medical device.
- It has not been established as a replacement for clinical diagnosis.
- Performance depends on the datasets, imaging conditions, and model generalization.
- Grad-CAM indicates model attention and does not independently confirm pathology.
- The fundus safety gateway does not rule out other retinal diseases.
- Larger prospective and multi-centre clinical validation would be required before real-world clinical deployment.
- Human ophthalmologist review remains important for uncertain, referable, or safety-triggered cases.

---

##  Future Scope

Future development can include:

- Larger-scale external validation
- Prospective clinical evaluation
- Multi-centre Indian retinal datasets
- Improved lesion-level explainability
- Additional retinal disease screening
- Edge/mobile deployment
- Camera integration
- Secure tele-ophthalmology connectivity
- Patient follow-up and screening history
- Clinical interoperability
- Model calibration and uncertainty estimation

---

