# Crab Species Detection with RF-DETR

## Project Overview
This repository contains an end-to-end object detection pipeline designed to identify and classify three species of crabs: the European Green Crab, Rock Crab, and Jonah Crab. The model was trained using the `RFDETRMedium` architecture over approximately 12 hours on Kaggle, leveraging a robust dataset created by merging five independent, open-source datasets. 

## Dataset Characteristics
The initial (base) dataset was aggregated from five free datasets and contains the following distributions before augmentation:
* **Total Base Images:** 2,889 images yielding 6,630 total object annotations.
* **Annotation Density:** The dataset averages 2.3 annotations per image. The vast majority of the images (2,405) contain exactly one annotated crab.
* **Image Dimensions:** The median image size is 386x342 pixels. Furthermore, 81.8% of the images are classified as "medium" resolution, while 12.4% are "jumbo" and 5.8% are "large".

### Class Distribution (Base Dataset)
The dataset maintains a relatively balanced distribution across its three target classes:
* **European Green Crab:** 2,557 total annotations (1,780 Train, 500 Validation, 277 Test).
* **Rock Crab:** 2,204 total annotations (1,538 Train, 417 Validation, 249 Test).
* **Jonah Crab:** 1,869 total annotations (1,285 Train, 358 Validation, 226 Test).

## Preprocessing & Augmentation Pipeline
To improve model generalization and prevent overfitting on the limited base dataset, a rigorous augmentation pipeline was applied, generating 5 variations per training example.

**Preprocessing Steps:**
* **Auto-Orient:** Applied to correct EXIF rotation flags.
* **Resize:** Images were resized to fit a 576x576 pixel bounding box, utilizing white-edge padding to preserve aspect ratios.

**Augmentation Settings:**
* **Rotation:** Randomly applied between -15° and +15°.
* **Brightness & Exposure:** Randomly adjusted between -15% and +15%.
* **Noise:** Random pixel noise applied to up to 2% of the image pixels.
* **Motion Blur:** Applied with a 20px length, 45° angle, across 2 frames.

**Final Augmented Splits:**
After applying the augmentation strategies exclusively to the training pool, the final dataset contains 10,977 total images:
* **Train Set:** 10,110 images (92%).
* **Validation Set:** 578 images (5%).
* **Test Set:** 289 images (3%).

## Model Architecture & Training Configuration
The pipeline relies on a transformer-based object detector configured for high-accuracy bounding box regression.
* **Architecture:** `RFDETRMedium` utilizing the `dinov2_windowed_small` encoder.
* **Optimizer:** AdamW with a base learning rate of 0.0001 (encoder learning rate of 0.00015) and a weight decay of 0.0001.
* **Batch Strategy:** Batch size of 4 with 4 gradient accumulation steps, resulting in a target effective batch size of 16.
* **Training Duration:** Configured for a maximum of 100 epochs, utilizing an early stopping patience of 10 epochs (min delta 0.001) to halt training once validation metrics plateaued.
* **Input Resolution:** The model was configured to natively process the preprocessed 576x576 inputs.

## Performance Highlights
*(Note: Refer to `eval_results/checkpoint_comparison.csv` and `assets/cm_best_total_test_raw.png` for exact metrics).* 
The model achieved highly robust generalization, typically scoring an mAP@50 of **>0.98** and an mAP@50:95 of **~0.86** on the withheld test set. The raw counts confusion matrix verifies that false negatives and cross-species misclassifications are exceptionally rare, proving the effectiveness of the chosen augmentations.

## Quickstart

```bash
pip install -r requirements.txt
python src/train.py
python src/evaluate_full.py