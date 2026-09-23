# Paper summary (ICORIS 2026)
Title: YOLOv8-Based River Debris Monitoring System With Meteorological Data for Flood Mitigation
Authors: Kelvin Alexander Harli, Ferdy Saputra (presenter; experiments, dataset, YOLOv8, mAP evaluation, SAW, CCTV integration, validation), Edi Purnomo Putra (supervisor). Information Systems, BINUS University.

## Problem & gap
- Urban floods in Indonesia are triggered by heavy rain plus floating debris blocking river flow (2,009 flood incidents in 2025).
- Prior work uses one approach only: visual debris detection (Ghost-YOLOv8, A2ANet) or hydrological sensing (ToF water-level sensor). Combining debris density with rainfall is rarely studied.

## Method
- Case study: Ciliwung River near Manggarai Station (CCTV imagery).
- Dataset: 2,249 images (own photos + Roboflow Universe: Meehirs Workspace, Trash SJ, Object Detection Debris), YOLO bounding boxes. Public on Zenodo: doi 10.5281/zenodo.20413204.
- 5 classes: bottle, paper, plastic, wood, other ("other" = catch-all to avoid class imbalance).
- Preprocessing: cleaning, resizing, normalization. Augmentation: brightness, random rotation, horizontal flip.
- Training: Google Colab, transfer learning from COCO weights, 150 epochs, random train/val/test split.
- Debris density: hybrid static-dynamic approach. Observation area calibrated to the CCTV frame (dynamic); river depth profile from historical literature (static). Debris pixel density vs depth profile = estimated remaining flow capacity.
- Rainfall: BMKG API (not yet real-time).
- Fusion: Simple Additive Weighting (SAW). S = 0.40 x R1 (debris density) + 0.60 x R2 (rainfall), both normalized. Rainfall weighted higher because it directly drives flooding. SAW chosen: computationally efficient, merges criteria into one index.
- Status: S >= 0.70 FLOOD DANGER; 0.40 <= S < 0.70 FLOOD WARNING; S < 0.40 SAFE. Shown on a monitoring dashboard.
- Hardware: Intel Core i5-11400H, 16 GB RAM, Windows 11, plus cloud for training.

## Results
- Training: box_loss 1.50 -> 0.50, dfl_loss 1.19 -> 0.83 over 150 epochs.
- Test set: 283 images, 702 instances. Overall P 0.788, R 0.773, mAP50 0.789, mAP50-95 0.445.
- Per class (P/R/mAP50/mAP50-95): Bottle 0.715/0.761/0.751/0.263; Other 0.751/0.683/0.645/0.385; Paper 0.876/0.705/0.800/0.381; Plastic 0.793/0.797/0.821/0.496 (394 instances, most frequent); Wood 0.804/0.920/0.926/0.701 (best).
- Low mAP50-95: debris is small in frame, so precise boxes are hard under strict IoU (common for floating objects).
- Errors: glare, reflections, lighting changes; "other" confused with other classes.
- Speed: 89.3 FPS end-to-end.
- Comparison (mAP50 / FPS): YOLOv7 0.800/179; YOLOv7+CBAM 0.901/49.5; YOLOv8-SST 0.912/86; A2ANet 0.841/26; ours 0.789/89.3 = optimized baseline for early warning.
- SAW validation: 36 labeled samples (20 SAFE, 5 WARNING, 11 DANGER). Accuracy 0.5556, precision 0.1852, recall 0.3333, F1 0.2381. ALL samples predicted SAFE -> current weights/thresholds underestimate risk; recalibration needed.

## Limitations & future work
- No water level or flow velocity yet. Cross-section uses static depth profile + pixel calibration (no morphology changes).
- Not exhaustively tested at night, in heavy rain, or on new rivers. BMKG data not real-time. Fixed SAW weights. No IoT sensors (budget/time).
- Future: IoT sensors, more hydrological parameters, adaptive weights, SAW sensitivity analysis against actual flood data.
