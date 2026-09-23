# Prepared Q&A (presenter's own answers; use these first)
Lines marked [FILL] = detail not in the paper; presenter must confirm. Never state those numbers unless filled in.

## Model & dataset
Q: Why YOLOv8 and not YOLOv10/11 or a transformer (RT-DETR)?
A: Fast, mature, well documented, easy to deploy on CCTV. 89.3 FPS suits real-time early warning. Our focus was the integration with rainfall, not a new detector. Newer versions are future work.

Q: Which YOLOv8 variant and input size?
A: [FILL: n/s/m, imgsz 640?]

Q: How big is the dataset and where is it from?
A: 2,249 images: our own photos plus Roboflow Universe (Meehirs Workspace, Trash SJ, Object Detection Debris). Public on Zenodo.

Q: Train/val/test split?
A: Random split; test set is 283 unseen images with 702 instances. [FILL: ratio, e.g. 80/10/10]

Q: Why is mAP50-95 only 0.445?
A: Debris is small and irregular in the frame, so exact boxes are hard under strict IoU. Common in floating-object detection. For early warning, detecting presence and amount (mAP50 0.789) matters more than pixel-perfect boxes.

Q: Your mAP50 is lower than YOLOv8-SST (0.912) and A2ANet (0.841). Why is this acceptable?
A: We trade some accuracy for speed and simplicity: 89.3 FPS vs 86 and 26 FPS. Our model is an optimized baseline; the contribution is the multi-parameter system. Modules like attention or small-target heads can be added later.

Q: Which class performs worst and why?
A: "Other", mAP50 0.645, recall 0.683: very diverse objects, no consistent visual pattern. Wood is best at 0.926. We kept "other" as a catch-all to avoid more class imbalance.

Q: How does it handle glare, night, or heavy rain?
A: Glare and lighting changes are the main error sources. Night and heavy rain are not yet exhaustively tested; that is a stated limitation. Future: more night/rain data and image enhancement.

Q: Overfitting risk?
A: Augmentation (brightness, rotation, flip), COCO transfer learning, and evaluation on unseen test data. Losses decreased steadily over 150 epochs.

Q: Where was 89.3 FPS measured?
A: End-to-end processing. [FILL: on Colab GPU or on the i5-11400H workstation?]

## Density & SAW
Q: How do you compute debris density?
A: Hybrid static-dynamic: observation area calibrated to the CCTV frame, river depth profile from historical literature. Debris pixel density is compared with the depth profile to estimate remaining flow capacity.

Q: Why SAW and not AHP, fuzzy logic, or machine learning?
A: SAW is transparent, computationally light, and easy to explain to disaster teams. With only 36 labeled samples, training an ML fusion model would overfit. AHP or fuzzy are good next steps.

Q: Why weights 0.60 rainfall and 0.40 debris?
A: Rainfall is the direct flood trigger, supported by literature; debris is an obstruction factor. Our validation shows these fixed weights need recalibration.

Q: Your SAW accuracy is only 0.556 and everything was classified SAFE. Doesn't that undermine the system?
A: Yes, honestly it is the main weakness, and we report it openly. Detection works well; the fusion thresholds are too conservative. Next step: sensitivity analysis of weights and thresholds against real flood data, and adding water level. The framework stays valid; the calibration must improve.

Q: Only 36 validation samples is small.
A: Agreed; flood events are rare and labeling needs synchronized video and rainfall. We plan to collect more events over rainy seasons.

Q: How is rainfall normalized?
A: [FILL: min-max over what range? mm/hour or BMKG category?]

Q: Is the BMKG data real-time?
A: Not yet; that is a limitation. Real-time integration is future work.

## System & impact
Q: How is it deployed?
A: CCTV frames go to YOLOv8, density is computed, combined with BMKG rainfall via SAW, and shown on a monitoring dashboard with status SAFE, WARNING, or DANGER.

Q: Why not use water-level sensors?
A: Budget and time. Our system complements sensors: it shows why flow may be blocked. IoT integration is planned.

Q: Can it generalize to other rivers?
A: Detection likely transfers; calibration of area and depth is river-specific. Tested only on Ciliwung near Manggarai so far.

Q: Who benefits?
A: Disaster response teams: remote monitoring and scheduling cleanups before heavy rain.

Q: Main novelty in one sentence?
A: Combining YOLOv8 debris density, which accounts for river cross-section, with BMKG rainfall through SAW into one flood-potential indicator on a dashboard.

## Safe fallback
Q: (anything not covered)
A: "Thank you, that's a great point. We haven't examined that yet, but it's a valuable direction for future work."
