# Detector and OCR upgrade policy

The current detector family is YOLOv8n. The first paired YOLO11n comparison
completed without demonstrating an upgrade; the deployed detector is retained.
Both model families can use the existing Ultralytics detection adapter. See
[development experiment results](EXPERIMENT_STATUS.md) for measured limits.

The architecture comparison starts each model from its own COCO-pretrained
checkpoint and uses the same reviewed development images, split, image size,
optimizer, schedule, seed and augmentation. Compare the trained YOLO11 candidate
both with the paired YOLOv8 control and with the deployed project detector.
Do not compare a generic COCO checkpoint with a plate-trained checkpoint.

Before promotion, measure localization, missed plates, duplicate boxes on a
single plate, false detections on reviewed background scenes, end-to-end plate
exact match, and inference latency under the same settings. Report detector
outputs before the application's maximum-plate cap as well as final results:
a cap of one can hide duplicate detections. Positive-only validation cannot
establish a background false-positive rate. Reused development cohorts do not
establish independent field accuracy.

CD/SN transcription is an OCR problem as well as a crop-quality problem.
Changing the detector does not establish that character confusions are fixed.
Evaluate those prefixes separately without automatically replacing ambiguous
characters. Historical diplomatic plate designs are excluded from the
current-format candidate dataset and current-format accuracy claims. Preserve
their provenance as excluded review material rather than silently relabeling
or deleting the originals.

External datasets remain quarantined until image, annotation, format, rights,
identity and duplicate reviews pass. Do not trust upstream train/validation
partitions as independent merely because their filenames differ. Keep raw
files unchanged and sealed test images unopened throughout development.

All images, labels, model weights, private experiment outputs and local camera
settings remain outside the public repository. Publish reviewed source and
documentation through the existing pull request; deployment requires measured
evidence, and the candidate experiment does not change runtime configuration.

Reference: [official YOLO11 model documentation](https://docs.ultralytics.com/models/yolo11/).
