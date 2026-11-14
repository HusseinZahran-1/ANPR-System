# Reviewed external detector development

Twenty-eight previously transcribed external full-scene sources were reviewed
for detector annotations. Eight have clipped or uncertain boundaries and stay
excluded. Twenty manual axis-aligned envelopes were checked against full images
and final crop overlays: ten external identities for training and ten different
identities for validation. Dealer-text bands were reduced where geometry allows;
oblique axis-aligned envelopes can still contain incidental neighboring text.
This is assistant visual annotation, not independent human signoff. Upstream
annotations were not adopted without inspection. No XML labels were used.

The existing 702 detector training and 14 validation images are retained.
The expanded development dataset contains 712 unique train images and 24
validation images. Selected external training sources repeat six times for
762 training draws. Every other external variant stays quarantined. Full-source
byte/pixel and configured dHash/pHash cross-split checks flag no overlaps.
Identity hints are disjoint; protected original test names/bytes are rejected
before decoding. External sealed test members remain unopened and untranscribed.
These checks do not prove absence of every near duplicate or unknown identity.
The external validation examples have prior OCR-development exposure, so they
are not an independent evaluation of the complete pipeline.

## First controlled warm-start

The deployed YOLOv8n checkpoint initializes the detector. Seed 42, 1280 resolution,
batch four, SGD learning rate 0.001, bias warmup learning rate 0.001, one warmup
epoch, no horizontal flip or mosaic, twelve maximum epochs and patience four
are fixed before training. This isolates data expansion; it is not a new
architecture comparison with YOLO11. Runtime weights and thresholds stay unchanged.

Training ends after nine epochs in 242.55 seconds, selecting epoch five by
validation mAP50-95. On the mixed 24 positive development scenes, mAP50 is
0.84779 and mAP50-95 is 0.58308. These AP metrics integrate confidence thresholds;
they do not imply a gain at the configured operating confidence.

At confidence 0.25, NMS IoU 0.5 and matching IoU 0.5, both baseline and candidate
produce 14 TP, zero FP and ten FN before the application plate cap: the ten
external validation plates remain missed. At confidence 0.01, the candidate
produces 24 TP and 18 FP, including six duplicate boxes; at 0.025, external
results are nine TP, three duplicate FP and one FN. No threshold is promoted.
The frozen camera cohort regresses from 112/113 to 111/113 readable exact,
while no-visible-plate empty results improve from 4/5 to 5/5. The candidate is
rejected. Positive-only validation cannot establish a background false-positive
rate; these reused camera counts do not establish an independent field rate.

A local loader-cache audit confirms that all sixty external draws representing
ten images were loaded with the intended class and normalized annotations.
A controlled source-sampling follow-up changes only external repeats from six
to seventy (1,402 total draws: 702 retained and 700 external), keeping unique
images, validation, initialization and other training settings fixed. The follow-up completes ten epochs in 437.89 seconds. Final validation mAP50
is 0.93375 and mAP50-95 is 0.61314. At confidence 0.25 its fixed-threshold
counts still remain 14 TP, zero FP and ten FN. At a development-calibrated
confidence of 0.05, counts become 18 TP, zero FP and six FN: four of the ten
external validation plates are detected, with no duplicates in this small cohort.
The frozen camera readings stay 112/113 and no-visible-plate empty results 4/5.

A direct replacement nonetheless regresses the actual highest-score pipeline on
four older scene cohorts: 40/45 to 39/45, 12/14 to 11/14, 5/5 to 4/5 and
10/16 to 8/16. Both model replacement and a global confidence change are rejected.
A separately validated primary-first composition preserves those existing
readings while adding new detections only after the primary returns no boxes.
See [optional primary-first fallback](FALLBACK_DETECTOR.md) for the resulting
application behavior and limits. This is a YOLOv8n data/sampling change, not a
YOLO11 architecture promotion.

## Reusable evaluation tool

Run with trusted single-class plate weights and your explicitly audited local
development dataset. The tool evaluates unique images in `train/images` or
`val/images`, with YOLO labels in the corresponding `labels` directory.

```powershell
python -m tools.evaluate_detector_development `
  --data-dir data/local/development `
  --split val `
  --protected-images data/local/sealed-test/images `
  --weights models/local/detector.pt `
  --output logs/local/detector-validation.json `
  --device cpu
```

Protected images are hashed, never decoded. The full source/annotation preflight
finishes before loading the detector or decoding development images. Missing
background annotations, invalid boxes, multi-class checkpoints and empty
protection inventories are refused. An empty label file explicitly declares a
background. Source/label hashes are rechecked before inference; the checkpoint
hash is checked for change during evaluation. Existing report files are never
overwritten. This is not a replacement for identity and near-duplicate audits.

The default confidence is 0.25 and image size is 1280; explicit `--confidence`
and `--imgsz` support recorded development diagnostics. Image size must be a
positive multiple of 32. A baseline resolution ablation on the same 24 scenes
yields 2 TP / 1 FP / 22 FN at 320, 8 TP / 1 FP / 16 FN at 640, versus
14 TP / 0 FP / 10 FN at 1280. No global resolution reduction is adopted. NMS remains active with maximum 300 boxes, while the
application one-plate cap is not applied. One-to-one matching separates duplicate
plate boxes from unrelated false detections. Confidence is not fused with OCR
or presented as end-to-end accuracy. Reports contain private paths and predictions
and must stay outside Git. No images, labels, checkpoints or private reports
are redistributed. Source attribution for admitted external images remains bau,
number plate recognition Jordan v4, CC BY 4.0, with hashes retained locally.
