# Development experiment results

These results describe selected, repeatedly inspected development cohorts.
They are not independent test results or production accuracy. Raw images are
unchanged; sealed test pixels were not opened. Weights and private data stay local.

## Paired detector architecture comparison

Both candidates started from their architecture's COCO checkpoint and used
702 reviewed positive training scenes and the same 14 positive validation
scenes: seed 42, image size 1280, batch 4, SGD learning rate 0.01, no flips,
60-epoch cap and patience 12. Early stopping ended YOLO11n at 14 epochs and
YOLOv8n at 22. Reported mAP values are from each selected best checkpoint.

| Candidate | mAP50 | mAP50–95 | TP | FP | FN | Duplicate boxes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11n | 0.995 | 0.6730 | 14 | 1 | 0 | 0 |
| Paired YOLOv8n control | 0.995 | 0.6969 | 14 | 0 | 0 | 0 |
| Deployed YOLOv8n detector | Not measured here | Not measured here | 14 | 0 | 0 | 0 |

Box counts use confidence 0.25, NMS IoU 0.5, matching IoU 0.5 and up to 100
detections per image, before the application's one-plate cap. A duplicate is
an extra prediction matching an already matched annotated plate. These 14
scenes do not reproduce the reported duplicate-detection failure, and there
are no background validation images to establish background false-positive
rate. On 120 reviewed camera views, with the same deployed OCR and one-plate
cap, both candidates retain 112/113 readable exact results. YOLO11n returns
empty text on 5/5 no-visible-plate views, the paired YOLOv8n control on 2/5,
and the deployed baseline on 4/5. These are selected reused views, not an
independent background rate. The YOLO11n extra validation detection was text
on clothing. A global confidence increase could also remove true plates:
one matched validation prediction has confidence approximately 0.251.
YOLO11n is not promoted; broader localization regression checks are still needed.

## Current-format OCR data and ablation

Two existing training views of one historical sidebar-format diplomatic plate
were excluded from the next candidate dataset. The old datasets and originals
are preserved. Historical public diplomatic photographs are also excluded
from current-format training and accuracy claims. Warm-start weights have
prior exposure to older training data; removing rows does not erase that history.

| OCR | Exact matches on unchanged 110-view validation | Current-format authority probe |
| --- | ---: | ---: |
| Deployed OCR baseline | 96/110 | 0/5 |
| Candidate after historical-format row exclusion | 95/110 | 0/5 |

On the reviewed camera cohort, the candidate retains 15/15 exact readings
outside its OCR training identities, changes training-identity fit from 97/98
to 96/98, and still produces text on one of five no-visible-plate scenes.
These reused camera views are not independent vehicles or field accuracy.
The candidate is not promoted, and CD/SN confusion remains unresolved.

A further 15 public numeric source groups were visually reviewed before model
predictions. One group was excluded for clipped characters; one variant from
each of 14 readable identities was admitted for a training experiment. Texts
do not overlap the checked prior training/development label inventories or
sealed-test filename identity hints. Exact and configured perceptual checks
found no cross-split duplicate candidates. This does not prove absence of
every near duplicate, or identity isolation against untranscribed sealed
external imagery. External test data remain sealed and unused for claims.

The expanded candidate has 1,018 training views and the unchanged 110 validation
views. It adds numeric diversity, not new CD/SN identities. Its selected best
checkpoint, at epoch 3 of 6, matches the baseline's 96/110 exact validation
readings and reduces character edits from 29 to 27 across 709 characters.
The authority probe remains 0/5 exact, with five character edits. On the 14
new training crops, exact fit increases from 9/14 to 12/14; that is a training
fit diagnostic. Camera views outside OCR training identities remain 15/15,
training-identity views improve from 97/98 to 98/98, and one of five empty
scenes still produces text. This does not establish an independent improvement.

| Reused scene cohort | Deployed OCR baseline | Expanded OCR candidate |
| --- | ---: | ---: |
| Established development scenes | 40/45 | 39/45 |
| Newer development scenes | 12/14 | 12/14 |
| Hard development scenes | 5/5 | 5/5 |
| Legacy validation scenes | 10/16 | 9/16 |

The degraded scene appears in both the established and legacy cohorts, so
these two regressions are not independent cases. The expanded candidate is
not promoted because the camera training-fit gain comes with that regression.
The verified new dataset remains available for the next controlled experiment.
An additional local pretrained English recognizer probe, with uppercase-only
and case-folded decoding, also reads 0/5 authority views exactly; it is not
adopted as an alternate recognizer.

Public source: [bau's Jordan plate dataset, version 4](https://universe.roboflow.com/bau-l8e0s/number-plate-recognition-jordan/dataset/4),
declared CC BY 4.0. Local receipts retain source attribution, original member
hashes, reviewed crop geometry and visual transcriptions. Images and labels
are not redistributed in this repository.
