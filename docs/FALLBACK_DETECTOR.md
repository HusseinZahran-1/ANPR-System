# Optional primary-first detector fallback

The application can keep its existing primary detector and try an independently
trained second detector only if the primary returns **no boxes**. Primary boxes
are returned unchanged, even when another model might assign a higher score.
Box sets and model scores are never merged. A primary inference exception
propagates; it is not treated as an empty scene. Existing application plate caps
are applied after this choice, and evaluation also inspects boxes before caps.
This addresses a measured external-image gap without changing working primary
crops. It does not correct existing primary false detections or verify country.

## Setup and provenance

Supply compatible trusted local primary/OCR weights and the separately trained
fallback checkpoint privately. Weights and private training data are not
included in this repository. The feature is off by default.

```dotenv
FALLBACK_DETECTOR_PATH=models/local/secondary-detector.pt
FALLBACK_DETECTION_CONF=0.05
OCR_LAYOUT_MODE=rectify_unfold_rows_v4
```

An empty fallback path disables the second pass. The primary detector retains
its own `DETECTION_CONF`, default 0.25. The API forwards the optional model to
both the general and camera profiles; `/demo/predict` and camera ingestion share
the configured camera primary. `/health` reports optional model presence without
loading it. A configured missing fallback is reported as unavailable rather
than silently ignoring configuration. Restart a manually launched server after
changing private settings. This installs no startup task.

The returned `meta.detector_source` states primary, fallback or no detection.
Primary and fallback thresholds are exposed separately, along with whether a
fallback was attempted. Fallback boxes carry `detector_source=fallback`; their
own confidence is retained. Every fallback reading is explicitly FLAG_REVIEW,
with country_verified=false and requires_human_review=true. A low model score
is not a probability of complete plate correctness or proof of country.

The measured local configuration retains the existing Phase38 primary and v4
CRNN, uses the Phase80 balanced-data YOLOv8n as fallback at 0.05, and combines
it with the optional verified English CD/SN reader and v4 two-row layout. Both
detectors are YOLOv8n. The earlier YOLO11 comparison remains unpromoted.
See [detector training and sampling](DETECTOR_EXPANSION.md) for data provenance.

## Measured development results

The prototype and actual constructor/API integration were checked separately.
On ten external detector-training sources, the old pipeline reads 0/10 exactly;
fallback with the older v3 layout reads 3/10 and with v4 layout reads 6/10.
These are training-fit diagnostics. On ten different external detector-validation
sources, the old pipeline reads 0/10, fallback with v3 reads 3/10 and with v4
reads 4/10. Four validation plates are detected and all four read correctly;
six are still missed. Every source has prior OCR-development exposure and has
been reused during development; these counts are not independent field accuracy.

Actual highest-score pipeline results on the four older scene cohorts remain
40/45, 12/14, 5/5 and 10/16. Before the application cap, those 80 scenes yield
80 TP, zero FP and zero duplicate boxes for both configurations. Boxes paired
by template matching are not independent hand-annotated localization ground truth.
The frozen camera cohort retains all prior texts: 112/113 readable exact and
4/5 no-visible-plate views empty, with no multiple-box events in the composed
configuration. Fifteen readable views are outside the current OCR training
manifest; the other 98 have training identities. This is not independent sampling.

The five reused CD/SN probe views still read 5/5, representing only three
physical plates. All reused authority/foreign full-scene texts remain unchanged.
Eight of nine foreign views still match a Jordanian text format; reliable
country rejection remains unresolved and a format match never verifies country.
Earlier sensitivity of authority prefixes to mild perturbations also remains.
The fallback can fail on unfamiliar backgrounds; the small development cohorts
do not establish a production background false-positive rate.

## HTTP/live verification and latency limits

Real-model integration posts all 25 selected scenes to camera ingestion and
the manual demo: five authority views plus ten external training and ten
external validation sources. Exact counts are 5/5, 6/10 and 4/10 respectively.
Every camera prediction matches the demo and direct pipeline. After each post,
the latest live event has the corresponding image/result; all 25 image retrievals
match the source bytes. A temporary archive records all 25 events and is then
deleted. The physical-camera event log hash stays unchanged. The feed deliberately
returns only the latest ten events. These are integration checks, not evidence
of physical-camera connectivity from another network. The legacy general
`/predict` profile retains 3/5 on those authority views and is reported separately.

On this laptop, eight measured warm runs after three warmups on each of two
selected development scenes have median pipeline times of 82.65 ms baseline /
88.36 ms composed for a camera-positive scene, and 21.98 ms baseline /
66.56 ms composed for an external-positive scene. The external baseline does
no OCR because it misses the plate. These two-scene warm measurements exclude
model startup and do not establish a production latency SLA.

Original and external held-out test pixels stay unopened. All raw-image hashes
remain unchanged. No source images, annotation transcripts, model weights,
camera settings or private per-image logs are published.

## Later recovery and fragment pilot

The counts above describe the earlier two-stage configuration. The later
optional recovery pass and within-model fragment unions raise both external
evaluation cohorts from 4/10 to 6/10 exact, with older gates unchanged.
See [configuration, raw/post-geometry metrics and limitations](RECOVERY_AND_FRAGMENTS.md).
