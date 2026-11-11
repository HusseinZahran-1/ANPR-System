# Optional visual CD/SN reading

The primary v4 CRNN checkpoint remains unchanged. Image geometry proposes the
first two glyphs; a pretrained English recognizer reads that image region.
Only a unique CD or SN prediction above the configured threshold is accepted.
The primary CRNN reads the geometric plate view; approximate CTC token positions
select a numeric suffix spatially after the prefix. No transcript substitutions
or ground-truth strings are used during inference. Missing or ambiguous evidence
retains the primary reading. This is an experimental review aid.

## Setup

Install the base requirements, then `pip install -r requirements-authority.txt`.
Supply trusted EasyOCR English generation-two weights privately, with filename
`english_g2.pth` and SHA256
`e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8`.
Weights are not distributed here; runtime downloads are disabled.

Set these private configuration values:

```dotenv
AUTHORITY_REGION_MODEL_PATH=models/pretrained/easyocr/english_g2.pth
AUTHORITY_PREFIX_MIN_CONF=0.8
```

An empty model path disables this feature by default. The threshold and model
scores are uncalibrated. Responses retain `primary_ocr_text`, expose separate
`authority_region` prefix/suffix evidence and `ocr_confidence_scope`; the original
OCR confidence is not confidence in the composed text. Every region reading is
FLAG_REVIEW with country_verified=false. A text format match does not verify country.

## Development evidence, November 11, 2026

Five repeatedly used authority views represent only three physical plates.
The baseline reads 0/5 exactly; the region reader reads 5/5 on both crops and full
scenes. These are development results, not independent accuracy estimates.
Across 1,113 numeric crop views (1,005 training and 108 validation), no readings
change. Overall validation remains 96/110. Four scene cohorts remain 40/45,
12/14, 5/5 and 10/16. The frozen camera cohort remains 112/113 readable exact
and 4/5 no-visible-plate views empty. Views are not independent identities.

Nine reused foreign views represent four identities. No readings change;
8/9 still falsely match a Jordanian text format. Foreign rejection remains
unresolved; these counts are not an independent false-acceptance-rate estimate.

Real-model HTTP checks with an isolated temporary archive read the five authority
scenes 5/5 through camera ingestion and 5/5 through `/demo/predict`. The older
general `/predict` configuration reads 3/5 and is retained separately. The demo
now uses the configured camera detector. The isolated feed contains all five
results; the physical-camera archive was unchanged. This does not establish
connectivity to the physical camera on another network.

## Local operation

The Windows launcher `./run_mvp.ps1` starts a manually operated loopback server
on port 8000 using `.venv`. It installs no startup task. Shared web pins are in
`requirements-web.txt`. Locally these web packages were installed into a project
venv while existing GPU/vision dependencies were inherited with
`--system-site-packages`; this is not a validated clean isolated installation.
The manual demo does not archive uploads; authenticated camera ingestion retains
its existing local archive. Private images, checkpoints and logs stay local.
