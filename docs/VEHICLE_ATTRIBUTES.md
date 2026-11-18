# Vehicle details pilot

The upload demo and live camera view show the full reading, registration code
and plate number in separate fields. The fields use the existing OCR and
splitting rules. Unresolved boundaries are displayed explicitly.

The gate profile can also detect vehicles and estimate their body style,
color and manufacturer. The detector uses COCO car, motorcycle, bus and
truck classes. Optional OpenAI CLIP ViT-B/32 classifies upright vehicle crops
against fixed label lists. Vehicle-specific thresholds can withhold an
attribute. A generic detector class remains available when a body style is
uncertain. Even a displayed make or color requires visual review.

## Local setup

Install the existing base requirements first. Set `VEHICLE_DETECTOR_PATH` to
a trusted COCO Ultralytics checkpoint, such as YOLOv8n. For experimental CLIP
attributes, install the optional dependency and provide the official OpenAI
ViT-B/32 weights:

```powershell
python -m pip install "setuptools==78.1.0" wheel
python -m pip install --no-build-isolation -r requirements-vehicle.txt
```

```dotenv
VEHICLE_DETECTOR_PATH=models/local/yolov8n.pt
VEHICLE_ATTRIBUTE_PATH=models/local/ViT-B-32.pt
```

The official ViT-B/32 SHA-256 is
`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
See [CLIP source](https://github.com/openai/CLIP/blob/main/clip/clip.py)
and [model card](https://github.com/openai/CLIP/blob/main/model-card.md).
Both paths are optional and resolve from the project root.
Restart the receiver after changing them. Runtime inference uses local model
files.

## Gate API and camera history

The gate pipeline publishes and classifies at most one vehicle. It prefers a
vehicle uniquely linked to the first returned plate. If none is linked, it
uses the largest visible detection, with detector score as the tie-breaker.
The UI displays separate category and color cards. Categories include SUV,
Sedan, Motorcycle, Pickup, Hatchback, Van, and other recognized body styles.

`/demo/predict`, `/v1/predict` and new camera events include:

- `vehicle_status` with `ready`, `detection_only`, `unavailable` or `disabled`.
- `vehicles[]` with the upright-image box, detector class and score, body type,
  optional color and make, and review flag. There is at most one entry.
- `vehicles[].plate_indices` with zero-based indexes into `detections` and
  `ocr_results`. A plate links to a vehicle when a unique vehicle box covers at
  least 85% of the plate. Ambiguous and missing links remain empty.

A scene may contain vehicles without a plate. Existing captures remain readable
and show vehicle details only when stored with those details. The legacy
`/predict` profile continues to return plate analysis.

## Validation scope

The CUDA smoke checks exercised both browser pages, a gate archive round trip,
the BMW and black labels on a known sample, a motorcycle detection, and an
SUV prediction. The vehicle-to-plate rule linked at least one plate in three
of six reused gate scenes. These checks exercise integration and do not
estimate field accuracy. The BMW example retained its generic car class
because the body-style evidence was inconclusive. Gray, silver and white
can be difficult to distinguish in changing light, and some vehicle makes
are absent from the candidate list. Gate images with independently reviewed
body styles, paint colors and makes are needed to measure coverage and errors.
