# Vehicle details pilot

The upload demo and live camera view show the full reading, registration code
and plate number in separate fields. The fields use the existing OCR and
splitting rules. Unresolved boundaries are displayed explicitly.

The gate profile can also detect vehicles and estimate their body style,
color and manufacturer. The detector uses COCO car, motorcycle, bus and
truck classes. Optional OpenAI CLIP ViT-B/32 evaluates several views of the selected upright
crop against fixed label lists. The optional supervised PaddleClas branch
adds paint and body evidence. Both use conservative agreement thresholds
that can withhold an attribute. A generic detector class remains available when a body style is
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
VEHICLE_STRUCTURED_ATTRIBUTE_PATH=models/local/vehicle_attribute.onnx
```

The official ViT-B/32 SHA-256 is
`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
See [CLIP source](https://github.com/openai/CLIP/blob/main/clip/clip.py)
and [model card](https://github.com/openai/CLIP/blob/main/model-card.md).
All paths are optional and resolve from the project root.
Restart the receiver after changing them. Runtime inference uses local model
files. The optional third path uses a converted
[PaddleClas vehicle attribute model](https://paddleclas-doc.readthedocs.io/zh/latest/PULC/PULC_vehicle_attribute_en.html)
with RGB 256×192 inputs and the documented PULC normalization. The tested
ONNX opset 14 SHA-256 is
`47770502245fe0971e9e0d77fddd5fb952c288ff26133fd5f199f11a001d25b8`.
The tested YOLO11s COCO checkpoint SHA-256 is
`85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`.

## Gate API and camera history

The gate pipeline publishes and classifies at most one vehicle. With a plate
detection, it requires a unique match to the first returned plate. With no
plate detection, it uses the largest visible detection, with detector score
as the tie-breaker.
The UI displays separate category and color cards. Categories include SUV,
Sedan, Motorcycle, Pickup, Hatchback, Van, and other recognized body styles.

`/demo/predict`, `/v1/predict` and new camera events include:

- `vehicle_status` with `ready`, `detection_only`, `unavailable`, `disabled`
  or `unassociated` for a plate with no unique vehicle match.
- `vehicles[]` with the upright-image box, detector class and score, body type,
  optional color and make, and review flag. There is at most one entry.
  `attribute_sources` records the method used for populated attributes.
- `vehicles[].plate_indices` with zero-based indexes into `detections` and
  `ocr_results`. A plate links to a vehicle when a unique vehicle box covers at
  least 85% of the plate. Ambiguous and missing links remain empty.

A scene may contain vehicles without a plate. Existing captures remain readable
and show vehicle details only when stored with those details. The legacy
`/predict` profile continues to return plate analysis.

## Matched development replay

Six reused gate frames and one public reference image were run with the old
YOLOv8n/single-view CLIP configuration and with YOLO11s, multi-view CLIP, and
supervised paint/body evidence. Each attribute was scored only where a
visual label was clear. A reading from a background vehicle counts as wrong.

| Reviewed measure | Before | After |
| --- | ---: | ---: |
| Correct vehicle linked at the gate | 3/6 | 6/6 |
| Correct color | 2/4 | 4/4 |
| Correct make | 2/5 | 4/5 |
| Correct detailed category | 0/3 | 1/3 |

For category, two wrong guesses and one unknown became one correct SUV and
two unknown results. The motorcycle remained a motorcycle. These are small,
reused development observations, with several BMW appearances. Independent field accuracy and score calibration require broader data. A dedicated
vehicle-grouped evaluation with night views, clipped vehicles and makes
unrepresented in the candidate labels remains open.
