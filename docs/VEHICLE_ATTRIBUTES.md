# Vehicle evidence

The gate profile selects at most one vehicle and returns its broad detector
type, estimated paint color and estimated make. With a plate detection it
requires a unique match to the first plate. With no plate detection it uses
the largest visible detection. Paint and make remain nullable when the crop
is small, obscured or ambiguous.

## Models

The vehicle detector accepts a compatible trusted COCO checkpoint, such as
YOLO11s. The optional OpenAI CLIP branch compares multiple views of the crop
against color and make labels. The optional supervised PaddleClas branch
adds paint evidence. It uses RGB 256×192 inputs and documented PULC
normalization from the [PaddleClas vehicle attribute guide](https://paddleclas-doc.readthedocs.io/zh/latest/PULC/PULC_vehicle_attribute_en.html).

Install the base requirements first. For color and make estimates install
the optional runtime and supply local model weights.

```powershell
python -m pip install "setuptools==78.1.0" wheel
python -m pip install --no-build-isolation -r requirements-vehicle.txt
```

```dotenv
VEHICLE_DETECTOR_PATH=models/local/yolo11s.pt
VEHICLE_ATTRIBUTE_PATH=models/local/ViT-B-32.pt
VEHICLE_STRUCTURED_ATTRIBUTE_PATH=models/local/vehicle_attribute.onnx
```

Checkpoint SHA-256 values for the tested stack:

| Checkpoint | SHA-256 |
| --- | --- |
| YOLO11s COCO | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` |
| OpenAI CLIP ViT-B/32 | `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` |
| PaddleClas ONNX, opset 14 | `47770502245fe0971e9e0d77fddd5fb952c288ff26133fd5f199f11a001d25b8` |

Paths resolve relative to the project root. Restart after changing model paths.

## API and exports

`vehicle_status` is `ready`, `detection_only`, `unavailable`, `disabled`,
or `unassociated` for a plate with no unique vehicle match. Each entry
includes the box, detector class and score, `vehicle_type` (Car, Motorcycle,
Bus or Truck), nullable `color` and `make`, the source of populated attributes,
and zero-based `plate_indices`. The XML and new filenames use the same broad
type. Older live entries are displayed with their broad type.

The prior matched development replay improved vehicle-to-plate links from
3/6 to 6/6, correct color from 2/4 to 4/4 and correct make from 2/5 to 4/5.
These reused views, with several appearances of one make, do not measure
independent field accuracy or performance on night scenes.
