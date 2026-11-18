# Inference API v1

Run the service with Python 3.12 and the supplied detector and OCR checkpoints.
Set the model paths in the service's private environment. Relative paths resolve
from the repository root. Use `DEVICE=cpu` with CPU PyTorch or `DEVICE=cuda`
with a wheel compatible with the NVIDIA GPU and driver. The manual launcher
supports `--require-models` to load the gate profile before opening its listener.

```powershell
.\run_mvp.ps1 -RequireModels
```

A cross-platform equivalent from the project root is
`python -m app.serve --require-models`. Use one worker for a small camera
deployment. Each worker caches its own models and serializes inference. Restart
when you change configuration or replace checkpoints. Store checkpoint files
read-only while a worker uses them.

## Routes

- `GET /docs` and `GET /openapi.json` describe the HTTP contract.
- `GET /v1/ready` loads the selected profile and returns its fingerprint. A
  missing or incompatible checkpoint returns 503. The legacy `/health` reports
  file availability.
- `POST /v1/predict` accepts one multipart `file` and returns JSON. It creates
  no camera record. The optional multipart field `save_xml=true` saves a paired
  original image and XML report. Requests with saving enabled create a new
  export on each successful submission, including repeated submissions.

The fixed profile identifier is `gate-camera-v1`. It uses
`CAMERA_DETECTOR_PATH` when set, or `MODEL_DETECTOR_PATH` as the default
primary, then uses the configured fallback, recovery and OCR options.
`CAMERA_RECOVERY_DETECTOR_PATH` and `CAMERA_MERGE_FALLBACK_FRAGMENTS`
override only this profile's recovery checkpoint and fragment behavior. The
camera still receiver and demo also use these overrides. The
response `checkpoints` maps enabled model roles to SHA-256 values. Use the
profile, checkpoint hashes and settings in `meta` together to track changes in
inference behavior. `/predict` keeps the general profile and `/demo/predict`
uses the gate profile.

Supply a dedicated `INFERENCE_API_KEY` when binding to a network interface.
Clients send it as `X-Inference-Key`. With no inference key or admin key configured, v1 accepts only
direct loopback access with a loopback bind. Set a key before putting the service
behind any proxy, terminate HTTPS there for remote backend clients, and route
only the intended endpoints. The legacy upload routes retain their existing
access behavior. Camera credentials have separate scope.

## Client examples

The examples use a provisioned key. For direct loopback with an empty server key,
omit the header.

```bash
curl --fail-with-body --connect-timeout 5 --max-time 60 \
  -H "X-Inference-Key: ${INFERENCE_API_KEY}" \
  -F "file=@scene.jpg;type=image/jpeg" \
  http://127.0.0.1:8000/v1/predict
```

```python
import os
from pathlib import Path

import httpx

base_url = os.environ["ANPR_BASE_URL"].rstrip("/")
headers = {"X-Inference-Key": os.environ["INFERENCE_API_KEY"]}
with httpx.Client(base_url=base_url, headers=headers,
                  timeout=httpx.Timeout(60, connect=5)) as client:
    client.get("/v1/ready").raise_for_status()
    image = Path("scene.jpg")
    with image.open("rb") as source:
        response = client.post("/v1/predict",
                               files={"file": (image.name, source, "image/jpeg")})
    response.raise_for_status()
    readings = [item["plate_parts"] for item in response.json()["ocr_results"]]
```

Use JPEG or PNG with its image MIME type. The default compressed limit is
10 MiB (`MAX_UPLOAD_BYTES`) and the decoded limit is 50 megapixels.
A reverse proxy should also cap the request body, accounting for multipart
overhead. Use a longer timeout for cold CPU instances if needed. Retry
connection errors, timeouts and temporary 503 responses with bounded backoff.
A persistent 503 requires checking model files or key configuration.

| HTTP | Meaning |
| --- | --- |
| 200 | Completed, including scenes with no detection |
| 401 | Missing or invalid inference key when one is configured |
| 413 | Image too large |
| 415 | Unsupported declared MIME type or image format |
| 422 | Missing multipart field or unreadable image |
| 500 | Inference failed |
| 503 | Profile unavailable or network deployment needs an inference key |

## Evidence

Detections and OCR readings are aligned by index. Boxes are pixel
`[x1, y1, x2, y2]` coordinates from the top-left of the upright image after
EXIF orientation. OCR carries its original `text`, confidence and available
regional evidence. The associated `plate_parts` has normalized `full_text`,
string `plate_code` and `plate_number` values that preserve leading zeros,
`split_basis` and `requires_human_review`. Ambiguous numeric boundaries
return null parts and `unresolved`. Seven contiguous digits use the declared
2+5 assumption. When the detector returns no plates, `detections` and
`ocr_results` are empty and `decisions` records the review outcome. Scores
express model confidence. A format match still requires review and
`country_verified` remains false.

The gate profile can also return [optional vehicle evidence](VEHICLE_ATTRIBUTES.md).


## API console and issued keys

Open `/api` to generate, list, test and revoke inference keys. Configure a
separate `INFERENCE_ADMIN_KEY` privately. Key-management calls use
`X-Inference-Admin-Key`; they return the new client key only once. The database
stores SHA-256 digests and key metadata. Set `INFERENCE_KEY_DB` to relocate it.
Existing `INFERENCE_API_KEY` clients continue working. Issued keys grant
access to `/v1/ready` and `/v1/predict`; the camera credentials and
administrator authority are separate.

See [capture exports](CAPTURE_EXPORTS.md) for layout and XML fields.
