# ANPR System

An experimental Automatic Number Plate Recognition application focused on
Jordanian plates. It locates plates in scene images, reads text with a CRNN,
checks text against Jordanian format rules, and returns structured results.
Development is ongoing. Every reading requires human review; a format match
does not establish the plate's country or authorize an access decision.

## Implemented features

- YOLO plate detection and CRNN/CTC OCR, with separate confidence signals.
- Crop preprocessing, layout handling, and Jordanian text validation.
- Optional visual CD/SN region reading with separate evidence and mandatory review.
- FastAPI JPEG/PNG upload endpoint and an English browser demo.
- Authenticated camera uploads and direct ISAPI2 multipart image ingestion.
- Live browser polling, durable local image archive, and quarantined reviews.
- Optional camera detector override and single-plate gate settings.
- Optional primary-first fallback detector with separate confidence and mandatory review.

## Requirements and Windows installation

Use Python 3.12, Git, and PowerShell. CPU execution is supported; a compatible
NVIDIA GPU and CUDA-enabled PyTorch are optional. Model weights are required
for inference and are deliberately not distributed here. Obtain trusted,
compatible detector and OCR checkpoints privately from the project owner.
There is no public dataset or checkpoint download included in this repository.
The OCR checkpoint must match the CRNN implementation and its metadata.
Do not load untrusted PyTorch checkpoints.

```powershell
git clone https://github.com/HusseinZahran-1/ANPR-System.git
cd ANPR-System
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

For GPU execution, install compatible CUDA PyTorch wheels instead of CPU
wheels before installing requirements. The pins reflect the local development
environment; a new dependency installation has not been validated in a clean
environment. Tests on the exported source are checked separately.

## Configuration and running

Set these values in your private `.env` after supplying compatible weights:

```dotenv
MODEL_DETECTOR_PATH=models/local/detector.pt
MODEL_OCR_PATH=models/local/ocr.pt
DEVICE=cpu
```

All `data/`, `models/`, and `logs/` content stays local and ignored. Model paths
may also point outside the repository. `CAMERA_DETECTOR_PATH` is optional.
Generate separate random values for `CAMERA_API_KEY` and `CAMERA_ISAPI_TOKEN`
before enabling camera ingestion. Never commit or share the management key.

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

Open http://127.0.0.1:8000/demo for manual uploads, `/live` for camera captures,
or `/docs` for the API schema. `/health` checks configured model-file presence;
it does not validate model compatibility or recognition accuracy. Without
weights, the UI and schema can open, but inference cannot work.

## Camera integration

On a trusted LAN, bind to `0.0.0.0` and configure the firewall deliberately.
Send `POST /camera/events` with multipart field `file`, optional `camera_id`,
and header `X-Camera-Key`. Direct ISAPI2 cameras can upload an image to
`POST /camera/isapi2/<your-private-token>`. The adapter selects the largest
valid image from multipart uploads. This receives still capture events,
not an RTSP video stream. Keep access logs disabled to avoid logging the URL
token. Use TLS and proper deployment controls before exposure outside a
trusted network. `/predict` itself has no authentication.

Captures are saved before inference in `data/processed/camera_inbox/`, with
date-partitioned images and JSON records. Optional operator reviews are saved
in `data/reviewed/live_feedback/` and remain quarantined. These directories
can grow indefinitely: manage retention and disk capacity locally.
This application does not install any automatic startup task.

## Project structure

```text
app/api/        configuration, upload validation, HTTP endpoints
app/camera/     ingestion, local archive, live events, reviewed samples
app/detector/   plate detector wrapper
app/ocr/        CRNN, decoders, preprocessing and layout handling
app/pipeline/   scene-to-result orchestration
app/validator/  text-format rules and review policy
app/demo/       English upload and live interfaces
tests/          public tests using synthetic inputs and model stubs
```

For reviewed public-source acquisition, see [Public data](docs/PUBLIC_DATA.md).
For measured detector/OCR comparisons and their limits, see
[Development experiment results](docs/EXPERIMENT_STATUS.md) and the
[model upgrade policy](docs/MODEL_UPGRADES.md).
The importer saves images locally in quarantine without admitting them to training.

Historical training scripts, private annotations, experiments, and deployment
notes remain in the owner's local workspace. They are not part of this public
source snapshot. No real plate image or capture is supplied as an example.

## Tests and CI

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Most tests use synthetic arrays/images and stub models. Checkpoint-specific
tests skip when the optional historical checkpoint is absent. Unit tests do
not establish real-camera OCR accuracy. The GitHub Actions workflow runs CPU
tests; local tests and remote CI are separate evidence.

## Limits and next work

Foreign plates can match Jordanian numeric formats. Confidence signals are
uncalibrated. CD/SN confusion, clipping, glare, angle and missed detections
remain development concerns. Multi-plate behavior exists, but field quality
is not independently established. No production accuracy figure is claimed.
Next steps are independent camera evaluation, more verified authority and
foreign examples, regression-controlled training, dependency installation
verification, and deployment hardening. See CONTRIBUTING.md for workflow.

## Licensing

A project license has not been selected. Public visibility does not grant a
general license to reuse the code. Review third-party dependency and model
licenses, including Ultralytics and OCR architecture provenance, before release
or redistribution. Model weights and datasets require a separate rights review.

## Visual authority reader

See [optional CD/SN region OCR](docs/AUTHORITY_REGION_OCR.md) for setup, measured
development results and limitations. Manual demo uploads use `/demo/predict`,
which shares the configured camera detector; general `/predict` keeps its own
detector configuration. Start the local MVP manually with `./run_mvp.ps1`.

## Experimental two-row preprocessing

An optional geometry-based layout fixes four reused training-crop readings but
does not fix detector misses on newly reviewed external full scenes. It remains
disabled by default. See [two-row layout evidence](docs/ROW_LAYOUT.md).

## Broader-image detection fallback

A validated optional second detector is tried only after the primary finds no
boxes. Combined with two-row preprocessing, it reads four of ten reused external
validation scenes previously missed entirely, while measured camera/older-scene
results remain unchanged. Six external scenes still fail; independent field
accuracy and country rejection remain unresolved. See [setup and evidence](docs/FALLBACK_DETECTOR.md).

The optional [recovery and aligned-fragment pilot](docs/RECOVERY_AND_FRAGMENTS.md)
addresses two-row misses and split plate boxes in the single-plate demo.
It remains a development pilot with mandatory review and separately reported
raw detector errors; the public defaults keep both new features off.

Final readings also expose [separate plate code and number](docs/PLATE_PARTS.md)
in the API, demo and live view, with unresolved boundaries left explicit.
