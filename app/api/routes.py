"""HTTP endpoints for scene-level Jordanian plate recognition."""

from __future__ import annotations

import hashlib
import hmac
import io
import tempfile
import threading
from functools import lru_cache
from ipaddress import ip_address
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Security, UploadFile
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.api.config import get_settings
from app.api.keys import valid_issued_key
from app.capture_exports import CaptureExport, save_export
from app.pipeline import Pipeline
from app.vehicle import Vehicle, VehicleStatus, vehicle_evidence
from app.ocr.authority_regions import AuthorityReaderUnavailable

router = APIRouter()
_pipeline: Pipeline | None = None
_camera_pipeline: Pipeline | None = None
_pipeline_lock = threading.Lock()
_inference_lock = threading.Lock()


def _get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                settings = get_settings()
                if not settings.detector_path.is_file():
                    raise FileNotFoundError(f"detector model not found: {settings.detector_path}")
                if not settings.ocr_path.is_file():
                    raise FileNotFoundError(f"OCR model not found: {settings.ocr_path}")
                if settings.authority_path is not None and not settings.authority_path.is_file():
                    raise FileNotFoundError('configured authority OCR model is missing')
                if settings.fallback_detector_path is not None and not settings.fallback_detector_path.is_file():
                    raise FileNotFoundError('configured fallback detector model is missing')
                if settings.recovery_detector_path is not None and not settings.recovery_detector_path.is_file():
                    raise FileNotFoundError('configured recovery detector model is missing')
                _pipeline = Pipeline(
                    detector_path=settings.detector_path,
                    ocr_path=settings.ocr_path,
                    det_conf=settings.detection_conf,
                    imgsz=settings.detection_imgsz,
                    nms_iou=settings.detection_nms_iou,
                    max_plates=settings.detection_max_plates,
                    layout_mode=settings.ocr_layout_mode,
                    device=settings.device,
                    authority_model_path=settings.authority_path,
                    authority_min_prefix_confidence=settings.authority_min_prefix_confidence,
                    fallback_detector_path=settings.fallback_detector_path,
                    fallback_det_conf=settings.fallback_detection_conf,
                    recovery_detector_path=settings.recovery_detector_path,
                    recovery_det_conf=settings.recovery_detection_conf,
                    recovery_layout_mode=settings.recovery_ocr_layout_mode,
                    merge_fallback_fragments=settings.merge_fallback_fragments,
                )
    return _pipeline


def _get_camera_pipeline() -> Pipeline:
    global _camera_pipeline
    settings = get_settings()
    detector_path = settings.camera_detector_path
    recovery_path = settings.camera_recovery_detector_path
    merge = settings.camera_merge_fallback_fragments
    if detector_path is None and recovery_path == settings.recovery_detector_path and merge is None:
        return _get_pipeline()
    detector_path = detector_path or settings.detector_path
    if merge is None:
        merge = settings.merge_fallback_fragments
    if _camera_pipeline is None:
        with _pipeline_lock:
            if _camera_pipeline is None:
                if not detector_path.is_file():
                    raise FileNotFoundError(f"camera detector model not found: {detector_path}")
                if not settings.ocr_path.is_file():
                    raise FileNotFoundError(f"OCR model not found: {settings.ocr_path}")
                if settings.authority_path is not None and not settings.authority_path.is_file():
                    raise FileNotFoundError('configured authority OCR model is missing')
                if settings.fallback_detector_path is not None and not settings.fallback_detector_path.is_file():
                    raise FileNotFoundError('configured fallback detector model is missing')
                if recovery_path is not None and not recovery_path.is_file():
                    raise FileNotFoundError('configured recovery detector model is missing')
                _camera_pipeline = Pipeline(
                    detector_path=detector_path,
                    ocr_path=settings.ocr_path,
                    det_conf=settings.detection_conf,
                    imgsz=settings.detection_imgsz,
                    nms_iou=settings.detection_nms_iou,
                    max_plates=settings.detection_max_plates,
                    layout_mode=settings.ocr_layout_mode,
                    device=settings.device,
                    authority_model_path=settings.authority_path,
                    authority_min_prefix_confidence=settings.authority_min_prefix_confidence,
                    fallback_detector_path=settings.fallback_detector_path,
                    fallback_det_conf=settings.fallback_detection_conf,
                    recovery_detector_path=recovery_path,
                    recovery_det_conf=settings.recovery_detection_conf,
                    recovery_layout_mode=settings.recovery_ocr_layout_mode,
                    merge_fallback_fragments=merge,
                )
    return _camera_pipeline


def _run_image(path: Path, filename: str, camera: bool = False) -> dict:
    with _inference_lock:
        result = (_get_camera_pipeline() if camera else _get_pipeline()).run(path)
        if camera:
            settings = get_settings()
            vehicles, status = vehicle_evidence(
                path, result["detections"], settings.vehicle_detector_path,
                settings.vehicle_attribute_path, settings.device,
                structured_model=settings.vehicle_structured_path,
            )
            result["vehicles"] = [vehicle.model_dump(mode="json") for vehicle in vehicles]
            result["vehicle_status"] = status
    result["image_path"] = filename
    return result


@router.get("/health")
def health() -> dict:
    """Report service and configured model-file availability without loading models."""
    settings = get_settings()
    detector_ok = settings.detector_path.is_file()
    camera_detector_ok = settings.camera_detector_path is None or settings.camera_detector_path.is_file()
    ocr_ok = settings.ocr_path.is_file()
    authority_ok = settings.authority_path is None or settings.authority_path.is_file()
    fallback_ok = settings.fallback_detector_path is None or settings.fallback_detector_path.is_file()
    recovery_ok = settings.recovery_detector_path is None or settings.recovery_detector_path.is_file()
    camera_recovery = settings.camera_recovery_detector_path
    camera_recovery_ok = camera_recovery is None or camera_recovery.is_file()
    ready = detector_ok and camera_detector_ok and ocr_ok and authority_ok and fallback_ok and recovery_ok and camera_recovery_ok
    return {
        "status": "ready" if ready else "model_missing",
        "models_loaded": _pipeline is not None,
        "camera_models_loaded": _camera_pipeline is not None,
        "detector_available": detector_ok,
        "camera_detector_available": camera_detector_ok,
        "ocr_available": ocr_ok,
        "authority_region_enabled": settings.authority_path is not None,
        "authority_region_available": authority_ok,
        "fallback_detector_enabled": settings.fallback_detector_path is not None,
        "fallback_detector_available": fallback_ok,
        "recovery_detector_enabled": settings.recovery_detector_path is not None,
        "recovery_detector_available": recovery_ok,
        "camera_recovery_detector_enabled": camera_recovery is not None,
        "camera_recovery_detector_available": camera_recovery_ok,
        "device": settings.device,
    }


async def analyze_upload(file: UploadFile, *, camera: bool = False, export_source: str | None = None) -> dict:
    """Analyze a checked upload with the general or configured camera detector."""
    settings = get_settings()
    declared_type = (file.content_type or "").split(";")[0].strip().lower()
    if declared_type not in settings.allowed_mime_types:
        await file.close()
        raise HTTPException(status_code=415, detail="only JPEG and PNG images are accepted")

    try:
        content = await file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="uploaded image exceeds the size limit")
        try:
            with Image.open(io.BytesIO(content)) as image:
                detected_type = image.format
                if detected_type not in {"JPEG", "PNG"}:
                    raise HTTPException(status_code=415, detail="uploaded file is not JPEG or PNG")
                if (image.width * image.height) > 50_000_000:
                    raise HTTPException(status_code=413, detail="image dimensions exceed the 50 megapixel limit")
                image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(status_code=422, detail="uploaded file is not a readable image") from exc

        suffix = ".jpg" if detected_type == "JPEG" else ".png"
        safe_name = Path(file.filename or "upload").name
        with tempfile.TemporaryDirectory(prefix="anpr_upload_") as temp_dir:
            image_path = Path(temp_dir) / f"input{suffix}"
            image_path.write_bytes(content)
            try:
                result = await run_in_threadpool(_run_image, image_path, safe_name, camera)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except AuthorityReaderUnavailable as exc:
                raise HTTPException(status_code=503,detail=str(exc)) from exc
            except (ValueError, OSError) as exc:
                raise HTTPException(status_code=422, detail=f"could not process image: {exc}") from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail="inference failed") from exc
            if export_source is not None:
                try:
                    result["export"] = await run_in_threadpool(
                        save_export, content, suffix, result, source=export_source,
                    )
                except OSError as exc:
                    raise HTTPException(status_code=503, detail="capture export could not be saved") from exc
            return result
    finally:
        await file.close()


@router.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """Accept one JPEG/PNG scene and return detections, OCR and Jordanian decisions."""
    return await analyze_upload(file)


@router.post('/demo/predict')
async def predict_demo(file: UploadFile = File(...)) -> dict:
    """Analyze a demo upload and save its original image and XML together."""
    return await analyze_upload(file, camera=True, export_source="upload")



class PlateParts(BaseModel):
    full_text: str
    plate_code: str | None
    plate_number: str | None
    split_basis: str
    requires_human_review: bool


class DetectionEvidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    box: tuple[float, float, float, float]
    det_conf: float


class ReadingEvidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    crop_box: tuple[float, float, float, float]
    text: str
    ocr_conf: float
    plate_parts: PlateParts


class InferenceResponse(BaseModel):
    api_version: str
    profile: str
    checkpoints: dict[str, str]
    image_path: str
    detections: list[DetectionEvidence]
    ocr_results: list[ReadingEvidence]
    vehicles: list[Vehicle] = Field(default_factory=list)
    vehicle_status: VehicleStatus = "disabled"
    export: CaptureExport | None = None
    validations: list[dict]
    decisions: list[dict]
    meta: dict


_inference_key_header = APIKeyHeader(name="X-Inference-Key", auto_error=False)


def _authorize_inference(request: Request, key: str | None = Security(_inference_key_header)) -> None:
    settings = get_settings()
    if settings.inference_api_key and key and hmac.compare_digest(
            key.encode(), settings.inference_api_key.encode()):
        return
    if settings.inference_admin_key and key and valid_issued_key(key):
        return
    if settings.inference_api_key or settings.inference_admin_key:
        raise HTTPException(status_code=401, detail="invalid inference API key")
    try:
        local = request.client is not None and ip_address(request.client.host).is_loopback
        loopback_bind = ip_address(settings.host).is_loopback
    except ValueError:
        local = loopback_bind = False
    if not local or not loopback_bind:
        raise HTTPException(status_code=503, detail="inference API key required for network deployment")


@lru_cache(maxsize=16)
def _checkpoint_digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_inference_profile() -> tuple[str, dict[str, str]]:
    """Load the configured gate profile once per worker and identify its weights."""
    settings = get_settings()
    paths = {
        "primary": settings.camera_detector_path or settings.detector_path,
        "ocr": settings.ocr_path,
        "fallback": settings.fallback_detector_path,
        "recovery": settings.camera_recovery_detector_path,
        "authority": settings.authority_path,
    }
    try:
        checkpoints = {role: _checkpoint_digest(path) for role, path in paths.items()
                       if path is not None}
        _get_camera_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="inference profile unavailable") from exc
    return "gate-camera-v1", checkpoints


@router.get("/v1/ready", tags=["inference-v1"], dependencies=[Depends(_authorize_inference)])
def inference_ready() -> dict:
    """Load and verify the versioned profile, returning 503 on initialization failure."""
    profile, checkpoints = load_inference_profile()
    return {"status": "ready", "api_version": "v1", "profile": profile, "checkpoints": checkpoints}


@router.post(
    "/v1/predict",
    tags=["inference-v1"],
    dependencies=[Depends(_authorize_inference)],
    response_model=InferenceResponse,
    responses={code: {"description": description} for code, description in {
        401: "Invalid inference key",
        413: "Image exceeds size or dimension limits",
        415: "Unsupported image type",
        422: "Missing or unreadable image",
        500: "Inference failed",
        503: "Profile unavailable or service key missing",
    }.items()},
)
async def predict_v1(file: UploadFile = File(...), save_xml: bool = Form(False)) -> dict:
    """Analyze a scene and optionally save an image/XML pair."""
    try:
        profile, checkpoints = await run_in_threadpool(load_inference_profile)
        result = await analyze_upload(file, camera=True, export_source="api" if save_xml else None)
    except HTTPException as exc:
        if exc.status_code == 422 and str(exc.detail).startswith("could not process image:"):
            raise HTTPException(status_code=422, detail="could not process image") from exc
        raise
    finally:
        await file.close()
    # Legacy responses carry local paths. The integration API publishes hashes instead.
    meta = {key: value for key, value in result["meta"].items()
            if key not in {"detector", "ocr", "fallback_detector", "recovery_detector"}}
    return {**result, "meta": meta, "api_version": "v1", "profile": profile,
            "checkpoints": checkpoints}
