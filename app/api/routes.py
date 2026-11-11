"""HTTP endpoints for scene-level Jordanian plate recognition."""

from __future__ import annotations

import io
import tempfile
import threading
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.api.config import get_settings
from app.pipeline import Pipeline
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
                )
    return _pipeline


def _get_camera_pipeline() -> Pipeline:
    global _camera_pipeline
    settings = get_settings()
    detector_path = settings.camera_detector_path
    if detector_path is None:
        return _get_pipeline()
    if _camera_pipeline is None:
        with _pipeline_lock:
            if _camera_pipeline is None:
                if not detector_path.is_file():
                    raise FileNotFoundError(f"camera detector model not found: {detector_path}")
                if not settings.ocr_path.is_file():
                    raise FileNotFoundError(f"OCR model not found: {settings.ocr_path}")
                if settings.authority_path is not None and not settings.authority_path.is_file():
                    raise FileNotFoundError('configured authority OCR model is missing')
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
                )
    return _camera_pipeline


def _run_image(path: Path, filename: str, camera: bool = False) -> dict:
    with _inference_lock:
        result = (_get_camera_pipeline() if camera else _get_pipeline()).run(path)
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
    ready = detector_ok and camera_detector_ok and ocr_ok and authority_ok
    return {
        "status": "ready" if ready else "model_missing",
        "models_loaded": _pipeline is not None,
        "camera_models_loaded": _camera_pipeline is not None,
        "detector_available": detector_ok,
        "camera_detector_available": camera_detector_ok,
        "ocr_available": ocr_ok,
        "authority_region_enabled": settings.authority_path is not None,
        "authority_region_available": authority_ok,
        "device": settings.device,
    }


async def analyze_upload(file: UploadFile, *, camera: bool = False) -> dict:
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
                return await run_in_threadpool(_run_image, image_path, safe_name, camera)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except AuthorityReaderUnavailable as exc:
                raise HTTPException(status_code=503,detail=str(exc)) from exc
            except (ValueError, OSError) as exc:
                raise HTTPException(status_code=422, detail=f"could not process image: {exc}") from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail="inference failed") from exc
    finally:
        await file.close()


@router.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """Accept one JPEG/PNG scene and return detections, OCR and Jordanian decisions."""
    return await analyze_upload(file)


@router.post('/demo/predict')
async def predict_demo(file: UploadFile = File(...)) -> dict:
    """Gate-camera demo inference using the configured camera detector, no archive."""
    return await analyze_upload(file,camera=True)
