"""Phase 12 - API configuration.

Environment-driven configuration for the FastAPI service. Deliberately a plain
:class:`dataclasses.dataclass` (no ``pydantic-settings`` dependency), per the
Phase 12 brief and ``AI_DEVELOPMENT_PLAN.md`` §36.

Every value can be overridden through an environment variable; the defaults
mirror the owner-approved Phase 11 pipeline settings and the frozen model
locations. Relative paths are resolved against the repository root (the
directory that contains ``app/``, ``models/`` and ``logs/``).

    MODEL_DETECTOR_PATH   detector weights   (default Phase 6 best.pt)
    CAMERA_DETECTOR_PATH  optional detector for camera uploads only
    MODEL_OCR_PATH        OCR weights        (default Phase 13 Jordan CRNN)
    DETECTION_CONF        detector threshold (default 0.25)
    DETECTION_IMGSZ       detector image size (default 1280)
    DETECTION_NMS_IOU     overlap suppression IoU for the MVP (default 0.50)
    DETECTION_MAX_PLATES  maximum returned plates in gate demo (default 1)
    CAMERA_API_KEY        shared key for camera ingestion and live viewing
    CAMERA_ISAPI_TOKEN    secret path token for cameras without custom headers
    MAX_UPLOAD_BYTES      upload cap         (default 10 MiB)
    ALLOWED_MIME_TYPES    comma-separated    (default image/jpeg,image/png)
    DEVICE                "cuda" / "cpu"     (default cuda if available else cpu)
    HOST                  bind host          (default 127.0.0.1)
    PORT                  bind port          (default 8000)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# app/api/config.py -> parents[2] == repository root
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=False)

DEFAULT_DETECTOR_PATH = "models/phase6_detector/yolov8n_plate_baseline/weights/best.pt"
DEFAULT_OCR_PATH = "models/phase13_ocr/jordan_crnn_v4_layout/weights/best.pt"
DEFAULT_DETECTION_CONF = 0.25
DEFAULT_DETECTION_IMGSZ = 1280
DEFAULT_DETECTION_NMS_IOU = 0.50
DEFAULT_DETECTION_MAX_PLATES = 1
DEFAULT_OCR_LAYOUT_MODE = "rectify_unfold_wide_v3"
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB
DEFAULT_ALLOWED_MIME_TYPES = ("image/jpeg", "image/png")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    parts = tuple(p.strip().lower() for p in raw.split(",") if p.strip())
    return parts or default


def default_device() -> str:
    """Return ``"cuda"`` when a CUDA device is usable, else ``"cpu"`` (§37)."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # torch missing / CUDA probe failed -> graceful CPU fallback
        return "cpu"


def resolve_path(path: str | os.PathLike) -> Path:
    """Resolve a possibly-relative path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


@dataclass
class Settings:
    """Runtime settings for the ANPR API service."""

    detector_model_path: str = DEFAULT_DETECTOR_PATH
    camera_detector_model_path: str = ""
    ocr_model_path: str = DEFAULT_OCR_PATH
    authority_model_path: str = ""
    authority_min_prefix_confidence: float = .8
    detection_conf: float = DEFAULT_DETECTION_CONF
    detection_imgsz: int = DEFAULT_DETECTION_IMGSZ
    detection_nms_iou: float = DEFAULT_DETECTION_NMS_IOU
    detection_max_plates: int = DEFAULT_DETECTION_MAX_PLATES
    camera_api_key: str = field(default="", repr=False)
    camera_isapi_token: str = field(default="", repr=False)
    ocr_layout_mode: str = DEFAULT_OCR_LAYOUT_MODE
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    allowed_mime_types: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_ALLOWED_MIME_TYPES
    )
    device: str = "cpu"
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT

    def __post_init__(self) -> None:
        from app.ocr.layout import SUPPORTED_LAYOUTS
        if self.ocr_layout_mode not in SUPPORTED_LAYOUTS:
            raise ValueError(f"OCR_LAYOUT_MODE must be one of {SUPPORTED_LAYOUTS}")
        if self.detection_imgsz <= 0 or self.detection_imgsz % 32:
            raise ValueError("DETECTION_IMGSZ must be a positive multiple of 32")
        if not 0 < self.detection_nms_iou < 1:
            raise ValueError("DETECTION_NMS_IOU must be between 0 and 1")
        if self.detection_max_plates < 1:
            raise ValueError("DETECTION_MAX_PLATES must be positive")
        if not 0 < self.authority_min_prefix_confidence <= 1:
            raise ValueError('AUTHORITY_PREFIX_MIN_CONF must be between zero and one')

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            detector_model_path=_env("MODEL_DETECTOR_PATH", DEFAULT_DETECTOR_PATH),
            camera_detector_model_path=_env("CAMERA_DETECTOR_PATH", ""),
            ocr_model_path=_env("MODEL_OCR_PATH", DEFAULT_OCR_PATH),
            authority_model_path=_env('AUTHORITY_REGION_MODEL_PATH',''),
            authority_min_prefix_confidence=_env_float('AUTHORITY_PREFIX_MIN_CONF',.8),
            detection_conf=_env_float("DETECTION_CONF", DEFAULT_DETECTION_CONF),
            detection_imgsz=_env_int("DETECTION_IMGSZ", DEFAULT_DETECTION_IMGSZ),
            detection_nms_iou=_env_float("DETECTION_NMS_IOU", DEFAULT_DETECTION_NMS_IOU),
            detection_max_plates=_env_int("DETECTION_MAX_PLATES", DEFAULT_DETECTION_MAX_PLATES),
            camera_api_key=_env("CAMERA_API_KEY", ""),
            camera_isapi_token=_env("CAMERA_ISAPI_TOKEN", ""),
            ocr_layout_mode=_env("OCR_LAYOUT_MODE", DEFAULT_OCR_LAYOUT_MODE),
            max_upload_bytes=_env_int("MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES),
            allowed_mime_types=_env_list(
                "ALLOWED_MIME_TYPES", DEFAULT_ALLOWED_MIME_TYPES
            ),
            device=_env("DEVICE", default_device()),
            host=_env("HOST", DEFAULT_HOST),
            port=_env_int("PORT", DEFAULT_PORT),
        )

    # Resolved (absolute) paths for the model loaders.
    @property
    def detector_path(self) -> Path:
        return resolve_path(self.detector_model_path)

    @property
    def camera_detector_path(self) -> Path | None:
        return resolve_path(self.camera_detector_model_path) if self.camera_detector_model_path else None

    @property
    def ocr_path(self) -> Path:
        return resolve_path(self.ocr_model_path)

    @property
    def authority_path(self) -> Path | None:
        return resolve_path(self.authority_model_path) if self.authority_model_path else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached process-wide settings built from the environment."""
    return Settings.from_env()


__all__ = [
    "Settings",
    "get_settings",
    "default_device",
    "resolve_path",
    "ROOT",
    "DEFAULT_DETECTOR_PATH",
    "DEFAULT_OCR_PATH",
    "DEFAULT_DETECTION_CONF",
    "DEFAULT_DETECTION_IMGSZ",
    "DEFAULT_DETECTION_NMS_IOU",
    "DEFAULT_DETECTION_MAX_PLATES",
    "DEFAULT_OCR_LAYOUT_MODE",
    "DEFAULT_MAX_UPLOAD_BYTES",
    "DEFAULT_ALLOWED_MIME_TYPES",
]
