"""Phase 11/12 - Detector application package.

    from app.detector import PlateDetector

Re-exports the detector wrapper used by the end-to-end pipeline.
"""

from __future__ import annotations

from .plate_detector import (
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    Detection,
    PlateDetector,
)

__all__ = [
    "PlateDetector",
    "Detection",
    "DEFAULT_IMGSZ",
    "DEFAULT_CONF",
    "DEFAULT_IOU",
]
