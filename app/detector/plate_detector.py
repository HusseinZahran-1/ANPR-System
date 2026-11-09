"""Phase 11/12 - Plate detector wrapper (Ultralytics YOLOv8n).

Thin, explicit interface around the frozen Phase 6 detector so the pipeline does
not call ``ultralytics`` directly and the exact inference call semantics used in
Phase 6/11 are preserved:

    source  -> the image **path** (not a numpy array), so Ultralytics applies the
               same byte-level preprocessing as Phase 6 evaluation and no
               RGB/BGR channel-order mismatch can occur;
    imgsz   -> 960  (Phase 6 training/eval size);
    conf    -> 0.25 (Phase 6 pre-registered reporting threshold; default);
    iou     -> 0.7  (Phase 6 NMS IoU);
    verbose -> False.

``detect()`` returns a list of ``{"box": [x1, y1, x2, y2], "det_conf": float}``
sorted by descending confidence. The wrapper performs **no** OCR and has no
knowledge of plate text or country rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Phase 6 pre-registered settings.
DEFAULT_IMGSZ = 960
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7


@dataclass
class Detection:
    """One detected plate box in pixel coordinates."""

    box: list[int]      # [x1, y1, x2, y2]
    det_conf: float


class PlateDetector:
    """Wraps an Ultralytics YOLO model with a stable, testable interface."""

    def __init__(self, weights, device=None, imgsz: int = DEFAULT_IMGSZ,
                 conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU):
        import torch
        from ultralytics import YOLO

        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f"detector weights not found: {path}")

        self.weights = str(path)
        self.device = torch.device(device) if device else torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.model = YOLO(str(path))

    def _device_arg(self):
        """Ultralytics accepts an int GPU index or the string ``"cpu"``."""
        return 0 if self.device.type == "cuda" else "cpu"

    def detect(self, image_path) -> list[dict]:
        """Run detection on an image path; return boxes sorted by confidence."""
        res = self.model.predict(
            source=str(image_path), imgsz=self.imgsz, conf=self.conf,
            iou=self.iou, verbose=False, device=self._device_arg())[0]
        dets: list[dict] = []
        if res.boxes is None or len(res.boxes) == 0:
            return dets
        xyxy = res.boxes.xyxy.detach().cpu().numpy()
        confs = res.boxes.conf.detach().cpu().numpy()
        for box, c in zip(xyxy, confs):
            dets.append({
                "box": [int(round(float(v))) for v in box.tolist()],
                "det_conf": round(float(c), 6),
            })
        dets.sort(key=lambda d: d["det_conf"], reverse=True)
        return dets


__all__ = ["PlateDetector", "Detection", "DEFAULT_IMGSZ", "DEFAULT_CONF", "DEFAULT_IOU"]
