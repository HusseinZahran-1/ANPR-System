"""Phase 11 - End-to-End plate pipeline (detector -> OCR -> validator -> policy).

Wires together the frozen artifacts of earlier phases, using **saved models
only** (no runtime training or fine-tuning). An optional, separately installed
English prefix reader supports visual CD/SN evidence:

    full scene image
      -> YOLOv8n detector           (models/phase6_detector/.../weights/best.pt)
      -> plate crop(s)
      -> Jordan-tuned CRNN (VGG+2xBiLSTM+CTC) (models/phase13_ocr/.../weights/best.pt)
      -> recognized text
      -> Phase 9 Jordanian validator (app/validator/jordanian_validator.py)
      -> Phase 10 foreign policy     (app/validator/foreign_policy.py)
      -> per-plate decision

The CRNN architecture and the greedy CTC decoder live in the application layer
(:mod:`app.ocr.ocr_engine`), relocated verbatim from the Phase 8 training script
so the inference path is the same model Phase 8 trained and evaluated, while the
pipeline no longer imports training code through ``sys.path`` manipulation. The
detector is accessed through :class:`app.detector.PlateDetector`. Only ``best.pt``
is used for the detector and primary OCR; optional prefix weights are verified
separately. Region readings retain their original OCR evidence and require review.

Confidence handling (DESIGN section D): three separate values are exposed and
**never fused** -- detector confidence, OCR confidence, and the rule-based
validator classification (which has no confidence). OCR confidence is an
uncalibrated proxy: the mean max-softmax score at the first timestep emitting
each greedy CTC character. Repeated frames no longer inflate the value. It is
``0.0`` when the decoder emits nothing. It must not be interpreted as a
probability that the complete plate text is correct.

Multi-plate support is fully generic (iterate over all detected boxes). The V1
dataset is single-plate-per-scene, so the multi-plate path is intentionally
implemented but not measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from app.ocr.layout import SUPPORTED_LAYOUTS, prepare_layout

# Application-layer modules (no sys.path manipulation, no training-script import).
from app.detector import DEFAULT_CONF, DEFAULT_IOU, PlateDetector
from app.ocr import conv_out_width, greedy_decode, load_crnn_checkpoint
from app.ocr.authority_review import possible_authority_text
from app.ocr.authority_regions import AuthorityRegionReader,load_prefix_reader
from app.ocr.spatial import vgg_ctc_tokens
from app.pipeline.io import (
    crop_from_box,
    load_scene,
    preprocess_for_ocr,
)
from app.validator.foreign_policy import decide
from app.validator.jordanian_validator import validate

ROOT = Path(__file__).resolve().parents[2]

SEED = 42
# The Phase 13 pipeline uses 1280 after same-cohort validation. The standalone
# detector retains its historical 960 default for reproducible Phase 6 usage.
DEFAULT_IMGSZ = 1280
FOREIGN_POLICY_ID = "phase9_validator+phase10_foreign_policy"


def _review_required(policy_decision: dict) -> dict:
    """Keep legacy format policy while declaring its operational limit."""
    return {**policy_decision, "country_verified": False,
            "requires_human_review": True}


DETECTOR_DEFAULT = (ROOT / "models" / "phase6_detector" /
                    "yolov8n_plate_baseline" / "weights" / "best.pt")
OCR_DEFAULT = (ROOT / "models" / "phase13_ocr" /
               "jordan_crnn_v4_layout" / "weights" / "best.pt")


@dataclass
class PipelineResult:
    """Typed optional view over :meth:`Pipeline.run`'s dict (same fields)."""

    image_path: str
    detections: list = field(default_factory=list)
    ocr_results: list = field(default_factory=list)
    validations: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineResult":
        return cls(
            image_path=d["image_path"],
            detections=d.get("detections", []),
            ocr_results=d.get("ocr_results", []),
            validations=d.get("validations", []),
            decisions=d.get("decisions", []),
            meta=d.get("meta", {}),
        )


def load_models(detector_path=DETECTOR_DEFAULT, ocr_path=OCR_DEFAULT,
                device=None, det_conf: float = DEFAULT_CONF,
                imgsz: int = DEFAULT_IMGSZ, nms_iou: float = DEFAULT_IOU):
    """Load the two frozen models and return them with their metadata.

    Returns a dict: ``detector`` (an :class:`app.detector.PlateDetector`),
    ``ocr`` (``CRNN`` in eval mode), ``alphabet``, ``height``, ``wmax``,
    ``num_class``, ``ckpt``, ``device``. No model is modified or retrained.
    """
    det_path = Path(detector_path)
    ocr_p = Path(ocr_path)
    if not det_path.is_file():
        raise FileNotFoundError(f"detector weights not found: {det_path}")
    if not ocr_p.is_file():
        raise FileNotFoundError(f"OCR weights not found: {ocr_p}")

    dev = torch.device(device) if device else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    detector = PlateDetector(det_path, device=dev, imgsz=imgsz, conf=det_conf,
                             iou=nms_iou)
    ocr_loaded = load_crnn_checkpoint(ocr_p, device=dev)

    return {
        "detector": detector,
        "ocr": ocr_loaded["model"],
        "alphabet": ocr_loaded["alphabet"],
        "height": ocr_loaded["height"],
        "wmax": ocr_loaded["wmax"],
        "min_resized_w": ocr_loaded["min_resized_w"],
        "pad_to_max_width": ocr_loaded["pad_to_max_width"],
        "layout_mode": ocr_loaded["layout_mode"],
        "num_class": ocr_loaded["num_class"],
        "ckpt": ocr_loaded["ckpt"],
        "device": ocr_loaded["device"],
    }


class Pipeline:
    """End-to-end scene -> decision pipeline using the frozen Phase 6/8 models."""

    def __init__(self, detector_path=DETECTOR_DEFAULT, ocr_path=OCR_DEFAULT,
                 det_conf: float = DEFAULT_CONF, imgsz: int = DEFAULT_IMGSZ,
                 nms_iou: float = DEFAULT_IOU, device=None,
                 crop_pad_pct: float = 0.0, layout_mode: str | None = None,
                 max_plates: int | None = None,
                 authority_model_path: str | Path | None = None,
                 authority_min_prefix_confidence: float = .8):
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.det_conf = float(det_conf)
        self.imgsz = int(imgsz)
        self.nms_iou = float(nms_iou)
        if max_plates is not None and max_plates < 1:
            raise ValueError("max_plates must be positive or None")
        self.max_plates = max_plates
        if crop_pad_pct < 0 or crop_pad_pct > 0.15:
            raise ValueError("crop_pad_pct must be between 0 and 0.15")
        self.crop_pad_pct = float(crop_pad_pct)

        loaded = load_models(detector_path, ocr_path, device=device,
                             det_conf=self.det_conf, imgsz=self.imgsz,
                             nms_iou=self.nms_iou)
        self.detector = loaded["detector"]        # app.detector.PlateDetector
        self.ocr = loaded["ocr"]
        self.alphabet = loaded["alphabet"]
        self.ocr_h = loaded["height"]
        self.ocr_wmax = loaded["wmax"]
        # Keep compatibility with lightweight injected model bundles used by
        # callers/tests predating checkpoint width metadata.
        self.ocr_min_width = loaded.get("min_resized_w", 8)
        self.ocr_pad_to_max_width = loaded.get("pad_to_max_width", False)
        checkpoint_layout = loaded.get("layout_mode", "none")
        self.ocr_layout_mode = layout_mode or checkpoint_layout
        if self.ocr_layout_mode not in SUPPORTED_LAYOUTS:
            raise ValueError(f"unsupported OCR layout preprocessing: {self.ocr_layout_mode}")
        self.device = loaded["device"]
        self.detector_path = str(Path(detector_path))
        self.ocr_path = str(Path(ocr_path))
        self.authority_reader = None
        if authority_model_path is not None:
            prefix_reader = load_prefix_reader(str(Path(authority_model_path).resolve()),str(self.device))
            self.authority_reader = AuthorityRegionReader(prefix_reader,self._ocr_spatial_tokens,
                                                          min_prefix_confidence=authority_min_prefix_confidence)

    # ------------------------------------------------------------------ #
    def _detect(self, image_path) -> list[dict]:
        """Run the detector on an image PATH and return boxes sorted desc.

        Delegates to :class:`app.detector.PlateDetector`, which passes the path
        (not a numpy array) to Ultralytics so the byte-level preprocessing is
        identical to Phase 6's evaluation.
        """
        return self.detector.detect(image_path)

    def _ocr_crop(self, crop_rgb: np.ndarray) -> tuple[str, float]:
        """Greedy-decode one crop; returns (text, ocr_conf)."""
        if crop_rgb.size == 0 or crop_rgb.shape[0] < 1 or crop_rgb.shape[1] < 1:
            return "", 0.0
        crop_rgb = np.array(prepare_layout(Image.fromarray(crop_rgb), self.ocr_layout_mode))
        tensor = preprocess_for_ocr(crop_rgb, target_h=self.ocr_h,
                                    max_w=self.ocr_wmax,
                                    min_resized_w=self.ocr_min_width,
                                    pad_to_max_width=self.ocr_pad_to_max_width)
        crop_h, crop_w = crop_rgb.shape[:2]
        valid_width = max(self.ocr_min_width,
                          min(int(round(self.ocr_h * crop_w / float(crop_h))),
                              self.ocr_wmax))
        valid_steps = conv_out_width(valid_width)
        with torch.no_grad():
            logits = self.ocr(tensor.to(self.device))      # (1, T, C)
        text, conf = greedy_decode(logits[0, :valid_steps].cpu())
        return text, round(float(conf), 6)

    def _ocr_spatial_tokens(self,plate_rgb: np.ndarray) -> list[dict]:
        """Read an explicitly localized single-row plate view without unfolding."""
        height,width = plate_rgb.shape[:2]
        resized = max(self.ocr_min_width,min(round(self.ocr_h*width/height),self.ocr_wmax))
        tensor = preprocess_for_ocr(plate_rgb,target_h=self.ocr_h,max_w=self.ocr_wmax,
                                    min_resized_w=self.ocr_min_width,pad_to_max_width=self.ocr_pad_to_max_width)
        with torch.no_grad():
            logits = self.ocr(tensor.to(self.device))[0,:conv_out_width(resized)]
        return vgg_ctc_tokens(logits,self.alphabet,source_width=width,resized_width=resized)

    def _recognize_crop(self,crop: np.ndarray) -> dict:
        text,confidence = self._ocr_crop(crop)
        entry = {'text':text,'ocr_conf':confidence}
        if self.authority_reader is not None:
            region = self.authority_reader.read(Image.fromarray(crop))
            if region is not None:
                entry.update(text=region['text'],primary_ocr_text=text,authority_region=region,
                             ocr_confidence_scope='primary_reading_before_region_ocr')
        return entry

    # ------------------------------------------------------------------ #
    def run(self, image_path) -> dict:
        """Full pipeline for ONE scene. Returns the structured result dict."""
        scene = load_scene(image_path)
        detections = self._detect(image_path)
        candidate_count = len(detections)
        if self.max_plates is not None:
            detections = sorted(detections, key=lambda det: det["det_conf"],
                                reverse=True)[:self.max_plates]

        ocr_results: list[dict] = []
        validations: list[dict] = []
        decisions: list[dict] = []

        for det in detections:
            box = det["box"]
            try:
                crop = crop_from_box(scene, box, pad_pct=self.crop_pad_pct)
            except ValueError:
                # Degenerate box (e.g. fully outside bounds): record no text.
                ocr_results.append({"crop_box": box, "text": "", "ocr_conf": 0.0,
                                    "crop_error": "empty crop after clipping"})
                v = validate("")
                validations.append({"text": "", **v})
                decisions.append(_review_required(decide(
                    v["classification"], v["matched_rule"], v["normalized"])))
                continue

            ocr_entry = {"crop_box": box, **self._recognize_crop(crop)}
            text = ocr_entry['text']
            suggestion = possible_authority_text(text)
            if suggestion:
                ocr_entry["possible_authority_text"] = suggestion
                ocr_entry["possible_authority_review_only"] = True
            ocr_results.append(ocr_entry)

            v = validate(text)
            validations.append({"text": text, **v})
            decision = _review_required(decide(v["classification"],v["matched_rule"],v["normalized"]))
            if 'authority_region' in ocr_entry:
                decision.update(decision='FLAG_REVIEW',reason='separate prefix and number reading; verify image and country')
            decisions.append(decision)

        if not detections:
            decisions.append({
                "decision": "FLAG_REVIEW",
                "reason": f"no plate detected above threshold {self.det_conf:g}",
                "matched_rule": None,
                "speculative": False,
                "country_verified": False,
                "requires_human_review": True,
            })

        return {
            "image_path": str(image_path),
            "detections": detections,
            "ocr_results": ocr_results,
            "validations": validations,
            "decisions": decisions,
            "meta": {
                "detector": self.detector_path,
                "ocr": self.ocr_path,
                "validator": FOREIGN_POLICY_ID,
                "country_verified": False,
                "requires_human_review": True,
                "det_conf_threshold": self.det_conf,
                "det_nms_iou": self.nms_iou,
                "max_plates": self.max_plates,
                "detector_candidate_count": candidate_count,
                "imgsz": self.imgsz,
                "ocr_min_resized_w": self.ocr_min_width,
                "ocr_pad_to_max_width": self.ocr_pad_to_max_width,
                "ocr_layout_mode": self.ocr_layout_mode,
                "authority_region_ocr": self.authority_reader is not None,
                "crop_pad_pct": self.crop_pad_pct,
                "device": str(self.device),
                "seed": SEED,
            },
        }


_DEFAULT_PIPELINE: Pipeline | None = None


def run(image_path, **kwargs) -> dict:
    """Convenience: run a module-level cached default :class:`Pipeline`.

    Extra keyword arguments configure the pipeline on **first** construction
    (they are ignored on later calls).
    """
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        _DEFAULT_PIPELINE = Pipeline(**kwargs)
    return _DEFAULT_PIPELINE.run(image_path)


__all__ = [
    "Pipeline",
    "PipelineResult",
    "load_models",
    "run",
    "DETECTOR_DEFAULT",
    "OCR_DEFAULT",
    "SEED",
    "DEFAULT_CONF",
]
