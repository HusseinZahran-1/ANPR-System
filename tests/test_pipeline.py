"""Unit/integration-lite tests - end-to-end Pipeline wiring (app.pipeline).

The detector and OCR models are replaced by deterministic stubs, so these tests
exercise *only* the pipeline's orchestration: cropping, per-plate iteration,
validator/policy chaining, the output schema, and the guarantee that the three
confidence-like values are never fused.

No real model is loaded and no protected test data is used (synthetic images).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import cv2
from PIL import Image

import app.pipeline.end_to_end as e2e
from app.ocr import BLANK, NUM_CLASS, encode
from app.pipeline import Pipeline
from app.pipeline.io import crop_from_box, load_scene


# --------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------
class StubDetector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, image_path):
        return [dict(d) for d in self._detections]


def test_region_reading_keeps_primary_evidence_and_forces_review(monkeypatch,scene_path):
    _install_stubs(monkeypatch,[{'box':[10,10,80,70],'det_conf':.9}],ocr_text='12')
    monkeypatch.setattr(e2e,'load_prefix_reader',lambda *args:None)
    class StubRegionReader:
        def __init__(self,*args,**kwargs):
            pass
        def read(self,image):
            return {'text':'CD12','prefix_text':'CD','numeric_suffix':'12',
                    'prefix_confidence':.9,'suffix_confidence':.8,'requires_human_review':True}
    monkeypatch.setattr(e2e,'AuthorityRegionReader',StubRegionReader)
    result=Pipeline(authority_model_path='synthetic.pth').run(scene_path)
    ocr=result['ocr_results'][0]
    assert ocr['text']=='CD12' and ocr['primary_ocr_text']=='12'
    assert ocr['ocr_confidence_scope']=='primary_reading_before_region_ocr'
    assert result['decisions'][0]['decision']=='FLAG_REVIEW'
    assert result['decisions'][0]['country_verified'] is False


class StubOCR(torch.nn.Module):
    """Returns fixed logits that greedy-decode to ``text``."""

    def __init__(self, text: str):
        super().__init__()
        ids = [BLANK] + [encode(ch)[0] for ch in text] + [BLANK]
        logits = torch.full((len(ids), NUM_CLASS), -10.0)
        for t, i in enumerate(ids):
            logits[t, i] = 10.0
        self.register_buffer("_fixed", logits.unsqueeze(0))  # (1, T, C)

    def forward(self, x):
        return self._fixed


def _install_stubs(monkeypatch, detections, ocr_text="1234567"):
    stub = {
        "detector": StubDetector(detections),
        "ocr": StubOCR(ocr_text),
        "alphabet": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "height": 64,
        "wmax": 256,
        "num_class": NUM_CLASS,
        "ckpt": {},
        "device": torch.device("cpu"),
    }
    monkeypatch.setattr(e2e, "load_models", lambda *a, **k: stub)
    return stub


@pytest.fixture()
def scene_path(tmp_path):
    arr = np.full((100, 100, 3), 128, dtype=np.uint8)
    p = tmp_path / "scene.png"
    Image.fromarray(arr).save(p)
    return p


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_zero_detections_yields_empty_stages(monkeypatch, scene_path):
    _install_stubs(monkeypatch, [])
    out = Pipeline(det_conf=0.4).run(scene_path)
    assert out["detections"] == []
    assert out["ocr_results"] == []
    assert out["validations"] == []
    # No plate detected -> a single explicit FLAG_REVIEW decision (no fabricated
    # plate text or OCR result).
    assert len(out["decisions"]) == 1
    assert out["decisions"][0]["decision"] == "FLAG_REVIEW"
    assert out["decisions"][0]["reason"] == "no plate detected above threshold 0.4"
    assert out["decisions"][0]["matched_rule"] is None
    assert out["decisions"][0]["requires_human_review"] is True
    assert out["decisions"][0]["country_verified"] is False


def test_single_detection_full_chain(monkeypatch, scene_path):
    _install_stubs(monkeypatch, [{"box": [10, 10, 60, 40], "det_conf": 0.9}])
    out = Pipeline().run(scene_path)

    assert len(out["ocr_results"]) == 1
    assert out["ocr_results"][0]["text"] == "1234567"
    assert out["ocr_results"][0]["crop_box"] == [10, 10, 60, 40]

    assert out["validations"][0]["classification"] == "VALID_JORDANIAN"
    assert out["decisions"][0]["decision"] == "ACCEPT_JORDANIAN"
    assert out["decisions"][0]["requires_human_review"] is True
    assert out["decisions"][0]["country_verified"] is False
    assert out["meta"]["requires_human_review"] is True


def test_multiple_detections_are_iterated(monkeypatch, scene_path):
    dets = [{"box": [10, 10, 60, 40], "det_conf": 0.9},
            {"box": [60, 10, 90, 40], "det_conf": 0.6}]
    _install_stubs(monkeypatch, dets)
    out = Pipeline().run(scene_path)
    assert len(out["detections"]) == 2
    assert len(out["ocr_results"]) == 2
    assert len(out["validations"]) == 2
    assert len(out["decisions"]) == 2


def test_authority_ambiguity_preserves_raw_ocr_and_requires_review(monkeypatch, scene_path):
    _install_stubs(monkeypatch, [{"box": [10, 10, 60, 40], "det_conf": 0.9}])
    pipeline = Pipeline()
    monkeypatch.setattr(pipeline, "_ocr_crop", lambda crop: ("C090001", 0.96))
    out = pipeline.run(scene_path)
    assert out["ocr_results"][0]["text"] == "C090001"
    assert out["ocr_results"][0]["possible_authority_text"] == "CD90001"
    assert out["ocr_results"][0]["possible_authority_review_only"] is True
    assert out["validations"][0]["normalized"] == "C090001"
    assert out["decisions"][0]["decision"] == "FLAG_REVIEW"


def test_gate_mode_keeps_highest_confidence_plate(monkeypatch, scene_path):
    dets = [{"box": [60, 10, 90, 40], "det_conf": 0.6},
            {"box": [10, 10, 60, 40], "det_conf": 0.9}]
    _install_stubs(monkeypatch, dets)
    out = Pipeline(max_plates=1).run(scene_path)
    assert [det["det_conf"] for det in out["detections"]] == [0.9]
    assert len(out["ocr_results"]) == len(out["decisions"]) == 1
    assert out["meta"]["max_plates"] == 1
    assert out["meta"]["detector_candidate_count"] == 2


def test_output_schema_and_meta(monkeypatch, scene_path):
    _install_stubs(monkeypatch, [{"box": [10, 10, 60, 40], "det_conf": 0.9}])
    out = Pipeline().run(scene_path)
    assert set(out) == {"image_path", "detections", "ocr_results",
                        "validations", "decisions", "meta"}
    for key in ("detector", "ocr", "validator", "det_conf_threshold",
                "det_nms_iou", "imgsz", "device", "seed"):
        assert key in out["meta"]


def test_confidences_are_not_fused(monkeypatch, scene_path):
    _install_stubs(monkeypatch, [{"box": [10, 10, 60, 40], "det_conf": 0.9}])
    out = Pipeline().run(scene_path)
    assert out["detections"][0]["det_conf"] == 0.9
    assert "ocr_conf" in out["ocr_results"][0]
    # the validator is rule-based and exposes no confidence
    assert "confidence" not in out["validations"][0]


def test_degenerate_box_is_recorded_not_crashed(monkeypatch, scene_path):
    # A box fully outside the 100x100 scene clips to empty -> ValueError path.
    _install_stubs(monkeypatch, [{"box": [1000, 1000, 1100, 1100],
                                  "det_conf": 0.5}])
    out = Pipeline().run(scene_path)
    assert len(out["ocr_results"]) == 1
    assert out["ocr_results"][0]["text"] == ""
    assert "crop_error" in out["ocr_results"][0]
    assert out["decisions"][0]["decision"] == "FLAG_REVIEW"


def test_run_does_not_import_training_script(monkeypatch, scene_path):
    import sys
    _install_stubs(monkeypatch, [])
    Pipeline().run(scene_path)
    assert "train_phase8_ocr" not in sys.modules


def test_exif_rotated_jpeg_crop_matches_detector_coordinates(tmp_path):
    # OpenCV/Ultralytics reports boxes in the EXIF-oriented coordinate frame.
    arr = np.zeros((120, 200, 3), dtype=np.uint8)
    arr[:, :100] = (250, 10, 10)
    arr[:, 100:] = (10, 250, 10)
    exif = Image.Exif()
    exif[274] = 6  # 90 degrees clockwise: left red half becomes top half.
    path = tmp_path / "oriented.jpg"
    Image.fromarray(arr).save(path, exif=exif, quality=95)
    original = path.read_bytes()

    scene = load_scene(path)
    assert scene.shape == (200, 120, 3)
    top = crop_from_box(scene, [10, 10, 110, 90])
    bottom = crop_from_box(scene, [10, 110, 110, 190])
    assert top[..., 0].mean() > top[..., 1].mean() + 150
    assert bottom[..., 1].mean() > bottom[..., 0].mean() + 150
    assert path.read_bytes() == original


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_scene_pixels_match_opencv_detector_frame(tmp_path, orientation):
    # OpenCV is the decoder used by the detector for image paths. Its eight
    # EXIF frames must agree with the scene passed to OCR, including mirrors.
    arr = np.zeros((80, 120, 3), dtype=np.uint8)
    arr[:40, :60] = (255, 0, 0)
    arr[:40, 60:] = (0, 255, 0)
    arr[40:, :60] = (0, 0, 255)
    arr[40:, 60:] = (255, 255, 0)
    exif = Image.Exif()
    exif[274] = orientation
    path = tmp_path / "oriented.jpg"
    Image.fromarray(arr).save(path, exif=exif, quality=100, subsampling=0)

    detector_rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    assert np.array_equal(load_scene(path), detector_rgb)


def test_optional_fallback_preserves_threshold_provenance_and_forces_review(monkeypatch,scene_path):
    _install_stubs(monkeypatch,[],ocr_text="1234567")
    weights=scene_path.with_suffix(".pt");weights.touch()
    monkeypatch.setattr(e2e,"PlateDetector",lambda *args,**kwargs:StubDetector([{"box":[10,10,80,70],"det_conf":.08}]))
    result=Pipeline(fallback_detector_path=weights,fallback_det_conf=.05).run(scene_path)
    assert result["ocr_results"][0]["text"]=="1234567"
    assert result["detections"][0]["det_conf"]==.08
    assert result["meta"]["detector_source"]=="fallback"
    assert result["meta"]["fallback_attempted"] is True
    assert result["meta"]["det_conf_threshold"]==.25
    assert result["meta"]["fallback_det_conf_threshold"]==.05
    assert result["decisions"][0]["decision"]=="FLAG_REVIEW"
    assert result["decisions"][0]["country_verified"] is False
    assert result["decisions"][0]["requires_human_review"] is True
