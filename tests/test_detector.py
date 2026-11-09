"""Unit tests - detector wrapper (app.detector.PlateDetector).

The Ultralytics ``YOLO`` class is replaced with a fake so these tests are fast,
deterministic and model-free. They pin the *interface* the pipeline relies on:
path-based inference, the Phase 6 settings (imgsz/conf/iou), rounded integer
boxes, and descending-confidence ordering.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from app.detector import DEFAULT_CONF, DEFAULT_IMGSZ, DEFAULT_IOU, PlateDetector


class _FakeBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4)
        self.conf = torch.tensor(conf, dtype=torch.float32).reshape(-1)

    def __len__(self):
        return int(self.conf.numel())


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYOLO:
    """Minimal stand-in for ``ultralytics.YOLO``."""

    last_instance = None

    def __init__(self, path):
        self.path = path
        self.calls = []
        self._result = _FakeResult(_FakeBoxes([[0, 0, 1, 1]], [0.5]))
        _FakeYOLO.last_instance = self

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [self._result]


@pytest.fixture()
def fake_ultralytics(monkeypatch, tmp_path):
    mod = types.ModuleType("ultralytics")
    mod.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"not-a-real-checkpoint")  # only existence is checked
    return weights


def test_detect_returns_sorted_rounded_boxes(fake_ultralytics):
    det = PlateDetector(fake_ultralytics, device="cpu")
    _FakeYOLO.last_instance._result = _FakeResult(
        _FakeBoxes([[10.4, 20.6, 30.5, 40.4], [1.2, 2.8, 3.1, 4.9]],
                   [0.31, 0.92]))
    out = det.detect("some/scene.jpg")

    assert out == [
        {"box": [1, 3, 3, 5], "det_conf": 0.92},
        {"box": [10, 21, 30, 40], "det_conf": 0.31},
    ]


def test_detect_uses_phase6_settings_and_path(fake_ultralytics):
    det = PlateDetector(fake_ultralytics, device="cpu", imgsz=960, conf=0.25,
                        iou=0.7)
    det.detect("scene.jpg")
    call = _FakeYOLO.last_instance.calls[-1]
    assert call["source"] == "scene.jpg"          # PATH, not a numpy array
    assert call["imgsz"] == 960
    assert call["conf"] == 0.25
    assert call["iou"] == 0.7
    assert call["verbose"] is False
    assert call["device"] == "cpu"


def test_detect_empty_boxes_returns_empty_list(fake_ultralytics):
    det = PlateDetector(fake_ultralytics, device="cpu")
    _FakeYOLO.last_instance._result = _FakeResult(_FakeBoxes([], []))
    assert det.detect("scene.jpg") == []


def test_missing_weights_raise_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        PlateDetector(tmp_path / "does_not_exist.pt", device="cpu")


def test_defaults_are_phase6_values():
    assert (DEFAULT_IMGSZ, DEFAULT_CONF, DEFAULT_IOU) == (960, 0.25, 0.7)
