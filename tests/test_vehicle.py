"""Vehicle evidence and gate integration without downloading model weights."""

from pathlib import Path

import torch

from app.api import routes
from app.api.config import Settings, resolve_path
from app.vehicle import Vehicle, VehicleAttributes, _assign_plates


def test_vehicle_association_requires_unique_containment():
    vehicles = [
        Vehicle(box=(0, 0, 120, 120), detection_class="car", det_conf=0.8, vehicle_type="SUV"),
        Vehicle(box=(90, 0, 210, 120), detection_class="car", det_conf=0.9, vehicle_type="Sedan"),
    ]
    _assign_plates(vehicles, [
        {"box": (20, 42, 40, 48)},
        {"box": (98, 42, 110, 48)},
        {"box": (150, 42, 175, 48)},
        {"box": (251, 42, 280, 48)},
    ])
    assert [vehicle.plate_indices for vehicle in vehicles] == [[0], [2]]


def test_close_attribute_scores_return_unknown():
    reader = VehicleAttributes.__new__(VehicleAttributes)
    reader.banks = {
        "make": (("BMW", "Toyota"), torch.tensor([[0.301, 0.0], [0.300, 0.0]])),
        "color": (("Black", "White"), torch.tensor([[0.31, 0.0], [0.26, 0.0]])),
    }
    feature = torch.tensor([1.0, 0.0])
    assert reader._label(feature, "make", minimum=0.24, margin=0.014) is None
    assert reader._label(feature, "color", minimum=0.22, margin=0.010) == "Black"


def test_gate_adds_vehicle_evidence_without_changing_legacy_prediction(monkeypatch):
    class StubPipeline:
        def run(self, path):
            return {"detections": [{"box": [20, 42, 40, 48]}], "ocr_results": []}

    class UnavailableDetector:
        def run(self, path, plates):
            raise RuntimeError("detector failed")

    def detect(path, plates, detector, attributes, device):
        from app.vehicle import vehicle_evidence
        assert detector == resolve_path("local/coco.pt")
        assert plates[0]["box"] == [20, 42, 40, 48]
        return vehicle_evidence(path, plates, detector, attributes, device)

    settings = Settings(vehicle_detector_model_path="local/coco.pt")
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    monkeypatch.setattr(routes, "_get_pipeline", StubPipeline)
    monkeypatch.setattr(routes, "_get_camera_pipeline", StubPipeline)
    monkeypatch.setattr("app.vehicle._load_analyzer", lambda *args: UnavailableDetector())
    monkeypatch.setattr(routes, "vehicle_evidence", detect)
    scene = Path("scene.jpg")

    gate = routes._run_image(scene, "scene.jpg", camera=True)
    assert gate["vehicle_status"] == "unavailable"
    assert gate["vehicles"] == []
    assert gate["detections"][0]["box"] == [20, 42, 40, 48]
    assert "vehicles" not in routes._run_image(scene, "scene.jpg")


def test_primary_vehicle_prefers_plate_then_largest_visible_vehicle():
    from app.vehicle import select_primary_vehicle

    vehicle = Vehicle(box=(0, 0, 100, 100), detection_class="car", det_conf=0.5, vehicle_type="SUV")
    background = Vehicle(box=(110, 0, 160, 40), detection_class="car",
                         det_conf=0.95, vehicle_type="Sedan")
    assert select_primary_vehicle([background, vehicle]) is vehicle
    background.plate_indices = [0]
    assert select_primary_vehicle([background, vehicle]) is background
    assert select_primary_vehicle([]) is None
