"""Optional vehicle detection and experimental visual attributes."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from app.pipeline.io import crop_from_box, load_scene

logger = logging.getLogger(__name__)

VehicleStatus = Literal["disabled", "detection_only", "ready", "unavailable"]
Box = tuple[int, int, int, int]
COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
BODY_STYLES = {
    "SUV": ("an SUV", "a sport utility vehicle"),
    "Sedan": ("a sedan", "a four-door sedan with a separate trunk"),
    "Hatchback": ("a hatchback", "a compact hatchback car"),
    "Pickup": ("a pickup truck", "a pickup with an open cargo bed"),
    "Van": ("a van", "a minivan or cargo van"),
    "Wagon": ("a station wagon", "an estate car"),
    "Coupe": ("a coupe", "a two-door coupe"),
    "Truck": ("a heavy truck", "a commercial lorry"),
}
COLORS = ("Black", "White", "Silver", "Gray", "Red", "Blue", "Green", "Brown",
          "Beige", "Yellow", "Orange", "Purple")
CAR_MAKES = ("BMW", "Mercedes-Benz", "Toyota", "Hyundai", "Kia", "Nissan", "Honda",
             "Ford", "Chevrolet", "Volkswagen", "Audi", "Lexus", "Mazda", "Mitsubishi",
             "Suzuki", "Renault", "Peugeot", "Skoda", "Jeep", "Land Rover", "Porsche",
             "Subaru", "Volvo", "Dodge", "GMC", "Tesla", "BYD", "Geely", "MG",
             "Changan", "Isuzu", "MAN", "Scania", "Iveco")
MOTORCYCLE_MAKES = ("Honda", "Yamaha", "Suzuki", "Kawasaki", "BMW", "Ducati",
                    "KTM", "Harley-Davidson", "Piaggio", "Triumph", "Bajaj", "TVS")


class Vehicle(BaseModel):
    box: Box
    detection_class: str
    det_conf: float
    vehicle_type: str
    color: str | None = None
    make: str | None = None
    plate_indices: list[int] = Field(default_factory=list)
    requires_human_review: bool = True


def _assign_plates(vehicles: list[Vehicle], plates: list[dict]) -> None:
    """Link a plate to a unique containing vehicle in upright image coordinates."""
    for index, plate in enumerate(plates):
        px1, py1, px2, py2 = plate["box"]
        area = (px2 - px1) * (py2 - py1)
        if px2 <= px1 or py2 <= py1:
            continue
        matches = []
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle.box
            width = max(0, min(px2, x2) - max(px1, x1))
            height = max(0, min(py2, y2) - max(py1, y1))
            if width * height >= 0.85 * area:
                matches.append(vehicle)
        if len(matches) == 1:
            matches[0].plate_indices.append(index)



def select_primary_vehicle(vehicles: list[Vehicle]) -> Vehicle | None:
    """Prefer the vehicle linked to the first plate, then the largest visible vehicle."""
    if not vehicles:
        return None
    return max(vehicles, key=lambda vehicle: (
        bool(vehicle.plate_indices),
        -min(vehicle.plate_indices, default=0),
        (vehicle.box[2] - vehicle.box[0]) * (vehicle.box[3] - vehicle.box[1]),
        vehicle.det_conf,
    ))


class VehicleAttributes:
    """Rank a fixed label set with OpenAI CLIP ViT-B/32 on detected crops."""

    def __init__(self, checkpoint: Path, device: str) -> None:
        import clip
        import torch
        from torch.nn.functional import normalize

        self.torch = torch
        self.device = device
        self.model, self.preprocess = clip.load(str(checkpoint), device=device)
        self.model.eval()
        self.banks = {}
        labels_and_prompts = {
            "car_make": (CAR_MAKES, ("a photo of a {} car", "a vehicle manufactured by {}")),
            "motorcycle_make": (MOTORCYCLE_MAKES, ("a photo of a {} motorcycle", "a motorcycle made by {}")),
        }
        for kind in ("car", "motorcycle", "bus", "truck"):
            labels_and_prompts[kind + "_color"] = (
                COLORS, (f"a photo of a {{}} {kind}", f"a {kind} with {{}} body paint")
            )
        descriptive_prompts = {
            name: {label: [template.format(label) for template in templates] for label in labels}
            for name, (labels, templates) in labels_and_prompts.items()
        }
        descriptive_prompts["type"] = {
            label: [f"a photo of {phrase}" for phrase in phrases] for label, phrases in BODY_STYLES.items()
        }
        with torch.inference_mode():
            for name, descriptions in descriptive_prompts.items():
                prompts = [prompt for group in descriptions.values() for prompt in group]
                encoded = self.model.encode_text(clip.tokenize(prompts).to(device)).float()
                groups = normalize(encoded).split([len(group) for group in descriptions.values()])
                self.banks[name] = (tuple(descriptions), normalize(torch.stack([g.mean(0) for g in groups])))

    def _label(self, features, bank: str, minimum: float, margin: float) -> str | None:
        labels, vectors = self.banks[bank]
        values, indices = (vectors @ features).topk(2)
        if values[0].item() < minimum or (values[0] - values[1]).item() < margin:
            return None
        return labels[indices[0].item()]

    def annotate(self, scene, vehicle: Vehicle) -> None:
        crop = crop_from_box(scene, vehicle.box)
        size = min(crop.shape[:2])
        if size < 40:
            return
        square = ImageOps.pad(Image.fromarray(crop), (224, 224), color=(123, 117, 104))
        with self.torch.inference_mode():
            image = self.preprocess(square).unsqueeze(0).to(self.device)
            features = self.torch.nn.functional.normalize(self.model.encode_image(image).float())
            feature = features[0]
            kind = vehicle.detection_class
            if kind in ("car", "truck"):
                style = self._label(feature, "type", minimum=0.22, margin=0.010)
                vehicle.vehicle_type = style or vehicle.vehicle_type
            vehicle.color = self._label(feature, kind + "_color", minimum=0.22, margin=0.010)
            if size >= 80 and kind != "bus":
                bank = "motorcycle_make" if kind == "motorcycle" else "car_make"
                vehicle.make = self._label(feature, bank, minimum=0.24, margin=0.014)


class VehicleAnalyzer:
    def __init__(self, detector: Path, attributes: VehicleAttributes | None, device: str) -> None:
        from ultralytics import YOLO

        if not detector.is_file():
            raise FileNotFoundError("vehicle detector checkpoint is missing")
        self.detector = YOLO(str(detector))
        if any(self.detector.names.get(k) != name for k, name in COCO_VEHICLES.items()):
            raise ValueError("vehicle detector must use the COCO class IDs")
        self.attributes = attributes
        self.device = device

    def run(self, path: Path, plates: list[dict]) -> tuple[list[Vehicle], VehicleStatus]:
        detection = self.detector.predict(str(path), classes=list(COCO_VEHICLES),
                                          imgsz=960, conf=0.30, iou=0.50, max_det=10,
                                          device=self.device, verbose=False)[0]
        height, width = detection.orig_shape
        vehicles = []
        for box, cls, conf in zip(detection.boxes.xyxy.cpu().tolist(),
                                  detection.boxes.cls.cpu().tolist(),
                                  detection.boxes.conf.cpu().tolist(), strict=True):
            x1, y1, x2, y2 = (int(round(value)) for value in box)
            pixels = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
            if pixels[0] < pixels[2] and pixels[1] < pixels[3]:
                kind = COCO_VEHICLES[int(cls)]
                vehicles.append(Vehicle(box=pixels, detection_class=kind,
                                        det_conf=round(conf, 6), vehicle_type=kind.title()))

        _assign_plates(vehicles, plates)
        primary = select_primary_vehicle(vehicles)
        vehicles = [primary] if primary is not None else []
        if self.attributes is None:
            return vehicles, "detection_only"
        if primary is not None:
            try:
                self.attributes.annotate(load_scene(path), primary)
            except Exception:
                logger.exception("Vehicle attribute inference failed")
                return vehicles, "detection_only"
        return vehicles, "ready"


@lru_cache(maxsize=1)
def _load_analyzer(detector: Path, attributes: Path | None, device: str) -> VehicleAnalyzer | None:
    try:
        analyzer = VehicleAnalyzer(detector, None, device)
    except Exception:
        logger.exception("Vehicle detector unavailable")
        return None
    if attributes is not None:
        try:
            if not attributes.is_file():
                raise FileNotFoundError("vehicle attribute checkpoint is missing")
            analyzer.attributes = VehicleAttributes(attributes, device)
        except Exception:
            logger.exception("Vehicle attribute model unavailable")
    return analyzer


def vehicle_evidence(path: Path, plates: list[dict], detector: Path | None,
                     attributes: Path | None, device: str) -> tuple[list[Vehicle], VehicleStatus]:
    if detector is None:
        return [], "disabled"
    analyzer = _load_analyzer(detector, attributes, device)
    if analyzer is not None:
        try:
            return analyzer.run(path, plates)
        except Exception:
            logger.exception("Vehicle detection failed")
    return [], "unavailable"
