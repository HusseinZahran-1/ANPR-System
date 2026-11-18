"""Optional vehicle detection and experimental visual attributes."""

from __future__ import annotations

from collections import Counter
import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps
from pydantic import BaseModel, Field, model_validator

from app.pipeline.io import crop_from_box, load_scene
from app.vehicle_attributes import PaintColor

logger = logging.getLogger(__name__)

VehicleStatus = Literal["disabled", "detection_only", "ready", "unavailable", "unassociated"]
Box = tuple[int, int, int, int]
COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
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
    attribute_sources: dict[str, str] = Field(default_factory=dict)
    plate_indices: list[int] = Field(default_factory=list)
    requires_human_review: bool = True


    @model_validator(mode="after")
    def normalize_vehicle_type(self) -> Vehicle:
        self.vehicle_type = self.detection_class.title()
        self.attribute_sources = {key: value for key, value in self.attribute_sources.items()
                                  if key in {"color", "make"}}
        return self


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

    def __init__(self, checkpoint: Path, device: str, structured: PaintColor | None = None) -> None:
        import clip
        import torch
        from torch.nn.functional import normalize

        self.structured = structured
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

    def _votes(self, features, bank: str) -> list[tuple[str, float, float]]:
        labels, vectors = self.banks[bank]
        values, indices = (features @ vectors.T).topk(2)
        return [(labels[index], first, first - second)
                for (index, _), (first, second) in zip(indices.tolist(), values.tolist(), strict=True)]

    @staticmethod
    def _agree(votes: list[tuple[str, float, float]], *, minimum: float,
               margin: float, required: int) -> str | None:
        counts = Counter(label for label, score, gap in votes if score >= minimum and gap >= margin)
        return next((label for label, count in counts.items() if count >= required
                     and all(other_count < count for other, other_count in counts.items() if other != label)), None)

    def annotate(self, scene, vehicle: Vehicle) -> None:
        crop = crop_from_box(scene, vehicle.box)
        height, width = crop.shape[:2]
        if min(height, width) < 40:
            return

        inset = crop[round(.08 * height):round(.91 * height), round(.08 * width):round(.92 * width)]
        upper = crop[:round(.80 * height), round(.12 * width):round(.88 * width)]
        full = Image.fromarray(crop)
        square = ImageOps.pad(full, (224, 224), color=(123, 117, 104))
        views = (square, full, Image.fromarray(inset), Image.fromarray(upper))
        with self.torch.inference_mode():
            batch = self.torch.stack([self.preprocess(view) for view in views]).to(self.device)
            features = self.torch.nn.functional.normalize(self.model.encode_image(batch).float())

        kind = vehicle.detection_class
        paint_votes = self._votes(features, kind + "_color")
        color = self._agree(paint_votes, minimum=0.22, margin=0.005, required=3)
        if color is None and len({vote[0] for vote in paint_votes}) == 1:
            if sum(vote[2] for vote in paint_votes) / len(paint_votes) >= 0.0035:
                color = paint_votes[0][0]
        if color:
            vehicle.color = color
            vehicle.attribute_sources["color"] = "clip_multiview"

        if kind in ("car", "truck") and self.structured:
            try:
                supervised_color = self.structured.classify(crop, inset)
            except Exception:
                logger.exception("Supervised vehicle attribute inference failed")
            else:
                # Accept a supervised paint result only when the visual alternatives support it.
                compatible = {"White", "Silver", "Gray"} if supervised_color == "White" else {supervised_color}
                if vehicle.color is None and supervised_color and any(
                    vote[0] in compatible for vote in paint_votes
                ):
                    vehicle.color = supervised_color
                    vehicle.attribute_sources["color"] = "paddleclas_with_clip"
        if min(height, width) >= 80 and kind != "bus":
            bank = "motorcycle_make" if kind == "motorcycle" else "car_make"
            votes = self._votes(features, bank)
            make = self._agree(votes, minimum=0.24, margin=0.010, required=2)
            primary_label, score, margin = votes[0]
            if (make is None and score >= 0.24 and margin >= 0.030
                    and all(label == primary_label or gap < 0.010 for label, _, gap in votes[1:])):
                make = primary_label
            if make:
                vehicle.make = make
                vehicle.attribute_sources["make"] = "clip_multiview"


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
        if plates:
            vehicles = [vehicle for vehicle in vehicles if 0 in vehicle.plate_indices]
            if not vehicles:
                return [], "unassociated"
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
def _load_analyzer(detector: Path, attributes: Path | None, device: str,
                   structured_model: Path | None = None) -> VehicleAnalyzer | None:
    try:
        analyzer = VehicleAnalyzer(detector, None, device)
    except Exception:
        logger.exception("Vehicle detector unavailable")
        return None
    if attributes is not None:
        try:
            if not attributes.is_file():
                raise FileNotFoundError("vehicle attribute checkpoint is missing")
            structured = None
            if structured_model is not None:
                try:
                    structured = PaintColor(structured_model)
                except Exception:
                    logger.exception("Supervised vehicle model unavailable")
            analyzer.attributes = VehicleAttributes(attributes, device, structured)
        except Exception:
            logger.exception("Vehicle attribute model unavailable")
    return analyzer


def vehicle_evidence(path: Path, plates: list[dict], detector: Path | None,
                     attributes: Path | None, device: str, *,
                     structured_model: Path | None = None) -> tuple[list[Vehicle], VehicleStatus]:
    if detector is None:
        return [], "disabled"
    analyzer = _load_analyzer(detector, attributes, device, structured_model)
    if analyzer is not None:
        try:
            return analyzer.run(path, plates)
        except Exception:
            logger.exception("Vehicle detection failed")
    return [], "unavailable"
