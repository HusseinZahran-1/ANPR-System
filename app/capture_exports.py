"""Write one original image and its XML analysis as a single capture bundle."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from uuid import uuid4
from xml.etree import ElementTree as ET
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from app.api.config import resolve_path
from app.validator.jordanian_validator import normalize_text


class CaptureExport(BaseModel):
    id: str
    folder: str
    image_filename: str
    xml_filename: str


def _add_fields(parent: ET.Element, pairs: dict) -> None:
    for name, value in pairs.items():
        field = ET.SubElement(parent, name)
        if value is not None:
            text = str(value).lower() if isinstance(value, bool) else str(value)
            field.text = re.sub(r"[^\u0009\u000a\u000d\u0020-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]", "", text)


def _add_box(parent: ET.Element, box: list | tuple | None) -> None:
    if box is not None:
        ET.SubElement(parent, "box", dict(zip(("x1", "y1", "x2", "y2"), map(str, box), strict=True)))


def _xml(export: CaptureExport, image: bytes, result: dict, source: str, captured_at: str,
         camera_id: str | None) -> bytes:
    root = ET.Element("anpr_capture", {"schema_version": "1", "id": export.id})
    _add_fields(root, {"source": source, "captured_at": captured_at, "camera_id": camera_id})
    ET.SubElement(root, "export", {"folder": export.folder, "xml_filename": export.xml_filename})
    image_node = ET.SubElement(root, "image", {"filename": export.image_filename,
                                               "sha256": hashlib.sha256(image).hexdigest(),
                                               "bytes": str(len(image))})
    _add_fields(image_node, {"original_filename": result.get("image_path")})
    plates = ET.SubElement(root, "plates")
    detections = result.get("detections", [])
    decisions = result.get("decisions", [])
    validations = result.get("validations", [])
    meta = result.get("meta", {})
    summary = ET.SubElement(root, "analysis")
    _add_fields(summary, {
        "plate_count": len(detections), "vehicle_count": len(result.get("vehicles", [])),
        "requires_human_review": meta.get("requires_human_review", True),
        "country_verified": meta.get("country_verified", False),
        "device": meta.get("device"), "ocr_layout_mode": meta.get("ocr_layout_mode"),
        "detector_candidate_count": meta.get("detector_candidate_count"),
        "fragment_input_candidate_count": meta.get("fragment_input_candidate_count"),
        "fragment_merge_applied": meta.get("fragment_merge_applied"),
    })
    if decisions and not detections:
        _add_fields(summary, {"decision": decisions[0].get("decision"),
                              "reason": decisions[0].get("reason")})
    for index, reading in enumerate(result.get("ocr_results", [])):
        plate = ET.SubElement(plates, "plate", {"index": str(index)})
        parts = reading.get("plate_parts") or {}
        _add_fields(plate, {
            "original_ocr": reading.get("text"),
            "full_text": parts.get("full_text", reading.get("text")),
            "registration_code": parts.get("plate_code"),
            "plate_number": parts.get("plate_number"),
            "split_basis": parts.get("split_basis"),
            "ocr_confidence": reading.get("ocr_conf"),
            "ocr_layout_mode": reading.get("ocr_layout_mode", meta.get("ocr_layout_mode")),
            "possible_authority_text": reading.get("possible_authority_text"),
            "primary_ocr_text": reading.get("primary_ocr_text"),
            "requires_human_review": parts.get("requires_human_review", True),
        })
        if index < len(detections):
            detection = detections[index]
            _add_box(plate, detection.get("box"))
            _add_fields(plate, {"detector_confidence": detection.get("det_conf"),
                                "detector_source": detection.get("detector_source", "primary")})
        if index < len(decisions):
            decision = decisions[index]
            _add_fields(plate, {"decision": decision.get("decision"),
                                "decision_reason": decision.get("reason"),
                                "country_verified": decision.get("country_verified", False)})
        if index < len(validations):
            validation = validations[index]
            _add_fields(ET.SubElement(plate, "validation"), {
                "classification": validation.get("classification"),
                "matched_rule": validation.get("matched_rule"), "reason": validation.get("reason"),
            })

    vehicles = ET.SubElement(root, "vehicles", {"status": result.get("vehicle_status", "disabled")})
    for value in result.get("vehicles", []):
        vehicle = ET.SubElement(vehicles, "vehicle")
        coarse_type = (value.get("detection_class") or "").title() or None
        category = value.get("vehicle_type")
        if coarse_type in {"Car", "Truck"} and category == coarse_type:
            category = None
        _add_fields(vehicle, {
            "type": value.get("vehicle_type"), "vehicle_category": category,
            "vehicle_type": coarse_type,
            "color": value.get("color"),
            "make": value.get("make"), "detection_class": value.get("detection_class"),
            "detector_confidence": value.get("det_conf"),
            "requires_human_review": value.get("requires_human_review", True),
        })
        _add_box(vehicle, value.get("box"))
        for index in value.get("plate_indices", []):
            ET.SubElement(vehicle, "plate_ref", {"index": str(index)})
    ET.indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _filename_attribute(value: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", value or "")[:24].strip("-") or "Unknown"


def export_basename(result: dict, capture_id: str) -> str:
    readings = result.get("ocr_results", [])
    reading = readings[0] if readings else {}
    parts = reading.get("plate_parts") or {}
    plate = normalize_text(parts.get("full_text") or reading.get("text", ""))
    vehicles = result.get("vehicles", [])
    primary = vehicles[0] if vehicles else {}
    coarse_type = (primary.get("detection_class") or "").title()
    category = primary.get("vehicle_type")
    type_label = f"{coarse_type}-{category}" if coarse_type and category and category != coarse_type else category or coarse_type
    attributes = [_filename_attribute(primary.get("color")), _filename_attribute(primary.get("make")),
                  _filename_attribute(type_label)]
    return "_".join(["ANPR", plate if 1 <= len(plate) <= 16 else capture_id, *attributes])


def export_root() -> Path:
    return resolve_path(os.getenv("ANPR_EXPORT_DIR") or "data/processed/anpr_exports")


def export_location(previous: dict) -> dict:
    """Resolve an older live event through the local migration index, when available."""
    if "folder" in previous:
        return previous
    index = export_root() / ".legacy-index.json"
    if not index.is_file():
        return previous
    try:
        return json.loads(index.read_text(encoding="utf-8")).get(previous.get("id"), previous)
    except (OSError, json.JSONDecodeError):
        return previous


def _reserve_name(folder: Path, basename: str) -> Path | None:
    reservation = folder / f".{basename}.lock"
    try:
        reservation.touch(exist_ok=False)
    except FileExistsError:
        return None
    if any((folder / (basename + ext)).exists() for ext in (".jpg", ".png", ".xml")):
        reservation.unlink()
        return None
    return reservation


def save_export(image: bytes, suffix: str, result: dict, *, source: str,
                capture_id: str | None = None, captured_at: str | None = None,
                camera_id: str | None = None) -> dict:
    """Save to the local day folder, publishing the XML after its original image."""
    capture_id = capture_id or uuid4().hex
    if suffix not in {".jpg", ".png"} or not re.fullmatch(r"[0-9a-f]{32}", capture_id):
        raise ValueError("invalid export ID or image extension")
    basename = export_basename(result, capture_id)
    now = datetime.now(timezone.utc)
    day = now.astimezone(ZoneInfo("Asia/Amman")).date().isoformat()
    folder = export_root() / day
    folder.mkdir(parents=True, exist_ok=True)

    reservation = _reserve_name(folder, basename)
    if reservation is None:
        basename += f"_{capture_id}"
        reservation = _reserve_name(folder, basename)
    if reservation is None:
        raise FileExistsError("capture name is already in use")

    export = CaptureExport(id=capture_id, folder=day, image_filename=basename + suffix,
                           xml_filename=basename + ".xml")
    image_path = folder / export.image_filename
    try:
        xml = _xml(export, image, result, source, captured_at or now.isoformat(), camera_id)
        with TemporaryDirectory(prefix=".pending-", dir=folder) as temporary:
            staging = Path(temporary)
            for name, content in ((export.image_filename, image), (export.xml_filename, xml)):
                with (staging / name).open("xb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
            (staging / export.image_filename).replace(image_path)
            try:
                (staging / export.xml_filename).replace(folder / export.xml_filename)
            except OSError:
                image_path.unlink(missing_ok=True)
                raise
    finally:
        reservation.unlink(missing_ok=True)
    return export.model_dump()
