"""Write one original image and its XML analysis as a single capture bundle."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from uuid import uuid4
from xml.etree import ElementTree as ET

from pydantic import BaseModel

from app.api.config import resolve_path
from app.validator.jordanian_validator import normalize_text


class CaptureExport(BaseModel):
    id: str
    image_filename: str
    xml_filename: str


def _add_fields(parent: ET.Element, pairs: dict) -> None:
    for name, value in pairs.items():
        field = ET.SubElement(parent, name)
        if value is not None:
            field.text = str(value).lower() if isinstance(value, bool) else str(value)


def _add_box(parent: ET.Element, box: list | tuple | None) -> None:
    if box is not None:
        ET.SubElement(parent, "box", dict(zip(("x1", "y1", "x2", "y2"), map(str, box), strict=True)))


def _xml(export: CaptureExport, image: bytes, result: dict, source: str, captured_at: str) -> bytes:
    root = ET.Element("anpr_capture", {"schema_version": "1", "id": export.id})
    _add_fields(root, {"source": source, "captured_at": captured_at})
    ET.SubElement(root, "image", {"filename": export.image_filename,
                                 "sha256": hashlib.sha256(image).hexdigest()})
    plates = ET.SubElement(root, "plates")
    detections = result.get("detections", [])
    decisions = result.get("decisions", [])
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
            "requires_human_review": parts.get("requires_human_review", True),
        })
        if index < len(detections):
            _add_box(plate, detections[index].get("box"))
        if index < len(decisions):
            _add_fields(plate, {"decision": decisions[index].get("decision")})

    vehicles = ET.SubElement(root, "vehicles", {"status": result.get("vehicle_status", "disabled")})
    for value in result.get("vehicles", []):
        vehicle = ET.SubElement(vehicles, "vehicle")
        _add_fields(vehicle, {
            "type": value.get("vehicle_type"), "color": value.get("color"),
            "make": value.get("make"), "detection_class": value.get("detection_class"),
            "detector_confidence": value.get("det_conf"),
            "requires_human_review": value.get("requires_human_review", True),
        })
        _add_box(vehicle, value.get("box"))
        for index in value.get("plate_indices", []):
            ET.SubElement(vehicle, "plate_ref", {"index": str(index)})
    ET.indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def save_export(image: bytes, suffix: str, result: dict, *, source: str,
                capture_id: str | None = None, captured_at: str | None = None) -> dict:
    """Publish both files together by renaming a staging directory on the same volume."""
    capture_id = capture_id or uuid4().hex
    if suffix not in {".jpg", ".png"} or not re.fullmatch(r"[0-9a-f]{32}", capture_id):
        raise ValueError("invalid export ID or image extension")
    readings = result.get("ocr_results", [])
    reading = readings[0] if readings else {}
    parts = reading.get("plate_parts") or {}
    plate = normalize_text(parts.get("full_text") or reading.get("text", ""))
    basename = f"ANPR_{plate if 1 <= len(plate) <= 16 else capture_id}"
    export = CaptureExport(id=capture_id, image_filename=basename + suffix,
                           xml_filename=basename + ".xml")
    timestamp = captured_at or datetime.now(timezone.utc).isoformat()
    xml = _xml(export, image, result, source, timestamp)
    root = resolve_path(os.getenv("ANPR_EXPORT_DIR") or "data/processed/anpr_exports")
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pending-", dir=root) as temporary:
        staging = Path(temporary)
        for name, content in ((export.image_filename, image), (export.xml_filename, xml)):
            with (staging / name).open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        staging.rename(root / capture_id)
    return export.model_dump()
