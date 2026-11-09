"""Durable, unverified camera capture inbox outside the immutable raw dataset."""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

from app.api.config import ROOT

ARCHIVE_DIR = ROOT / "data" / "processed" / "camera_inbox"
_lock = threading.Lock()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, record: dict) -> None:
    _atomic_write(path, (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))


def _validate_image(image: bytes, content_type: str) -> str:
    declared = content_type.split(";", 1)[0].strip().lower()
    if declared not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="camera image must be JPEG or PNG")
    try:
        with Image.open(io.BytesIO(image)) as source:
            actual = {"JPEG": "image/jpeg", "PNG": "image/png"}.get(source.format)
            if actual != declared:
                raise HTTPException(status_code=415, detail="camera image MIME type does not match its bytes")
            if source.width * source.height > 50_000_000:
                raise HTTPException(status_code=413, detail="camera image exceeds 50 megapixels")
            source.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=422, detail="invalid camera image") from exc
    return ".jpg" if declared == "image/jpeg" else ".png"


def save_capture(*, image: bytes, content_type: str, camera_id: str,
                 original_filename: str) -> dict:
    """Persist one valid image before inference, returning its stable record."""
    suffix = _validate_image(image, content_type)
    now = datetime.now(timezone.utc)
    event_id = uuid4().hex
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    relative = Path("images") / now.strftime("%Y-%m-%d") / f"{stamp}_{event_id}{suffix}"
    image_path = ARCHIVE_DIR / relative
    metadata_path = ARCHIVE_DIR / "records" / now.strftime("%Y-%m-%d") / f"{event_id}.json"
    record = {
        "event_id": event_id,
        "captured_at": now.isoformat(),
        "camera_id": camera_id,
        "original_filename": Path(original_filename).name[:200],
        "image_path": relative.as_posix(),
        "content_type": content_type,
        "image_bytes": len(image),
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "status": "received",
        "label_status": "unverified_camera_capture",
    }
    _atomic_write(image_path, image)
    _atomic_json(metadata_path, record)
    return record


def complete_capture(record: dict, *, result: dict | None = None,
                     error: Exception | None = None) -> None:
    """Write final analysis metadata and a one-line index for the capture."""
    if (result is None) == (error is None):
        raise ValueError("provide exactly one of result or error")
    completed = dict(record)
    completed["completed_at"] = datetime.now(timezone.utc).isoformat()
    if error is None:
        completed["status"] = "analyzed"
        completed["result"] = result
    else:
        completed["status"] = "analysis_failed"
        completed["error_type"] = type(error).__name__
    day = record["captured_at"][:10]
    metadata_path = ARCHIVE_DIR / "records" / day / f"{record['event_id']}.json"
    line = {
        "event_id": record["event_id"],
        "captured_at": record["captured_at"],
        "camera_id": record["camera_id"],
        "image_path": record["image_path"],
        "image_sha256": record["image_sha256"],
        "status": completed["status"],
        "detected_plates": len(result.get("detections", [])) if result else 0,
        "ocr_texts_unverified": [item.get("text", "") for item in result.get("ocr_results", [])] if result else [],
    }
    with _lock:
        _atomic_json(metadata_path, completed)
        log = ARCHIVE_DIR / "events.jsonl"
        with log.open("ab") as target:
            target.write((json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8"))
            target.flush()
            os.fsync(target.fileno())


def recent_captures(limit: int = 10) -> list[tuple[dict, bytes]]:
    """Load the newest analyzed captures so the live page survives restarts."""
    log = ARCHIVE_DIR / "events.jsonl"
    if not log.is_file():
        return []
    recent: deque[dict] = deque(maxlen=limit)
    with log.open(encoding="utf-8") as source:
        for line in source:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("status") == "analyzed":
                recent.append(item)
    loaded = []
    root = ARCHIVE_DIR.resolve()
    for item in recent:
        event_id = item.get("event_id", "")
        if len(event_id) != 32 or any(ch not in "0123456789abcdef" for ch in event_id):
            continue
        day = item.get("captured_at", "")[:10]
        record_path = root / "records" / day / f"{event_id}.json"
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            image_path = (root / record["image_path"]).resolve()
            if not image_path.is_relative_to(root):
                continue
            image = image_path.read_bytes()
            if hashlib.sha256(image).hexdigest() != record["image_sha256"]:
                continue
            if record.get("status") != "analyzed" or not isinstance(record.get("result"), dict):
                continue
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
        loaded.append((record, image))
    return loaded


def _reverse_log_lines():
    """Read the newest log lines without rescanning the growing camera log."""
    path = ARCHIVE_DIR / "events.jsonl"
    if not path.is_file():
        return
    with path.open("rb") as source:
        source.seek(0, os.SEEK_END)
        position = source.tell()
        pending = b""
        while position:
            size = min(position, 8192)
            position -= size
            source.seek(position)
            parts = (source.read(size) + pending).split(b"\n")
            pending = parts[0]
            for line in reversed(parts[1:]):
                if line:
                    yield line
        if pending:
            yield pending


def _record_from_index(item: dict) -> dict | None:
    event_id = item.get("event_id", "")
    if not isinstance(event_id, str) or len(event_id) != 32 or any(
            ch not in "0123456789abcdef" for ch in event_id):
        return None
    try:
        day = datetime.fromisoformat(item["captured_at"]).date().isoformat()
        record = json.loads((ARCHIVE_DIR / "records" / day / f"{event_id}.json")
                            .read_text(encoding="utf-8"))
        if record.get("event_id") != event_id or record.get("status") != "analyzed":
            return None
        if not isinstance(record.get("result"), dict):
            return None
        image_path = (ARCHIVE_DIR / record["image_path"]).resolve()
        if not image_path.is_relative_to(ARCHIVE_DIR.resolve()) or not image_path.is_file():
            return None
        return record
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def recent_records(limit: int = 10) -> list[dict]:
    """Return newest completed camera analyses from the shared disk inbox."""
    found = []
    for line in _reverse_log_lines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("status") != "analyzed":
            continue
        record = _record_from_index(item)
        if record is not None:
            found.append(record)
        if len(found) >= limit:
            break
    return found


def stored_capture(event_id: str) -> tuple[dict, bytes] | None:
    """Find one archived event by its opaque ID and verify its image hash."""
    if len(event_id) != 32 or any(ch not in "0123456789abcdef" for ch in event_id):
        return None
    for path in (ARCHIVE_DIR / "records").glob(f"*/{event_id}.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("event_id") != event_id or record.get("status") != "analyzed":
                continue
            image_path = (ARCHIVE_DIR / record["image_path"]).resolve()
            if not image_path.is_relative_to(ARCHIVE_DIR.resolve()):
                continue
            image = image_path.read_bytes()
            if hashlib.sha256(image).hexdigest() == record["image_sha256"]:
                return record, image
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
    return None
