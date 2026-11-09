"""Persist user-checked camera captures separately from training data."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel

from app.api.config import ROOT

REVIEW_DIR = ROOT / "data/reviewed/live_feedback"
_write_lock = threading.Lock()
PLACEHOLDER_TEXTS = {"NONE", "UNKNOWN", "UNREADABLE", "NOPLATE", "NA", "NULL"}


class ReviewPayload(BaseModel):
    reviewer: str
    plate_present: bool
    plate_readable: bool
    corrected_text: str = ""
    country: Literal["JORDAN", "FOREIGN", "UNKNOWN"]
    confirmed_from_pixels: bool


def _checked(payload: ReviewPayload) -> tuple[str, str]:
    reviewer = payload.reviewer.strip()
    raw_text = payload.corrected_text.strip()
    normalized = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    if not payload.confirmed_from_pixels:
        raise HTTPException(status_code=422, detail="confirm that you checked the image pixels")
    if not 2 <= len(reviewer) <= 64:
        raise HTTPException(status_code=422, detail="reviewer ID must be 2-64 characters")
    if not payload.plate_present:
        if payload.plate_readable or raw_text or payload.country != "UNKNOWN":
            raise HTTPException(status_code=422, detail="no-plate review must have no text and unknown country")
    elif payload.plate_readable:
        if not 1 <= len(normalized) <= 16 or len(raw_text) > 64:
            raise HTTPException(status_code=422, detail="readable plate needs 1-16 Latin letters or digits")
        if normalized in PLACEHOLDER_TEXTS:
            raise HTTPException(status_code=422, detail="use the plate-present/readable fields instead of placeholder text")
    elif raw_text:
        raise HTTPException(status_code=422, detail="unreadable plate must have empty text")
    return reviewer, normalized


def save_review(event: dict, payload: ReviewPayload) -> dict:
    """Save one opt-in review and its exact source image; never alter data/raw."""
    reviewer, normalized = _checked(payload)
    event_id = event["id"]
    if not re.fullmatch(r"[0-9a-f]{32}", event_id):
        raise ValueError("invalid camera event ID")
    image = event["image"]
    suffix = ".jpg" if event["content_type"] == "image/jpeg" else ".png"
    record = {
        "schema_version": 1,
        "review_id": event_id,
        "event_id": event_id,
        "camera_id": event["camera_id"],
        "captured_at": event["captured_at"],
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "reviewer": reviewer,
        "plate_present": payload.plate_present,
        "plate_readable": payload.plate_readable,
        "corrected_text": payload.corrected_text.strip(),
        "normalized_text": normalized,
        "country": payload.country,
        "confirmed_from_pixels": True,
        "source_image_filename": event["filename"],
        "saved_image_filename": event_id + suffix,
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "model_result": event["result"],
        "training_status": "quarantined_pending_independent_audit",
    }
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    image_path = REVIEW_DIR / record["saved_image_filename"]
    record_path = REVIEW_DIR / f"{event_id}.json"
    with _write_lock:
        if image_path.exists() or record_path.exists():
            raise HTTPException(status_code=409, detail="this capture has already been reviewed")
        with image_path.open("xb") as stream:
            stream.write(image)
        try:
            with record_path.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        except Exception:
            image_path.unlink(missing_ok=True)
            raise
    return {"review_id": event_id, "saved": True,
            "training_status": record["training_status"]}
