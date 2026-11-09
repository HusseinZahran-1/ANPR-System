"""Authenticated camera image ingestion and live-view endpoints."""

from __future__ import annotations

import hmac
from pathlib import Path

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response

from app.api.config import get_settings
from app.api.routes import analyze_upload
from app.camera.archive import complete_capture, recent_records, save_capture, stored_capture
from app.camera.events import events
from app.camera.reviews import ReviewPayload, save_review

router = APIRouter(prefix="/camera", tags=["camera"])


def _authorize(key: str | None) -> None:
    expected = get_settings().camera_api_key
    if not expected:
        raise HTTPException(status_code=503, detail="CAMERA_API_KEY is not configured")
    if not key or not hmac.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="invalid camera API key")


@router.post("/events", status_code=201)
async def receive_camera_image(
    file: UploadFile = File(...),
    camera_id: str = Form("gate-1"),
    x_camera_key: str | None = Header(None),
) -> dict:
    """Receive a camera capture, run ANPR, and publish it to the live page."""
    _authorize(x_camera_key)
    if not camera_id or len(camera_id) > 64:
        await file.close()
        raise HTTPException(status_code=422, detail="camera_id must be 1-64 characters")
    size_limit = get_settings().max_upload_bytes
    image = await file.read(size_limit + 1)
    if len(image) > size_limit:
        await file.close()
        raise HTTPException(status_code=413, detail="uploaded image exceeds the size limit")
    await file.seek(0)
    filename = Path(file.filename or "camera-capture").name
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    archived = save_capture(image=image, content_type=content_type,
                            camera_id=camera_id, original_filename=filename)
    try:
        result = await analyze_upload(file, camera=True)
    except Exception as exc:
        complete_capture(archived, error=exc)
        raise
    complete_capture(archived, result=result)
    return events.add(camera_id=camera_id, filename=filename,
                      content_type=content_type, image=image, result=result,
                      event_id=archived["event_id"], captured_at=archived["captured_at"])


@router.get("/events")
def list_camera_events(limit: int = 10, x_camera_key: str | None = Header(None)) -> dict:
    _authorize(x_camera_key)
    if not 1 <= limit <= 10:
        raise HTTPException(status_code=422, detail="limit must be 1-10")
    visible = {item["id"]: item for item in events.list(limit)}
    for record in recent_records(limit):
        visible[record["event_id"]] = {
            "id": record["event_id"], "captured_at": record["captured_at"],
            "camera_id": record["camera_id"], "filename": record["original_filename"],
            "result": record["result"],
        }
    return {"events": sorted(visible.values(), key=lambda item: item["captured_at"],
                             reverse=True)[:limit]}


@router.get("/events/{event_id}/image")
def get_camera_image(event_id: str, x_camera_key: str | None = Header(None)) -> Response:
    _authorize(x_camera_key)
    found = events.image(event_id)
    if found is None:
        archived = stored_capture(event_id)
        if archived is not None:
            record, image = archived
            found = (image, record["content_type"])
    if found is None:
        raise HTTPException(status_code=404, detail="camera image not found")
    image, content_type = found
    return Response(image, media_type=content_type,
                    headers={"Cache-Control": "no-store"})


@router.post("/events/{event_id}/review", status_code=201)
def review_camera_image(event_id: str, payload: ReviewPayload,
                        x_camera_key: str | None = Header(None)) -> dict:
    """Save an explicitly checked capture for later dataset audit."""
    _authorize(x_camera_key)
    event = events.snapshot(event_id)
    if event is None:
        archived = stored_capture(event_id)
        if archived is not None:
            record, image = archived
            event = {"id": event_id, "captured_at": record["captured_at"],
                     "camera_id": record["camera_id"], "filename": record["original_filename"],
                     "content_type": record["content_type"], "image": image,
                     "result": record["result"]}
    if event is None:
        raise HTTPException(status_code=404, detail="camera capture not found or expired")
    return save_review(event, payload)
