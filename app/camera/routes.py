"""Authenticated camera image ingestion and live-view endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from app.api.config import get_settings
from app.api.routes import analyze_upload
from app.camera.archive import complete_capture, save_capture_export, recent_records, save_capture, stored_capture
from app.camera.events import events
from app.camera.reviews import ReviewPayload, save_review
from app.camera.settings import authorize_camera_key, camera_destination
from app.camera.viewer import (
    AUTO_DISABLED_COOKIE, COOKIE, HEADER, MAX_AGE,
    authorize_viewer, has_viewer_session, is_local_viewer, make_viewer_session,
)
from app.capture_exports import export_location
from app.validator.plate_parts import split_plate_text
from app.vehicle import Vehicle, select_primary_vehicle

router = APIRouter(prefix="/camera", tags=["camera"])


def _authorize(key: str | None) -> None:
    authorize_camera_key(key, get_settings())


@router.post("/events", status_code=201)
async def receive_camera_image(
    file: UploadFile = File(...),
    camera_id: str = Form("gate-1"),
    x_camera_key: str | None = Header(None),
) -> dict:
    """Receive a camera capture, run ANPR, and publish it to the live page."""
    _authorize(x_camera_key)
    return await _store_camera_image(file, camera_id)


async def _store_camera_image(file: UploadFile, camera_id: str) -> dict:
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
        result["export"] = await run_in_threadpool(save_capture_export, archived, image, result)
    except Exception as exc:
        complete_capture(archived, error=exc)
        raise
    complete_capture(archived, result=result)
    return events.add(camera_id=camera_id, filename=filename,
                      content_type=content_type, image=image, result=result,
                      event_id=archived["event_id"], captured_at=archived["captured_at"])


@router.post("/viewer/session", status_code=204)
def start_viewer_session(request: Request, x_camera_key: str | None = Header(None)) -> Response:
    settings = get_settings()
    local = not x_camera_key and request.headers.get(HEADER) == "1" and is_local_viewer(request, settings)
    session = make_viewer_session(x_camera_key, settings, local=local)
    response = Response(status_code=204, headers={"Cache-Control": "no-store"})
    cookie_options = {"path": "/camera", "httponly": True, "samesite": "strict",
                      "secure": request.url.scheme == "https"}
    response.set_cookie(COOKIE, session, max_age=MAX_AGE, **cookie_options)
    response.delete_cookie(AUTO_DISABLED_COOKIE, **cookie_options)
    return response


@router.get("/viewer/session")
def check_viewer_session(request: Request, response: Response) -> dict:
    response.headers["Cache-Control"] = "no-store"
    settings = get_settings()
    return {
        "connected": has_viewer_session(request, settings),
        "local_access": is_local_viewer(request, settings),
        "auto_connect": is_local_viewer(request, settings) and AUTO_DISABLED_COOKIE not in request.cookies,
    }


@router.delete("/viewer/session", status_code=204)
def end_viewer_session(request: Request) -> Response:
    if request.headers.get(HEADER) != "1":
        raise HTTPException(status_code=403, detail="viewer request required")
    response = Response(status_code=204, headers={"Cache-Control": "no-store"})
    cookie_options = {"path": "/camera", "httponly": True, "samesite": "strict",
                      "secure": request.url.scheme == "https"}
    response.delete_cookie(COOKIE, **cookie_options)
    response.set_cookie(AUTO_DISABLED_COOKIE, "1", max_age=MAX_AGE, **cookie_options)
    return response


@router.post("/viewer/captures", status_code=201)
async def send_viewer_test_capture(
    request: Request, file: UploadFile = File(...), x_camera_key: str | None = Header(None),
) -> dict:
    authorize_viewer(request, x_camera_key, get_settings())
    return await _store_camera_image(file, "manual-test")


@router.get("/bridge/config")
def get_camera_bridge(request: Request, response: Response, x_camera_key: str | None = Header(None)) -> dict:
    authorize_viewer(request, x_camera_key, get_settings())
    response.headers["Cache-Control"] = "no-store"
    return camera_destination(get_settings())


@router.post("/bridge/{alias}", status_code=201)
async def receive_camera_alias(
    alias: str, file: UploadFile = File(...), camera_id: str = Form("gate-1"),
    x_camera_key: str | None = Header(None),
) -> dict:
    if camera_destination(get_settings())["bridge_path"] != f"/camera/bridge/{alias}":
        await file.close()
        raise HTTPException(status_code=404, detail="camera bridge path not configured")
    return await receive_camera_image(file=file, camera_id=camera_id, x_camera_key=x_camera_key)


@router.get("/events")
def list_camera_events(request: Request, response: Response, limit: int = 10,
                       x_camera_key: str | None = Header(None)) -> dict:
    authorize_viewer(request, x_camera_key, get_settings())
    response.headers["Cache-Control"] = "no-store"
    if not 1 <= limit <= 10:
        raise HTTPException(status_code=422, detail="limit must be 1-10")
    visible = {item["id"]: item for item in events.list(limit)}
    for record in recent_records(limit):
        visible[record["event_id"]] = {
            "id": record["event_id"], "captured_at": record["captured_at"],
            "camera_id": record["camera_id"], "filename": record["original_filename"],
            "result": record["result"],
        }
    latest = sorted(visible.values(), key=lambda item: item["captured_at"], reverse=True)[:limit]
    for index, event in enumerate(latest):
        result = event["result"]
        readings = result.get("ocr_results", [])
        if any(reading.get("plate_parts") is None for reading in readings):
            # Derive the display fields from the saved OCR without editing the archive.
            readings = [
                {**reading, "plate_parts": split_plate_text(reading.get("text", ""))}
                if reading.get("plate_parts") is None else reading
                for reading in readings
            ]
            result = {**result, "ocr_results": readings}
        vehicles = result.get("vehicles") or []
        if vehicles:
            primary = select_primary_vehicle([Vehicle.model_validate(vehicle) for vehicle in vehicles])
            result = {**result, "vehicles": [primary.model_dump(mode="json")]}
        if result.get("export"):
            location = export_location(result["export"])
            if location is not result["export"]:
                result = {**result, "export": location}
        if result is not event["result"]:
            latest[index] = {**event, "result": result}
    return {"events": latest}


@router.get("/events/{event_id}/image")
def get_camera_image(event_id: str, request: Request, x_camera_key: str | None = Header(None)) -> Response:
    authorize_viewer(request, x_camera_key, get_settings())
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
def review_camera_image(event_id: str, payload: ReviewPayload, request: Request,
                        x_camera_key: str | None = Header(None)) -> dict:
    """Save an explicitly checked capture for later dataset audit."""
    authorize_viewer(request, x_camera_key, get_settings())
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
