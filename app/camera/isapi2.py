"""Receive ISAPI-style camera pushes without assuming a fixed multipart field.

The camera's exact wire format is not available yet. This adapter accepts a
direct JPEG/PNG or a multipart request containing one or more image parts,
selects the largest valid scene image, and records packet diagnostics. Valid
images are archived as unverified camera captures before inference.
"""

from __future__ import annotations

import hmac
import io
import socket
import threading
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.datastructures import Headers

from app.api.config import get_settings
from app.api.routes import analyze_upload
from app.camera.archive import complete_capture, save_capture
from app.camera.events import events

router = APIRouter(prefix="/camera/isapi2", tags=["camera"])
_diagnostic_lock = threading.Lock()
_last_packet: dict = {}


def _record_packet(summary: dict) -> None:
    with _diagnostic_lock:
        _last_packet.clear()
        _last_packet.update(summary)


def _image_info(data: bytes) -> tuple[int, str] | None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG"}:
                return None
            if image.width * image.height > 50_000_000:
                return None
            return image.width * image.height, "image/jpeg" if image.format == "JPEG" else "image/png"
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return None


def _extract(body: bytes, content_type: str, max_image_bytes: int) -> tuple[bytes | None, str, str, list[dict]]:
    """Return the largest valid image and safe metadata for diagnostics."""
    parts: list[tuple[str, str, bytes]] = []
    if content_type.lower().startswith("multipart/"):
        header = ("MIME-Version: 1.0\r\nContent-Type: " + content_type + "\r\n\r\n").encode("latin-1")
        message = BytesParser(policy=policy.default).parsebytes(header + body)
        if not message.is_multipart():
            raise HTTPException(status_code=422, detail="camera multipart boundary could not be parsed")
        for part in message.walk():
            if part.is_multipart():
                continue
            name = part.get_filename() or part.get_param("name", header="content-disposition") or "unnamed"
            parts.append((str(name)[:80], part.get_content_type(), part.get_payload(decode=True) or b""))
    else:
        parts.append(("body", content_type.split(";")[0].strip().lower(), body))

    summary = []
    candidates = []
    for index, (name, part_type, data) in enumerate(parts):
        summary.append({"name": name, "content_type": part_type, "bytes": len(data)})
        if len(data) > max_image_bytes:
            continue
        info = _image_info(data)
        if info is not None:
            area, mime = info
            candidates.append((area, index, data, mime, name))
    if not candidates:
        return None, "", "", summary
    _, _, image, mime, name = max(candidates, key=lambda item: (item[0], -item[1]))
    suffix = ".jpg" if mime == "image/jpeg" else ".png"
    filename = Path(name).name
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        filename = "camera-capture" + suffix
    return image, mime, filename, summary


@router.post("/{token}")
async def receive_isapi2(request: Request, token: str) -> dict:
    """Accept a camera push authenticated by an opaque URL path token."""
    settings = get_settings()
    if not settings.camera_isapi_token:
        raise HTTPException(status_code=503, detail="CAMERA_ISAPI_TOKEN is not configured")
    if not hmac.compare_digest(token, settings.camera_isapi_token):
        raise HTTPException(status_code=401, detail="invalid camera endpoint")
    content_type = request.headers.get("content-type", "").strip()
    max_body = 2 * settings.max_upload_bytes + 1024 * 1024
    chunks = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > max_body:
            raise HTTPException(status_code=413, detail="camera payload exceeds the size limit")
        chunks.append(chunk)
    body = b"".join(chunks)
    extracted = _extract(body, content_type, settings.max_upload_bytes)
    image, mime, filename, parts = extracted
    diagnostic = {"received_at": datetime.now(timezone.utc).isoformat(),
                  "content_type": content_type[:120], "bytes": size,
                  "parts": parts, "image_selected": image is not None}
    _record_packet(diagnostic)
    if image is None:
        # Camera heartbeats carry no picture and should not create an ANPR event.
        if b"heartbeat" in body[:100_000].lower():
            return {"status": "ok", "message": "heartbeat received"}
        raise HTTPException(status_code=422, detail="no JPEG or PNG scene image found in camera payload")
    upload = UploadFile(file=io.BytesIO(image), filename=filename,
                        headers=Headers({"content-type": mime}))
    camera_id = "isapi2-" + (request.client.host if request.client else "camera")
    camera_id = camera_id[:64]
    archived = save_capture(image=image, content_type=mime,
                            camera_id=camera_id, original_filename=filename)
    try:
        result = await analyze_upload(upload, camera=True)
    except Exception as exc:
        complete_capture(archived, error=exc)
        raise
    complete_capture(archived, result=result)
    event = events.add(camera_id=camera_id[:64], filename=filename,
                       content_type=mime, image=image, result=result,
                       event_id=archived["event_id"], captured_at=archived["captured_at"])
    return {"status": "ok", "event_id": event["id"], "camera_id": event["camera_id"]}


@router.get("/status/last")
def last_isapi2_packet(x_camera_key: str | None = Header(None)) -> dict:
    """Expose image-free packet diagnostics to the authenticated live viewer."""
    expected = get_settings().camera_api_key
    if not expected or not x_camera_key or not hmac.compare_digest(x_camera_key, expected):
        raise HTTPException(status_code=401, detail="invalid camera API key")
    with _diagnostic_lock:
        return dict(_last_packet)


@router.get("/config")
def isapi2_setup(x_camera_key: str | None = Header(None)) -> dict:
    """Give an authenticated operator the exact camera-side settings."""
    settings = get_settings()
    if not settings.camera_api_key or not x_camera_key or not hmac.compare_digest(
            x_camera_key, settings.camera_api_key):
        raise HTTPException(status_code=401, detail="invalid camera API key")
    if not settings.camera_isapi_token:
        raise HTTPException(status_code=503, detail="CAMERA_ISAPI_TOKEN is not configured")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            connection.connect(("192.0.2.1", 1))
            host_ip = connection.getsockname()[0]
    except OSError:
        host_ip = ""
    return {"version": "HTTP", "host_ip": host_ip, "host_port": settings.port,
            "host_url": "/camera/isapi2/" + settings.camera_isapi_token,
            "heartbeat_interval_s": 0, "uploaded_picture_type": "Detection Picture",
            "authentication_mode": "None", "platform_response_verification": False}
