"""Signed browser-only access to the live camera dashboard."""

from __future__ import annotations

import hashlib
import hmac
import time

from fastapi import HTTPException, Request

from app.api.config import Settings
from app.camera.settings import authorize_camera_key, camera_preferences

COOKIE = "anpr_camera_viewer"
HEADER = "X-Camera-Viewer"
MAX_AGE = 30 * 24 * 60 * 60


def _signature(expires: str, settings: Settings, credential: str) -> str:
    secret = settings.site_admin_key or credential
    key = hmac.digest(secret.encode(), f"camera-viewer:{credential}".encode(), hashlib.sha256)
    return hmac.new(key, expires.encode(), hashlib.sha256).hexdigest()


def make_viewer_session(key: str | None, settings: Settings) -> str:
    _, override = camera_preferences()
    credential = override or settings.camera_api_key
    if not credential:
        raise HTTPException(status_code=503, detail="camera viewer is unavailable")
    if override:
        candidate = hashlib.sha256(key.encode()).hexdigest() if key else ""
    else:
        candidate = key or ""
    if not hmac.compare_digest(candidate.encode(), credential.encode()):
        raise HTTPException(status_code=401, detail="invalid camera API key")
    expires = str(int(time.time()) + MAX_AGE)
    return f"{expires}.{_signature(expires, settings, credential)}"


def has_viewer_session(request: Request, settings: Settings) -> bool:
    if request.headers.get(HEADER) != "1":
        return False
    expires, separator, supplied = request.cookies.get(COOKIE, "").partition(".")
    if not separator or not expires.isascii() or not expires.isdigit() or len(expires) > 12:
        return False
    if int(expires) <= time.time():
        return False
    _, override = camera_preferences()
    credential = override or settings.camera_api_key
    return bool(credential and hmac.compare_digest(
        supplied.encode(), _signature(expires, settings, credential).encode()
    ))


def authorize_viewer(request: Request, key: str | None, settings: Settings) -> None:
    if key or not has_viewer_session(request, settings):
        authorize_camera_key(key, settings)
