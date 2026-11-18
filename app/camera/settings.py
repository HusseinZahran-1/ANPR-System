"""Persistent camera endpoint and credential settings for the local operator."""

from __future__ import annotations

from contextlib import closing
import hashlib
import hmac
from ipaddress import ip_address
import os
import re
import socket
import sqlite3
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

from app.api.config import Settings, get_settings, resolve_path

router = APIRouter(prefix="/ops/camera-settings", tags=["operator"])
_admin_header = APIKeyHeader(name="X-Site-Admin-Key", auto_error=False)
DEFAULT_BRIDGE = "/camera/events"


class CameraPreferences(BaseModel):
    public_host: str = ""
    public_port: int | None = Field(default=None, ge=1, le=65535)
    transport: Literal["HTTP", "HTTPS"] | None = None
    bridge_alias: str = Field(default="", pattern=r"^(?:[a-z0-9][a-z0-9-]{0,39})?$")

    @field_validator("public_host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        value = value.strip()
        if not value:
            return value
        try:
            return str(ip_address(value))
        except ValueError:
            labels = value.split(".")
            if len(value) <= 253 and all(
                re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label) for label in labels
            ):
                return value.lower()
        raise ValueError("enter an IP address or DNS name without a scheme or port")


class CameraUpdate(CameraPreferences):
    camera_key: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{16,128}$")


def _path():
    return resolve_path(os.getenv("CAMERA_SETTINGS_DB") or "data/processed/camera_settings.sqlite3")


def _saved(database: sqlite3.Connection) -> tuple[CameraPreferences, str]:
    row = database.execute("SELECT preferences, key_digest FROM camera_settings WHERE id = 1").fetchone()
    return (CameraPreferences.model_validate_json(row[0]), row[1]) if row else (CameraPreferences(), "")


def _ensure_table(database: sqlite3.Connection) -> None:
    database.execute(
        "CREATE TABLE IF NOT EXISTS camera_settings ("
        "id INTEGER PRIMARY KEY CHECK(id = 1), preferences TEXT NOT NULL, key_digest TEXT NOT NULL)"
    )


def camera_preferences() -> tuple[CameraPreferences, str]:
    path = _path()
    if not path.is_file():
        return CameraPreferences(), ""
    with closing(sqlite3.connect(path)) as database:
        _ensure_table(database)
        return _saved(database)


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def authorize_camera_key(key: str | None, settings: Settings) -> None:
    _, digest = camera_preferences()
    if not digest and not settings.camera_api_key:
        raise HTTPException(status_code=503, detail="CAMERA_API_KEY is not configured")
    if key and hmac.compare_digest(_digest(key), digest or _digest(settings.camera_api_key)):
        return
    raise HTTPException(status_code=401, detail="invalid camera API key")


def camera_destination(settings: Settings, preferences: CameraPreferences | None = None) -> dict:
    if preferences is None:
        preferences, _ = camera_preferences()
    host = preferences.public_host or settings.camera_public_host or settings.host
    if host in {"0.0.0.0", "::"}:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
                connection.connect(("192.0.2.1", 1))
                host = connection.getsockname()[0]
        except OSError:
            host = ""
    port = preferences.public_port or settings.camera_public_port or settings.port
    transport = preferences.transport or settings.camera_public_transport
    bridge = f"/camera/bridge/{preferences.bridge_alias}" if preferences.bridge_alias else DEFAULT_BRIDGE
    authority = f"[{host}]" if ":" in host else host
    return {"host": host, "port": port, "transport": transport, "bridge_path": bridge,
            "bridge_url": f"{transport.lower()}://{authority}:{port}{bridge}" if host else None}


def _require_admin(key: str | None = Security(_admin_header)) -> None:
    expected = get_settings().site_admin_key
    if not expected:
        raise HTTPException(status_code=503, detail="site settings administration is disabled")
    if not key or not hmac.compare_digest(_digest(key), _digest(expected)):
        raise HTTPException(status_code=401, detail="invalid site administrator key")


def _describe(settings: Settings, preferences: CameraPreferences, digest: str) -> dict:
    return {"overrides": preferences.model_dump(), "effective": camera_destination(settings, preferences),
            "camera_key_configured": bool(digest or settings.camera_api_key), "canonical_path": DEFAULT_BRIDGE}


@router.get("", dependencies=[Depends(_require_admin)])
def get_camera_settings(response: Response) -> dict:
    response.headers["Cache-Control"] = "no-store"
    return _describe(get_settings(), *camera_preferences())


@router.put("", dependencies=[Depends(_require_admin)])
def update_camera_settings(update: CameraUpdate, response: Response) -> dict:
    preferences = CameraPreferences(**update.model_dump(exclude={"camera_key"}))
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as database, database:
        _ensure_table(database)
        database.execute("BEGIN IMMEDIATE")
        _, previous_digest = _saved(database)
        digest = _digest(update.camera_key) if update.camera_key else previous_digest
        database.execute(
            "INSERT INTO camera_settings (id, preferences, key_digest) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET preferences=excluded.preferences, key_digest=excluded.key_digest",
            (preferences.model_dump_json(), digest),
        )
    response.headers["Cache-Control"] = "no-store"
    return _describe(get_settings(), preferences, digest)
