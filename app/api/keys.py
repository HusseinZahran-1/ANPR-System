"""Issued inference keys. Cleartext is returned only by the creation request."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import hmac
import os
import secrets
import sqlite3
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Response, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, StringConstraints
from typing import Annotated

from app.api.config import get_settings, resolve_path

_admin_header = APIKeyHeader(name="X-Inference-Admin-Key", auto_error=False)
router = APIRouter(prefix="/v1/keys", tags=["inference-keys"])


def _admin(key: str | None = Security(_admin_header)) -> None:
    expected = get_settings().inference_admin_key
    if not expected:
        raise HTTPException(status_code=503, detail="inference key management is disabled")
    if not key or not hmac.compare_digest(key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="invalid inference admin key")


def _database_path():
    return resolve_path(os.getenv("INFERENCE_KEY_DB") or "data/processed/inference_keys.sqlite3")


def _open_database():
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(path)
    try:
        database.row_factory = sqlite3.Row
        database.execute(
            "CREATE TABLE IF NOT EXISTS inference_keys "
            "(id TEXT PRIMARY KEY, label TEXT NOT NULL, digest TEXT UNIQUE NOT NULL, created_at TEXT NOT NULL)"
        )
    except Exception:
        database.close()
        raise
    return database


def valid_issued_key(key: str) -> bool:
    if not key.startswith("anpr_") or len(key) != 48 or not _database_path().is_file():
        return False
    with closing(_open_database()) as database:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return database.execute("SELECT 1 FROM inference_keys WHERE digest = ?", (digest,)).fetchone() is not None


class KeyRequest(BaseModel):
    label: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]


@router.post("", status_code=201, dependencies=[Depends(_admin)])
def create_key(payload: KeyRequest, response: Response) -> dict:
    """Generate a revocable key for the versioned inference API."""
    key = "anpr_" + secrets.token_urlsafe(32)
    digest = hashlib.sha256(key.encode()).hexdigest()
    meta = {"id": uuid4().hex, "label": payload.label, "created_at": datetime.now(timezone.utc).isoformat()}
    with closing(_open_database()) as database, database:
        database.execute("INSERT INTO inference_keys(id, label, digest, created_at) VALUES (?, ?, ?, ?)",
                         (meta["id"], meta["label"], digest, meta["created_at"]))
    response.headers["Cache-Control"] = "no-store"
    return {**meta, "key": key}


@router.get("", dependencies=[Depends(_admin)])
def list_keys(response: Response) -> list[dict]:
    response.headers["Cache-Control"] = "no-store"
    with closing(_open_database()) as database:
        return [dict(row) for row in database.execute(
            "SELECT id, label, created_at FROM inference_keys ORDER BY created_at DESC"
        )]


@router.delete("/{key_id}", status_code=204, dependencies=[Depends(_admin)])
def revoke_key(key_id: str) -> Response:
    with closing(_open_database()) as database, database:
        deleted = database.execute("DELETE FROM inference_keys WHERE id = ?", (key_id,)).rowcount
    if not deleted:
        raise HTTPException(status_code=404, detail="inference key not found")
    return Response(status_code=204, headers={"Cache-Control": "no-store"})
