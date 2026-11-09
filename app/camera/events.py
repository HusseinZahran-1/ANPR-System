"""Small in-memory event buffer for the local camera MVP."""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from uuid import uuid4


class CameraEvents:
    def __init__(self, max_events: int = 10, max_image_bytes: int = 30 * 1024 * 1024):
        self.max_events = max_events
        self.max_image_bytes = max_image_bytes
        self._lock = threading.Lock()
        self._events: deque[dict] = deque()
        self._image_bytes = 0

    def add(self, *, camera_id: str, filename: str, content_type: str,
            image: bytes, result: dict, event_id: str | None = None,
            captured_at: str | None = None) -> dict:
        event = {
            "id": event_id or uuid4().hex,
            "captured_at": captured_at or datetime.now(timezone.utc).isoformat(),
            "camera_id": camera_id,
            "filename": filename,
            "result": result,
            "content_type": content_type,
            "image": image,
        }
        with self._lock:
            self._events.appendleft(event)
            self._image_bytes += len(image)
            while (len(self._events) > self.max_events or
                   self._image_bytes > self.max_image_bytes):
                old = self._events.pop()
                self._image_bytes -= len(old["image"])
        return self.public(event)

    @staticmethod
    def public(event: dict) -> dict:
        return {key: value for key, value in event.items()
                if key not in {"image", "content_type"}}

    def list(self, limit: int = 10) -> list[dict]:
        with self._lock:
            return [self.public(e) for e in list(self._events)[:limit]]

    def image(self, event_id: str) -> tuple[bytes, str] | None:
        with self._lock:
            for event in self._events:
                if event["id"] == event_id:
                    return event["image"], event["content_type"]
        return None

    def snapshot(self, event_id: str) -> dict | None:
        """Return a stable copy of an event for an explicit review submission."""
        with self._lock:
            for event in self._events:
                if event["id"] == event_id:
                    return dict(event)
        return None

    def clear(self) -> None:
        """Clear volatile events; useful for isolated tests."""
        with self._lock:
            self._events.clear()
            self._image_bytes = 0


events = CameraEvents()
