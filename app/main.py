"""FastAPI application entry point."""

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse

from app.api.routes import router
from app.camera.routes import router as camera_router
from app.camera.isapi2 import router as isapi2_router
from app.camera.archive import recent_captures
from app.camera.events import events


@asynccontextmanager
async def lifespan(_app: FastAPI):
    for record, image in recent_captures():
        events.add(camera_id=record["camera_id"], filename=record["original_filename"],
                   content_type=record["content_type"], image=image,
                   result=record["result"], event_id=record["event_id"],
                   captured_at=record["captured_at"])
    yield

app = FastAPI(
    title="Jordan ANPR",
    description="Jordan-focused license plate detection, OCR, and validation.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)
app.include_router(camera_router)
app.include_router(isapi2_router)


@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    return RedirectResponse(url="/demo", status_code=307)


@app.get("/demo", include_in_schema=False)
def demo() -> FileResponse:
    """Serve the self-contained company demo on the same origin as the API."""
    return FileResponse(Path(__file__).parent / "demo" / "index.html",
                        media_type="text/html; charset=utf-8")


@app.get("/live", include_in_schema=False)
def live() -> FileResponse:
    """Serve the local camera monitor."""
    return FileResponse(Path(__file__).parent / "demo" / "live.html",
                        media_type="text/html; charset=utf-8")
