"""Phase 11 - End-to-End pipeline package.

    from app.pipeline import Pipeline, run
    Pipeline().run("scene.jpg")

Re-exports :func:`run` (cached default pipeline), the :class:`Pipeline` class,
the optional typed view :class:`PipelineResult`, and :func:`load_models`.
"""

from __future__ import annotations

from .end_to_end import (
    DEFAULT_CONF,
    DETECTOR_DEFAULT,
    OCR_DEFAULT,
    SEED,
    Pipeline,
    PipelineResult,
    load_models,
    run,
)

__all__ = [
    "Pipeline",
    "PipelineResult",
    "load_models",
    "run",
    "DETECTOR_DEFAULT",
    "OCR_DEFAULT",
    "SEED",
    "DEFAULT_CONF",
]
