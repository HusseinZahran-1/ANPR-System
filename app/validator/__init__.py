"""Jordanian plate validator package (Phase 9 + Phase 10).

ML-free, text-only rule-based validation of Jordanian vs foreign-looking plate
text (Phase 9) plus the foreign-plate rejection policy (Phase 10). See
``logs/phase9_validator/DESIGN.md`` and ``logs/phase10_foreign/DESIGN.md``.

    from app.validator import validate, decide, decide_text
    validate("1234567")     # classification
    decide_text("2111GUD")  # classification + decision
"""

from __future__ import annotations

from .foreign_policy import decide, decide_text
from .jordanian_validator import normalize_text, validate
from .rules import RULES

__all__ = ["validate", "normalize_text", "RULES", "decide", "decide_text"]
