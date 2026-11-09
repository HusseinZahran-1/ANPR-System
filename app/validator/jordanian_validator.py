"""ML-free, text-only Jordanian plate validator (Phase 9 - Step 2).

Exactly one public entry point, :func:`validate`, plus the normalization helper
:func:`normalize_text`. The validator is a deterministic ruleset skeleton:
it re-applies the Phase 7 normalization, applies global length gates, then walks
the rule table top-down and returns the first match. There is no confidence
score, no image input, no ML, and no rejection concept (rejection is Phase 10).

See ``logs/phase9_validator/DESIGN.md`` for the design and the owner decisions,
and ``app/validator/rules.py`` for the rule table.
"""

from __future__ import annotations

import json
import re
import sys

try:  # package import (``from app.validator.jordanian_validator import validate``)
    from .rules import RULES
except ImportError:  # direct execution: ``python app/validator/jordanian_validator.py``
    from rules import RULES  # type: ignore

# Characters permitted in a normalized plate string.
ALNUM = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

# Separators removed by the Phase 7 normalization rule (rule 3). Includes the
# ASCII punctuation set plus en/em dashes and the middle-dot/bullet variants that
# were observed in reviewer input.
SEPARATORS = "-_\u2013\u2014/\\.,;:|\u00b7\u2022"

# Appended to ``reason`` for rules flagged speculative in rules.py. The em dash
# matches the owner-approved wording exactly.
SPECULATIVE_NOTE = "[SPECULATIVE \u2014 owner insight; confirm before ground truth]"

# Project-level length bounds (owner-approved, DESIGN.md section C).
MIN_DECIDABLE_LENGTH = 3
MAX_PLATE_LENGTH = 8

# Pre-compile the anchored patterns once, preserving rule-table order.
_COMPILED = [(rule, re.compile(rule["pattern"])) for rule in RULES]


def normalize_text(text: str) -> str:
    """Re-apply the exact Phase 7 Step 6 normalization rule, idempotently.

    uppercase -> strip whitespace -> strip separators -> keep ``[A-Z0-9]``.

    Never performs silent corrections: no ``0 <-> O``, no ``1 <-> I``, no
    leading-zero edits, no reordering. Returns ``""`` when nothing survives
    (including for non-string input).
    """
    if not isinstance(text, str):
        return ""
    s = text.upper()                                     # 1. uppercase
    s = "".join(ch for ch in s if not ch.isspace())      # 2. drop whitespace
    for ch in SEPARATORS:                                # 3. drop separators
        s = s.replace(ch, "")
    s = "".join(ch for ch in s if ch in ALNUM)           # 4. keep [A-Z0-9]
    return s


def _result(classification: str, matched_rule: str | None,
            normalized: str, reason: str) -> dict:
    return {
        "classification": classification,
        "matched_rule": matched_rule,
        "normalized": normalized,
        "reason": reason,
    }


def validate(text: str) -> dict:
    """Classify a plate string.

    Returns a dict with exactly four keys:

    ``classification``
        one of ``VALID_JORDANIAN``, ``VALID_GOVERNMENT``, ``FOREIGN_LIKELY``,
        ``INVALID_FORMAT``, ``UNKNOWN``.
    ``matched_rule``
        the rule ID that fired, or ``None`` when no rule matched.
    ``normalized``
        the normalized text actually evaluated (never a "corrected" plate).
    ``reason``
        short human-readable explanation; speculative rules include the
        ``[SPECULATIVE ...]`` note.

    There is deliberately **no confidence score** and **no rejection**: a
    ``FOREIGN_LIKELY`` result is a flag only.
    """
    normalized = normalize_text(text)

    # Global gate G1: nothing left after normalization.
    if not normalized:
        return _result("UNKNOWN", None, normalized, "empty after normalization")

    n = len(normalized)

    # Global gate G2: too short to decide.
    if n < MIN_DECIDABLE_LENGTH:
        return _result("UNKNOWN", None, normalized,
                       "too short to decide (length 1-2)")

    # Global gate G3: longer than the project maximum.
    if n > MAX_PLATE_LENGTH:
        return _result("INVALID_FORMAT", None, normalized,
                       f"exceeds project max length {MAX_PLATE_LENGTH}")

    # Rule table: top-down, first match wins.
    for rule, rx in _COMPILED:
        if rx.match(normalized):
            reason = f"matched {rule['id']} ({rule['classification']})"
            if rule.get("speculative"):
                reason = f"{reason} {SPECULATIVE_NOTE}"
            return _result(rule["classification"], rule["id"], normalized,
                           reason)

    # Fallback: well-formed length 3-8 over [A-Z0-9], but no known pattern.
    return _result("INVALID_FORMAT", None, normalized, "no known pattern")


def _main() -> int:
    """Read one plate string from stdin and print the JSON verdict."""
    text = sys.stdin.read().rstrip("\r\n")
    print(json.dumps(validate(text), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
