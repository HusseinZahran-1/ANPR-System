"""Flag known authority-prefix OCR confusions without rewriting model output."""

from __future__ import annotations

import re

from app.validator.jordanian_validator import validate


def possible_authority_text(text: str) -> str | None:
    """Suggest an unverified CD/SN reading for a narrowly matched confusion.

    The suggestion is for a human inspecting the image. It never replaces OCR
    text or changes the validator decision and is not a country prediction.
    """
    normalized = re.sub(r"[^A-Z0-9]", "", text.upper())
    if re.fullmatch(r"C0\d{1,6}", normalized):
        candidate = "CD" + normalized[2:]
    elif re.fullmatch(r"5N\d{1,6}", normalized):
        candidate = "SN" + normalized[2:]
    elif re.fullmatch(r"5S\d{1,6}", normalized):
        candidate = "SN" + normalized[2:]
    else:
        return None
    return candidate if validate(candidate)["classification"] == "VALID_GOVERNMENT" else None
