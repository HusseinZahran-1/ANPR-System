"""Separate recognized text using explicit syntax and local format assumptions.

This is presentation metadata, not another OCR prediction or country evidence.
Numeric strings with more than one possible boundary remain unresolved.
"""
from __future__ import annotations

import re

from .jordanian_validator import normalize_text
from .rules import GOV_CODES


def split_plate_text(text: str) -> dict:
    full = normalize_text(text)
    result = {
        "full_text": full,
        "plate_code": None,
        "plate_number": None,
        "split_basis": "unresolved",
        "requires_human_review": True,
    }
    if not isinstance(text, str):
        return result
    # Preserve an explicit boundary before the normalizer removes separators.
    explicit = re.fullmatch(r"\s*([0-9]{1,2})\s*[-/\u2013\u2014]\s*([0-9]{1,5})\s*", text)
    authority = re.fullmatch(r"(" + "|".join(GOV_CODES) + r")([0-9]{1,6})", full)
    if explicit:
        code, number = explicit.groups()
        basis = "explicit_separator"
    elif authority:
        code, number = authority.groups()
        basis = "recognized_authority_prefix"
    elif re.fullmatch(r"[0-9]{7}", full):
        # Unique 2+5 boundary under the project's 1-2 code / 1-5 number model.
        # This assumption does not establish that the plate is Jordanian.
        code, number = full[:2], full[2:]
        basis = "numeric_2_plus_5_assumption"
    else:
        return result
    result.update(plate_code=code, plate_number=number, split_basis=basis)
    return result
