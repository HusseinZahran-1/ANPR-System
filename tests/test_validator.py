"""Unit tests - Phase 9 Jordanian validator + Phase 10 foreign policy.

These are **unit** tests (pure functions, ML-free, no models, no dataset):
they pin the rule-table contract and the decision mapping. They do not touch
``data/prepared/test`` or any protected evaluation artifact.

Historical ad-hoc self-tests live in ``scripts/test_phase9_validator.py`` and
``scripts/test_phase10_foreign.py`` and are intentionally left untouched.
"""

from __future__ import annotations

import pytest

from app.validator import decide, decide_text, normalize_text, validate
from app.validator.foreign_policy import (
    ACCEPT_JORDANIAN,
    FLAG_REVIEW,
    REJECT_FOREIGN,
)


# --------------------------------------------------------------------------
# normalize_text
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        (" 12-34 5 ", "12345"),          # whitespace + separator stripped
        ("ab c", "ABC"),                  # uppercased, space dropped
        ("1\u20132/3.4", "1234"),         # en-dash, slash, dot are separators
        ("", ""),                         # empty stays empty
        ("---", ""),                      # separators only -> empty
        (None, ""),                       # non-string -> ""
        (123, ""),                        # non-string -> ""
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_normalize_text_is_idempotent():
    assert normalize_text(normalize_text(" 12 34 ")) == normalize_text(" 12 34 ")


# --------------------------------------------------------------------------
# validate - classification contract
# --------------------------------------------------------------------------
def test_validate_has_exactly_four_keys_and_no_confidence():
    out = validate("1234567")
    assert set(out) == {"classification", "matched_rule", "normalized", "reason"}
    assert "confidence" not in out  # validator never invents a confidence


@pytest.mark.parametrize(
    "text, classification, rule",
    [
        ("1234567", "VALID_JORDANIAN", "JOR_PRIVATE_7"),
        ("123456", "VALID_JORDANIAN", "JOR_PRIVATE_6"),
        ("12345", "VALID_JORDANIAN", "JOR_PRIVATE_5"),
        ("1234", "VALID_JORDANIAN", "JOR_PRIVATE_4"),
        ("123", "VALID_JORDANIAN", "JOR_PRIVATE_3"),
        ("GV12345", "VALID_GOVERNMENT", "JOR_GOV_SPEC_5D"),
        ("T56454", "VALID_GOVERNMENT", "JOR_OBS_T"),
        ("X78572", "VALID_GOVERNMENT", "JOR_OBS_X"),   # before UAE_LIKELY_L5D
        ("2111GUD", "FOREIGN_LIKELY", "KSA_LIKELY"),
        ("12A34567", "FOREIGN_LIKELY", "IRQ_LIKELY_2L5D"),
        ("G12345", "FOREIGN_LIKELY", "UAE_LIKELY_L5D"),
    ],
)
def test_validate_classifications(text, classification, rule):
    out = validate(text)
    assert out["classification"] == classification
    assert out["matched_rule"] == rule


def test_validate_normalizes_before_matching():
    out = validate(" 12-345 67 ")
    assert out["normalized"] == "1234567"
    assert out["classification"] == "VALID_JORDANIAN"


@pytest.mark.parametrize("text", ["", "   ", "!", "-"])
def test_validate_empty_after_normalization_is_unknown(text):
    out = validate(text)
    assert out["classification"] == "UNKNOWN"
    assert out["matched_rule"] is None


@pytest.mark.parametrize("text", ["A", "AB"])
def test_validate_too_short_is_unknown(text):
    assert validate(text)["classification"] == "UNKNOWN"


def test_validate_too_long_is_invalid():
    out = validate("ABCDEFGHIJ")  # length 10 > MAX_PLATE_LENGTH (8)
    assert out["classification"] == "INVALID_FORMAT"
    assert out["matched_rule"] is None


def test_validate_wellformed_no_pattern_is_invalid():
    out = validate("ABCD")  # 4 letters: length ok, no rule matches
    assert out["classification"] == "INVALID_FORMAT"
    assert out["matched_rule"] is None


def test_speculative_rule_appends_note_to_reason():
    out = validate("X78572")
    assert "SPECULATIVE" in out["reason"]


def test_non_speculative_rule_has_no_note():
    assert "SPECULATIVE" not in validate("1234567")["reason"]


# --------------------------------------------------------------------------
# decide - Phase 10 policy mapping
# --------------------------------------------------------------------------
def test_decide_has_exactly_four_keys():
    out = decide("VALID_JORDANIAN", "JOR_PRIVATE_7", "1234567")
    assert set(out) == {"decision", "reason", "matched_rule", "speculative"}


@pytest.mark.parametrize(
    "classification, rule, expected",
    [
        ("VALID_JORDANIAN", "JOR_PRIVATE_7", ACCEPT_JORDANIAN),
        ("VALID_GOVERNMENT", "JOR_GOV_SPEC_5D", ACCEPT_JORDANIAN),
        ("FOREIGN_LIKELY", "KSA_LIKELY", REJECT_FOREIGN),
        ("FOREIGN_LIKELY", "UAE_LIKELY_L5D", REJECT_FOREIGN),
        ("INVALID_FORMAT", None, FLAG_REVIEW),
        ("UNKNOWN", None, FLAG_REVIEW),
        ("SOMETHING_NEW", None, FLAG_REVIEW),   # defensive fallback
    ],
)
def test_decide_mapping(classification, rule, expected):
    assert decide(classification, rule, "ABC")["decision"] == expected


def test_decide_flags_speculative_rule():
    out = decide("FOREIGN_LIKELY", "UAE_LIKELY_L5D", "G12345")
    assert out["speculative"] is True
    assert "SPECULATIVE" in out["reason"]


def test_decide_text_chains_validate_and_decide():
    out = decide_text("2111GUD")
    assert out["classification"] == "FOREIGN_LIKELY"
    assert out["decision"] == REJECT_FOREIGN
    assert out["normalized"] == "2111GUD"


def test_decide_text_accepts_private_plate():
    out = decide_text("1234567")
    assert out["decision"] == ACCEPT_JORDANIAN
