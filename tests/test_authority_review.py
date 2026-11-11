"""An OCR ambiguity must never be silently turned into a confirmed plate."""

from app.ocr.authority_review import possible_authority_text
from app.validator.jordanian_validator import validate


def test_known_authority_prefix_confusions_are_review_suggestions():
    assert possible_authority_text("C090001") == "CD90001"
    assert possible_authority_text("5N90002") == "SN90002"
    assert possible_authority_text("5S90002") == "SN90002"
    assert validate("C090001")["classification"] != "VALID_GOVERNMENT"


def test_unrelated_or_malformed_readings_are_not_rewritten():
    for text in ("CD90001", "SN90003", "1234567", "C0ABC", "5N", "5N1234567", "50-12345"):
        assert possible_authority_text(text) is None
