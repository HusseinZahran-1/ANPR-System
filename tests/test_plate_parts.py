import pytest
from app.validator.plate_parts import split_plate_text


@pytest.mark.parametrize("text,code,number,basis", [
    ("12-90001", "12", "90001", "explicit_separator"),
    ("1/0042", "1", "0042", "explicit_separator"),
    ("1290001", "12", "90001", "numeric_2_plus_5_assumption"),
    ("CD90001", "CD", "90001", "recognized_authority_prefix"),
    ("sn 00042", "SN", "00042", "recognized_authority_prefix"),
])
def test_split_preserves_strings_and_reports_assumption(text,code,number,basis):
    parts=split_plate_text(text)
    assert parts["plate_code"]==code and parts["plate_number"]==number
    assert parts["full_text"]==code+number
    assert parts["split_basis"]==basis and parts["requires_human_review"] is True


@pytest.mark.parametrize("text", ["190001", "90001", "C090001", "5N90001", "5S90001", "2111GUD", "", None])
def test_ambiguous_numeric_unknown_and_ocr_confusions_do_not_invent_code(text):
    parts=split_plate_text(text)
    assert parts["plate_code"] is None and parts["plate_number"] is None
    assert parts["split_basis"]=="unresolved"
