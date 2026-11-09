import torch

from app.ocr.ctc_beam import beam_decode


def test_ctc_beam_keeps_repeated_character_only_after_blank():
    adjacent = torch.tensor([[0.0, 8.0, -4.0], [0.0, 8.0, -4.0]])
    separated = torch.tensor([[0.0, 8.0, -4.0], [8.0, 0.0, -4.0],
                              [0.0, 8.0, -4.0]])
    assert beam_decode(adjacent, "AB") == "A"
    assert beam_decode(separated, "AB") == "AA"


def test_ctc_beam_rejects_wrong_alphabet_shape():
    logits = torch.zeros(2, 4)
    try:
        beam_decode(logits, "AB")
    except ValueError as exc:
        assert "blank" in str(exc)
    else:
        raise AssertionError("mismatched output alphabet was accepted")
