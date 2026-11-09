"""Unit tests - relocated OCR engine (app.ocr).

Covers the architecture constants, the CTC greedy decoder, the CNN width helper
and strict checkpoint compatibility. The checkpoint test is skipped if the Phase
8 weights are absent, so the suite stays runnable without models.

No protected test data is used here (synthetic tensors only).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from app.ocr import (
    ALPHABET,
    BLANK,
    NUM_CLASS,
    BidirectionalLSTM,
    CRNN,
    VGGFeatureExtractor,
    build_from_checkpoint,
    conv_out_width,
    encode,
    greedy_decode,
    load_crnn_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
OCR_WEIGHTS = (ROOT / "models" / "phase8_ocr" / "easyocr_crnn" /
               "weights" / "best.pt")


# --------------------------------------------------------------------------
# constants / tokenizer
# --------------------------------------------------------------------------
def test_constants():
    assert ALPHABET == "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert len(ALPHABET) == 36
    assert BLANK == 0
    assert NUM_CLASS == 37


def test_encode_maps_into_1_based_alphabet():
    assert encode("0") == [1]
    assert encode("A") == [11]
    assert encode("Z") == [36]
    for i, ch in enumerate(ALPHABET):
        assert encode(ch) == [i + 1]


def test_encode_rejects_unknown_character():
    with pytest.raises(ValueError):
        encode("a")  # lowercase not in alphabet


# --------------------------------------------------------------------------
# conv width bookkeeping
# --------------------------------------------------------------------------
@pytest.mark.parametrize("w", [16, 32, 64, 128, 256])
def test_conv_out_width_matches_forward(w):
    model = CRNN().eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 64, w))
    assert out.shape[1] == conv_out_width(w)
    assert out.shape == (1, conv_out_width(w), NUM_CLASS)


# --------------------------------------------------------------------------
# greedy CTC decode
# --------------------------------------------------------------------------
def _one_hot(ids: list[int]) -> torch.Tensor:
    logits = torch.full((len(ids), NUM_CLASS), -10.0)
    for t, i in enumerate(ids):
        logits[t, i] = 10.0
    return logits


def test_greedy_decode_removes_blanks_and_repeats():
    a, b, c = encode("A")[0], encode("B")[0], encode("C")[0]
    ids = [BLANK, a, a, BLANK, b, BLANK, c, BLANK]
    text, conf = greedy_decode(_one_hot(ids))
    assert text == "ABC"
    assert 0.9 < conf <= 1.0


def test_greedy_decode_all_blank_is_empty_and_zero_conf():
    text, conf = greedy_decode(_one_hot([BLANK, BLANK, BLANK]))
    assert text == ""
    assert conf == 0.0


def test_greedy_decode_digits():
    ids = [encode(ch)[0] for ch in "1234567"]
    text, _ = greedy_decode(_one_hot(ids))
    assert text == "1234567"


# --------------------------------------------------------------------------
# architecture shapes
# --------------------------------------------------------------------------
def test_vgg_and_lstm_shapes():
    vgg = VGGFeatureExtractor(1).eval()
    with torch.no_grad():
        feat = vgg(torch.zeros(1, 1, 64, 64))
    assert feat.shape == (1, 256, 3, 15)

    lstm = BidirectionalLSTM(256, 256, 256).eval()
    with torch.no_grad():
        out = lstm(torch.zeros(1, 7, 256))
    assert out.shape == (1, 7, 256)


def test_crnn_state_dict_keys_are_stable():
    keys = set(CRNN().state_dict().keys())
    assert len(keys) == 44
    assert "FeatureExtraction.ConvNet.0.weight" in keys
    assert "SequenceModeling.1.rnn.weight_ih_l0" in keys
    assert "Prediction.weight" in keys


# --------------------------------------------------------------------------
# application layer must not import the training script
# --------------------------------------------------------------------------
def test_app_ocr_does_not_import_training_script():
    assert "train_phase8_ocr" not in sys.modules
    import app.ocr.ocr_engine as engine
    app_root = (Path(__file__).resolve().parents[1] / "app").resolve()
    assert Path(engine.__file__).resolve().is_relative_to(app_root)


# --------------------------------------------------------------------------
# checkpoint compatibility (skipped when weights are absent)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not OCR_WEIGHTS.is_file(), reason="Phase 8 OCR weights absent")
def test_checkpoint_loads_strictly():
    loaded = load_crnn_checkpoint(OCR_WEIGHTS, device="cpu")
    ckpt = loaded["ckpt"]
    model = CRNN(num_class=int(ckpt["num_class"]),
                 input_channel=int(ckpt.get("input_channel", 1)))
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"],
                                                strict=False)
    assert not missing and not unexpected

    assert loaded["alphabet"] == ALPHABET
    assert (loaded["height"], loaded["wmax"], loaded["num_class"]) == (64, 256, 37)
    assert loaded["model"].training is False          # eval mode


@pytest.mark.skipif(not OCR_WEIGHTS.is_file(), reason="Phase 8 OCR weights absent")
def test_build_from_checkpoint_round_trips():
    ckpt = torch.load(str(OCR_WEIGHTS), map_location="cpu", weights_only=False)
    model = build_from_checkpoint(ckpt)
    assert isinstance(model, CRNN)
    assert model.Prediction.out_features == int(ckpt["num_class"])


@pytest.mark.skipif(not OCR_WEIGHTS.is_file(), reason="Phase 8 OCR weights absent")
def test_checkpoint_logits_deterministic():
    loaded = load_crnn_checkpoint(OCR_WEIGHTS, device="cpu")
    x = torch.zeros(1, 1, 64, 128)
    with torch.no_grad():
        a = loaded["model"](x)
        b = loaded["model"](x)
    assert torch.equal(a, b)
