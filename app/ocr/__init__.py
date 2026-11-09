"""Phase 11/12 - OCR application package.

    from app.ocr import CRNN, greedy_decode, load_crnn_checkpoint

Re-exports the relocated Phase 8 CRNN architecture, the greedy CTC decoder and
the checkpoint loader so the pipeline never imports the training script.
"""

from __future__ import annotations

from .ocr_engine import (
    ALPHABET,
    BLANK,
    CRNN,
    HEIGHT,
    INPUT_CHANNEL,
    NUM_CLASS,
    WMAX,
    BidirectionalLSTM,
    VGGFeatureExtractor,
    build_from_checkpoint,
    conv_out_width,
    encode,
    greedy_decode,
    load_crnn_checkpoint,
)

__all__ = [
    "ALPHABET",
    "BLANK",
    "NUM_CLASS",
    "INPUT_CHANNEL",
    "HEIGHT",
    "WMAX",
    "CRNN",
    "VGGFeatureExtractor",
    "BidirectionalLSTM",
    "conv_out_width",
    "encode",
    "greedy_decode",
    "build_from_checkpoint",
    "load_crnn_checkpoint",
]
