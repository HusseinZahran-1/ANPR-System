"""Phase 11/12 - Production OCR engine: CRNN architecture + CTC decoder.

This module hosts the **inference-side** model definition and decoding utilities
that previously lived only inside ``scripts/train_phase8_ocr.py`` (a training
script). The application layer must not depend on training code through
``sys.path`` manipulation, so the architecture is relocated here.

Fidelity guarantees
-------------------
The class names, module nesting, layer order/arguments, and constants are copied
**verbatim** from the Phase 8 training script. As a result a Phase 8 checkpoint
loads with a *strict* state_dict match and produces identical logits:

    VGGFeatureExtractor.ConvNet.0 ... .12      (22 tensors)
    SequenceModeling.0/.1 (BidirectionalLSTM)  (20 tensors)
    Prediction.{weight,bias}                   (2 tensors)
    -> 44 tensors total, 0 missing / 0 unexpected

Verified against ``models/phase8_ocr/easyocr_crnn/weights/best.pt``
(``num_class=37``, ``input_channel=1``, ``height=64``, ``wmax=256``,
``alphabet="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"``).

Architecture (DTRB / EasyOCR ``english_g2``): VGG backbone + 2xBiLSTM + linear
CTC head. Input is grayscale (1 channel), height 64, width <= 256, replicate-pad,
normalized as ``(x/255 - 0.5) / 0.5``.

Only ``torch`` (already a project dependency) and ``numpy`` are used here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Phase 8 constants (verbatim) ------------------------------------------
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # 36 symbols
BLANK = 0                                            # CTC blank index
NUM_CLASS = len(ALPHABET) + 1                        # 37
INPUT_CHANNEL = 1
OUTPUT_CHANNEL = 256
HIDDEN_SIZE = 256

# Preprocessing conventions (see app/pipeline/io.py).
HEIGHT = 64
WMAX = 256


# --------------------------------------------------------------------------
# Model - exact DTRB / EasyOCR VGG-BiLSTM-CTC architecture (checkpoint-exact)
# --------------------------------------------------------------------------
class VGGFeatureExtractor(nn.Module):
    """Replica of ``easyocr.model.modules.VGG_FeatureExtractor`` (layer indices
    and shapes must match the ``english_g2`` state_dict exactly)."""

    def __init__(self, input_channel: int, output_channel: int = 256):
        super().__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))

    def forward(self, x):
        return self.ConvNet(x)


class BidirectionalLSTM(nn.Module):
    """Replica of ``easyocr.model.modules.BidirectionalLSTM``."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        return self.linear(recurrent)


class CRNN(nn.Module):
    """CNN backbone + 2-layer BiLSTM + linear CTC head (37 classes)."""

    def __init__(self, num_class: int = NUM_CLASS, input_channel: int = INPUT_CHANNEL,
                 output_channel: int = OUTPUT_CHANNEL, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.FeatureExtraction = VGGFeatureExtractor(input_channel, output_channel)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(output_channel, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.Prediction = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        visual = self.FeatureExtraction(x)                       # (B,256,H',W')
        visual = self.AdaptiveAvgPool(
            visual.permute(0, 3, 1, 2)).squeeze(3)               # (B,W',256)
        contextual = self.SequenceModeling(visual)               # (B,W',256)
        return self.Prediction(contextual)                       # (B,W',num_class)


# --------------------------------------------------------------------------
# CNN width bookkeeping: VGG downsamples width to (w//2)//2 - 1
# --------------------------------------------------------------------------
def conv_out_width(w: int) -> int:
    w = w // 2      # MaxPool2d(2,2)
    w = w // 2      # MaxPool2d(2,2)
    w = max(1, w - 1)   # Conv2d(256,256,2,1,0)
    return w


def encode(text: str) -> list[int]:
    """Map an ``[A-Z0-9]`` string to CTC targets (blank occupies index 0)."""
    return [ALPHABET.index(ch) + 1 for ch in text]     # +1: blank occupies 0


def greedy_decode(logits: torch.Tensor) -> tuple[str, float]:
    """Greedy CTC decode.

    ``logits``: ``(T, num_class)``. Returns ``(text, mean_char_confidence)``.
    Confidence averages the max softmax probability at the first timestep of
    each emitted CTC character. Repeated frames therefore do not count as extra
    evidence. Returns ``0.0`` when nothing is emitted. This is an uncalibrated
    model confidence proxy, not a probability that the full plate is correct.
    """
    probs = F.softmax(logits, dim=1)
    idx = probs.argmax(dim=1).tolist()
    conf = probs.max(dim=1).values.tolist()
    out, emitted_conf = [], []
    prev = None
    for i, score in zip(idx, conf):
        if i != prev and i != BLANK:
            out.append(ALPHABET[i - 1])
            emitted_conf.append(score)
        prev = i
    mean_conf = float(np.mean(emitted_conf)) if emitted_conf else 0.0
    return "".join(out), mean_conf


# --------------------------------------------------------------------------
# Checkpoint loading
# --------------------------------------------------------------------------
def build_from_checkpoint(ckpt: dict) -> CRNN:
    """Construct a :class:`CRNN` from a loaded Phase 8 checkpoint dict."""
    model = CRNN(num_class=int(ckpt["num_class"]),
                 input_channel=int(ckpt.get("input_channel", INPUT_CHANNEL)))
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def load_crnn_checkpoint(path, device=None) -> dict:
    """Load a Phase 8 OCR checkpoint into eval mode.

    Returns a dict with ``model`` (``CRNN`` in eval mode, moved to ``device``),
    ``alphabet``, ``height``, ``wmax``, ``num_class``, ``input_channel``,
    ``ckpt`` (the raw checkpoint) and ``device`` (a ``torch.device``).
    """
    dev = torch.device(device) if device else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(Path(path)), map_location="cpu", weights_only=False)
    model = build_from_checkpoint(ckpt)
    model.to(dev).eval()
    return {
        "model": model,
        "alphabet": ckpt["alphabet"],
        "height": int(ckpt["height"]),
        "wmax": int(ckpt.get("wmax", WMAX)),
        "min_resized_w": int(ckpt.get("min_resized_w", 8)),
        "pad_to_max_width": bool(ckpt.get("pad_to_max_width", False)),
        "layout_mode": ckpt.get("layout_mode", "none"),
        "num_class": int(ckpt["num_class"]),
        "input_channel": int(ckpt.get("input_channel", INPUT_CHANNEL)),
        "ckpt": ckpt,
        "device": dev,
    }


__all__ = [
    "ALPHABET",
    "BLANK",
    "NUM_CLASS",
    "INPUT_CHANNEL",
    "OUTPUT_CHANNEL",
    "HIDDEN_SIZE",
    "HEIGHT",
    "WMAX",
    "VGGFeatureExtractor",
    "BidirectionalLSTM",
    "CRNN",
    "conv_out_width",
    "encode",
    "greedy_decode",
    "build_from_checkpoint",
    "load_crnn_checkpoint",
]
