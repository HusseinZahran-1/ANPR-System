"""Phase 11 - End-to-End pipeline: I/O and crop-preprocessing utilities.

Only ``stdlib + PIL + numpy`` are used here, plus ``torch`` **solely** to build
the returned tensor for :func:`preprocess_for_ocr` (no ``ultralytics`` /
``easyocr`` / model loading in this module).

The OCR preprocessing is written to match Phase 8's training/inference pipeline
(``scripts/train_phase8_ocr.py`` ``PlateCrops`` + ``collate``):

    grayscale (PIL ``convert("L")``, equivalently 0.299 R + 0.587 G + 0.114 B)
    -> resize to H=64 preserving aspect ratio (PIL BICUBIC)
    -> clamp width to <= 256
    -> replicate (border) right-pad to the batch width
    -> normalize (x/255 - 0.5) / 0.5

New checkpoints store a fixed, label-independent minimum width and inference
reads that value from the checkpoint. Older Phase 8 checkpoints did not store
the bound and retain the legacy inference floor of 8 pixels.

Functions
---------
load_scene(image_path) -> np.ndarray
    Read an image as an RGB ``uint8`` array of shape ``(H, W, 3)``.
crop_from_box(scene_rgb, box_xyxy, pad_pct=0.0) -> np.ndarray
    Crop ``[x1, y1, x2, y2]`` (clipped to image bounds) from the scene, with an
    optional relative padding, returning an RGB ``uint8`` array.
preprocess_for_ocr(crop_rgb, target_h=64, max_w=256,
                   min_resized_w=8) -> torch.Tensor
    Convert a crop to the Phase 8 input tensor of shape ``(1, 1, target_h, W)``
    float32.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageOps

# Phase 8 conventions (see models/phase8_ocr/easyocr_crnn/training_config.json).
DEFAULT_TARGET_H = 64
DEFAULT_MAX_W = 256
# Inference-time minimal resized width floor. Training used
# ``max(rw, 4 * (len(gt_text) + 4))``; the ground-truth text is unknown at
# inference, so we use the smallest floor that still guarantees the VGG
# width-downsampler emits at least one timestep: conv_out_width(8) = 1.
MIN_RESIZED_W = 8


def load_scene(image_path) -> np.ndarray:
    """Load an image as RGB ``uint8``, shape ``(H, W, 3)``.

    Apply EXIF orientation before converting so crop coordinates match the
    detector's OpenCV/Ultralytics image loading. Any mode (L / P / RGBA / CMYK
    ...) is converted to RGB. The array is a fresh copy (writable) so
    downstream drawing/cropping never mutates a memory map.
    """
    with Image.open(image_path) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(rgb, dtype=np.uint8).copy()


def crop_from_box(scene_rgb: np.ndarray, box_xyxy, pad_pct: float = 0.0) -> np.ndarray:
    """Crop an RGB scene to ``[x1, y1, x2, y2]`` (pixel ints, clipped).

    ``box_xyxy`` may be int or float; it is rounded to int and clipped to
    ``[0, W] x [0, H]``. ``pad_pct`` (default ``0.0``) expands each side by
    ``pad_pct`` of the box's own width/height before clipping. Returns an RGB
    ``uint8`` array. Raises ``ValueError`` if the clipped box is empty.
    """
    h, w = scene_rgb.shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in box_xyxy)
    if pad_pct:
        bw, bh = (x2 - x1), (y2 - y1)
        px, py = int(round(bw * float(pad_pct))), int(round(bh * float(pad_pct)))
        x1, y1, x2, y2 = x1 - px, y1 - py, x2 + px, y2 + py
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty crop after clipping: {(x1, y1, x2, y2)} "
                         f"for scene {w}x{h}")
    return scene_rgb[y1:y2, x1:x2].copy()


def replicate_pad_right(t: torch.Tensor, target_w: int) -> torch.Tensor:
    """Replicate (border) right-pad a ``(1, 1, H, W)`` tensor to ``target_w``.

    Mirrors Phase 8 ``collate``: the last column is repeated. No-op when
    ``t.shape[-1] >= target_w``.
    """
    cur = t.shape[-1]
    if cur >= target_w:
        return t
    pad = t[:, :, :, -1:].expand(t.shape[0], t.shape[1], t.shape[2], target_w - cur)
    return torch.cat([t, pad], dim=3)


def preprocess_for_ocr(crop_rgb: np.ndarray, target_h: int = DEFAULT_TARGET_H,
                       max_w: int = DEFAULT_MAX_W,
                       min_resized_w: int = MIN_RESIZED_W,
                       pad_to_max_width: bool = False) -> torch.Tensor:
    """Convert an RGB crop to a Phase 8 input tensor ``(1, 1, target_h, W)``.

    Steps (matching Phase 8 training; see module docstring):

    1. grayscale via PIL ``convert("L")`` == ``0.299 R + 0.587 G + 0.114 B``;
    2. resize to ``(W', target_h)`` with ``W' = round(target_h * w / h)``,
       preserving aspect ratio, PIL BICUBIC;
    3. clamp ``W'`` to ``[min_resized_w, max_w]``. Both bounds are checkpoint
       metadata so training and inference use the same label-independent width;
    4. normalize ``(x/255 - 0.5) / 0.5``; optionally replicate-pad to ``max_w``
       when the checkpoint was trained on a fixed-width input.

    New fixed-width checkpoints set ``pad_to_max_width=True``. Legacy
    checkpoints keep their variable-width input behavior.
    """
    if crop_rgb.ndim != 3 or crop_rgb.shape[2] != 3:
        raise ValueError(f"expected RGB (H, W, 3), got shape {crop_rgb.shape}")
    h, w = crop_rgb.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"empty crop {w}x{h}")
    if min_resized_w < 1 or max_w < min_resized_w:
        raise ValueError(f"invalid width bounds: min={min_resized_w}, max={max_w}")

    # `mode=` is deprecated (Pillow 13) and is unnecessary: an (H, W, 3) uint8
    # array is inferred as RGB identically.
    gray = Image.fromarray(crop_rgb).convert("L")
    rw = int(round(target_h * w / float(h)))
    rw = max(min_resized_w, min(rw, max_w))
    gray = gray.resize((rw, target_h), Image.BICUBIC)

    a = np.asarray(gray, dtype=np.float32) / 255.0        # (H, W')
    t = torch.from_numpy(a).unsqueeze(0).unsqueeze(0)     # (1, 1, H, W')
    t = (t - 0.5) / 0.5                                   # Phase 8 NormalizePAD
    if pad_to_max_width:
        t = replicate_pad_right(t, max_w)
    return t.contiguous()


__all__ = [
    "load_scene",
    "crop_from_box",
    "preprocess_for_ocr",
    "replicate_pad_right",
    "DEFAULT_TARGET_H",
    "DEFAULT_MAX_W",
    "MIN_RESIZED_W",
]
