"""Optional supervised paint and body evidence from PaddleClas vehicle attributes."""

from pathlib import Path

import cv2
import numpy as np

COLORS = ("Yellow", "Orange", "Green", "Gray", "Red", "Blue", "White", "Gold", "Brown", "Black")
CATEGORIES = ("Sedan", "SUV", "Van", "Hatchback", "MPV", "Pickup", "Bus", "Truck", "Wagon")
_ALLOWED = {
    "car": {"Sedan", "SUV", "Van", "Hatchback", "MPV", "Pickup", "Wagon"},
    "truck": {"Van", "Pickup", "Truck"},
}
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PaintAndBody:
    def __init__(self, checkpoint: Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(checkpoint), sess_options=options, providers=["CPUExecutionProvider"]
        )
        model_input = self.session.get_inputs()[0]
        if model_input.shape[-3:] != [3, 192, 256] or self.session.get_outputs()[0].shape[-1] != 19:
            raise ValueError("expected the PaddleClas vehicle attribute architecture")
        self.input_name = model_input.name

    def classify(self, crop: np.ndarray, inset: np.ndarray, detection_class: str) -> tuple[str | None, str | None]:
        allowed = _ALLOWED.get(detection_class)
        if not allowed:
            return None, None
        inputs = []
        for view in (crop, inset):
            rgb = cv2.resize(view, (256, 192)).astype(np.float32) / 255
            inputs.append(((rgb - _MEAN) / _STD).transpose(2, 0, 1))
        scores = self.session.run(None, {self.input_name: np.stack(inputs)})[0]
        colors, categories = scores[:, :len(COLORS)], scores[:, len(COLORS):]
        color = _supported_label(colors, COLORS, set(COLORS), minimum=0.90, margin=0.35)
        category = _supported_label(categories, CATEGORIES, allowed, minimum=0.70, margin=0.15)
        return color, category


def _supported_label(scores: np.ndarray, labels: tuple[str, ...], allowed: set[str],
                     *, minimum: float, margin: float) -> str | None:
    candidates = [i for i, label in enumerate(labels) if label in allowed]
    ranked = scores[:, candidates].argsort(axis=1)
    first = [candidates[index] for index in ranked[:, -1]]
    if first[0] != first[1]:
        return None
    winner = first[0]
    votes = scores[:, winner]
    runners = scores[np.arange(len(scores)), [candidates[index] for index in ranked[:, -2]]]
    if (votes.min() < minimum or (votes - runners).min() < margin
            or (scores.max(axis=1) - votes).max() > 0.10):
        return None
    return labels[winner]
