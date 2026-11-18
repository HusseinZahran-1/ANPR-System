"""Optional supervised paint evidence from PaddleClas vehicle attributes."""

from pathlib import Path

import cv2
import numpy as np

COLORS = ("Yellow", "Orange", "Green", "Gray", "Red", "Blue", "White", "Gold", "Brown", "Black")
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PaintColor:
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

    def classify(self, crop: np.ndarray, inset: np.ndarray) -> str | None:
        inputs = []
        for view in (crop, inset):
            rgb = cv2.resize(view, (256, 192)).astype(np.float32) / 255
            inputs.append(((rgb - _MEAN) / _STD).transpose(2, 0, 1))
        scores = self.session.run(None, {self.input_name: np.stack(inputs)})[0]
        return _supported_color(scores[:, :len(COLORS)])


def _supported_color(scores: np.ndarray) -> str | None:
    ranked = scores.argsort(axis=1)
    selected = ranked[:, -1]
    if selected[0] != selected[1]:
        return None
    rows = np.arange(len(scores))
    votes = scores[rows, selected]
    runners = scores[rows, ranked[:, -2]]
    if votes.min() < 0.90 or (votes - runners).min() < 0.35:
        return None
    return COLORS[selected[0]]
