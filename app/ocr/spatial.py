"""Approximate horizontal CTC emission positions for this project's VGG CRNN."""
from __future__ import annotations
import torch
from app.ocr.ocr_engine import conv_out_width


def vgg_ctc_tokens(logits: torch.Tensor, alphabet: str, *,
                   source_width: int, resized_width: int) -> list[dict]:
    """Collapse CTC runs and locate each run's peak on the input x axis.

    The VGG encoder has horizontal stride four and first receptive-field center
    at input coordinate four. The contextual LSTM may shift alignments, so these
    are approximate emission centers, not character bounding boxes. Caller must
    exclude padded timesteps and supply the width before right padding.
    """
    if (logits.ndim != 2 or logits.shape[1] != len(alphabet)+1
            or len(set(alphabet)) != len(alphabet)):
        raise ValueError('expected time by blank-plus-alphabet logits')
    if source_width <= 0 or resized_width < 8 or logits.shape[0] > conv_out_width(resized_width):
        raise ValueError('invalid widths or padded timesteps')
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('non-finite CTC logits')
    probabilities=logits.detach().float().softmax(dim=1).cpu().numpy()
    indices=probabilities.argmax(axis=1)
    tokens=[]
    start=0
    while start < len(indices):
        end=start+1
        while end < len(indices) and indices[end] == indices[start]:
            end+=1
        symbol=int(indices[start])
        if symbol:
            peak=start+int(probabilities[start:end,symbol].argmax())
            tokens.append({'character':alphabet[symbol-1],'time_index':peak,
                           'source_x_center':4*(peak+1)*source_width/resized_width,
                           'confidence':float(probabilities[peak,symbol])})
        start=end
    return tokens
