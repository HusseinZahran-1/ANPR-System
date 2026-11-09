"""Small, language-model-free CTC prefix beam decoder for OCR experiments."""

from __future__ import annotations

import numpy as np
import torch


def beam_decode(logits: torch.Tensor, alphabet: str, *, width: int = 8,
                symbols_per_step: int = 12) -> str:
    """Decode ``(time, blank + characters)`` logits with blank index zero.

    This sums alignments for each prefix. ``symbols_per_step`` bounds compute;
    the blank symbol is always retained. No plate-format prior is applied.
    """
    if logits.ndim != 2 or logits.shape[1] != len(alphabet) + 1:
        raise ValueError("logits must have one blank and one column per character")
    if width < 1 or symbols_per_step < 1:
        raise ValueError("beam width and symbol cap must be positive")
    if logits.shape[0] == 0:
        return ""
    scores = torch.log_softmax(logits.float(), dim=1).cpu().numpy()
    negative_inf = float("-inf")
    beam: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, negative_inf)}

    def add(state, prefix, blank, nonblank):
        old_blank, old_nonblank = state.get(prefix, (negative_inf, negative_inf))
        state[prefix] = (float(np.logaddexp(old_blank, blank)),
                         float(np.logaddexp(old_nonblank, nonblank)))

    for row in scores:
        n = min(symbols_per_step, len(alphabet))
        candidates = np.argpartition(row[1:], -n)[-n:] + 1
        nxt: dict[tuple[int, ...], tuple[float, float]] = {}
        for prefix, (p_blank, p_nonblank) in beam.items():
            total = float(np.logaddexp(p_blank, p_nonblank))
            add(nxt, prefix, total + float(row[0]), negative_inf)
            for symbol in candidates:
                c = int(symbol)
                emission = float(row[c])
                if prefix and prefix[-1] == c:
                    # A repeated character without an intervening blank
                    # collapses to the same output prefix.
                    add(nxt, prefix, negative_inf, p_nonblank + emission)
                    add(nxt, prefix + (c,), negative_inf, p_blank + emission)
                else:
                    add(nxt, prefix + (c,), negative_inf, total + emission)
        beam = dict(sorted(nxt.items(),
                           key=lambda item: float(np.logaddexp(*item[1])),
                           reverse=True)[:width])
    winner = max(beam, key=lambda prefix: float(np.logaddexp(*beam[prefix])))
    return "".join(alphabet[index - 1] for index in winner)
