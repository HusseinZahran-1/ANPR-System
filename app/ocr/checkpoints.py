"""Atomic checkpoint replacement, preserving the previous file on save failure."""
import os
from pathlib import Path
import tempfile

import torch


def atomic_torch_save(payload: dict, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + '.',
                                     suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
