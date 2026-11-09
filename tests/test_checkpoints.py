import pytest
import torch

from app.ocr.checkpoints import atomic_torch_save


def test_atomic_checkpoint_replaces_previous_version(tmp_path):
    path = tmp_path / 'weights' / 'last.pt'
    atomic_torch_save({'epoch': 1}, path)
    atomic_torch_save({'epoch': 2, 'val_cer': .15}, path)
    assert torch.load(path, weights_only=True) == {'epoch': 2, 'val_cer': .15}
    assert list(path.parent.iterdir()) == [path]


def test_failed_save_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'best.pt'
    atomic_torch_save({'epoch': 1}, path)
    before = path.read_bytes()

    def fail(payload, destination):
        destination.write_bytes(b'partial write')
        raise OSError('simulated disk failure')

    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError, match='disk failure'):
        atomic_torch_save({'epoch': 2}, path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
