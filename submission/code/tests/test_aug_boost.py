"""Per-class augmentation boost: applies to the named classes and nobody else.

Runs on synthetic in-memory images — no data dir or GPU needed: pytest tests/
"""

import pandas as pd
import pytest
import torch

from src.data import TrashDataset, build_eval_transform, build_train_transform

CFG = {"img_size": 64, "aug": "basic", "random_erasing": 0.0}


class Recorder:
    """Stand-in transform that records which samples routed through it."""

    def __init__(self):
        self.calls = 0

    def __call__(self, img):
        self.calls += 1
        return torch.zeros(3, 8, 8)


def make_frame():
    return pd.DataFrame({
        "path": [f"data/raw/x/{i}.jpg" for i in range(4)],
        "label": ["trash", "paper", "trash", "glass"],
    })


def patched_dataset(monkeypatch, **kwargs):
    """TrashDataset over fake paths — stub out the actual image open."""
    from src import data as data_mod

    class FakeImage:
        def convert(self, _mode):
            return self

    monkeypatch.setattr(data_mod.Image, "open", lambda _p: FakeImage())
    return TrashDataset(make_frame(), **kwargs)


def test_boost_routes_only_named_classes(monkeypatch):
    base, boost = Recorder(), Recorder()
    ds = patched_dataset(monkeypatch, transform=base, boost_transform=boost,
                         boost_classes=["trash"])
    for i in range(len(ds)):
        ds[i]
    assert boost.calls == 2, "both trash images should take the boost transform"
    assert base.calls == 2, "paper and glass should take the normal transform"


def test_no_boost_classes_means_no_branch(monkeypatch):
    base, boost = Recorder(), Recorder()
    ds = patched_dataset(monkeypatch, transform=base, boost_transform=boost,
                         boost_classes=[])
    for i in range(len(ds)):
        ds[i]
    assert boost.calls == 0
    assert base.calls == 4


def test_eval_dataset_cannot_boost(monkeypatch):
    """An eval dataset is built without a boost transform — no augmentation path."""
    ds = patched_dataset(monkeypatch, transform=Recorder())
    assert ds.boost is None


def test_unknown_boost_class_is_rejected():
    from src.data import build_loaders

    frame = make_frame()
    with pytest.raises(ValueError, match="unknown classes"):
        build_loaders({**CFG, "batch_size": 2, "num_workers": 0,
                       "aug_boost_classes": ["rubbish"]}, frame, frame)


def test_boost_transform_is_strictly_harder():
    """Same cfg, boost on: wider crop scale and more ops than the normal one."""
    normal = build_train_transform(CFG)
    boosted = build_train_transform(CFG, boost=True)
    assert len(boosted.transforms) > len(normal.transforms)
    assert boosted.transforms[0].scale[0] < normal.transforms[0].scale[0]
    # random_erasing is 0 in cfg but the boost floor turns it on
    assert any(t.__class__.__name__ == "RandomErasing" for t in boosted.transforms)
    assert not any(t.__class__.__name__ == "RandomErasing" for t in normal.transforms)


def test_eval_transform_has_no_random_ops():
    names = [t.__class__.__name__ for t in build_eval_transform(64).transforms]
    assert not [n for n in names if n.startswith("Random")]
