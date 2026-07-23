"""Shared fixtures. Puts submission/code on sys.path so `import evaluate` works,
and builds a throwaway ConvNeXt checkpoint / tiny image set once per session."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))


@pytest.fixture(scope="session")
def convnet_ckpt(tmp_path_factory):
    """A randomly-initialised 6-class ConvNeXt state dict — a valid checkpoint
    for exercising the load + eval pipeline without the real 106 MB weights."""
    import timm
    import torch

    model = timm.create_model(
        "convnextv2_tiny.fcmae_ft_in22k_in1k_384", pretrained=False, num_classes=6
    )
    path = tmp_path_factory.mktemp("ckpt") / "convnet.pt"
    torch.save(model.state_dict(), path)
    return path


@pytest.fixture
def tiny_dataset(tmp_path):
    """A folder of class sub-directories with a couple of random JPEGs each."""
    import evaluate

    root = tmp_path / "data"
    for cls in evaluate.CLASSES_6:
        cdir = root / cls
        cdir.mkdir(parents=True)
        for i in range(2):
            pixels = (np.random.rand(48, 48, 3) * 255).astype("uint8")
            Image.fromarray(pixels).save(cdir / f"{cls}_{i}.jpg")
    return root
