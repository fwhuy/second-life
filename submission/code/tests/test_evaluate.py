"""Behavioural tests for evaluate.py: the class-masking contract, deterministic
transforms, a CPU end-to-end run, and every documented failure mode."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
import numpy as np

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
import evaluate  # noqa: E402

EVAL = CODE_DIR / "evaluate.py"


def _run(*args):
    return subprocess.run([sys.executable, str(EVAL), *args],
                          capture_output=True, text=True)


def test_shared_indices_map_ten_classes_to_six():
    # The six shared columns pulled from the ten-class logit vector must line up
    # exactly with the ConvNeXt class order, or the comparison is meaningless.
    assert evaluate.SHARED_IN_10 == [2, 4, 5, 6, 7, 9]
    assert [evaluate.CLASSES_10[i] for i in evaluate.SHARED_IN_10] == evaluate.CLASSES_6


def test_eval_transforms_are_deterministic():
    img = Image.fromarray((np.random.rand(400, 300, 3) * 255).astype("uint8"))
    for transform, size in ((evaluate.convnet_transform(), 384),
                            (evaluate.swin_transform(), 224)):
        a, b = transform(img), transform(img)
        assert torch.equal(a, b)              # no randomness in the eval pipeline
        assert a.shape == (3, size, size)


def test_metrics_are_correct_on_a_known_case():
    # Two classes, one deliberate mistake: hand-check accuracy and the matrix.
    labels = np.array([0, 0, 1, 1])
    logits = torch.tensor([[3.0, 0.0], [3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])  # last is wrong
    m = evaluate.compute_metrics(labels, logits, ["a", "b"])
    assert m["accuracy"] == 0.75
    assert m["confusion"].tolist() == [[2, 0], [1, 1]]
    assert m["per_class"]["a"]["recall"] == 1.0
    assert m["per_class"]["b"]["recall"] == 0.5


def test_end_to_end_cpu_writes_all_artifacts(tiny_dataset, convnet_ckpt, tmp_path):
    out = tmp_path / "out"
    r = _run("--model", "convnet", "--checkpoint", str(convnet_ckpt),
             "--data-root", str(tiny_dataset), "--out-dir", str(out),
             "--device", "cpu", "--no-plots")
    assert r.returncode == 0, r.stderr
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["n_images"] == 12                       # 6 classes x 2 images
    assert metrics["classes"] == evaluate.CLASSES_6
    assert metrics["deterministic"] is True
    for name in ("predictions.csv", "confusion_matrix.json",
                 "errors_highest_loss.csv", "errors_confident_wrong.csv"):
        assert (out / name).is_file(), name


def test_missing_checkpoint_exits_2(tiny_dataset, tmp_path):
    r = _run("--model", "convnet", "--checkpoint", str(tmp_path / "nope.pt"),
             "--data-root", str(tiny_dataset), "--out-dir", str(tmp_path / "o"),
             "--device", "cpu")
    assert r.returncode == 2
    assert "checkpoint not found" in r.stderr


def test_wrong_class_count_exits_2(tiny_dataset, tmp_path):
    import timm
    wrong = timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k_384",
                              pretrained=False, num_classes=5)
    ckpt = tmp_path / "wrong.pt"
    torch.save(wrong.state_dict(), ckpt)
    r = _run("--model", "convnet", "--checkpoint", str(ckpt),
             "--data-root", str(tiny_dataset), "--out-dir", str(tmp_path / "o"),
             "--device", "cpu")
    assert r.returncode == 2
    assert "incompatible" in r.stderr


def test_empty_split_exits_2(convnet_ckpt, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    r = _run("--model", "convnet", "--checkpoint", str(convnet_ckpt),
             "--data-root", str(empty), "--out-dir", str(tmp_path / "o"),
             "--device", "cpu")
    assert r.returncode == 2


def test_manifest_with_bad_label_exits_2(tiny_dataset, convnet_ckpt, tmp_path):
    manifest = tmp_path / "m.csv"
    manifest.write_text("path,label\nsomewhere.jpg,not_a_class\n")
    r = _run("--model", "convnet", "--checkpoint", str(convnet_ckpt),
             "--split-manifest", str(manifest), "--out-dir", str(tmp_path / "o"),
             "--device", "cpu")
    assert r.returncode == 2
