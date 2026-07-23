"""Validate the current submission's metrics and canonical checkpoint files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SUBMISSION = HERE.parent
RESULTS = SUBMISSION / "results"
EXPECTED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = load_json(RESULTS / "manifest.json")
    models = manifest["models"]

    conv = load_json(RESULTS / models["convnextv2_tiny"]["metrics"])
    assert conv["model"] == models["convnextv2_tiny"]["architecture"]
    assert conv["classes"] == EXPECTED_CLASSES
    assert conv["image_size"] == 384
    assert conv["parameter_count"] == 27_871_110
    assert abs(conv["validation_accuracy"] - 0.9829605963791267) < 1e-12

    swin = load_json(RESULTS / models["swin_b"]["metrics"])
    assert swin["swin_b"]["params"] == 86_753_474
    assert abs(swin["swin_b"]["test_acc"] - 0.9760739532354541) < 1e-12
    assert abs(swin["swin_b"]["tta_acc"] - 0.9782490483958673) < 1e-12
    assert set(swin["per_class_tta"]) == {
        "battery", "biological", "cardboard", "clothes", "glass",
        "metal", "paper", "plastic", "shoes", "trash",
    }

    for name, record in models.items():
        checkpoint = (RESULTS / record["checkpoint"]).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{name}: missing checkpoint {checkpoint}")
        actual = sha256(checkpoint)
        if actual != record["sha256"]:
            raise RuntimeError(
                f"{name}: checkpoint hash mismatch\n"
                f"expected {record['sha256']}\nactual   {actual}"
            )
        print(f"OK {name}: {checkpoint.name} ({actual[:12]}...)")

    print("OK metrics: model identities, class orders, scores, and parameters")


if __name__ == "__main__":
    main()
