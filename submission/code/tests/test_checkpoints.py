"""Checkpoint-integrity and class-order consistency tests. The heavy ones skip
cleanly when the real weights are not present (e.g. a fresh clone)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
import evaluate  # noqa: E402

SUBMISSION = CODE_DIR.parent
RESULTS = SUBMISSION / "results"
MANIFEST = RESULTS / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_class_orders_match_recorded_metadata():
    conv = json.loads((RESULTS / "convnextv2_tiny" / "metrics.json").read_text())
    assert conv["classes"] == evaluate.CLASSES_6
    swin = json.loads((RESULTS / "swin_b" / "metrics.json").read_text())
    assert set(swin["per_class_tta"]) == set(evaluate.CLASSES_10)


@pytest.mark.parametrize("key, builder, expected_params", [
    ("convnet", evaluate.build_convnet, 27_871_110),
    ("transformer", lambda: evaluate.SwinBClassifier(10), 86_753_474),
])
def test_checkpoint_loads_with_expected_param_count(key, builder, expected_params):
    ckpt = evaluate.CANONICAL_CKPT[key]
    if not ckpt.is_file():
        pytest.skip(f"checkpoint not present: {ckpt}")
    model = builder()
    evaluate.load_checkpoint(model, ckpt, key)   # exits(2) on any mismatch
    assert sum(p.numel() for p in model.parameters()) == expected_params


def test_checkpoint_hashes_match_manifest():
    models = json.loads(MANIFEST.read_text())["models"]
    checked = 0
    for name, record in models.items():
        ckpt = (MANIFEST.parent / record["checkpoint"]).resolve()
        if not ckpt.is_file():
            continue
        assert _sha256(ckpt) == record["sha256"], f"{name} hash mismatch"
        checked += 1
    if checked == 0:
        pytest.skip("no checkpoints present to hash")


def test_verify_artifacts_script_passes():
    models = json.loads(MANIFEST.read_text())["models"]
    for record in models.values():
        if not (MANIFEST.parent / record["checkpoint"]).resolve().is_file():
            pytest.skip("checkpoints absent; verify_artifacts would fail on hashes")
    result = subprocess.run([sys.executable, str(CODE_DIR / "verify_artifacts.py")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
