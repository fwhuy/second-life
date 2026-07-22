"""Unified-corpus guarantees: provenance recovery and spent-test quarantine.

The unified corpus contains the spent 361-image test set under disguised filenames,
so these tests pin the machinery that keeps it out of training. Synthetic frames
only — no corpus, no GPU: pytest tests/
"""

import pandas as pd
import pytest

from src.model import build_model
from src.unified_data import _split_indices, conflicting_label_paths
from src.utils import assert_old_test_quarantined, load_split_frames


# ---------------------------------------------------------------------------
# Provenance: the zip-order reconstruction that identifies disguised files
# ---------------------------------------------------------------------------

def test_zip_member_order_is_lexicographic_not_numeric():
    """source_index came from `sorted(names)`, so trash10 precedes trash2."""
    from scripts.map_trashnet_provenance import zip_member_order

    groups = pd.DataFrame({
        "path": ["data/raw/trash/trash2.jpg", "data/raw/trash/trash10.jpg",
                 "data/raw/glass/glass1.jpg"],
        "label": ["trash", "trash", "glass"],
    })
    order = zip_member_order(groups)
    assert list(order["member"]) == [
        "dataset-resized/glass/glass1.jpg",
        "dataset-resized/trash/trash10.jpg",
        "dataset-resized/trash/trash2.jpg",
    ]
    assert list(order["source_index"]) == [0, 1, 2]


def test_conflicting_labels_flag_every_copy():
    """Identical pixels under two labels: we cannot tell which is right, so both go."""
    manifest = pd.DataFrame({
        "included": [1, 1, 1, 1],
        "status": ["stored", "stored", "stored", "duplicate"],
        "pixel_sha256": ["aaa", "aaa", "bbb", "ccc"],
        "mapped_label": ["glass", "plastic", "paper", "metal"],
        "relative_path": ["p/glass.jpg", "p/plastic.jpg", "p/paper.jpg", "p/dup.jpg"],
    })
    assert conflicting_label_paths(manifest) == {"p/glass.jpg", "p/plastic.jpg"}


# ---------------------------------------------------------------------------
# Quarantine assertion
# ---------------------------------------------------------------------------

def make_split(test_paths, fold_paths):
    test = pd.DataFrame({"path": test_paths, "label": "glass", "group": range(len(test_paths))})
    folds = pd.DataFrame({"path": fold_paths, "label": "glass",
                          "group": range(100, 100 + len(fold_paths)), "fold": 0})
    return folds, test


def test_quarantined_split_passes():
    folds, test = make_split(["t1.jpg", "t2.jpg"], ["f1.jpg"])
    assert_old_test_quarantined(folds, test, ["t1.jpg", "t2.jpg"])


def test_spent_test_image_in_training_is_caught():
    """The disguised-filename failure mode: a test image sitting in a train fold."""
    folds, test = make_split(["t1.jpg"], ["f1.jpg", "t2.jpg"])
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_old_test_quarantined(folds, test, ["t1.jpg", "t2.jpg"])


def test_spent_test_image_silently_dropped_is_caught():
    """Quarantined images must be *held* in test, not quietly filtered away."""
    folds, test = make_split(["t1.jpg"], ["f1.jpg"])
    with pytest.raises(AssertionError, match="QUARANTINE BREACH"):
        assert_old_test_quarantined(folds, test, ["t1.jpg", "t2.jpg"])


def test_load_split_frames_enforces_quarantine_file(tmp_path):
    """Every entry point inherits the guarantee, not just the split builder."""
    folds, test = make_split(["t1.jpg"], ["f1.jpg", "t2.jpg"])
    folds.to_csv(tmp_path / "folds.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"path": ["t1.jpg", "t2.jpg"]}).to_csv(tmp_path / "quarantine.csv", index=False)
    with pytest.raises(AssertionError, match="LEAKAGE"):
        load_split_frames(tmp_path)


def test_load_split_frames_without_quarantine_file_is_unaffected(tmp_path):
    """The original TrashNet splits have no quarantine.csv and must still load."""
    folds, test = make_split(["t1.jpg"], ["f1.jpg"])
    folds.to_csv(tmp_path / "folds.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    loaded_folds, loaded_test = load_split_frames(tmp_path)
    assert list(loaded_folds["path"]) == ["f1.jpg"]
    assert list(loaded_test["path"]) == ["t1.jpg"]


# ---------------------------------------------------------------------------
# Stratification + the parameter cap
# ---------------------------------------------------------------------------

def test_a_rare_stratum_does_not_break_split_construction():
    """label x source keeps all domains in every fold; a 2-member stratum must not
    stop 5 folds from being produced (sklearn warns here rather than raising, so
    the label-only fallback is a safety net for the cases where it does raise)."""
    frame = pd.DataFrame({
        "label": ["glass"] * 20,
        "stratum": ["glass|rare"] * 2 + ["glass|common"] * 18,
        "group": range(20),
    })
    folds = _split_indices(frame, n_splits=5, seed=42)
    assert len(folds) == 5


# ---------------------------------------------------------------------------
# One-shot test-set guard
# ---------------------------------------------------------------------------

def test_spent_guard_is_keyed_per_split_set():
    """Spending the old TrashNet test must not block the unified corpus."""
    from src.utils import test_spent_path

    assert test_spent_path("data/splits").name == "TEST_SPENT.json"
    assert test_spent_path("data/splits_unified").name == "TEST_SPENT_splits_unified.json"


def test_second_final_eval_is_refused(tmp_path, monkeypatch):
    from src import utils

    monkeypatch.setattr(utils, "REPO_ROOT", tmp_path)
    utils.assert_test_not_spent("data/splits_unified")  # fresh: allowed
    utils.mark_test_spent("data/splits_unified", checkpoint="ck.pth",
                          metrics={"acc": 0.94, "macro_f1": 0.93}, n_images=1739)
    with pytest.raises(AssertionError, match="ALREADY SPENT"):
        utils.assert_test_not_spent("data/splits_unified")
    # a different split set is unaffected
    utils.assert_test_not_spent("data/splits")


def test_param_cap_rejects_an_oversized_backbone():
    with pytest.raises(ValueError, match="over the 30M competition cap"):
        build_model({"model": "convnext_base.fb_in22k_ft_in1k"}, pretrained=False)


def test_param_cap_can_be_disabled_explicitly():
    model = build_model({"model": "resnet18.a1_in1k", "max_params": 0}, pretrained=False)
    assert model is not None
