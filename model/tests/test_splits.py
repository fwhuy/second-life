"""Unit tests for the leakage assertions and group-aware split logic.

These run on synthetic frames — no images or GPU needed: pytest tests/
"""

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    assert_folds_group_disjoint,
    assert_no_leakage,
    train_val_from_folds,
)


def make_frame(n=60, n_groups=20, seed=0):
    rng = np.random.RandomState(seed)
    groups = rng.randint(0, n_groups, size=n)
    labels = np.array(["glass", "paper", "trash"])[groups % 3]
    return pd.DataFrame({
        "path": [f"data/raw/x/{i}.jpg" for i in range(n)],
        "label": labels,
        "group": groups,
    })


def split_by_group(df, test_groups):
    test = df[df["group"].isin(test_groups)].reset_index(drop=True)
    folds = df[~df["group"].isin(test_groups)].reset_index(drop=True)
    folds["fold"] = folds["group"] % 5
    return folds, test


def test_clean_split_passes():
    folds, test = split_by_group(make_frame(), test_groups={0, 1, 2})
    assert_no_leakage(folds, test)
    assert_folds_group_disjoint(folds)


def test_shared_image_is_caught():
    folds, test = split_by_group(make_frame(), test_groups={0, 1, 2})
    leaked = pd.concat([folds, test.head(1)], ignore_index=True)
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_leakage(leaked, test)


def test_shared_group_is_caught():
    """Different image files, same near-duplicate group → still leakage."""
    df = make_frame()
    folds, test = split_by_group(df, test_groups={0, 1, 2})
    # planted: a brand-new image whose group is quarantined in test
    plant = pd.DataFrame([{"path": "data/raw/x/new.jpg", "label": "glass",
                           "group": 0, "fold": 1}])
    with pytest.raises(AssertionError, match="group"):
        assert_no_leakage(pd.concat([folds, plant], ignore_index=True), test)


def test_group_spanning_folds_is_caught():
    folds, _ = split_by_group(make_frame(), test_groups={0})
    folds.loc[folds.index[:2], "fold"] = [0, 1]
    folds.loc[folds.index[:2], "group"] = 99
    with pytest.raises(AssertionError, match="folds"):
        assert_folds_group_disjoint(folds)


def test_train_val_split_group_disjoint():
    folds, _ = split_by_group(make_frame(), test_groups={0, 1})
    train, val = train_val_from_folds(folds, val_fold=0)
    assert len(train) + len(val) == len(folds)
    assert set(train["group"]).isdisjoint(set(val["group"]))
    assert (val["fold"] == 0).all()
