"""Tests for the split invariants ConvNeXt relies on. The training corpus is
rebuilt on the GPU host rather than committed, so these verify the *method*
(group-disjoint, reproducible) that keeps perceptual duplicates out of both
sides of a split — the property that makes the validation number trustworthy."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def _fold_zero(n=600, n_groups=200, seed=42):
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, 6, size=n)
    groups = rng.randint(0, n_groups, size=n)   # many images share a group id
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(splitter.split(np.zeros(n), labels, groups))
    return train_idx, val_idx, groups


def test_no_group_crosses_the_train_val_boundary():
    train_idx, val_idx, groups = _fold_zero()
    assert set(groups[train_idx]).isdisjoint(set(groups[val_idx]))
    assert set(train_idx).isdisjoint(set(val_idx))


def test_every_index_is_assigned_exactly_once():
    train_idx, val_idx, groups = _fold_zero()
    assert len(train_idx) + len(val_idx) == len(groups)
    assert set(train_idx) | set(val_idx) == set(range(len(groups)))


def test_split_is_reproducible_under_the_fixed_seed():
    a_train, a_val, _ = _fold_zero()
    b_train, b_val, _ = _fold_zero()
    assert np.array_equal(a_train, b_train)
    assert np.array_equal(a_val, b_val)
