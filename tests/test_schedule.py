"""Tests for the LR schedule.

The Stage E v2/v3/v3.1 sweeps were confounded because `lr_at` annealed over
`max_iters` while early stopping halted runs at wildly different fractions of
their cap — so two cells with the same best step but different caps got
different effective schedules (worth 0.144 bpc in one measured pair, larger
than the 8K architecture gap those sweeps reported). `lr_horizon` decouples
the anneal from the stopping backstop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch  # noqa: F401  (train imports torch; keep the dependency explicit)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train  # noqa: E402


def cfg(**kw) -> train.TrainConfig:
    base = dict(lr=3e-4, min_lr=3e-5, warmup_iters=100, max_iters=10000)
    base.update(kw)
    return train.TrainConfig(**base)


def test_horizon_defaults_to_max_iters():
    """Unset lr_horizon must reproduce the pre-v4 schedule exactly."""
    legacy = cfg(max_iters=7000)
    explicit = cfg(max_iters=7000, lr_horizon=7000)
    for step in (0, 50, 100, 1000, 3500, 7000, 9000):
        assert train.lr_at(step, legacy) == train.lr_at(step, explicit)


def test_horizon_is_independent_of_the_stopping_cap():
    """The whole point: same horizon, different caps -> identical LR at every
    step. This is the invariant the confounded sweeps violated."""
    a = cfg(max_iters=2000, lr_horizon=10000)
    b = cfg(max_iters=30000, lr_horizon=10000)
    for step in (0, 100, 500, 1999, 5000, 9999):
        assert train.lr_at(step, a) == train.lr_at(step, b)


def test_cap_alone_changes_the_schedule_without_a_horizon():
    """Guard the negative case, so the confound can't silently return."""
    a = cfg(max_iters=2500)
    b = cfg(max_iters=8000)
    assert train.lr_at(2000, a) != train.lr_at(2000, b)


def test_warmup_then_anneal_endpoints():
    c = cfg(lr_horizon=10000)
    assert train.lr_at(0, c) == pytest.approx(3e-4 / 100)
    assert train.lr_at(99, c) == pytest.approx(3e-4)
    assert train.lr_at(10000, c) == pytest.approx(3e-5)
    # Cosine midpoint sits at the mean of lr and min_lr.
    assert train.lr_at(5050, c) == pytest.approx((3e-4 + 3e-5) / 2, rel=1e-6)


def test_lr_never_leaves_bounds_past_the_horizon():
    c = cfg(lr_horizon=1000)
    for step in (1000, 5000, 100000):
        assert train.lr_at(step, c) == pytest.approx(3e-5)


def test_wsd_is_flat_then_decays():
    c = cfg(lr_horizon=10000, lr_schedule="wsd", wsd_decay_frac=0.2)
    # Stable phase holds peak LR.
    for step in (100, 3000, 7000, 8000):
        assert train.lr_at(step, c) == pytest.approx(3e-4)
    # Decay phase is linear from lr to min_lr over the last 20%.
    # progress is measured from the end of warmup, so the decay midpoint
    # (progress = 0.9) is step 100 + 0.9 * 9900 = 9010, not 9000.
    assert train.lr_at(9010, c) == pytest.approx((3e-4 + 3e-5) / 2, rel=1e-6)
    assert train.lr_at(10000, c) == pytest.approx(3e-5)


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError):
        train.lr_at(500, cfg(lr_schedule="triangular"))
