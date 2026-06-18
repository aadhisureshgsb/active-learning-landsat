"""Unit tests for query strategies (synthetic data, no network)."""

import numpy as np

from active_learning.strategies import STRATEGIES


def _probs():
    # row 0: very confident; row 1: near-tie (ambiguous); row 2: mild.
    return np.array(
        [
            [0.90, 0.05, 0.05],
            [0.40, 0.38, 0.22],
            [0.60, 0.30, 0.10],
        ]
    )


def test_registry_populated():
    for name in ("random", "margin", "entropy", "least_confidence"):
        assert name in STRATEGIES


def test_each_strategy_returns_k_unique_indices():
    rng = np.random.default_rng(0)
    p = _probs()
    for fn in STRATEGIES.values():
        idx = fn(p, rng, 2)
        assert len(idx) == 2
        assert len(set(np.asarray(idx).tolist())) == 2
        assert all(0 <= i < len(p) for i in idx)


def test_margin_picks_the_ambiguous_row_first():
    rng = np.random.default_rng(0)
    idx = STRATEGIES["margin"](_probs(), rng, 1)
    assert idx[0] == 1  # the near-tie row is the most ambiguous


def test_least_confidence_picks_the_ambiguous_row_first():
    rng = np.random.default_rng(0)
    idx = STRATEGIES["least_confidence"](_probs(), rng, 1)
    assert idx[0] == 1  # lowest max-probability
