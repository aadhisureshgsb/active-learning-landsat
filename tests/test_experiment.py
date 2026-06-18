"""Integration test for the active-learning loop (synthetic, no network)."""

import numpy as np
from sklearn.datasets import make_classification

from active_learning.experiment import run_once
from active_learning.strategies import STRATEGIES


def _toy():
    X, y = make_classification(
        n_samples=800, n_features=12, n_informative=8,
        n_classes=4, n_clusters_per_class=1, random_state=0,
    )
    return X[:600], y[:600] + 1, X[600:], y[600:] + 1


def test_loop_runs_and_improves_accuracy():
    X_pool, y_pool, X_test, y_test = _toy()
    n_labels, accs = run_once(
        X_pool, y_pool, X_test, y_test,
        STRATEGIES["margin"], seed=0,
        seed_per_class=2, batch=10, rounds=15,
    )
    assert len(n_labels) == len(accs)
    assert np.all(np.diff(n_labels) > 0)          # label count grows
    assert accs[-1] >= accs[0]                     # accuracy does not regress
    assert 0.0 <= accs.min() and accs.max() <= 1.0
