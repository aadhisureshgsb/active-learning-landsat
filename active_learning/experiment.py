"""The active-learning loop and the multi-seed experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from active_learning.strategies import STRATEGIES, QueryFn


def default_classifier() -> ClassifierMixin:
    """A fast, probabilistic classifier suitable for uncertainty queries."""
    return LogisticRegression(max_iter=2000, C=1.0)


@dataclass
class Curve:
    """Mean +/- std test-accuracy learning curve over seeds."""

    n_labels: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def run_once(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    query_fn: QueryFn,
    seed: int,
    seed_per_class: int = 2,
    batch: int = 10,
    rounds: int = 38,
    classifier: Optional[ClassifierMixin] = None,
):
    """Run one active-learning trajectory; return ``(n_labels, accuracies)``."""
    rng = np.random.default_rng(seed)
    base_clf = classifier if classifier is not None else default_classifier()

    # Stratified warm-start so every class is present from round 0.
    labeled: list[int] = []
    for c in np.unique(y_pool):
        idx = np.where(y_pool == c)[0]
        labeled.extend(rng.choice(idx, size=seed_per_class, replace=False))
    pool = list(set(range(len(y_pool))) - set(labeled))

    n_labels, accs = [], []
    for _ in range(rounds + 1):
        clf = clone(base_clf)
        clf.fit(X_pool[labeled], y_pool[labeled])
        accs.append(accuracy_score(y_test, clf.predict(X_test)))
        n_labels.append(len(labeled))
        if not pool:
            break
        probs = clf.predict_proba(X_pool[pool])
        k = min(batch, len(pool))
        chosen = [pool[i] for i in query_fn(probs, rng, k)]
        labeled.extend(chosen)
        pool = list(set(pool) - set(chosen))

    return np.asarray(n_labels), np.asarray(accs)


def run_experiment(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    strategies=None,
    n_seeds: int = 12,
    **run_kwargs,
) -> Dict[str, Curve]:
    """Run every strategy over ``n_seeds`` seeds; return name -> Curve."""
    strategies = strategies or {
        "Random (baseline)": STRATEGIES["random"],
        "Uncertainty: margin": STRATEGIES["margin"],
        "Uncertainty: entropy": STRATEGIES["entropy"],
    }

    results: Dict[str, Curve] = {}
    for name, fn in strategies.items():
        runs = [
            run_once(X_pool, y_pool, X_test, y_test, fn, seed=s, **run_kwargs)
            for s in range(n_seeds)
        ]
        n_labels = runs[0][0]
        stack = np.vstack([r[1] for r in runs])
        results[name] = Curve(n_labels, stack.mean(0), stack.std(0))
    return results


def labels_to_target(curve: Curve, target_acc: float) -> Optional[int]:
    """Fewest labels at which the mean curve first reaches ``target_acc``."""
    hit = np.where(curve.mean >= target_acc)[0]
    return int(curve.n_labels[hit[0]]) if len(hit) else None
