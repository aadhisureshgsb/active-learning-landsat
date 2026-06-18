"""Pool-based query strategies.

Each strategy has signature ``(probs, rng, k) -> np.ndarray`` and returns the
indices (into the pool) of the ``k`` items to label next.

    probs : (n_pool, n_classes) predicted class probabilities for the pool
    rng   : numpy Generator (used by stochastic strategies)
    k     : number of items to acquire this round
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

QueryFn = Callable[[np.ndarray, np.random.Generator, int], np.ndarray]

STRATEGIES: Dict[str, QueryFn] = {}


def register_strategy(name: str) -> Callable[[QueryFn], QueryFn]:
    """Decorator to add a strategy to the global registry."""

    def _wrap(fn: QueryFn) -> QueryFn:
        STRATEGIES[name] = fn
        return fn

    return _wrap


@register_strategy("random")
def random_sampling(probs: np.ndarray, rng: np.random.Generator, k: int) -> np.ndarray:
    """Baseline: sample ``k`` pool items uniformly at random."""
    return rng.choice(len(probs), size=k, replace=False)


@register_strategy("margin")
def margin_sampling(probs: np.ndarray, rng: np.random.Generator, k: int) -> np.ndarray:
    """Smallest gap between the top-two class probabilities = most ambiguous."""
    top2 = np.sort(probs, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    return np.argsort(margin)[:k]


@register_strategy("entropy")
def entropy_sampling(probs: np.ndarray, rng: np.random.Generator, k: int) -> np.ndarray:
    """Highest predictive entropy = most uncertain overall.

    Included deliberately as a cautionary baseline: maximizing raw uncertainty
    chases ambiguous outliers and ignores representativeness, and on this
    benchmark it underperforms even random sampling.
    """
    ent = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    return np.argsort(ent)[::-1][:k]


@register_strategy("least_confidence")
def least_confidence(probs: np.ndarray, rng: np.random.Generator, k: int) -> np.ndarray:
    """Lowest top-class probability = least confident prediction."""
    conf = probs.max(axis=1)
    return np.argsort(conf)[:k]
