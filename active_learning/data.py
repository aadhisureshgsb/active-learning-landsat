"""Loading the Statlog (Landsat Satellite) benchmark.

Data: Srinivasan, A. (1993). Statlog (Landsat Satellite). UCI Machine Learning
Repository. https://doi.org/10.24432/C55887  (CC BY 4.0)

36 features = 4 Landsat MSS spectral bands x a 3x3 pixel neighbourhood.
6 surface classes (class label 6, "mixture", is absent from the dataset).
6,435 samples: 4,435 train (used here as the unlabeled-able pool) + 2,000 test.
"""

from __future__ import annotations

import os
import urllib.request
from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

# Mirror of the UCI files (the canonical UCI endpoint is not always reachable
# programmatically). Files are tiny ASCII and licensed CC BY 4.0.
_BASE = (
    "https://raw.githubusercontent.com/"
    "sanskriti23/Statlog-Landsat-Satellite-Dataset/main"
)
_FILES = ("sat.trn", "sat.tst")

CLASS_NAMES = {
    1: "red soil",
    2: "cotton crop",
    3: "grey soil",
    4: "damp grey soil",
    5: "soil w/ vegetation stubble",
    7: "very damp grey soil",
}


def ensure_data(data_dir: str = "data") -> None:
    """Download the benchmark files into ``data_dir`` if they are not present."""
    os.makedirs(data_dir, exist_ok=True)
    for name in _FILES:
        path = os.path.join(data_dir, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        url = f"{_BASE}/{name}"
        print(f"[data] downloading {name} ...")
        urllib.request.urlretrieve(url, path)


def load_landsat(
    data_dir: str = "data", standardize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X_pool, y_pool, X_test, y_test)``.

    ``X_pool`` is the training split used as the pool of unlabeled-able tiles;
    ``X_test`` is the fixed 2,000-tile held-out evaluation set. When
    ``standardize`` is True the scaler is fit on the pool *features* only —
    legitimate, since features are available without labels.
    """
    ensure_data(data_dir)
    tr = np.loadtxt(os.path.join(data_dir, "sat.trn"))
    te = np.loadtxt(os.path.join(data_dir, "sat.tst"))

    X_pool, y_pool = tr[:, :-1], tr[:, -1].astype(int)
    X_test, y_test = te[:, :-1], te[:, -1].astype(int)

    if standardize:
        scaler = StandardScaler().fit(X_pool)
        X_pool = scaler.transform(X_pool)
        X_test = scaler.transform(X_test)

    return X_pool, y_pool, X_test, y_test
