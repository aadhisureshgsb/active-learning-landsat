"""Label-efficient active learning on the Landsat (Statlog) satellite benchmark."""

from active_learning.data import load_landsat, CLASS_NAMES
from active_learning.strategies import STRATEGIES, register_strategy
from active_learning.experiment import run_once, run_experiment, labels_to_target

__version__ = "0.1.0"

__all__ = [
    "load_landsat",
    "CLASS_NAMES",
    "STRATEGIES",
    "register_strategy",
    "run_once",
    "run_experiment",
    "labels_to_target",
    "__version__",
]
