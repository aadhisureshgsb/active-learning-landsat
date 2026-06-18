"""Plotting the label-efficiency learning curves."""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from active_learning.experiment import Curve  # noqa: E402

_COLORS = {
    "Random (baseline)": "#9aa0a6",
    "Uncertainty: margin": "#d93025",
    "Uncertainty: entropy": "#1a73e8",
    "Uncertainty: least_confidence": "#188038",
}


def plot_learning_curves(
    results: Dict[str, Curve],
    target_acc: float,
    out_path: str = "results/active_learning_landsat.png",
    dpi: int = 150,
) -> str:
    """Render mean curves with +/-1 std bands; save to ``out_path``."""
    plt.figure(figsize=(9, 5.5))
    palette = iter(["#d93025", "#1a73e8", "#188038", "#f29900"])

    for name, curve in results.items():
        color = _COLORS.get(name) or next(palette)
        front = "Random" not in name
        plt.plot(curve.n_labels, curve.mean, label=name, color=color, lw=2.2,
                 zorder=3 if front else 2)
        plt.fill_between(curve.n_labels, curve.mean - curve.std,
                         curve.mean + curve.std, color=color, alpha=0.13)

    last_x = next(iter(results.values())).n_labels[-1]
    plt.axhline(target_acc, ls="--", lw=1, color="#5f6368")
    plt.text(last_x, target_acc + 0.004, f"{target_acc:.0%} target",
             ha="right", va="bottom", fontsize=9, color="#5f6368")

    plt.xlabel("Number of labeled training tiles")
    plt.ylabel("Test accuracy (2,000 held-out tiles)")
    plt.title("Active learning on real Landsat satellite tiles\n"
              "model-chosen labels vs random labeling")
    plt.legend(loc="lower right", frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    return out_path
