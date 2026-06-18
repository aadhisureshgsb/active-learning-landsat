"""Command-line entry point: ``python -m active_learning``."""

from __future__ import annotations

import argparse

from active_learning.data import load_landsat
from active_learning.experiment import labels_to_target, run_experiment
from active_learning.plotting import plot_learning_curves


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="active_learning",
        description="Active-learning label-efficiency study on the Landsat "
                    "(Statlog) satellite benchmark.",
    )
    p.add_argument("--data-dir", default="data", help="where to cache the dataset")
    p.add_argument("--seeds", type=int, default=12, help="number of random seeds")
    p.add_argument("--batch", type=int, default=10, help="labels acquired per round")
    p.add_argument("--rounds", type=int, default=38, help="acquisition rounds")
    p.add_argument("--seed-per-class", type=int, default=2,
                   help="warm-start labels per class")
    p.add_argument("--target-acc", type=float, default=0.82,
                   help="accuracy threshold for the label-budget comparison")
    p.add_argument("--out", default="results/active_learning_landsat.png",
                   help="output path for the learning-curve figure")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    X_pool, y_pool, X_test, y_test = load_landsat(args.data_dir)
    print(f"Pool: {len(y_pool)}   Test: {len(y_test)}   "
          f"Features: {X_pool.shape[1]}   Classes: {sorted(set(y_pool))}")

    results = run_experiment(
        X_pool, y_pool, X_test, y_test,
        n_seeds=args.seeds, batch=args.batch, rounds=args.rounds,
        seed_per_class=args.seed_per_class,
    )

    print(f"\nLabels to reach {args.target_acc:.0%} test accuracy:")
    budgets = {}
    for name, curve in results.items():
        budgets[name] = labels_to_target(curve, args.target_acc)
        shown = f"{budgets[name]} labels" if budgets[name] else "not reached"
        print(f"  {name:24s}: {shown}")

    base = budgets.get("Random (baseline)")
    margin = budgets.get("Uncertainty: margin")
    if base and margin:
        print(f"\nLabel savings (margin vs random): {base - margin} fewer labels "
              f"({100 * (base - margin) / base:.0f}% reduction)")

    path = plot_learning_curves(results, args.target_acc, args.out)
    print(f"\nSaved figure -> {path}")


if __name__ == "__main__":
    main()
