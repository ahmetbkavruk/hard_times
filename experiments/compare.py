"""Compare trade off/on at each gamma using the same seeds and policy."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ht import Config
from experiments.runner import condition_directory, run_experiment, write_csv


def run(
    seeds=(7, 11, 17),
    gammas=(1.0, 1.5, 2.0),
    steps: int = 600,
    results_dir: Path = Path("results/comparison"),
    *,
    deficit_move_probability: float = 0.12,
    trade_efficiency: float = 1.0,
    max_trade_per_pair: float = 1.0,
    matchings=("neighborhood",),
    make_plots: bool = True,
):
    base_config = Config(
        deficit_move_probability=deficit_move_probability,
        trade_efficiency=trade_efficiency,
        max_trade_per_pair=max_trade_per_pair,
    )
    conditions = [replace(base_config, trade_enabled=False)]
    conditions.extend(
        replace(base_config, trade_enabled=True, trade_matching=matching)
        for matching in dict.fromkeys(matchings)
    )
    summaries = []
    for gamma in dict.fromkeys(gammas):
        for seed in dict.fromkeys(seeds):
            for condition in conditions:
                config = replace(
                    condition,
                    production_specialization_exponent=gamma,
                )
                output_dir = condition_directory(results_dir, config, seed)
                world = run_experiment(config, seed, steps, output_dir, make_plots=make_plots)
                final = world.history[-1]
                uses = final["cumulative_recipe_uses"]
                summaries.append({
                    "seed": seed,
                    "production_specialization_exponent": gamma,
                    "trade_enabled": config.trade_enabled,
                    "trade_matching": config.trade_matching,
                    "trade_efficiency": trade_efficiency,
                    "max_trade_per_pair": max_trade_per_pair,
                    "deficit_move_probability": deficit_move_probability,
                    "requested_steps": steps,
                    **final,
                    "fraction_recipe_uses_with_received_stock": (
                        final["cumulative_recipe_uses_with_received_stock"] / uses
                        if uses else 0.0
                    ),
                })
    if not summaries:
        raise ValueError("at least one seed and gamma are required")
    write_csv(results_dir / "summary.csv", summaries, summaries[0].keys())
    print(f"wrote {results_dir / 'summary.csv'}")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 11, 17))
    parser.add_argument("--gammas", type=float, nargs="+", default=(1.0, 1.5, 2.0))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--results-dir", type=Path, default=Path("results/comparison"))
    parser.add_argument("--deficit-move-probability", type=float, default=0.12)
    parser.add_argument("--trade-efficiency", type=float, default=1.0)
    parser.add_argument("--max-trade-per-pair", type=float, default=1.0)
    parser.add_argument("--matchings", nargs="+", choices=("neighborhood", "random", "compatible"), default=("neighborhood",))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run(
        seeds=args.seeds,
        gammas=args.gammas,
        steps=args.steps,
        results_dir=args.results_dir,
        deficit_move_probability=args.deficit_move_probability,
        trade_efficiency=args.trade_efficiency,
        max_trade_per_pair=args.max_trade_per_pair,
        matchings=args.matchings,
        make_plots=not args.no_plots,
    )
