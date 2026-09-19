"""The explicitly configured no-trade control."""

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ht import Config
from experiments.runner import condition_directory, run_experiment


def run(
    seed: int = 7,
    steps: int = 600,
    results_dir: Path = Path("results"),
    *,
    gamma: float = 1.0,
    deficit_move_probability: float = 0.12,
    finite_resource_fraction: float = 0.0,
    resource_seed: int | None = None,
    base_capacity: float = 8.0,
    regen_rate: float = 0.03,
    social_enabled: bool = False,
    make_plots: bool = True,
):
    config = Config(
        trade_enabled=False,
        production_specialization_exponent=gamma,
        deficit_move_probability=deficit_move_probability,
        finite_resource_fraction=finite_resource_fraction, resource_seed=resource_seed,
        base_capacity=base_capacity, regen_rate=regen_rate,
        social_enabled=social_enabled,
    )
    output_dir = condition_directory(results_dir, config, seed)
    world = run_experiment(config, seed, steps, output_dir, make_plots=make_plots)
    print(f"wrote {output_dir}")
    return world


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and plot the no-trade baseline")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--deficit-move-probability", type=float, default=0.12)
    parser.add_argument("--finite-fraction", type=float, default=0.0)
    parser.add_argument("--resource-seed", type=int)
    parser.add_argument("--base-capacity", type=float, default=8.0)
    parser.add_argument("--regen-rate", type=float, default=0.03)
    parser.add_argument("--social", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run(
        seed=args.seed,
        steps=args.steps,
        results_dir=args.results_dir,
        gamma=args.gamma,
        deficit_move_probability=args.deficit_move_probability,
        finite_resource_fraction=args.finite_fraction, resource_seed=args.resource_seed,
        base_capacity=args.base_capacity, regen_rate=args.regen_rate,
        social_enabled=args.social,
        make_plots=not args.no_plots,
    )
