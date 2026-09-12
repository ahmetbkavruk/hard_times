from pathlib import Path
import argparse
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ht import Config, World
from experiments.plotting import plot_baseline


def _allocation_snapshot(world):
    return [a.production_allocation for a in world.agents.values()]


def run(seed: int = 7, steps: int = 600, results_dir: Path = Path("results")):
    cfg = Config()
    world = World(cfg, seed=seed)

    snapshot_times = {0, steps // 3, (2 * steps) // 3, steps}
    allocation_snapshots = {0: _allocation_snapshot(world)}
    while world.t < steps and world.agents:
        world.step()
        if world.t in snapshot_times:
            allocation_snapshots[world.t] = _allocation_snapshot(world)
    allocation_snapshots[world.t] = _allocation_snapshot(world)

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"baseline_seed_{seed}.csv"

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=world.history[0].keys())
        writer.writeheader()
        writer.writerows(world.history)

    figures_dir = results_dir / "figures"
    plot_baseline(world.history, allocation_snapshots, seed, figures_dir)

    final = world.history[-1]
    print(f"seed={seed} steps={final['t']}")
    print(f"population={final['population']}")
    print(f"births={final['births']} deaths={final['deaths']}")
    print(f"surviving founder lineages={final['lineages']}")
    print(f"mean generation={final['mean_generation']:.2f} max generation={final['max_generation']}")
    print(
        "mean traits: "
        f"hA={final['mean_hA']:.3f}, "
        f"s={final['mean_production_allocation']:.3f}, "
        f"v={final['mean_v']:.3f}"
    )
    print(f"wrote {out_path}")
    print(f"wrote figures to {figures_dir}")

    return world


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and plot the no-trade baseline")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(seed=args.seed, steps=args.steps, results_dir=args.results_dir)
