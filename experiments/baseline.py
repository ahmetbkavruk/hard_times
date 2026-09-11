from pathlib import Path
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ht import Config, World


def run(seed: int = 7, steps: int = 600):
    cfg = Config()
    world = World(cfg, seed=seed)
    world.run(steps)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"baseline_seed_{seed}.csv"

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=world.history[0].keys())
        writer.writeheader()
        writer.writerows(world.history)

    final = world.history[-1]
    print(f"seed={seed} steps={final['t']}")
    print(f"population={final['population']}")
    print(f"births={final['births']} deaths={final['deaths']}")
    print(f"surviving founder lineages={final['lineages']}")
    print(f"mean generation={final['mean_generation']:.2f} max generation={final['max_generation']}")
    print(
        "mean traits: "
        f"hA={final['mean_hA']:.3f}, "
        f"hB={final['mean_hB']:.3f}, "
        f"hC={final['mean_hC']:.3f}, "
        f"v={final['mean_v']:.3f}"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    run()
