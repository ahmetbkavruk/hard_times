"""Run one explicit configuration and save its measurements."""

import csv
from dataclasses import asdict
import json
from pathlib import Path

from ht import Config, World


def condition_directory(results_dir: Path, config: Config, seed: int) -> Path:
    condition = "trade_on" if config.trade_enabled else "trade_off"
    return (
        results_dir
        / f"gamma_{config.production_specialization_exponent:g}"
        / condition
        / f"seed_{seed}"
    )


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _trade_rows(events):
    for event in events:
        donors = {value["good"]: agent_id for agent_id, value in event["surrendered"].items()}
        yield {
            "t": event["t"],
            "cell_x": event["cell"][0],
            "cell_y": event["cell"][1],
            "B_donor_id": donors["B"],
            "C_donor_id": donors["C"],
            "gross_B": event["surrendered"][donors["B"]]["gross"],
            "gross_C": event["surrendered"][donors["C"]]["gross"],
            "net_B": event["received"][donors["C"]]["net"],
            "net_C": event["received"][donors["B"]]["net"],
            "loss_B": event["transaction_loss"]["B"],
            "loss_C": event["transaction_loss"]["C"],
        }


def run_experiment(
    config: Config,
    seed: int,
    steps: int,
    output_dir: Path,
    *,
    make_plots: bool = True,
) -> World:
    if steps < 0:
        raise ValueError("steps must be nonnegative")

    world = World(config, seed=seed)
    snapshot_times = {0, steps // 3, (2 * steps) // 3, steps}
    snapshots = {0: [agent.production_allocation for agent in world.agents.values()]}
    while world.t < steps and world.agents:
        world.step()
        if world.t in snapshot_times:
            snapshots[world.t] = [agent.production_allocation for agent in world.agents.values()]
    snapshots[world.t] = [agent.production_allocation for agent in world.agents.values()]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "history.csv", world.history, world.history[0].keys())
    write_csv(
        output_dir / "trades.csv",
        _trade_rows(world.trades),
        (
            "t", "cell_x", "cell_y", "B_donor_id", "C_donor_id",
            "gross_B", "gross_C", "net_B", "net_C", "loss_B", "loss_C",
        ),
    )
    metadata = {
        "seed": seed,
        "requested_steps": steps,
        "completed_steps": world.t,
        "config": asdict(config),
    }
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output_dir / "allocation_snapshots.json").write_text(
        json.dumps(snapshots, indent=2) + "\n", encoding="utf-8"
    )
    if make_plots:
        from experiments.plotting import plot_run

        plot_run(world.history, snapshots, seed, output_dir / "figures")

    final = world.history[-1]
    print(
        f"seed={seed} gamma={config.production_specialization_exponent:g} "
        f"trade={config.trade_enabled} steps={world.t} "
        f"population={final['population']} trades={final['cumulative_trade_events']}"
    )
    return world
