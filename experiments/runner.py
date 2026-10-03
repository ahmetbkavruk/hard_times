"""Run one explicit configuration and save its measurements."""

import csv
from dataclasses import asdict
import json
from pathlib import Path

from ht import Config, World
from ht.fields import FIELD_CHANNELS


def condition_directory(results_dir: Path, config: Config, seed: int) -> Path:
    if config.fields_enabled:
        results_dir = results_dir / f"fields_decay_{config.field_decay:g}_diffusion_{config.field_diffusion:g}"
    if config.social_enabled:
        results_dir = results_dir / "social"
    condition = f"trade_{config.trade_matching}" if config.trade_enabled else "trade_off"
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
            "B_donor_x": event["partner_cells"][donors["B"]][0],
            "B_donor_y": event["partner_cells"][donors["B"]][1],
            "C_donor_x": event["partner_cells"][donors["C"]][0],
            "C_donor_y": event["partner_cells"][donors["C"]][1],
            "gross_B": event["surrendered"][donors["B"]]["gross"],
            "gross_C": event["surrendered"][donors["C"]]["gross"],
            "net_B": event["received"][donors["C"]]["net"],
            "net_C": event["received"][donors["B"]]["net"],
            "loss_B": event["transaction_loss"]["B"],
            "loss_C": event["transaction_loss"]["C"],
            "energy_cost_per_agent": event["energy_cost_per_agent"],
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

    save_results(world, snapshots, steps, output_dir, make_plots=make_plots)
    return world


def save_results(
    world: World, snapshots: dict, steps: int, output_dir: Path, *, make_plots: bool = True
) -> None:
    """Save a completed run, including runs with deliberately prepared founders."""
    config, seed = world.cfg, world.seed
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "history.csv", world.history, world.history[0].keys())
    write_csv(
        output_dir / "trades.csv",
        _trade_rows(world.trades),
        (
            "t", "cell_x", "cell_y", "B_donor_id", "C_donor_id",
            "B_donor_x", "B_donor_y", "C_donor_x", "C_donor_y",
            "gross_B", "gross_C", "net_B", "net_C", "loss_B", "loss_C",
            "energy_cost_per_agent",
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
    resource_map = {
        "t": world.t,
        "renewable_mask": world.renewable_mask.tolist(),
        "initial_A": world.capacity[0].tolist(),
        "remaining_A": world.resources[0].tolist(),
    }
    (output_dir / "resource_map.json").write_text(
        json.dumps(resource_map) + "\n", encoding="utf-8"
    )
    (output_dir / "interactions.json").write_text(
        json.dumps(world.interactions) + "\n", encoding="utf-8"
    )
    (output_dir / "agent_traits.json").write_text(
        json.dumps({"founders": world.founder_traits, "survivors": world.agent_traits()}) + "\n", encoding="utf-8"
    )
    (output_dir / "carcasses.json").write_text(
        json.dumps([asdict(body) for body in world.carcasses.values()]) + "\n", encoding="utf-8"
    )
    if world.fields is not None:
        write_csv(output_dir / "field_history.csv", world.fields.history, world.fields.history[0].keys())
        field_map = {
            "t": world.fields.t, "axis_order": "y, x", "unit": "event weight",
            "decay": config.field_decay, "diffusion": config.field_diffusion,
            "fields": {channel: world.fields.layer(channel).tolist() for channel in FIELD_CHANNELS},
        }
        (output_dir / "field_map.json").write_text(json.dumps(field_map) + "\n", encoding="utf-8")
    else:
        # A caller may deliberately reuse an output directory for a control.
        for name in ("field_map.json", "field_history.csv"):
            (output_dir / name).unlink(missing_ok=True)
    if make_plots:
        from experiments.plotting import plot_run

        plot_run(world.history, snapshots, seed, output_dir / "figures")

    final = world.history[-1]
    print(
        f"seed={seed} gamma={config.production_specialization_exponent:g} "
        f"trade={config.trade_enabled} matching={config.trade_matching} steps={world.t} "
        f"finite_fraction={config.finite_resource_fraction:g} "
        f"social={config.social_enabled} "
        f"population={final['population']} trades={final['cumulative_trade_events']}"
    )
