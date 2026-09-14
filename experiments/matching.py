"""Test local matching at fixed gamma=2 with two specialist founder lineages."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ht import Config, World
from experiments.runner import condition_directory, save_results, write_csv


def _group_metrics(world, births, generalists):
    counts = {"B_specialist": 0, "C_specialist": 0, "generalist": 0}
    specialist_energy = 0.0
    for agent in world.agents.values():
        group = _group(agent.lineage)
        counts[group] += 1
        if agent.lineage in (0, 1):
            specialist_energy += agent.energy
    specialists = counts["B_specialist"] + counts["C_specialist"]
    population = len(world.agents)
    row = {
        "t": world.t,
        "specialists_alive": specialists,
        "specialist_fraction": specialists / population if population else 0.0,
        "mean_specialist_energy": specialist_energy / specialists if specialists else np.nan,
    }
    for group, initial in (("B_specialist", 1), ("C_specialist", 1), ("generalist", generalists)):
        row[f"{group}s_alive"] = counts[group]
        row[f"{group}_births"] = births[group]
        row[f"{group}_deaths"] = initial + births[group] - counts[group]
    return row


def _group(lineage):
    if lineage == 0:
        return "C_specialist"
    if lineage == 1:
        return "B_specialist"
    return "generalist"


def _run_condition(config, seed, steps, generalists, output_dir, make_plots):
    world = World(config, seed=seed)
    for agent in world.agents.values():
        agent.production_allocation = (0.0, 1.0)[agent.id] if agent.id < 2 else 0.5
        agent.genotype = np.array([1.0, 0.0])
    # Replace the constructor's random-founder metrics with the prepared state.
    world.history.clear()
    world.record_metrics()
    founders = [
        {"id": agent.id, "lineage": agent.lineage,
         "production_allocation": agent.production_allocation,
         "genotype": agent.genotype.tolist(), "energy": agent.energy}
        for agent in world.agents.values()
    ]
    births = {"B_specialist": 0, "C_specialist": 0, "generalist": 0}
    group_history = [_group_metrics(world, births, generalists)]
    snapshots = {0: [agent.production_allocation for agent in world.agents.values()]}
    snapshot_times = {steps // 3, 2 * steps // 3, steps}
    specialist_extinction_step = None
    capacity_reached_step = 0 if len(world.agents) >= config.max_agents else None
    while world.t < steps and world.agents:
        first_child_id = world.next_agent_id
        world.step()
        # Newborns have positive energy and do not act until the next step.
        for agent in world.agents.values():
            if agent.id >= first_child_id:
                births[_group(agent.lineage)] += 1
        row = _group_metrics(world, births, generalists)
        group_history.append(row)
        if row["specialists_alive"] == 0 and specialist_extinction_step is None:
            specialist_extinction_step = world.t
        if len(world.agents) >= config.max_agents and capacity_reached_step is None:
            capacity_reached_step = world.t
        if world.t in snapshot_times:
            snapshots[world.t] = [agent.production_allocation for agent in world.agents.values()]
    snapshots[world.t] = [agent.production_allocation for agent in world.agents.values()]
    save_results(world, snapshots, steps, output_dir, make_plots=make_plots)
    write_csv(output_dir / "lineage_history.csv", group_history, group_history[0].keys())
    (output_dir / "founders.json").write_text(
        json.dumps(founders, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "seed": seed,
        "trade_enabled": config.trade_enabled,
        "trade_matching": config.trade_matching,
        "production_specialization_exponent": config.production_specialization_exponent,
        "requested_steps": steps,
        "initial_specialist_fraction": 2 / config.initial_agents,
        "specialist_extinction_step": specialist_extinction_step,
        "capacity_reached_step": capacity_reached_step,
        **world.history[-1],
        **group_history[-1],
    }


def run(
    seeds=(7, 11, 17),
    survival_steps: int = 1000,
    reproduction_steps: int = 200,
    generalists: int = 18,
    max_agents: int = 1000,
    results_dir: Path = Path("results/matching"),
    *,
    make_plots: bool = True,
):
    if generalists < 0 or max_agents < generalists + 2:
        raise ValueError("generalists must be nonnegative and max_agents must fit all founders")
    if survival_steps < 0 or reproduction_steps < 0:
        raise ValueError("experiment durations must be nonnegative")
    seeds = tuple(dict.fromkeys(seeds))
    if not seeds:
        raise ValueError("at least one seed is required")
    # Supply exceeds the maximum possible aggregate A harvest every step.
    supply = max(1000.0, 2.0 * max_agents)
    base = Config(
        L=1, initial_agents=generalists + 2, max_agents=max_agents,
        base_capacity=supply, capacity_contrast=0.0, regen_rate=supply,
        production_specialization_exponent=2.0,
        deficit_move_probability=0.0, move_exploration=0.0,
        mutation_sigma=0.0, allocation_mutation_sigma=0.0,
    )
    phases = (
        ("survival", survival_steps, replace(
            base, max_agents=base.initial_agents,
            reproduce_probability=0.0, reproduce_threshold=1e9,
        )),
        ("reproduction", reproduction_steps, base),
    )
    summaries = []
    for phase, steps, phase_config in phases:
        conditions = (
            replace(phase_config, trade_enabled=False),
            replace(phase_config, trade_enabled=True, trade_matching="random"),
            replace(phase_config, trade_enabled=True, trade_matching="compatible"),
        )
        for seed in seeds:
            for config in conditions:
                output_dir = condition_directory(results_dir / phase, config, seed)
                summary = _run_condition(config, seed, steps, generalists, output_dir, make_plots)
                summaries.append({"phase": phase, **summary})
    write_csv(results_dir / "summary.csv", summaries, summaries[0].keys())
    print(f"wrote {results_dir / 'summary.csv'}")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 11, 17))
    parser.add_argument("--survival-steps", type=int, default=1000)
    parser.add_argument("--reproduction-steps", type=int, default=200)
    parser.add_argument("--generalists", type=int, default=18)
    parser.add_argument("--max-agents", type=int, default=1000)
    parser.add_argument("--results-dir", type=Path, default=Path("results/matching"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run(
        seeds=args.seeds, survival_steps=args.survival_steps,
        reproduction_steps=args.reproduction_steps, generalists=args.generalists,
        max_agents=args.max_agents, results_dir=args.results_dir,
        make_plots=not args.no_plots,
    )
