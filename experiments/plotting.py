"""Read-only plots for either explicitly configured experiment condition."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _series(history, key):
    return [row[key] for row in history]


def _save(fig, output_dir: Path, filename: str):
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150)
    plt.close(fig)


def plot_run(history, allocation_snapshots, seed: int, output_dir: Path):
    """Write figures without changing the world or its agents."""
    output_dir.mkdir(parents=True, exist_ok=True)
    time = _series(history, "t")

    fig, ax = plt.subplots()
    ax.plot(time, _series(history, "population"), label="Population")
    ax.plot(time, _series(history, "births"), label="Cumulative births")
    ax.plot(time, _series(history, "deaths"), label="Cumulative deaths")
    ax.set(xlabel="Timestep", ylabel="Agents", title="Population dynamics")
    ax.legend()
    _save(fig, output_dir, f"population_seed_{seed}.png")

    fig, ax = plt.subplots()
    ax.plot(time, _series(history, "mean_energy"))
    ax.set(xlabel="Timestep", ylabel="Mean energy", title="Energy dynamics")
    _save(fig, output_dir, f"energy_seed_{seed}.png")

    fig, ax = plt.subplots()
    mean_h_a = _series(history, "mean_hA")
    mean_v = _series(history, "mean_v")
    std_h_a = _series(history, "std_hA")
    std_v = _series(history, "std_v")
    ax.plot(time, mean_h_a, label="Mean h_A")
    ax.fill_between(
        time,
        [mean - std for mean, std in zip(mean_h_a, std_h_a)],
        [mean + std for mean, std in zip(mean_h_a, std_h_a)],
        alpha=0.15,
    )
    ax.plot(time, mean_v, label="Mean v")
    ax.fill_between(
        time,
        [mean - std for mean, std in zip(mean_v, std_v)],
        [mean + std for mean, std in zip(mean_v, std_v)],
        alpha=0.15,
    )
    ax.set(xlabel="Timestep", ylabel="Trait value", title="Ecological traits")
    ax.legend()
    _save(fig, output_dir, f"ecological_traits_seed_{seed}.png")

    fig, ax = plt.subplots()
    ax.plot(time, _series(history, "mean_production_allocation"), label="Mean")
    ax.plot(time, _series(history, "std_production_allocation"), label="Standard deviation")
    ax.set(
        xlabel="Timestep",
        ylabel="Production allocation",
        title="Production allocation dynamics",
    )
    ax.legend()
    _save(fig, output_dir, f"production_allocation_seed_{seed}.png")

    fig, ax = plt.subplots()
    for snapshot_time, values in sorted(allocation_snapshots.items()):
        if values:
            ax.hist(
                values,
                bins=20,
                range=(0.0, 1.0),
                histtype="step",
                linewidth=1.5,
                label=f"t={snapshot_time}",
            )
    ax.set(
        xlabel="Production allocation",
        ylabel="Agent count",
        title="Production allocation snapshots",
        xlim=(0.0, 1.0),
    )
    if allocation_snapshots:
        ax.legend()
    _save(fig, output_dir, f"production_allocation_histograms_seed_{seed}.png")

    fig, ax = plt.subplots()
    ax.plot(time, _series(history, "total_B_stock"), label="Total B stock")
    ax.plot(time, _series(history, "total_C_stock"), label="Total C stock")
    ax.set(xlabel="Timestep", ylabel="Inventory", title="Manufactured inventories")
    ax.legend()
    _save(fig, output_dir, f"manufactured_inventories_seed_{seed}.png")

    fig, ax = plt.subplots()
    ax.plot(time, _series(history, "renewable_A_remaining"), label="Renewable ground", color="#527b62")
    ax.plot(time, _series(history, "finite_A_remaining"), label="Finite deposits", color="#927256")
    ax.set(xlabel="Timestep", ylabel="Resource A remaining", title="Environmental resource stocks")
    ax.legend()
    _save(fig, output_dir, f"environmental_resources_seed_{seed}.png")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    axes[0].plot(time, _series(history, "cumulative_trade_events"), label="Trades")
    axes[0].plot(
        time, _series(history, "cumulative_recipe_uses_with_received_stock"),
        label="Recipe uses containing receipts",
    )
    axes[0].set(ylabel="Cumulative events", title="Exchange and use of received goods")
    axes[0].legend()
    for quantity, label in (("gross", "Surrendered"), ("received", "Received"), ("loss", "Lost")):
        amounts = [
            row[f"cumulative_trade_{quantity}_B"] + row[f"cumulative_trade_{quantity}_C"]
            for row in history
        ]
        axes[1].plot(time, amounts, label=label)
    axes[1].set(xlabel="Timestep", ylabel="Cumulative B + C units")
    axes[1].legend()
    _save(fig, output_dir, f"trade_activity_seed_{seed}.png")
