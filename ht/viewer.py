"""A live, read-only view of the existing lattice simulation.

Only advancing the simulation calls World.step(). Drawing, selecting, pausing,
and changing playback speed never draw from the world's random generator.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from math import ceil, sqrt

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button, Slider
import numpy as np

from .actions import ActionKind
from .config import Config
from .world import World


BACKGROUND = "#10151b"
PANEL = "#19222c"
TEXT = "#e8edf2"
MUTED = "#9aabbc"
EATING = "#c9ff76"
TRADING = "#f58be0"
RESOURCE_COLORS = LinearSegmentedColormap.from_list(
    "resource_amount", ["#151c22", "#29443e", "#527b62"]
)
AGENT_COLORS = LinearSegmentedColormap.from_list(
    "production_allocation", ["#50caff", "#ebedf1", "#ffb454"]
)


class LiveViewer:
    def __init__(
        self, config: Config, seed: int = 7, *, speed: int = 10,
        steps: int | None = None, paused: bool = False,
    ):
        if not 1 <= speed <= 60:
            raise ValueError("speed must be between 1 and 60 steps per second")
        if steps is not None and steps < 1:
            raise ValueError("steps must be positive")
        self.config = config
        self.seed = seed
        self.steps = steps
        self.paused = paused
        self.world = World(config, seed=seed, record_actions=True)
        self.selected_agent = None
        self.selected_died_at = None
        self.recent_trades = []
        self.eating_cells = []
        self.death_cells = []
        self.agent_positions = {}

        with plt.rc_context({"font.family": "DejaVu Sans", "toolbar": "None"}):
            self.figure = plt.figure(figsize=(12, 8), facecolor=BACKGROUND)
        self.figure.canvas.manager.set_window_title("Hard Times | Live world")
        self.figure.text(0.045, 0.945, "HARD TIMES", color=TEXT, fontsize=23, weight="bold")
        self.figure.text(0.045, 0.908, "LIVE WORLD  /  observe a seeded simulation", color=MUTED, fontsize=10)
        self.status = self.figure.text(0.955, 0.943, "", color=EATING, fontsize=11, ha="right")

        self.grid = self.figure.add_axes([0.045, 0.215, 0.59, 0.655], facecolor=BACKGROUND)
        self.grid.set_aspect("equal")
        self.grid.set_xlim(-0.5, config.L - 0.5)
        self.grid.set_ylim(config.L - 0.5, -0.5)
        self.grid.set_xticks([])
        self.grid.set_yticks([])
        for spine in self.grid.spines.values():
            spine.set_color("#35434e")
        self.resource_image = self.grid.imshow(
            self.world.resources[0], origin="upper", interpolation="nearest",
            cmap=RESOURCE_COLORS, vmin=0, vmax=max(float(self.world.capacity.max()), 1e-12),
        )
        self.dots = self.grid.scatter(
            [], [], marker="s", c=[], cmap=AGENT_COLORS, vmin=0, vmax=1,
            edgecolors=BACKGROUND, linewidths=0.3, zorder=3,
        )
        self.eating_marks = self.grid.scatter(
            [], [], marker="s", facecolors="none", edgecolors=EATING,
            linewidths=1.3, zorder=4,
        )
        self.trade_marks = self.grid.scatter(
            [], [], marker="o", facecolors="none", edgecolors=TRADING,
            linewidths=1.5, zorder=5,
        )
        self.death_marks = self.grid.scatter(
            [], [], marker="x", color="#fa8279", linewidths=1.1, zorder=5,
        )
        self.selection = self.grid.scatter(
            [], [], marker="s", facecolors="none", edgecolors="white",
            linewidths=1.7, zorder=6,
        )

        self.sidebar = self.figure.add_axes([0.675, 0.215, 0.285, 0.655], facecolor=PANEL)
        self.sidebar.set_xticks([])
        self.sidebar.set_yticks([])
        for spine in self.sidebar.spines.values():
            spine.set_visible(False)
        self.sidebar.text(0.065, 0.955, "WORLD", color=MUTED, fontsize=10, weight="bold")
        self.world_text = self.sidebar.text(
            0.065, 0.895, "", va="top", color=TEXT, fontsize=11, linespacing=1.65,
            fontfamily="DejaVu Sans Mono",
        )
        self.sidebar.axhline(0.54, xmin=0.065, xmax=0.935, color="#34424f", linewidth=0.8)
        self.sidebar.text(0.065, 0.50, "AGENT INSPECTOR", color=MUTED, fontsize=10, weight="bold")
        self.agent_text = self.sidebar.text(
            0.065, 0.445, "", va="top", color=TEXT, fontsize=9.5, linespacing=1.6,
            fontfamily="DejaVu Sans Mono",
        )

        color_axis = self.figure.add_axes([0.12, 0.172, 0.25, 0.012])
        color_axis.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=AGENT_COLORS)
        color_axis.set_axis_off()
        self.figure.text(0.045, 0.171, "C output", color="#50caff", fontsize=9)
        self.figure.text(0.38, 0.171, "B output", color="#ffb454", fontsize=9)
        self.figure.text(0.045, 0.144, "Ground: brighter = more resource A", color=MUTED, fontsize=9)
        self.figure.text(0.675, 0.18, "Green square: ate a recipe", color=EATING, fontsize=9)
        self.figure.text(0.675, 0.153, "Pink ring: traded   /   Red cross: died", color=TRADING, fontsize=9)
        self.figure.text(0.675, 0.126, "Highlights show the latest step.", color=MUTED, fontsize=8.5)

        self.play_button = self._button([0.045, 0.06, 0.12, 0.05], "Pause", self.toggle_play)
        self.step_button = self._button([0.175, 0.06, 0.12, 0.05], "Step", self.single_step)
        self.reset_button = self._button([0.305, 0.06, 0.12, 0.05], "Restart seed", self.reset)
        slider_axis = self.figure.add_axes([0.545, 0.078, 0.325, 0.015], facecolor=PANEL)
        self.speed_slider = Slider(
            slider_axis, "Steps / sec", 1, 60, valinit=speed, valstep=1,
            valfmt="%d", color="#71b9b1", track_color="#34424f",
        )
        self.speed_slider.label.set_color(MUTED)
        self.speed_slider.valtext.set_color(TEXT)
        self.figure.text(
            0.045, 0.023,
            "SPACE  play / pause     N  one step     R  restart     Click a cell to inspect; click again to cycle agents.",
            color=MUTED, fontsize=8.5,
        )

        self.timer = self.figure.canvas.new_timer(interval=max(1, round(1000 / speed)))
        self.timer.add_callback(self.tick)
        self.speed_slider.on_changed(self.change_speed)
        self.figure.canvas.mpl_connect("button_press_event", self.on_click)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.figure.canvas.mpl_connect("close_event", lambda event: self.timer.stop())
        self.figure.canvas.mpl_connect("resize_event", lambda event: self.refresh())
        self.refresh()

    def _button(self, bounds, label, callback):
        button = Button(self.figure.add_axes(bounds), label, color=PANEL, hovercolor="#30434f")
        button.label.set_color(TEXT)
        button.label.set_fontsize(10)
        for spine in button.ax.spines.values():
            spine.set_color("#34424f")
        button.on_clicked(callback)
        return button

    @property
    def finished(self):
        return not self.world.agents or (self.steps is not None and self.world.t >= self.steps)

    def toggle_play(self, event=None):
        if not self.finished:
            self.paused = not self.paused
        self.refresh()

    def single_step(self, event=None):
        self.paused = True
        self.advance()

    def reset(self, event=None):
        self.world = World(self.config, seed=self.seed, record_actions=True)
        self.selected_agent = None
        self.selected_died_at = None
        self.recent_trades = []
        self.eating_cells = []
        self.death_cells = []
        self.paused = True
        self.refresh()

    def change_speed(self, value):
        self.timer.interval = max(1, round(1000 / float(value)))

    def tick(self):
        if not self.paused and not self.finished:
            self.advance()
        return True  # Keep the timer registered while paused or finished.

    def advance(self):
        if self.finished:
            self.refresh()
            return
        previous_agents = dict(self.world.agents)
        trade_start = len(self.world.trades)
        self.world.step()
        self.recent_trades = self.world.trades[trade_start:]
        self.eating_cells = [
            (previous_agents[aid].x, previous_agents[aid].y)
            for aid in sorted(self.world.last_recipe_uses)
        ]
        self.death_cells = [
            (agent.x, agent.y) for aid, agent in previous_agents.items()
            if aid not in self.world.agents
        ]
        if (
            self.selected_agent is not None
            and self.selected_agent.id not in self.world.agents
            and self.selected_died_at is None
        ):
            self.selected_died_at = self.world.t
        self.refresh()

    @staticmethod
    def _offsets(points):
        return np.asarray(points, dtype=float).reshape(-1, 2)

    def refresh(self):
        """Read world state and update artists; never advance or randomize it."""
        world = self.world
        self.resource_image.set_data(world.resources[0])
        occupancy = defaultdict(list)
        for agent in world.agents.values():
            occupancy[(agent.x, agent.y)].append(agent)
        positions, allocations, sizes = [], [], []
        self.agent_positions = {}
        # Subcell offsets reveal co-location without changing physical positions.
        cell_points = min(self.grid.bbox.width, self.grid.bbox.height) * 72 / self.figure.dpi / self.config.L
        for (x, y), agents in sorted(occupancy.items()):
            agents.sort(key=lambda agent: agent.id)
            columns = ceil(sqrt(len(agents)))
            rows = ceil(len(agents) / columns)
            for index, agent in enumerate(agents):
                dx = ((index % columns + 0.5) / columns - 0.5) * 0.8
                dy = ((index // columns + 0.5) / rows - 0.5) * 0.8
                position = (x + dx, y + dy)
                self.agent_positions[agent.id] = position
                positions.append(position)
                allocations.append(agent.production_allocation)
                sizes.append((0.65 * cell_points / columns) ** 2)
        self.dots.set_offsets(self._offsets(positions))
        self.dots.set_array(np.asarray(allocations))
        self.dots.set_sizes(sizes)
        for artist, points, factor in (
            (self.eating_marks, self.eating_cells, 0.92),
            (self.trade_marks, sorted({tuple(trade["cell"]) for trade in self.recent_trades}), 1.1),
            (self.death_marks, self.death_cells, 0.75),
        ):
            artist.set_offsets(self._offsets(points))
            artist.set_sizes([(cell_points * factor) ** 2])
        selected_position = []
        if self.selected_agent is not None and self.selected_agent.id in self.agent_positions:
            selected_position = [self.agent_positions[self.selected_agent.id]]
        self.selection.set_offsets(self._offsets(selected_position))
        self.selection.set_sizes([(cell_points * 0.95) ** 2])

        if not world.agents:
            state = "EXTINCT"
        elif self.finished:
            state = "STEP LIMIT REACHED"
        else:
            state = "PAUSED" if self.paused else "RUNNING"
        self.status.set_text(state)
        self.play_button.label.set_text("Finished" if self.finished else ("Play" if self.paused else "Pause"))
        trade_mode = self.config.trade_matching if self.config.trade_enabled else "off"
        self.world_text.set_text(
            f"Step        {world.t:,}" + (f" / {self.steps:,}" if self.steps else "")
            + f"\nPopulation  {len(world.agents):,}"
            + f"\nBorn / died {world.total_births:,} / {world.total_deaths:,}"
            + f"\nSeed        {self.seed}"
            + f"\nGamma       {self.config.production_specialization_exponent:g}"
            + f"\nTrade       {trade_mode}"
            + f"\nAte / traded this step:"
            + f"\n{len(world.last_recipe_uses)} agents / {len(self.recent_trades)} pairs"
        )
        self._inspect_agent()
        self.figure.canvas.draw_idle()

    def _inspect_agent(self):
        agent = self.selected_agent
        if agent is None:
            self.agent_text.set_text(
                "Click an occupied cell.\n\nWatch its agent's energy,\nstocks and latest action.\n\nSeveral agents can share\na cell. Click again to\ncycle through them."
            )
            return
        alive = agent.id in self.world.agents
        action = self.world.last_actions.get(agent.id)
        labels = {
            ActionKind.MOVE: "Move", ActionKind.HARVEST: "Harvest A",
            ActionKind.PRODUCE: "Produce B/C", ActionKind.REPRODUCE: "Try reproduction",
            ActionKind.USE: "Eat recipe",
        }
        latest = labels[action.kind] if action else ("Newborn" if agent.age == 0 and self.world.t else "No action yet")
        if action and action.kind is ActionKind.USE and agent.id not in self.world.last_recipe_uses:
            latest = "Eat (incomplete recipe)"
        if not alive:
            latest = f"Died at step {self.selected_died_at}"
        partners = []
        for trade in self.recent_trades:
            if agent.id in trade["partner_ids"]:
                partners.extend(str(aid) for aid in trade["partner_ids"] if aid != agent.id)
        self.agent_text.set_text(
            f"#{agent.id}  |  cell ({agent.x}, {agent.y})"
            + f"\nEnergy  {agent.energy:.2f}   Age {agent.age}"
            + f"\nLineage {agent.lineage}   Gen {agent.generation}"
            + f"\nAllocation B {agent.production_allocation:.2f}"
            + f"\nA stock       {agent.stock(0):.2f}"
            + "\n            own  received"
            + f"\nB stock    {agent.self_produced_inventory[0]:5.2f}   {agent.received_inventory[0]:5.2f}"
            + f"\nC stock    {agent.self_produced_inventory[1]:5.2f}   {agent.received_inventory[1]:5.2f}"
            + f"\n{latest}"
            + (f"\nTraded with #{', #'.join(partners)}" if partners else "")
        )

    def on_click(self, event):
        if event.inaxes is not self.grid or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        x, y = int(np.floor(event.xdata + 0.5)), int(np.floor(event.ydata + 0.5))
        agents = sorted(
            (agent for agent in self.world.agents.values() if (agent.x, agent.y) == (x, y)),
            key=lambda agent: agent.id,
        )
        if not agents:
            self.selected_agent = None
        else:
            ids = [agent.id for agent in agents]
            current = self.selected_agent.id if self.selected_agent is not None else None
            index = (ids.index(current) + 1) % len(ids) if current in ids else 0
            self.selected_agent = agents[index]
        self.selected_died_at = None
        self.refresh()

    def on_key(self, event):
        callbacks = {" ": self.toggle_play, "n": self.single_step, "r": self.reset}
        callback = callbacks.get(event.key)
        if callback:
            callback()

    def show(self):
        backend = plt.get_backend().lower()
        if backend in {"agg", "pdf", "pgf", "ps", "svg", "template", "cairo"} or "matplotlib_inline" in backend:
            raise RuntimeError(
                "The live viewer needs a desktop Matplotlib backend (such as TkAgg or QtAgg). "
                "Run it from a local desktop terminal with Tk or Qt installed; "
                f"the current backend is {plt.get_backend()}."
            )
        self.timer.start()
        plt.show()


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="Watch the Hard Times simulation on a live grid.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=positive_integer, help="stop at this timestep (default: until extinction or close)")
    parser.add_argument("--size", type=positive_integer, default=40, help="grid side length")
    parser.add_argument("--agents", type=positive_integer, default=250, help="initial population")
    parser.add_argument("--speed", type=int, choices=range(1, 61), default=10, metavar="1..60", help="target steps per second")
    parser.add_argument("--paused", action="store_true", help="start paused at step zero")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--trade", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trade-matching", choices=("random", "compatible"), default="random")
    parser.add_argument("--deficit-move-probability", type=float, default=0.12)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.agents > Config().max_agents:
        parser.error(f"--agents must not exceed the population cap ({Config().max_agents})")
    try:
        config = Config(
            L=args.size, initial_agents=args.agents,
            production_specialization_exponent=args.gamma,
            trade_enabled=args.trade, trade_matching=args.trade_matching,
            deficit_move_probability=args.deficit_move_probability,
        )
    except ValueError as error:
        parser.error(str(error))
    viewer = LiveViewer(config, seed=args.seed, speed=args.speed, steps=args.steps, paused=args.paused)
    try:
        viewer.show()
    except RuntimeError as error:
        plt.close(viewer.figure)
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
