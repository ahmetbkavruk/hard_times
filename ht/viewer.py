"""A live, read-only view of the existing lattice simulation.

Only advancing the simulation calls World.step(). Drawing, selecting, pausing,
and changing playback speed never draw from the world's random generator.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from math import ceil, isfinite, sqrt

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider
import numpy as np

from .actions import ActionKind
from .config import Config
from .fields import FIELD_CHANNELS
from .social import ALIGNMENT_COLORS, Alignment
from .world import World


BACKGROUND = "#10151b"
PANEL = "#19222c"
TEXT = "#e8edf2"
MUTED = "#9aabbc"
EATING = "#c9ff76"
TRADING = "#f58be0"
FIELD_COLORS = {"movement": "#63cfff", "trade": "#f58be0", "violence": "#ff625c"}
RESOURCE_COLORS = LinearSegmentedColormap.from_list(
    "resource_amount", ["#151c22", "#29443e", "#527b62"]
)
FINITE_RESOURCE_COLORS = LinearSegmentedColormap.from_list(
    "finite_resource_amount", ["#261b18", "#594333", "#927256"]
)
AGENT_COLORS = LinearSegmentedColormap.from_list(
    "production_allocation", ["#50caff", "#ebedf1", "#ffb454"]
)


class LiveViewer:
    def __init__(
        self, config: Config, seed: int = 7, *, speed: int = 10,
        steps: int | None = None, paused: bool = False,
        color_by: str | None = None,
        field_overlay: str | None = None, field_scale: float = 5.0,
    ):
        if not 1 <= speed <= 60:
            raise ValueError("speed must be between 1 and 60 steps per second")
        if steps is not None and steps < 1:
            raise ValueError("steps must be positive")
        self.config = config
        self.seed = seed
        self.steps = steps
        self.paused = paused
        self.color_by = color_by or ("alignment" if config.social_enabled else "production")
        if self.color_by not in ("alignment", "production"):
            raise ValueError("color_by must be 'alignment' or 'production'")
        self.field_overlay = field_overlay or ("movement" if config.fields_enabled else "none")
        if self.field_overlay not in ("none", *FIELD_CHANNELS):
            raise ValueError("unknown field overlay")
        if not config.fields_enabled and self.field_overlay != "none":
            raise ValueError("field overlays require --fields")
        if not isfinite(field_scale) or field_scale <= 0:
            raise ValueError("field scale must be finite and positive")
        self.field_scale = field_scale
        self.world = World(config, seed=seed, record_actions=True)
        self.selected_agent = None
        self.selected_died_at = None
        self.recent_trades = []
        self.recent_interactions = []
        self.eating_cells = []
        self.death_cells = []
        self.agent_positions = {}

        with plt.rc_context({"font.family": "DejaVu Sans", "toolbar": "None"}):
            self.figure = plt.figure(figsize=(14, 10), facecolor=BACKGROUND)
        self.figure.canvas.manager.set_window_title("Hard Times | Live world")
        self.figure.text(0.045, 0.945, "HARD TIMES", color=TEXT, fontsize=23, weight="bold")
        self.figure.text(0.045, 0.908, "LIVE WORLD  /  observe a seeded simulation", color=MUTED, fontsize=10)
        self.status = self.figure.text(0.955, 0.943, "", color=EATING, fontsize=11, ha="right")

        self.grid = self.figure.add_axes([0.035, 0.275, 0.61, 0.61], facecolor=BACKGROUND)
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
        self.field_image = self.grid.imshow(np.zeros((config.L, config.L, 4)), origin="upper", interpolation="nearest", zorder=1)
        self.field_text = self.figure.text(0.045, 0.888, "", color=MUTED, fontsize=8.5)
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
        self.body_marks = self.grid.scatter([], [], marker=".", color="#b7a893", zorder=2)
        self.attack_marks = self.grid.scatter(
            [], [], marker="D", facecolors="none", edgecolors="#ff555f", linewidths=1.3, zorder=5,
        )
        self.predation_marks = self.grid.scatter(
            [], [], marker="^", facecolors="none", edgecolors="#ffe6ad", linewidths=1.4, zorder=5,
        )
        self.flee_marks = self.grid.scatter(
            [], [], marker=">", facecolors="none", edgecolors="#83e8ff", linewidths=1.3, zorder=5,
        )
        self.selection = self.grid.scatter(
            [], [], marker="s", facecolors="none", edgecolors="white",
            linewidths=1.7, zorder=6,
        )

        self.sidebar = self.figure.add_axes([0.675, 0.235, 0.285, 0.65], facecolor=PANEL)
        self.sidebar.set_xticks([])
        self.sidebar.set_yticks([])
        for spine in self.sidebar.spines.values():
            spine.set_visible(False)
        self.sidebar.text(0.065, 0.955, "WORLD", color=MUTED, fontsize=10, weight="bold")
        self.world_text = self.sidebar.text(
            0.065, 0.90, "", va="top", color=TEXT, fontsize=10, linespacing=1.4,
            fontfamily="DejaVu Sans Mono",
        )
        self.sidebar.axhline(0.55, xmin=0.065, xmax=0.935, color="#34424f", linewidth=0.8)
        self.sidebar.text(0.065, 0.515, "AGENT INSPECTOR", color=MUTED, fontsize=10, weight="bold")
        self.agent_text = self.sidebar.text(
            0.065, 0.475, "", va="top", color=TEXT, fontsize=9, linespacing=1.4,
            fontfamily="DejaVu Sans Mono",
        )

        color_axis = self.figure.add_axes([0.12, 0.17, 0.25, 0.012])
        color_axis.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=AGENT_COLORS)
        color_axis.set_axis_off()
        self.production_legend = [color_axis,
            self.figure.text(0.045, 0.17, "C output", color="#50caff", fontsize=9),
            self.figure.text(0.38, 0.17, "B output", color="#ffb454", fontsize=9),
        ]
        self.alignment_legend = self.figure.legend(
            handles=[Line2D([], [], marker="s", linestyle="none", color=ALIGNMENT_COLORS[a],
                            markersize=7, label=a.label) for a in Alignment],
            ncol=3, loc="lower left", bbox_to_anchor=(0.035, 0.135),
            frameon=False, labelcolor=TEXT, fontsize=9, columnspacing=1.5,
        )
        self.figure.text(
            0.045, 0.246, "Ground: green = renewable, brown = finite; brighter = more A",
            color=MUTED, fontsize=9,
        )
        self.resource_text = self.figure.text(0.045, 0.222, "", color=MUTED, fontsize=8.5)
        self.figure.text(0.675, 0.21, "Green square: recipe   /   Pink ring: trade", color=MUTED, fontsize=9)
        self.figure.text(0.675, 0.187, "Red diamond: attack   /   Red cross: death", color=MUTED, fontsize=9)
        self.figure.text(0.675, 0.164, "Blue arrow: flee   /   Pale triangle: ate remains", color=MUTED, fontsize=9)
        self.figure.text(0.675, 0.14, "Highlights: latest step. Small beige dots: remains.", color=MUTED, fontsize=8.5)

        self.play_button = self._button([0.045, 0.06, 0.095, 0.05], "Pause", self.toggle_play)
        self.step_button = self._button([0.15, 0.06, 0.08, 0.05], "Step", self.single_step)
        self.reset_button = self._button([0.24, 0.06, 0.11, 0.05], "Restart seed", self.reset)
        self.color_button = self._button([0.36, 0.06, 0.135, 0.05], "Colors", self.toggle_colors)
        self.field_button = self._button([0.505, 0.06, 0.14, 0.05], "Field", self.toggle_field)
        slider_axis = self.figure.add_axes([0.78, 0.078, 0.125, 0.015], facecolor=PANEL)
        self.speed_slider = Slider(
            slider_axis, "Steps / sec", 1, 60, valinit=speed, valstep=1,
            valfmt="%d", color="#71b9b1", track_color="#34424f",
        )
        self.speed_slider.label.set_color(MUTED)
        self.speed_slider.valtext.set_color(TEXT)
        self.figure.text(
            0.045, 0.023,
            "SPACE  play / pause     N  one step     R  restart     C  colors     F  field overlay     Click a cell to inspect; click again to cycle agents.",
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

    def toggle_colors(self, event=None):
        self.color_by = "production" if self.color_by == "alignment" else "alignment"
        self.refresh()

    def toggle_field(self, event=None):
        if self.world.fields is not None:
            choices = ("none", *FIELD_CHANNELS)
            self.field_overlay = choices[(choices.index(self.field_overlay) + 1) % len(choices)]
        self.refresh()

    def reset(self, event=None):
        self.world = World(self.config, seed=self.seed, record_actions=True)
        self.selected_agent = None
        self.selected_died_at = None
        self.recent_trades = []
        self.recent_interactions = []
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
        interaction_start = len(self.world.interactions)
        self.world.step()
        self.recent_trades = self.world.trades[trade_start:]
        self.recent_interactions = self.world.interactions[interaction_start:]
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
        intensity = np.clip(world.resources[0] / self.resource_image.get_clim()[1], 0, 1)
        colors = RESOURCE_COLORS(intensity)
        colors[~world.renewable_mask] = FINITE_RESOURCE_COLORS(intensity[~world.renewable_mask])
        self.resource_image.set_data(colors)
        self._refresh_field()
        resource_state = world.resource_metrics()
        self.resource_text.set_text(
            f"A remaining: {resource_state['renewable_A_remaining']:,.1f} renewable"
            f"  /  {resource_state['finite_A_remaining']:,.1f} finite"
            f"    Empty finite cells: {resource_state['exhausted_finite_cells']} / {resource_state['finite_cells']}"
        )
        occupancy = defaultdict(list)
        for agent in world.agents.values():
            occupancy[(agent.x, agent.y)].append(agent)
        positions, allocations, alignment_colors, sizes = [], [], [], []
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
                alignment_colors.append(ALIGNMENT_COLORS[agent.alignment])
                sizes.append((0.65 * cell_points / columns) ** 2)
        self.dots.set_offsets(self._offsets(positions))
        if self.color_by == "alignment":
            self.dots.set_array(None)
            self.dots.set_facecolors(alignment_colors)
        else:
            self.dots.set_array(np.asarray(allocations))
        self.alignment_legend.set_visible(self.color_by == "alignment")
        for artist in self.production_legend:
            artist.set_visible(self.color_by == "production")
        self.color_button.label.set_text(f"Colors: {self.color_by}")
        self.dots.set_sizes(sizes)
        for artist, points, factor in (
            (self.eating_marks, self.eating_cells, 0.92),
            (self.trade_marks, sorted({
                cell for trade in self.recent_trades for cell in trade["partner_cells"].values()
            }), 1.1),
            (self.death_marks, self.death_cells, 0.75),
            (self.body_marks, [(b.x, b.y) for b in world.carcasses.values()], 0.45),
            (self.attack_marks, sorted({tuple(event[key]) for event in self.recent_interactions
                                       if event["kind"] == "attack" for key in ("cell", "target_cell")}), 1.1),
            (self.predation_marks, [event["cell"] for event in self.recent_interactions
                                   if event["kind"] == "eat_agent"], 1.1),
            (self.flee_marks, [event["cell"] for event in self.recent_interactions
                              if event["kind"] == "flee"], 1.1),
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
            + f"\nPolicy      {'social' if self.config.social_enabled else 'baseline'}"
            + f"\nAte / traded this step:"
            + f"\n{len(world.last_recipe_uses)} agents / {len(self.recent_trades)} pairs"
            + f"\nAttacks / fled / ate remains: "
            + " / ".join(str(sum(event["kind"] == kind for event in self.recent_interactions))
                         for kind in ("attack", "flee", "eat_agent"))
        )
        self._inspect_agent()
        self.figure.canvas.draw_idle()

    def _refresh_field(self):
        """Use a fixed display scale; switching layers never updates the fields."""
        enabled = self.world.fields is not None
        visible = enabled and self.field_overlay != "none"
        self.field_image.set_visible(visible)
        self.field_button.label.set_text(f"Field: {self.field_overlay}" if enabled else "Fields disabled")
        if not visible:
            self.field_text.set_text("Passive fields recorded; F to show an overlay." if enabled else "Passive fields off; start with --fields to record event traces.")
            self.field_text.set_color(MUTED)
            return
        values = self.world.fields.layer(self.field_overlay)
        colors = np.empty((*values.shape, 4))
        colors[:] = to_rgba(FIELD_COLORS[self.field_overlay])
        colors[:, :, 3] = 0.85 * np.clip(values / self.field_scale, 0, 1)
        self.field_image.set_data(colors)
        self.field_text.set_color(FIELD_COLORS[self.field_overlay])
        self.field_text.set_text(f"{self.field_overlay.title()} trace | full tint at {self.field_scale:g} events/cell | total {values.sum():.1f} | agents cannot sense it")

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
            ActionKind.ATTACK: "Try attack", ActionKind.FLEE: "Flee", ActionKind.EAT_AGENT: "Try eating remains",
        }
        latest = labels[action.kind] if action else ("Newborn" if agent.age == 0 and self.world.t else "No action yet")
        if action and action.kind is ActionKind.USE and agent.id not in self.world.last_recipe_uses:
            latest = "Eat (incomplete recipe)"
        if action and action.target_id is not None:
            if any(e["actor_id"] == agent.id and e["kind"] == action.kind.value for e in self.recent_interactions):
                latest = "Attack" if action.kind is ActionKind.ATTACK else "Ate remains of"
            latest += f" #{action.target_id}"
        if not alive:
            latest = f"Died at step {self.selected_died_at}"
        partners = []
        for trade in self.recent_trades:
            if agent.id in trade["partner_ids"]:
                partners.extend(str(aid) for aid in trade["partner_ids"] if aid != agent.id)
        partner_text = ""
        if partners:
            partner_text = (
                f"\nTraded with #{', #'.join(partners)}" if len(partners) <= 3
                else f"\nTraded with {len(partners)} agents"
            )
        self.agent_text.set_text(
            f"#{agent.id}  |  cell ({agent.x}, {agent.y})"
            + f"\n{agent.alignment.label}"
            + f"\nLawfulness {agent.lawfulness:+.2f}  Concern {agent.concern:+.2f}"
            + f"\nEnergy  {agent.energy:.2f}   Age {agent.age}"
            + f"\nLineage {agent.lineage}   Gen {agent.generation}"
            + f"\nAllocation B {agent.production_allocation:.2f}"
            + f"\nA stock       {agent.stock(0):.2f}"
            + "\n            own  received"
            + f"\nB stock    {agent.self_produced_inventory[0]:5.2f}   {agent.received_inventory[0]:5.2f}"
            + f"\nC stock    {agent.self_produced_inventory[1]:5.2f}   {agent.received_inventory[1]:5.2f}"
            + f"\nGround: {'renewable' if self.world.renewable_mask[agent.y, agent.x] else 'finite'}"
            + f"\nBody {agent.body_energy:.1f}  Memory {len(agent.memory)} contacts"
            + f"\n{latest}"
            + partner_text
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
        callbacks = {" ": self.toggle_play, "n": self.single_step, "r": self.reset, "c": self.toggle_colors, "f": self.toggle_field}
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
    parser.add_argument("--social", action="store_true", help="enable local social decisions, combat and predation")
    parser.add_argument("--fields", action="store_true", help="record passive local event traces (agents cannot sense them)")
    parser.add_argument("--field-decay", type=float, default=0.02, help="fraction of old trace lost per step (0 to 1)")
    parser.add_argument("--field-diffusion", type=float, default=0.2, help="fraction mixed with cardinal neighbors per step (0 to 1)")
    parser.add_argument("--field-overlay", choices=("none", *FIELD_CHANNELS), help="initial overlay; requires --fields (default: movement)")
    parser.add_argument("--field-scale", type=float, default=5.0, help="fixed event weight per cell at full overlay tint")
    parser.add_argument("--color-by", choices=("alignment", "production"), help="initial agent colors (default: alignment with --social)")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--finite-fraction", type=float, default=0.0, help="fraction of cells with nonrenewable A (0 to 1)")
    parser.add_argument("--resource-seed", type=int, help="change the source-type map independently of starting agents")
    parser.add_argument("--base-capacity", type=float, default=8.0, help="base initial A per cell")
    parser.add_argument("--regen-rate", type=float, default=0.03, help="A restored per renewable cell per step")
    parser.add_argument("--trade", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trade-matching", choices=("neighborhood", "random", "compatible"), default="neighborhood")
    parser.add_argument("--deficit-move-probability", type=float, default=0.12)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.agents > Config().max_agents:
        parser.error(f"--agents must not exceed the population cap ({Config().max_agents})")
    try:
        config = Config(
            L=args.size, initial_agents=args.agents,
            finite_resource_fraction=args.finite_fraction, resource_seed=args.resource_seed,
            base_capacity=args.base_capacity, regen_rate=args.regen_rate,
            production_specialization_exponent=args.gamma,
            trade_enabled=args.trade, trade_matching=args.trade_matching,
            deficit_move_probability=args.deficit_move_probability,
            social_enabled=args.social,
            fields_enabled=args.fields, field_decay=args.field_decay, field_diffusion=args.field_diffusion,
        )
        viewer = LiveViewer(config, seed=args.seed, speed=args.speed, steps=args.steps, paused=args.paused, color_by=args.color_by, field_overlay=args.field_overlay, field_scale=args.field_scale)
    except ValueError as error:
        parser.error(str(error))
    try:
        viewer.show()
    except RuntimeError as error:
        plt.close(viewer.figure)
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
