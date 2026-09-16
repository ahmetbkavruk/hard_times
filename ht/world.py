from __future__ import annotations
from typing import Dict, List
import numpy as np

from .actions import Action, ActionKind
from .agent import Agent
from .config import Config
from .policies import Observation, Policy, RuleBasedPolicy


class World:
    def __init__(
        self, config: Config, seed: int = 0, policy: Policy | None = None,
        *, record_actions: bool = False,
    ):
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.t = 0
        self.next_agent_id = 0
        self.total_births = 0
        self.total_deaths = 0
        self.policy = policy if policy is not None else RuleBasedPolicy(config)
        # Optional observations of the latest step, never policy inputs. Recording
        # does not consume random draws or change action/event ordering.
        self.record_actions = record_actions
        self.last_actions: Dict[int, Action] = {}
        self.last_recipe_uses: set[int] = set()

        self.capacity = self._make_capacities()
        self.resources = self.capacity.copy()
        self.agents: Dict[int, Agent] = {}

        for _ in range(self.cfg.initial_agents):
            self._spawn_founder()

        self.history: List[dict] = []
        self.trades: List[dict] = []
        self.trade_totals = {
            "trade_events": 0,
            "trade_gross_B": 0.0,
            "trade_gross_C": 0.0,
            "trade_received_B": 0.0,
            "trade_received_C": 0.0,
            "trade_loss_B": 0.0,
            "trade_loss_C": 0.0,
        }
        self.usage_totals = {
            "recipe_uses": 0,
            "recipe_uses_with_received_stock": 0,
            "received_B_used": 0.0,
            "received_C_used": 0.0,
        }
        self.record_metrics()

    def _make_capacities(self) -> np.ndarray:
        L = self.cfg.L
        yy, xx = np.mgrid[0:L, 0:L]

        centers = [
            (0.22 * L, 0.25 * L),
            (0.76 * L, 0.30 * L),
            (0.52 * L, 0.78 * L),
        ]

        sigma = 0.19 * L
        caps = np.zeros((self.cfg.num_resources, L, L), dtype=float)

        for k, (cx, cy) in enumerate(centers[: self.cfg.num_resources]):
            dx = np.minimum(np.abs(xx - cx), L - np.abs(xx - cx))
            dy = np.minimum(np.abs(yy - cy), L - np.abs(yy - cy))
            d2 = dx * dx + dy * dy
            bump = np.exp(-d2 / (2.0 * sigma * sigma))
            caps[k] = self.cfg.base_capacity + self.cfg.capacity_contrast * bump

        return caps

    def _random_genotype(self) -> np.ndarray:
        # The developmental budget now covers only A harvesting and movement.
        g = self.rng.gamma(shape=1.0, scale=1.0, size=2)
        return Agent.normalize_genotype(g, self.cfg.min_trait_norm)

    def _spawn_founder(self):
        aid = self.next_agent_id
        self.next_agent_id += 1
        agent = Agent(
            id=aid,
            x=int(self.rng.integers(self.cfg.L)),
            y=int(self.rng.integers(self.cfg.L)),
            energy=self.cfg.initial_energy,
            inventory=np.zeros(self.cfg.num_resources, dtype=float),
            genotype=self._random_genotype(),
            production_allocation=float(self.rng.random()),
            self_produced_inventory=np.zeros(2, dtype=float),
            received_inventory=np.zeros(2, dtype=float),
            age=0,
            lineage=aid,
            generation=0,
        )
        self.agents[aid] = agent

    def regenerate_resources(self):
        self.resources = np.minimum(
            self.capacity,
            self.resources + self.cfg.regen_rate,
        )

    def local_resources(self, agent: Agent) -> np.ndarray:
        return self.resources[:, agent.y, agent.x]

    def do_harvest(self, agent: Agent, k: int):
        available = self.resources[k, agent.y, agent.x]
        amount = min(available, self.cfg.q_max * agent.trait_for_resource(k))
        self.resources[k, agent.y, agent.x] -= amount
        agent.inventory[k] += amount
        agent.energy -= self.cfg.harvest_cost

    def do_produce(self, agent: Agent):
        """Allocate fixed effort; output is linear only when gamma is one."""
        s = agent.production_allocation
        amount = self.cfg.production_amount
        gamma = self.cfg.production_specialization_exponent
        agent.self_produced_inventory[0] += amount * s**gamma
        agent.self_produced_inventory[1] += amount * (1.0 - s)**gamma
        agent.energy -= self.cfg.production_cost

    def do_use(self, agent: Agent):
        """Use one complete A/B/C recipe, or make no state change at all."""
        recipe = self.cfg.recipe
        if any(agent.stock(k) < recipe[k] for k in range(3)):
            return False

        agent.inventory[0] -= recipe[0]
        used_received_stock = False
        for k, good in ((1, "B"), (2, "C")):
            needed = recipe[k]
            own = min(needed, agent.self_produced_inventory[k - 1])
            agent.self_produced_inventory[k - 1] -= own
            received = needed - own
            agent.received_inventory[k - 1] -= received
            self.usage_totals[f"received_{good}_used"] += received
            used_received_stock = used_received_stock or received > 0.0
        agent.energy += self.cfg.recipe_energy_gain
        agent.energy -= self.cfg.use_cost
        self.usage_totals["recipe_uses"] += 1
        self.usage_totals["recipe_uses_with_received_stock"] += int(used_received_stock)
        return True

    def do_move(self, agent: Agent):
        dx, dy = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(self.rng.integers(4))]
        agent.x = (agent.x + dx) % self.cfg.L
        agent.y = (agent.y + dy) % self.cfg.L
        agent.energy -= self.cfg.move_cost_scale * (1.0 - agent.v)

    def do_reproduce(self, agent: Agent):
        if agent.energy < self.cfg.reproduce_threshold:
            return
        if len(self.agents) >= self.cfg.max_agents:
            return

        total_cost = self.cfg.reproduce_cost + self.cfg.child_energy
        if agent.energy < total_cost:
            return
        agent.energy -= total_cost

        mutation = self.rng.normal(0.0, self.cfg.mutation_sigma, size=2)
        child_g = Agent.normalize_genotype(
            agent.genotype + mutation, self.cfg.min_trait_norm
        )

        aid = self.next_agent_id
        self.next_agent_id += 1

        child = Agent(
            id=aid,
            x=agent.x,
            y=agent.y,
            energy=self.cfg.child_energy,
            inventory=np.zeros(self.cfg.num_resources, dtype=float),
            genotype=child_g,
            production_allocation=Agent.reflect_production_allocation(
                agent.production_allocation
                + self.rng.normal(0.0, self.cfg.allocation_mutation_sigma)
            ),
            self_produced_inventory=np.zeros(2, dtype=float),
            received_inventory=np.zeros(2, dtype=float),
            age=0,
            lineage=agent.lineage,
            generation=agent.generation + 1,
        )
        self.agents[aid] = child
        self.total_births += 1

    def exchange(self) -> None:
        """Exchange among surviving local contacts using current stocks."""
        if not self.cfg.trade_enabled:
            return

        occupancy: Dict[tuple[int, int], List[Agent]] = {}
        for agent in self.agents.values():
            occupancy.setdefault((agent.x, agent.y), []).append(agent)

        if self.cfg.trade_matching == "neighborhood":
            self._exchange_neighborhood(occupancy)
            return

        # Original one-partner rules remain explicit experimental controls.
        for cell, occupants in occupancy.items():
            self.rng.shuffle(occupants)
            if self.cfg.trade_matching == "random":
                pairs = (
                    occupants[start : start + 2]
                    for start in range(0, len(occupants) - 1, 2)
                )
            else:
                # Every eligible B donor can trade with every eligible C donor.
                # Filtering the shuffled order gives random priority within each
                # side without spending matches on incompatible bystanders.
                recipe_b, recipe_c = self.cfg.recipe[1:]
                b_donors, c_donors = [], []
                for agent in occupants:
                    if (
                        agent.self_produced_inventory[0] > recipe_b
                        and agent.stock(2) < recipe_c
                    ):
                        b_donors.append(agent)
                    elif (
                        agent.self_produced_inventory[1] > recipe_c
                        and agent.stock(1) < recipe_b
                    ):
                        c_donors.append(agent)
                pairs = zip(b_donors, c_donors)

            for first, second in pairs:
                self._exchange_pair(first, second, cell)

    def _exchange_neighborhood(self, occupancy) -> None:
        """Visit each same-cell/cardinal-neighbor pair once, with many partners."""
        agents = list(self.agents.values())
        self.rng.shuffle(agents)
        visited = set()
        length = self.cfg.L
        for first in agents:
            visited.add(first.id)
            x, y = first.x, first.y
            # Deduplicate wrapped cells, including on grids of side 1 or 2.
            cells = sorted({
                (x, y), ((x - 1) % length, y), ((x + 1) % length, y),
                (x, (y - 1) % length), (x, (y + 1) % length),
            })
            partners = [
                second for cell in cells for second in occupancy.get(cell, ())
                if second.id not in visited
            ]
            self.rng.shuffle(partners)
            for second in partners:
                # Earlier exchanges can remove surplus or fill a deficit.
                # _exchange_pair recalculates both from the current inventory.
                self._exchange_pair(first, second, (x, y))

    def _exchange_pair(
        self, first: Agent, second: Agent, cell: tuple[int, int]
    ) -> bool:
        """Exchange B from one partner for C from the other, if complementary."""
        recipe_b, recipe_c = self.cfg.recipe[1:]

        def quantities(b_donor: Agent, c_donor: Agent) -> float:
            export_b = max(0.0, b_donor.self_produced_inventory[0] - recipe_b)
            export_c = max(0.0, c_donor.self_produced_inventory[1] - recipe_c)
            need_c = max(0.0, recipe_c - b_donor.stock(2))
            need_b = max(0.0, recipe_b - c_donor.stock(1))
            return min(
                export_b,
                export_c,
                need_c / self.cfg.trade_efficiency,
                need_b / self.cfg.trade_efficiency,
                self.cfg.max_trade_per_pair,
            )

        q = quantities(first, second)
        if q > 0.0:
            b_donor, c_donor = first, second
        else:
            q = quantities(second, first)
            if q <= 0.0:
                return False
            b_donor, c_donor = second, first

        net = self.cfg.trade_efficiency * q
        loss = (1.0 - self.cfg.trade_efficiency) * q
        # All limits have been established before either partner is mutated.
        # Debiting only self-produced stock also prevents receipt re-export.
        b_donor.self_produced_inventory[0] -= q
        c_donor.self_produced_inventory[1] -= q
        b_donor.received_inventory[1] += net
        c_donor.received_inventory[0] += net

        self.trades.append(
            {
                "t": self.t,
                "cell": cell,
                "partner_ids": (first.id, second.id),
                "partner_cells": {
                    first.id: (first.x, first.y),
                    second.id: (second.x, second.y),
                },
                "surrendered": {
                    b_donor.id: {"good": "B", "gross": q},
                    c_donor.id: {"good": "C", "gross": q},
                },
                "received": {
                    b_donor.id: {"good": "C", "net": net},
                    c_donor.id: {"good": "B", "net": net},
                },
                "transaction_loss": {"B": loss, "C": loss},
            }
        )
        self.trade_totals["trade_events"] += 1
        for good in ("B", "C"):
            self.trade_totals[f"trade_gross_{good}"] += q
            self.trade_totals[f"trade_received_{good}"] += net
            self.trade_totals[f"trade_loss_{good}"] += loss
        return True

    def step(self):
        self.last_actions.clear()
        self.last_recipe_uses.clear()
        self.regenerate_resources()

        ids = np.array(list(self.agents.keys()), dtype=int)
        self.rng.shuffle(ids)

        for aid in ids:
            agent = self.agents.get(int(aid))
            if agent is None:
                continue

            observation = Observation(
                energy=agent.energy,
                inventory=tuple(agent.inventory),
                genotype=tuple(agent.genotype),
                production_allocation=agent.production_allocation,
                self_produced_inventory=tuple(agent.self_produced_inventory),
                received_inventory=tuple(agent.received_inventory),
                local_resources=tuple(self.local_resources(agent)),
                population=len(self.agents),
            )
            action = self.policy.choose_action(observation, self.rng)
            if self.record_actions:
                self.last_actions[agent.id] = action

            if action.kind is ActionKind.HARVEST:
                if action.resource is None:
                    raise ValueError("harvest actions require a resource index")
                if action.resource != 0:
                    raise ValueError(f"invalid harvest resource index: {action.resource}")
                self.do_harvest(agent, action.resource)
            elif action.kind is ActionKind.USE:
                used = self.do_use(agent)
                if used and self.record_actions:
                    self.last_recipe_uses.add(agent.id)
            elif action.kind is ActionKind.PRODUCE:
                self.do_produce(agent)
            elif action.kind is ActionKind.MOVE:
                self.do_move(agent)
            elif action.kind is ActionKind.REPRODUCE:
                self.do_reproduce(agent)

            agent.energy -= self.cfg.metabolism
            agent.age += 1

        dead = [aid for aid, a in self.agents.items() if a.energy <= 0.0]
        for aid in dead:
            del self.agents[aid]
        self.total_deaths += len(dead)

        self.t += 1
        self.exchange()
        self.record_metrics()

    def record_metrics(self):
        n = len(self.agents)

        if n:
            G = np.stack([a.genotype for a in self.agents.values()])
            allocations = np.array([a.production_allocation for a in self.agents.values()])
            energies = np.array([a.energy for a in self.agents.values()], dtype=float)
            ages = np.array([a.age for a in self.agents.values()], dtype=float)
            mean_g = G.mean(axis=0)
            std_g = G.std(axis=0)
            mean_energy = float(energies.mean())
            mean_age = float(ages.mean())
            lineages = len({a.lineage for a in self.agents.values()})
            mean_generation = float(np.mean([a.generation for a in self.agents.values()]))
            max_generation = int(max(a.generation for a in self.agents.values()))
            total_b_stock = float(sum(a.stock(1) for a in self.agents.values()))
            total_c_stock = float(sum(a.stock(2) for a in self.agents.values()))
        else:
            mean_g = np.full(2, np.nan)
            std_g = np.full(2, np.nan)
            mean_energy = np.nan
            mean_age = np.nan
            lineages = 0
            mean_generation = np.nan
            max_generation = 0
            total_b_stock = 0.0
            total_c_stock = 0.0

        self.history.append(
            {
                "t": self.t,
                "population": n,
                "births": self.total_births,
                "deaths": self.total_deaths,
                "mean_energy": mean_energy,
                "mean_age": mean_age,
                "lineages": lineages,
                "mean_generation": mean_generation,
                "max_generation": max_generation,
                "mean_hA": float(mean_g[0]),
                "mean_production_allocation": float(allocations.mean()) if n else np.nan,
                # Descriptive endpoint bias, not a measure of dependence on trade.
                "mean_specialization": float(np.mean(np.abs(2 * allocations - 1)))
                if n else np.nan,
                "mean_v": float(mean_g[1]),
                "std_hA": float(std_g[0]),
                "std_production_allocation": float(allocations.std()) if n else np.nan,
                "std_v": float(std_g[1]),
                "total_B_stock": total_b_stock,
                "total_C_stock": total_c_stock,
                **{f"cumulative_{key}": value for key, value in self.trade_totals.items()},
                **{f"cumulative_{key}": value for key, value in self.usage_totals.items()},
            }
        )

    def run(self, steps: int):
        for _ in range(steps):
            if not self.agents:
                break
            self.step()
        return self.history
