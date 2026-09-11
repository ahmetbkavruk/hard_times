from __future__ import annotations
from typing import Dict, List
import numpy as np

from .actions import ActionKind
from .agent import Agent
from .config import Config
from .policies import Observation, Policy, RuleBasedPolicy


class World:
    def __init__(self, config: Config, seed: int = 0, policy: Policy | None = None):
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.t = 0
        self.next_agent_id = 0
        self.total_births = 0
        self.total_deaths = 0
        self.policy = policy if policy is not None else RuleBasedPolicy(config)

        self.capacity = self._make_capacities()
        self.resources = self.capacity.copy()
        self.agents: Dict[int, Agent] = {}

        for _ in range(self.cfg.initial_agents):
            self._spawn_founder()

        self.history: List[dict] = []
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
        """Produce a fixed total quantity, split between B and C by ``s``."""
        s = agent.production_allocation
        amount = self.cfg.production_amount
        agent.self_produced_inventory[0] += amount * s
        agent.self_produced_inventory[1] += amount * (1.0 - s)
        agent.energy -= self.cfg.production_cost

    def do_use(self, agent: Agent):
        """Use one complete A/B/C recipe, or make no state change at all."""
        recipe = self.cfg.recipe
        if any(agent.stock(k) < recipe[k] for k in range(3)):
            return False

        agent.inventory[0] -= recipe[0]
        for k in (1, 2):
            needed = recipe[k]
            own = min(needed, agent.self_produced_inventory[k - 1])
            agent.self_produced_inventory[k - 1] -= own
            agent.received_inventory[k - 1] -= needed - own
        agent.energy += self.cfg.recipe_energy_gain
        agent.energy -= self.cfg.use_cost
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

    def step(self):
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

            if action.kind is ActionKind.HARVEST:
                if action.resource is None:
                    raise ValueError("harvest actions require a resource index")
                if action.resource != 0:
                    raise ValueError(f"invalid harvest resource index: {action.resource}")
                self.do_harvest(agent, action.resource)
            elif action.kind is ActionKind.USE:
                self.do_use(agent)
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
        self.record_metrics()

    def record_metrics(self):
        n = len(self.agents)

        if n:
            G = np.stack([a.genotype for a in self.agents.values()])
            energies = np.array([a.energy for a in self.agents.values()], dtype=float)
            ages = np.array([a.age for a in self.agents.values()], dtype=float)
            mean_g = G.mean(axis=0)
            std_g = G.std(axis=0)
            mean_energy = float(energies.mean())
            mean_age = float(ages.mean())
            lineages = len({a.lineage for a in self.agents.values()})
            mean_generation = float(np.mean([a.generation for a in self.agents.values()]))
            max_generation = int(max(a.generation for a in self.agents.values()))
        else:
            mean_g = np.full(2, np.nan)
            std_g = np.full(2, np.nan)
            mean_energy = np.nan
            mean_age = np.nan
            lineages = 0
            mean_generation = np.nan
            max_generation = 0

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
                "mean_production_allocation": float(np.mean(
                    [a.production_allocation for a in self.agents.values()]
                )) if n else np.nan,
                "mean_v": float(mean_g[1]),
                "std_hA": float(std_g[0]),
                "std_production_allocation": float(np.std(
                    [a.production_allocation for a in self.agents.values()]
                )) if n else np.nan,
                "std_v": float(std_g[1]),
            }
        )

    def run(self, steps: int):
        for _ in range(steps):
            if not self.agents:
                break
            self.step()
        return self.history
