from __future__ import annotations
from collections import Counter
from typing import Dict, List
import numpy as np

from .actions import Action, ActionKind
from .agent import Agent
from .config import Config
from .fields import FieldSystem
from .policies import Neighbor, NearbyBody, Observation, Policy, RuleBasedPolicy, TradeOffer
from .social import Alignment, Carcass, Encounter, reflect_trait
from .social_policy import DIRECTIONS, SocialPolicy


class World:
    def __init__(
        self, config: Config, seed: int = 0, policy: Policy | None = None,
        *, record_actions: bool = False,
    ):
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self.trait_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(2,)))
        self.seed = seed
        self.t = 0
        self.next_agent_id = 0
        self.total_births = 0
        self.total_deaths = 0
        self.policy = policy if policy is not None else (
            SocialPolicy(config) if config.social_enabled else RuleBasedPolicy(config)
        )
        self._acting = False
        self._occupancy = None
        self.carcasses: Dict[int, Carcass] = {}
        self._body_cells: Dict[tuple[int, int], set[int]] = {}
        self.interactions: List[dict] = []
        self.social_totals = {
            "attacks": 0, "combat_deaths": 0, "fleeing": 0, "predation_events": 0,
            "trade_refusals": 0, "attack_energy_cost": 0.0, "combat_energy_damage": 0.0,
            "flee_energy_cost": 0.0, "trade_energy_cost": 0.0,
            "body_energy_born": 0.0, "body_energy_eaten": 0.0,
            "predation_energy_received": 0.0, "predation_energy_loss": 0.0,
            "eating_energy_cost": 0.0,
        }
        # Optional observations of the latest step, never policy inputs. Recording
        # does not consume random draws or change action/event ordering.
        self.record_actions = record_actions
        self.last_actions: Dict[int, Action] = {}
        self.last_recipe_uses: set[int] = set()
        self.fields = FieldSystem(config.L, decay=config.field_decay, diffusion=config.field_diffusion) if config.fields_enabled else None

        self.capacity = self._make_capacities()
        self.resources = self.capacity.copy()
        self.renewable_mask = self._make_renewable_mask()
        self.initial_finite_A = float(self.resources[0, ~self.renewable_mask].sum())
        self.cumulative_finite_A_harvested = 0.0
        self.agents: Dict[int, Agent] = {}

        for _ in range(self.cfg.initial_agents):
            self._spawn_founder()
        self.initial_body_energy = sum(agent.body_energy for agent in self.agents.values())
        self.founder_traits = self.agent_traits()

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

    def _make_renewable_mask(self) -> np.ndarray:
        """Make clustered source types without consuming the agent RNG stream."""
        length = self.cfg.L
        finite_count = int(round(self.cfg.finite_resource_fraction * length * length))
        mask = np.ones((length, length), dtype=bool)
        if finite_count == length * length:
            mask[:] = False
        elif finite_count:
            seed = self.seed if self.cfg.resource_seed is None else self.cfg.resource_seed
            map_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
            field = map_rng.random((length, length))
            # Smooth across periodic neighbors to form patches rather than
            # independently scattered cells. This only constructs the initial map.
            for _ in range(4):
                field = (
                    field + np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0)
                    + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)
                ) / 5
            indices = np.argsort(field, axis=None, kind="stable")[:finite_count]
            mask.flat[indices] = False
        mask.setflags(write=False)
        return mask

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
            lawfulness=float(self.trait_rng.uniform(-1, 1)),
            concern=float(self.trait_rng.uniform(-1, 1)),
            body_energy=self.cfg.body_energy if self.cfg.social_enabled else 0.0,
        )
        self.agents[aid] = agent

    def regenerate_resources(self):
        self.resources = np.minimum(
            self.capacity,
            self.resources + self.cfg.regen_rate * self.renewable_mask,
        )

    def local_resources(self, agent: Agent) -> np.ndarray:
        return self.resources[:, agent.y, agent.x]

    def do_harvest(self, agent: Agent, k: int):
        available = self.resources[k, agent.y, agent.x]
        amount = min(available, self.cfg.q_max * agent.trait_for_resource(k))
        self.resources[k, agent.y, agent.x] -= amount
        agent.inventory[k] += amount
        if k == 0 and not self.renewable_mask[agent.y, agent.x]:
            self.cumulative_finite_A_harvested += amount
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

    def do_move(self, agent: Agent, direction=None, *, fleeing=False):
        if direction is not None and direction not in DIRECTIONS:
            raise ValueError("movement requires a cardinal one-cell direction")
        if fleeing and not self.cfg.social_enabled:
            return False
        dx, dy = direction if direction is not None else DIRECTIONS[int(self.rng.integers(4))]
        previous = (agent.x, agent.y)
        agent.x = (agent.x + dx) % self.cfg.L
        agent.y = (agent.y + dy) % self.cfg.L
        if self.fields is not None and previous != (agent.x, agent.y):
            self.fields.deposit("movement", (agent.x, agent.y))
        agent.last_direction = (dx, dy)
        if self._occupancy is not None:
            self._occupancy[previous].discard(agent.id)
            self._occupancy.setdefault((agent.x, agent.y), set()).add(agent.id)
        agent.energy -= self.cfg.move_cost_scale * (1.0 - agent.v)
        if fleeing:
            agent.energy -= self.cfg.flee_cost
            self.social_totals["fleeing"] += 1
            self.social_totals["flee_energy_cost"] += self.cfg.flee_cost
            self.interactions.append({
                "t": self.event_time, "kind": "flee", "actor_id": agent.id,
                "from_cell": previous, "cell": (agent.x, agent.y),
                "energy_cost": self.cfg.move_cost_scale * (1 - agent.v) + self.cfg.flee_cost,
            })
        return True

    def do_reproduce(self, agent: Agent):
        if agent.energy < self.cfg.reproduce_threshold:
            return
        if len(self.agents) >= self.cfg.max_agents:
            return

        body_energy = self.cfg.body_energy if self.cfg.social_enabled else 0.0
        total_cost = self.cfg.reproduce_cost + self.cfg.child_energy + body_energy
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
            lawfulness=reflect_trait(agent.lawfulness + self.trait_rng.normal(0, self.cfg.social_mutation_sigma)),
            concern=reflect_trait(agent.concern + self.trait_rng.normal(0, self.cfg.social_mutation_sigma)),
            body_energy=body_energy,
        )
        self.agents[aid] = child
        if self._occupancy is not None:
            self._occupancy.setdefault((child.x, child.y), set()).add(aid)
        self.total_births += 1
        self.social_totals["body_energy_born"] += body_energy
        if self.cfg.social_enabled:
            self.interactions.append({
                "t": self.event_time, "kind": "birth", "actor_id": agent.id, "target_id": child.id,
                "cell": (child.x, child.y), "parent_energy_cost": total_cost,
                "child_energy": child.energy, "body_energy": body_energy,
                "dissipated_energy": self.cfg.reproduce_cost, "lineage": child.lineage,
                "generation": child.generation, "lawfulness": child.lawfulness,
                "concern": child.concern, "alignment": child.alignment.value,
            })

    @property
    def event_time(self):
        return self.t + int(self._acting)

    def _offset(self, origin, destination):
        length = self.cfg.L
        offset = (destination - origin) % length
        return offset if offset <= length // 2 else offset - length

    def _in_contact(self, first, second):
        return abs(self._offset(first.x, second.x)) + abs(self._offset(first.y, second.y)) <= 1

    def _local_cells(self, agent):
        radius = min(self.cfg.perception_radius, self.cfg.L)
        return sorted({
            ((agent.x + dx) % self.cfg.L, (agent.y + dy) % self.cfg.L)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1) if abs(dx) + abs(dy) <= radius
        })

    def _neighbor(self, observer, other):
        memory = observer.memory.get(other.id, Encounter())
        cutoff = self.event_time - self.cfg.memory_duration
        return Neighbor(
            id=other.id, dx=self._offset(observer.x, other.x), dy=self._offset(observer.y, other.y),
            energy=other.energy, body_energy=other.body_energy,
            self_produced_inventory=tuple(other.self_produced_inventory),
            received_inventory=tuple(other.received_inventory),
            recent_trades=memory.trades if memory.last_trade >= 0 and memory.last_trade > cutoff else 0,
            harmed_me=memory.last_harm >= 0 and memory.last_harm > cutoff,
        )

    def observe(self, agent: Agent) -> Observation:
        """Copy local state into immutable values, exposing neither labels nor a map."""
        neighbors, bodies, adjacent = (), (), ()
        if self.cfg.social_enabled:
            occupancy = self._occupancy
            if occupancy is None:
                occupancy = {}
                for other in self.agents.values():
                    occupancy.setdefault((other.x, other.y), set()).add(other.id)
            cells = self._local_cells(agent)
            neighbors = tuple(
                self._neighbor(agent, self.agents[aid])
                for cell in cells for aid in sorted(occupancy.get(cell, ()))
                if aid != agent.id and aid in self.agents and self.agents[aid].energy > 0
            )
            bodies = tuple(
                NearbyBody(aid, self._offset(agent.x, cell[0]), self._offset(agent.y, cell[1]),
                           self.carcasses[aid].edible_energy)
                for cell in cells for aid in sorted(self._body_cells.get(cell, ()))
            )
            adjacent = tuple((dx, dy, float(self.resources[0, (agent.y + dy) % self.cfg.L,
                                                             (agent.x + dx) % self.cfg.L]))
                             for dx, dy in DIRECTIONS)
        return Observation(
            energy=agent.energy, inventory=tuple(agent.inventory), genotype=tuple(agent.genotype),
            production_allocation=agent.production_allocation,
            self_produced_inventory=tuple(agent.self_produced_inventory),
            received_inventory=tuple(agent.received_inventory),
            local_resources=tuple(self.local_resources(agent)), population=len(self.agents),
            lawfulness=agent.lawfulness, concern=agent.concern, neighbors=neighbors,
            adjacent_resources=adjacent, bodies=bodies, last_direction=agent.last_direction,
            grid_size=self.cfg.L,
        )

    def _prune_memory(self, agent):
        cutoff = self.event_time - self.cfg.memory_duration
        agent.memory = {aid: memory for aid, memory in agent.memory.items() if memory.last_event > cutoff}
        while len(agent.memory) > self.cfg.memory_capacity:
            oldest = min(agent.memory, key=lambda aid: (agent.memory[aid].last_event, aid))
            del agent.memory[oldest]

    def _remember(self, agent, other_id, *, harmed=False):
        self._prune_memory(agent)
        old = agent.memory.get(other_id, Encounter())
        recent_trades = old.trades if old.last_trade > self.event_time - self.cfg.memory_duration else 0
        agent.memory[other_id] = (
            Encounter(old.last_trade, self.event_time, old.trades) if harmed
            else Encounter(self.event_time, old.last_harm, recent_trades + 1)
        )
        self._prune_memory(agent)

    def _remove_dead(self, *, killed_by=None, candidates=None):
        if candidates is None:
            candidates = self.agents.values()
        dead = [agent for agent in candidates if agent.energy <= 0 and agent.id in self.agents]
        for agent in dead:
            del self.agents[agent.id]
            if self._occupancy is not None:
                self._occupancy[(agent.x, agent.y)].discard(agent.id)
            if self.cfg.social_enabled:
                if agent.body_energy > 0:
                    self.carcasses[agent.id] = Carcass(agent.id, agent.x, agent.y, agent.body_energy)
                    self._body_cells.setdefault((agent.x, agent.y), set()).add(agent.id)
                self.interactions.append({
                    "t": self.event_time, "kind": "death", "actor_id": agent.id,
                    "cell": (agent.x, agent.y), "killed_by": (killed_by or {}).get(agent.id),
                    "body_energy": agent.body_energy,
                    "discarded_inventory": [agent.stock(k) for k in range(3)],
                })
                agent.body_energy = 0.0
            self.total_deaths += 1

    def do_attack(self, agent: Agent, target_id: int | None) -> bool:
        target = self.agents.get(target_id)
        if (not self.cfg.social_enabled or self.agents.get(agent.id) is not agent or target is None
                or target is agent or target.energy <= 0 or agent.energy <= self.cfg.attack_cost
                or self.cfg.attack_damage <= 0 or not self._in_contact(agent, target)):
            return False
        damage = min(target.energy, self.cfg.attack_damage * (0.5 + agent.hA))
        agent.energy -= self.cfg.attack_cost
        target.energy -= damage
        if self.fields is not None:
            self.fields.deposit("violence", (target.x, target.y))
        self._remember(target, agent.id, harmed=True)
        killed = target.energy <= 0
        self.social_totals["attacks"] += 1
        self.social_totals["combat_deaths"] += int(killed)
        self.social_totals["attack_energy_cost"] += self.cfg.attack_cost
        self.social_totals["combat_energy_damage"] += damage
        self.interactions.append({
            "t": self.event_time, "kind": "attack", "actor_id": agent.id, "target_id": target.id,
            "cell": (agent.x, agent.y), "target_cell": (target.x, target.y),
            "energy_cost": self.cfg.attack_cost, "damage": damage, "killed": bool(killed),
        })
        self._remove_dead(killed_by={target.id: agent.id} if killed else None, candidates=(target,))
        return True

    def do_eat_agent(self, agent: Agent, target_id: int | None) -> bool:
        body = self.carcasses.get(target_id)
        if (not self.cfg.social_enabled or self.agents.get(agent.id) is not agent or body is None
                or agent.energy <= self.cfg.eating_cost or not self._in_contact(agent, body)):
            return False
        gross = min(body.edible_energy, self.cfg.bite_energy)
        if gross <= 0:
            return False
        net = self.cfg.predation_efficiency * gross
        body.edible_energy -= gross
        agent.energy += net - self.cfg.eating_cost
        self.social_totals["predation_events"] += 1
        self.social_totals["body_energy_eaten"] += gross
        self.social_totals["predation_energy_received"] += net
        self.social_totals["predation_energy_loss"] += gross - net
        self.social_totals["eating_energy_cost"] += self.cfg.eating_cost
        self.interactions.append({
            "t": self.event_time, "kind": "eat_agent", "actor_id": agent.id, "target_id": target_id,
            "cell": (agent.x, agent.y), "target_cell": (body.x, body.y),
            "gross_energy": gross, "received_energy": net, "loss": gross - net,
            "energy_cost": self.cfg.eating_cost,
        })
        if body.edible_energy <= 0:
            del self.carcasses[target_id]
            self._body_cells[(body.x, body.y)].discard(target_id)
            if not self._body_cells[(body.x, body.y)]:
                del self._body_cells[(body.x, body.y)]
        return True

    def agent_traits(self):
        return [dict(id=a.id, lineage=a.lineage, generation=a.generation,
                     lawfulness=a.lawfulness, concern=a.concern, alignment=a.alignment.value)
                for a in self.agents.values()]

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
        if self.cfg.social_enabled:
            consent = getattr(self.policy, "willing_to_trade", None)
            if consent is None or min(first.energy, second.energy) <= self.cfg.trade_energy_cost:
                return False
            b_offer = TradeOffer(self._neighbor(b_donor, c_donor), 1, 2, q, net)
            c_offer = TradeOffer(self._neighbor(c_donor, b_donor), 2, 1, q, net)
            b_agrees = consent(self.observe(b_donor), b_offer, self.rng)
            c_agrees = consent(self.observe(c_donor), c_offer, self.rng)
            if not (b_agrees and c_agrees):
                self.social_totals["trade_refusals"] += 1
                self.interactions.append({
                    "t": self.event_time, "kind": "trade_refused", "actor_id": b_donor.id,
                    "target_id": c_donor.id, "consented": [bool(b_agrees), bool(c_agrees)],
                    "cell": (b_donor.x, b_donor.y), "target_cell": (c_donor.x, c_donor.y),
                })
                return False
            first.energy -= self.cfg.trade_energy_cost
            second.energy -= self.cfg.trade_energy_cost
            self.social_totals["trade_energy_cost"] += 2 * self.cfg.trade_energy_cost
            self._remember(first, second.id)
            self._remember(second, first.id)
        # All limits have been established before either partner is mutated.
        # Debiting only self-produced stock also prevents receipt re-export.
        b_donor.self_produced_inventory[0] -= q
        c_donor.self_produced_inventory[1] -= q
        b_donor.received_inventory[1] += net
        c_donor.received_inventory[0] += net
        if self.fields is not None:
            self.fields.deposit("trade", (first.x, first.y), 0.5)
            self.fields.deposit("trade", (second.x, second.y), 0.5)

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
                "energy_cost_per_agent": self.cfg.trade_energy_cost if self.cfg.social_enabled else 0.0,
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
        self._acting = True
        if self.cfg.social_enabled:
            self._occupancy = {}
            for agent in self.agents.values():
                self._occupancy.setdefault((agent.x, agent.y), set()).add(agent.id)
                self._prune_memory(agent)
        try:
            self._act_agents(ids)
            self._remove_dead()
            self.t += 1
            self._acting = False
            self.exchange()
            if self.fields is not None:
                self.fields.advance()
            self.record_metrics()
        finally:
            self._acting = False
            self._occupancy = None

    def _act_agents(self, ids):
        for aid in ids:
            agent = self.agents.get(int(aid))
            if agent is None:
                continue
            if self.cfg.social_enabled and agent.energy <= 0:
                self._remove_dead(candidates=(agent,))
                continue
            observation = self.observe(agent)
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
                if action.direction is None:
                    self.do_move(agent)
                else:
                    self.do_move(agent, action.direction)
            elif action.kind is ActionKind.REPRODUCE:
                self.do_reproduce(agent)
            elif action.kind is ActionKind.ATTACK:
                self.do_attack(agent, action.target_id)
            elif action.kind is ActionKind.FLEE:
                self.do_move(agent, action.direction, fleeing=True)
            elif action.kind is ActionKind.EAT_AGENT:
                self.do_eat_agent(agent, action.target_id)

            agent.energy -= self.cfg.metabolism
            agent.age += 1
            if self.cfg.social_enabled:
                self._remove_dead(candidates=(agent,))

    def resource_metrics(self) -> dict:
        """Observe remaining ground stock, independent of agents' inventories."""
        finite = ~self.renewable_mask
        stock = self.resources[0]
        return {
            "renewable_A_remaining": float(stock[self.renewable_mask].sum()),
            "finite_A_remaining": float(stock[finite].sum()),
            "initial_finite_A": self.initial_finite_A,
            "finite_cells": int(finite.sum()),
            "exhausted_finite_cells": int(np.count_nonzero(finite & (stock <= 0))),
            "cumulative_finite_A_harvested": self.cumulative_finite_A_harvested,
        }

    def record_metrics(self):
        n = len(self.agents)
        alignment_counts = Counter(a.alignment for a in self.agents.values())

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
                **self.resource_metrics(),
                **{f"cumulative_{key}": value for key, value in self.trade_totals.items()},
                **{f"cumulative_{key}": value for key, value in self.usage_totals.items()},
                **{f"cumulative_{key}": value for key, value in self.social_totals.items()},
                "initial_body_energy": self.initial_body_energy,
                "living_body_energy": sum(a.body_energy for a in self.agents.values()),
                "carcass_energy": sum(b.edible_energy for b in self.carcasses.values()),
                "mean_lawfulness": float(np.mean([a.lawfulness for a in self.agents.values()])) if n else np.nan,
                "mean_concern": float(np.mean([a.concern for a in self.agents.values()])) if n else np.nan,
                **{f"population_{alignment.value}": alignment_counts[alignment]
                   for alignment in Alignment},
            }
        )

    def run(self, steps: int):
        for _ in range(steps):
            if not self.agents:
                break
            self.step()
        return self.history
