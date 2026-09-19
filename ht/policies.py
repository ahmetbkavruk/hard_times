"""Agent policies that choose actions without mutating simulation state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.random import Generator

from .actions import Action, ActionKind
from .config import Config


@dataclass(frozen=True)
class Neighbor:
    id: int
    dx: int
    dy: int
    energy: float
    body_energy: float
    self_produced_inventory: tuple[float, float]
    received_inventory: tuple[float, float]
    recent_trades: int = 0
    harmed_me: bool = False

    def stock(self, good: int) -> float:
        return self.self_produced_inventory[good - 1] + self.received_inventory[good - 1]


@dataclass(frozen=True)
class NearbyBody:
    agent_id: int
    dx: int
    dy: int
    edible_energy: float


@dataclass(frozen=True)
class TradeOffer:
    partner: Neighbor
    give_good: int
    receive_good: int
    gross: float
    net: float


@dataclass(frozen=True)
class Observation:
    """The portion of world state currently exposed to a policy."""

    energy: float
    inventory: tuple[float, ...]
    genotype: tuple[float, ...]
    local_resources: tuple[float, ...]
    population: int
    production_allocation: float = 0.5
    self_produced_inventory: tuple[float, float] = (0.0, 0.0)
    received_inventory: tuple[float, float] = (0.0, 0.0)
    lawfulness: float = 0.0
    concern: float = 0.0
    neighbors: tuple[Neighbor, ...] = ()
    # (dx, dy, amount of A) in the four immediately adjacent cells.
    adjacent_resources: tuple[tuple[int, int, float], ...] = ()
    bodies: tuple[NearbyBody, ...] = ()
    last_direction: tuple[int, int] | None = None
    grid_size: int = 1


class Policy(Protocol):
    """Interface shared by hand-written and, later, learned controllers."""

    def choose_action(self, observation: Observation, rng: Generator) -> Action:
        """Return a decision without changing the agent or world."""


class RuleBasedPolicy:
    """Local homeostatic rules shared by trade and no-trade conditions."""

    def __init__(self, config: Config):
        self.config = config

    def choose_action(self, observation: Observation, rng: Generator) -> Action:
        cfg = self.config

        bc = np.asarray(observation.self_produced_inventory) + np.asarray(
            observation.received_inventory
        )
        has_recipe = (
            observation.inventory[0] >= cfg.recipe[0]
            and bc[0] >= cfg.recipe[1]
            and bc[1] >= cfg.recipe[2]
        )
        if has_recipe and observation.energy < cfg.reproduce_threshold + cfg.child_energy:
            return Action(ActionKind.USE)

        if bc[0] < cfg.recipe[1] or bc[1] < cfg.recipe[2]:
            # A specialist cannot manufacture its missing good at an endpoint.
            # Permit exploration without observing partners or granting trade
            # experiments a different controller. Zero restores the old rule.
            if (
                cfg.deficit_move_probability > 0.0
                and rng.random() < cfg.deficit_move_probability
            ):
                return Action(ActionKind.MOVE)
            return Action(ActionKind.PRODUCE)

        if (
            observation.energy >= cfg.reproduce_threshold
            and rng.random() < cfg.reproduce_probability
            and observation.population < cfg.max_agents
        ):
            return Action(ActionKind.REPRODUCE)

        genotype = np.asarray(observation.genotype)
        local_resources = np.asarray(observation.local_resources)
        # B and C are manufactured, so only A participates in harvesting.
        harvest_weights = genotype[:1] * local_resources[:1]
        move_weight = observation.genotype[1] + cfg.move_exploration
        weights = np.concatenate([harvest_weights, [move_weight]])
        total = float(weights.sum())

        if total <= 0:
            return Action(ActionKind.MOVE)

        choice = int(rng.choice(len(weights), p=weights / total))
        if choice < 1:
            return Action(ActionKind.HARVEST, resource=0)
        return Action(ActionKind.MOVE)
