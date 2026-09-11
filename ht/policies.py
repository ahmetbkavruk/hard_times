"""Agent policies that choose actions without mutating simulation state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.random import Generator

from .actions import Action, ActionKind
from .config import Config


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


class Policy(Protocol):
    """Interface shared by hand-written and, later, learned controllers."""

    def choose_action(self, observation: Observation, rng: Generator) -> Action:
        """Return a decision without changing the agent or world."""


class RuleBasedPolicy:
    """Local homeostatic rules for production without trade."""

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

        if (bc[0] < cfg.recipe[1] or bc[1] < cfg.recipe[2]):
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
