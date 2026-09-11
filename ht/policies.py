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


class Policy(Protocol):
    """Interface shared by hand-written and, later, learned controllers."""

    def choose_action(self, observation: Observation, rng: Generator) -> Action:
        """Return a decision without changing the agent or world."""


class RuleBasedPolicy:
    """Local homeostatic rules used by the ecological control experiment."""

    def __init__(self, config: Config):
        self.config = config

    def choose_action(self, observation: Observation, rng: Generator) -> Action:
        cfg = self.config

        if (
            observation.inventory[0] > 1e-12
            and observation.energy < cfg.reproduce_threshold + cfg.child_energy
        ):
            return Action(ActionKind.EAT)

        if (
            observation.energy >= cfg.reproduce_threshold
            and rng.random() < cfg.reproduce_probability
            and observation.population < cfg.max_agents
        ):
            return Action(ActionKind.REPRODUCE)

        genotype = np.asarray(observation.genotype)
        local_resources = np.asarray(observation.local_resources)
        harvest_weights = genotype[: cfg.num_resources] * local_resources
        move_weight = observation.genotype[3] + cfg.move_exploration
        weights = np.concatenate([harvest_weights, [move_weight]])
        total = float(weights.sum())

        if total <= 0:
            return Action(ActionKind.MOVE)

        choice = int(rng.choice(len(weights), p=weights / total))
        if choice < cfg.num_resources:
            return Action(ActionKind.HARVEST, resource=choice)
        return Action(ActionKind.MOVE)
