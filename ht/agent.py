from __future__ import annotations
from dataclasses import dataclass, field
from math import isfinite
import numpy as np

from .social import Alignment, Encounter, alignment_for

@dataclass
class Agent:
    id: int
    x: int
    y: int
    energy: float
    inventory: np.ndarray
    genotype: np.ndarray
    production_allocation: float
    self_produced_inventory: np.ndarray
    received_inventory: np.ndarray
    age: int = 0
    lineage: int = -1
    generation: int = 0
    lawfulness: float = 0.0
    concern: float = 0.0
    body_energy: float = 0.0
    memory: dict[int, Encounter] = field(default_factory=dict)
    last_direction: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        # Allocation is a scalar trait rather than part of the normalized
        # harvesting/movement budget.
        self.production_allocation = float(
            np.clip(self.production_allocation, 0.0, 1.0)
        )
        alignment_for(self.lawfulness, self.concern)
        if not isfinite(self.body_energy) or self.body_energy < 0:
            raise ValueError("body_energy must be finite and nonnegative")

    @property
    def alignment(self) -> Alignment:
        return alignment_for(self.lawfulness, self.concern)

    @property
    def hA(self) -> float:
        return float(self.genotype[0])

    @property
    def v(self) -> float:
        return float(self.genotype[1])

    def trait_for_resource(self, k: int) -> float:
        if k != 0:
            raise ValueError("only environmental resource A can be harvested")
        return self.hA

    def stock(self, k: int) -> float:
        """Return usable stock, keeping produced and received B/C auditable."""
        if k == 0:
            return float(self.inventory[0])
        return float(self.self_produced_inventory[k - 1] + self.received_inventory[k - 1])

    @staticmethod
    def normalize_genotype(g: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        g = np.clip(np.asarray(g, dtype=float), 0.0, None)
        norm = np.linalg.norm(g)
        if norm < eps:
            g = np.ones_like(g)
            norm = np.linalg.norm(g)
        return g / norm

    @staticmethod
    def reflect_production_allocation(value: float) -> float:
        """Reflect an arbitrary real value into the closed interval [0, 1]."""
        folded = float(value) % 2.0
        return folded if folded <= 1.0 else 2.0 - folded
