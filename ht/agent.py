from __future__ import annotations
from dataclasses import dataclass
import numpy as np

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

    def __post_init__(self) -> None:
        # Allocation is a scalar trait rather than part of the normalized
        # harvesting/movement budget.
        self.production_allocation = float(
            np.clip(self.production_allocation, 0.0, 1.0)
        )

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
