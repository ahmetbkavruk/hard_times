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
    age: int = 0
    lineage: int = -1
    generation: int = 0

    @property
    def hA(self) -> float:
        return float(self.genotype[0])

    @property
    def hB(self) -> float:
        return float(self.genotype[1])

    @property
    def hC(self) -> float:
        return float(self.genotype[2])

    @property
    def v(self) -> float:
        return float(self.genotype[3])

    def trait_for_resource(self, k: int) -> float:
        return float(self.genotype[k])

    @staticmethod
    def normalize_genotype(g: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        g = np.clip(np.asarray(g, dtype=float), 0.0, None)
        norm = np.linalg.norm(g)
        if norm < eps:
            g = np.ones_like(g)
            norm = np.linalg.norm(g)
        return g / norm
