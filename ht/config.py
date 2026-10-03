from dataclasses import dataclass
from math import isfinite
from numbers import Integral

@dataclass(frozen=True)
class Config:
    L: int = 40
    initial_agents: int = 250
    max_agents: int = 5000

    # Only A is environmental; manufactured B/C have separate agent stocks.
    num_resources: int = 1
    base_capacity: float = 8.0
    capacity_contrast: float = 7.0
    regen_rate: float = 0.03
    # Fraction of grid cells with a finite initial supply of A and no renewal.
    finite_resource_fraction: float = 0.0
    # An independent map seed; None derives it from the simulation seed.
    resource_seed: int | None = None

    q_max: float = 1.2
    recipe_energy_gain: float = 4.0
    # One use consumes this (A, B, C) recipe atomically.
    recipe: tuple[float, float, float] = (1.0, 0.5, 0.5)
    production_amount: float = 1.0
    production_cost: float = 0.08
    production_specialization_exponent: float = 1.0

    trade_enabled: bool = True
    trade_matching: str = "neighborhood"
    trade_efficiency: float = 1.0
    max_trade_per_pair: float = 1.0

    # Passive environmental event traces, never inputs to an agent policy.
    fields_enabled: bool = False
    field_decay: float = 0.02
    field_diffusion: float = 0.2

    # Opt-in local decision model. False preserves the earlier experiment control.
    social_enabled: bool = False
    social_mutation_sigma: float = 0.06
    perception_radius: int = 2
    memory_capacity: int = 8
    memory_duration: int = 30
    decision_temperature: float = 0.7
    trade_energy_cost: float = 0.02  # per participant, per accepted exchange
    attack_cost: float = 0.4
    attack_damage: float = 2.0
    flee_cost: float = 0.1  # in addition to normal movement cost
    body_energy: float = 4.0  # edible reserve, funded by the parent at birth
    bite_energy: float = 2.0
    predation_efficiency: float = 0.75
    eating_cost: float = 0.1

    initial_energy: float = 12.0
    metabolism: float = 0.28
    move_cost_scale: float = 0.20
    harvest_cost: float = 0.05
    use_cost: float = 0.01
    reproduce_cost: float = 3.0
    child_energy: float = 5.0

    reproduce_threshold: float = 22.0
    mutation_sigma: float = 0.06
    allocation_mutation_sigma: float = 0.06

    hungry_energy: float = 8.0
    reproduce_probability: float = 0.12
    move_exploration: float = 0.12
    # Chance to move instead of repeatedly producing while B or C is missing.
    # This policy parameter applies in both trade and no-trade experiments.
    deficit_move_probability: float = 0.12

    min_trait_norm: float = 1e-12

    def __post_init__(self) -> None:
        for name in ("field_decay", "field_diffusion"):
            value = getattr(self, name)
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        for name in ("perception_radius", "memory_capacity", "memory_duration"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "social_mutation_sigma", "trade_energy_cost", "attack_cost", "attack_damage",
            "flee_cost", "body_energy", "bite_energy", "eating_cost",
            "reproduce_cost", "child_energy",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isfinite(self.decision_temperature) or self.decision_temperature <= 0:
            raise ValueError("decision_temperature must be finite and positive")
        if not 0 < self.predation_efficiency <= 1:
            raise ValueError("predation_efficiency must be in (0, 1]")
        if not 0.0 <= self.finite_resource_fraction <= 1.0:
            raise ValueError("finite_resource_fraction must be between 0 and 1")
        if self.resource_seed is not None and (
            not isinstance(self.resource_seed, Integral) or self.resource_seed < 0
        ):
            raise ValueError("resource_seed must be a nonnegative integer or None")
        for name in ("base_capacity", "regen_rate", "q_max"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isfinite(self.capacity_contrast) or self.base_capacity + min(0, self.capacity_contrast) < 0:
            raise ValueError("capacity_contrast must be finite and keep capacities nonnegative")
        if self.trade_matching not in ("neighborhood", "random", "compatible"):
            raise ValueError("trade_matching must be 'neighborhood', 'random' or 'compatible'")
        if not (
            isfinite(self.production_specialization_exponent)
            and self.production_specialization_exponent >= 1.0
        ):
            raise ValueError("production_specialization_exponent must be finite and >= 1")
        if not 0.0 <= self.deficit_move_probability <= 1.0:
            raise ValueError("deficit_move_probability must be between 0 and 1")
        if not (0.0 < self.trade_efficiency <= 1.0):
            raise ValueError("trade_efficiency must satisfy 0 < trade_efficiency <= 1")
        if not self.max_trade_per_pair >= 0.0:
            raise ValueError("max_trade_per_pair must be nonnegative")
