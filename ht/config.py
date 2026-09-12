from dataclasses import dataclass

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

    q_max: float = 1.2
    recipe_energy_gain: float = 4.0
    # One use consumes this (A, B, C) recipe atomically.
    recipe: tuple[float, float, float] = (1.0, 0.5, 0.5)
    production_amount: float = 1.0
    production_cost: float = 0.08

    trade_enabled: bool = True
    trade_efficiency: float = 1.0
    max_trade_per_pair: float = 1.0

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

    min_trait_norm: float = 1e-12

    def __post_init__(self) -> None:
        if not (0.0 < self.trade_efficiency <= 1.0):
            raise ValueError("trade_efficiency must satisfy 0 < trade_efficiency <= 1")
        if not self.max_trade_per_pair >= 0.0:
            raise ValueError("max_trade_per_pair must be nonnegative")
