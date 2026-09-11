from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    L: int = 40
    initial_agents: int = 250
    max_agents: int = 5000

    num_resources: int = 3
    base_capacity: float = 8.0
    capacity_contrast: float = 7.0
    regen_rate: float = 0.03

    q_max: float = 1.2
    food_energy: float = 4.0
    eat_amount: float = 1.0

    initial_energy: float = 12.0
    metabolism: float = 0.28
    move_cost_scale: float = 0.20
    harvest_cost: float = 0.05
    eat_cost: float = 0.01
    reproduce_cost: float = 3.0
    child_energy: float = 5.0

    reproduce_threshold: float = 22.0
    mutation_sigma: float = 0.06

    hungry_energy: float = 8.0
    reproduce_probability: float = 0.12
    move_exploration: float = 0.12

    min_trait_norm: float = 1e-12
