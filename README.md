# Endogenous Niche World v0.1

A minimal evolutionary ecology for studying whether specialization and interdependence can emerge without a global fitness function.

## Scientific role of v0.1

This increment adds individual production without trade. Energy is obtained only by
atomically using a complete A/B/C recipe; B or C inventory alone has no energetic
value.

## Implemented mechanics

- 2D periodic lattice
- One renewable, spatially heterogeneous environmental resource A
- Agents with heritable developmental genotype `g = (h_A, v)` and an independent,
  bounded production allocation `s`
- Nonnegative unit-norm developmental budget: `||g||_2 = 1`
- Local movement, A harvesting, fixed-capacity B/C production, recipe use,
  reproduction, and death
- Energy-conserving reproduction: the parent funds the child's initial energy plus a dissipative reproductive overhead
- Gaussian mutation followed by clipping and genotype renormalization
- Founder lineage and generation tracking
- Deterministic trajectories for a fixed random seed
- No `fitness` variable anywhere in the agent state
- Typed action values and a replaceable policy interface

## Architecture

This first step keeps the existing small package rather than creating many mostly
empty subpackages. `agent.py` owns persistent individual state, `world.py` owns
environment state and applies consequences, `actions.py` defines the decisions
passed between those layers, and `policies.py` chooses decisions from a limited
observation. This boundary is useful because a learned controller can later replace
the rule-based policy without gaining permission to mutate the world directly.

This milestone contains the production-without-trade condition. Local exchange is
deliberately deferred. Analysis and visualization should become their own modules
when later increments create real responsibilities for them; empty folders would
add navigation without clarity.

## Governing bookkeeping

One complete configured A/B/C recipe supplies energy:

`E(t+1) = E(t) + epsilon_recipe - metabolism - action_cost`.

Agents die when `E <= 0`. Reproductive success is measured retrospectively; it is not assigned as an objective.

Resource regeneration obeys

`R_k(x,t+1) = min(K_k(x), R_k(x,t) + r_k)`.

Environmental A harvesting obeys

`Delta I_A = min(R_A(x,t), q_max * h_A)`.

Each production action creates exactly `production_amount`, divided as
`B = production_amount * s` and `C = production_amount * (1 - s)`. Self-produced
and received B/C are stored separately so later exchange mechanics can preserve
their provenance.

Because successful recipe use requires A, B, and C, all three inputs are
survival-relevant in this milestone. B and C are individually produced, but there
is no exchange or specialization incentive yet.

## Run

```bash
python run_demo.py
```

or run the baseline experiment and write a CSV:

```bash
python experiments/baseline.py
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover genotype normalization, resource carrying capacities, periodic space, deterministic seeding, absence of a global fitness field, reproduction energy accounting, generation tracking, and baseline demographic turnover.

## Next experiment

Add local exchange, then compare the production-without-trade and
production-plus-trade conditions. The key question is whether new persistent
behavioral or genotypic clusters appear only when agents can become mutually useful
to one another.
