# Endogenous Niche World v0.1

A minimal evolutionary ecology for studying whether specialization and interdependence can emerge without a global fitness function.

## Scientific role of v0.1

v0.1 is the ecological control. It intentionally excludes trade, production, money, neural networks, and reinforcement learning. Because only resource A currently supplies metabolic energy, ordinary selection should favor A-harvesting ability. If the baseline instead produced stable occupational diversity, that would be suspicious.

## Implemented mechanics

- 2D periodic lattice
- Three renewable, spatially heterogeneous resources A, B, C
- Agents with heritable genotype `g = (h_A, h_B, h_C, v)`
- Nonnegative unit-norm developmental budget: `||g||_2 = 1`
- Local movement, harvesting, eating, reproduction, and death
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

The current version is deliberately an **ecological control**, not yet the complete
first milestone. The next smallest scientific increment is to add one production
chain and local exchange behind the same action/policy boundary, then compare runs
with exchange enabled and disabled. Analysis and visualization should become their
own modules when that increment creates real responsibilities for them; empty
folders would add navigation without clarity.

## Governing bookkeeping

Only A is edible in this control:

`E(t+1) = E(t) + epsilon_A * consumed_A - metabolism - action_cost`.

Agents die when `E <= 0`. Reproductive success is measured retrospectively; it is not assigned as an objective.

Resource regeneration obeys

`R_k(x,t+1) = min(K_k(x), R_k(x,t) + r_k)`.

Harvesting obeys

`Delta I_k = min(R_k(x,t), q_max * h_k)`.

## Baseline sanity result

With the default parameters, independent fixed-seed runs show sustained birth/death turnover rather than a frozen founder population or immediate saturation. After 400 steps in two test seeds, populations were about 550 agents, cumulative births about 2,950, cumulative deaths about 2,640, and only about 40-50 of the original 250 founder lineages survived. Mean `h_A` rose to about 0.86, as expected when A is the only resource with direct survival value.

This is a useful negative control: before adding economic interdependence, the system tends toward the one currently valuable capability.

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

Add a manufactured good and local exchange, then compare three controlled worlds: ecology only; production without trade; production plus trade. The key question is whether new persistent behavioral/genotypic clusters appear only when agents can become mutually useful to one another.
