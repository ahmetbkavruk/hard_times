## Hard Times

A minimal evolutionary ecology for studying whether specialization and interdependence can emerge without a global fitness function.

## Implemented mechanics

- A periodic two-dimensional lattice with renewable, spatially heterogeneous resource A.
- A heritable developmental genotype `g = (h_A, v)`, with nonnegative components and `||g||_2 = 1`, controlling A harvesting and movement cost.
- An independent, bounded production allocation `s` controlling manufactured B/C output.
- Separate self-produced and received B/C inventories.
- Local movement, harvesting, production, atomic recipe use, reproduction, and death.
- Reciprocal local exchange, with configurable efficiency and per-pair limits.
- Parent-funded child energy, dissipative reproductive overhead, mutation, lineage tracking, and deterministic runs for a fixed seed and configuration.

`agent.py` owns individual state, `world.py` applies physical consequences, `actions.py` defines decisions, and `policies.py` chooses actions from immutable observations. There is no assigned fitness variable. Reproductive success is measured from the resulting history.

## Production and energy bookkeeping

Let `P = production_amount`, `s = production_allocation` in [0, 1], and
`gamma = production_specialization_exponent`. Each production action adds

```text
B = P * s**gamma
C = P * (1 - s)**gamma
```

Gamma must be finite and at least 1; its default is 1. For gamma = 1, total
output is exactly P, preserving the linear production law. For gamma > 1,
a full specialist still produces P, while an even split produces only
`P * 2**(1 - gamma)` in total. Outputs are not normalized back to P.
Since `s**gamma <= s` on [0, 1], total output never exceeds P.

This models a productivity advantage from concentrating effort at the same
production action cost. It does not guarantee survival or the evolution of
specialists: raising gamma also makes mixed producers less productive.

Energy is obtained only by atomically consuming the configured A/B/C recipe:

```text
energy change on successful use = recipe_energy_gain - use_cost
```

An incomplete recipe changes neither inventories nor energy. Self-produced
stock is consumed before received stock. B/C inventory alone gives no energy.
Every existing agent also pays metabolism each timestep, and agents die at
energy <= 0. Reproduction transfers child_energy from the parent and dissipates
reproduce_cost.

Environmental A regeneration is capped by local carrying capacity, and each
harvest transfers at most `q_max * h_A` from the cell to the agent.

## Exchange and exploration

After existing agents act and dead agents are removed, the world builds
occupancy from final positions. It shuffles each occupied cell using the world
random generator and pairs consecutive occupants. Pairs are disjoint; each
agent has at most one partner in that timestep.

For a B donor and C donor, exchange requires self-produced surplus above one
recipe and a deficit of the opposite good. Deficits include received inventory.
The common gross amount surrendered by each donor is

```text
q = min(B surplus, C surplus, C deficit / efficiency,
        B deficit / efficiency, max_trade_per_pair)
```

Each recipient gets `efficiency * q`; each good loses
`(1 - efficiency) * q`. Received goods cannot be re-exported, and exchange
does not alter energy. Configuration requires `0 < trade_efficiency <= 1`
and `max_trade_per_pair >= 0`.

When B or C is missing, the policy moves with `deficit_move_probability`
(default 0.12), otherwise it produces. This allows a deficient specialist to
explore instead of producing forever at one position. Movement is random:
agents receive no information about partners. The same policy and probability
apply with trade enabled and disabled. A usable recipe still has priority
when energy is below the policy's target.

Set `deficit_move_probability=0` to recover the previous always-produce
deficit rule. The new default changes trajectories even at gamma = 1;
comparisons must keep this probability fixed.

## Run the no-trade control

Install dependencies and run:

```bash
python -m pip install -r requirements.txt
python experiments/baseline.py --seed 7 --steps 600
```

This runner explicitly uses `trade_enabled=False`. The general-purpose
`Config()` default remains trade enabled, so `python run_demo.py` exercises
the trading world.

The baseline accepts `--gamma`, `--deficit-move-probability`,
`--results-dir`, and `--no-plots`. For the earlier linear no-trade policy:

```bash
python experiments/baseline.py --gamma 1 --deficit-move-probability 0
```

## Compare trade and production laws

Run both trade conditions at every gamma and seed:

```bash
python experiments/compare.py --gammas 1 1.5 2 --seeds 7 11 17 --steps 600
```

The default gamma list is 1, 1.5, 2; gamma is never raised implicitly in the
model. The comparison also accepts `--trade-efficiency`,
`--max-trade-per-pair`, `--deficit-move-probability`, `--results-dir`,
and `--no-plots`.

Each pair has the same configuration except for trade_enabled and starts from
the same seeded population and environment. Exchange consumes random draws,
so later action draws are not synchronized between conditions. Use multiple
seeds and compare outcomes across runs.

## Saved measurements

Each run writes to
`RESULTS_DIR/gamma_G/trade_on_or_off/seed_S/`:

| File | Contents |
| --- | --- |
| `config.json` | Full configuration, seed, requested duration, and completed duration |
| `history.csv` | Population, energy, traits, inventories, cumulative trade and recipe-use measurements |
| `trades.csv` | Time, cell, B/C donor IDs, gross transfers, net receipts, and losses per event |
| `allocation_snapshots.json` | Allocation samples at initialization, one-third, two-thirds, and the last simulated step |
| `figures/` | Population, energy, traits, allocation distributions, stocks, and trade activity |

The comparison additionally writes `summary.csv`, one final row per run.
Baseline output defaults to `results/`; comparison output defaults to
`results/comparison/`. Runs are separated by gamma, trade condition, and seed.
Use a distinct results directory for different durations, efficiencies, caps,
or exploration probabilities; rerunning the same path replaces those files.

All history fields beginning with `cumulative_` accumulate since initialization.
Gross transfer equals net receipt plus transaction loss separately for each good.
In `trades.csv`, the C donor receives net_B and the B donor receives net_C.
A no-trade run still writes the CSV header, with no event rows.

`mean_specialization` is the population mean of `abs(2*s - 1)`: zero for
balanced producers and one for endpoint producers. It is a descriptive measure
of allocation, not evidence of dependence on trade. It is NaN after extinction.

`cumulative_recipe_uses_with_received_stock` counts successful recipes using
any received B/C; `cumulative_received_B_used` and
`cumulative_received_C_used` measure actual amounts consumed. The summary
also reports the fraction of successful recipes containing received goods
(zero if there were no successful recipes). Receiving goods and using them
are different measurements; even receipt use alone does not establish that
an agent could not survive without exchange.

Assess population survival, births and deaths, allocation distributions, and
receipt use together. A larger gamma with trade is an experiment, not a
preselected successful outcome.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover production at gamma = 1 and gamma > 1, atomic recipe use and receipt
consumption, configurable exploration, reproduction, bounded mutation, seeded
reproducibility, exchange pairing/provenance/loss accounting, event order, and
the saved experiment controls and measurements.
