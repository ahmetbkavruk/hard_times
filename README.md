## Hard Times

A minimal evolutionary ecology for studying whether specialization and interdependence can emerge without a global fitness function.

## Persistent local event traces

Record and watch passive movement, trade and violence traces on top of the
social model:

```bash
python run_viewer.py --social --fields --seed 7 --steps 600 --finite-fraction 0.5
```

Press **F** or click **Field** to cycle the overlays. Traces persist, decay and
spread only to the four neighboring cells; agents cannot sense them. Recording
does not alter the seeded trajectory. Use `--field-diffusion 0` for traces that
stay at the event locations, and `--field-decay 0` for cumulative counts.
`--field-scale 5` sets the fixed display saturation in event units per cell.

The comparison runner also accepts `--fields`, `--field-decay` and
`--field-diffusion`, exporting `field_history.csv` and `field_map.json` in each
run directory. Recording works with either policy and is off by default.
See [FIELD_MODEL.md](FIELD_MODEL.md) for event definitions, timing, accounting,
measurements and interpretation limits.

## Watch the live world

From a local desktop terminal in the repository directory:

```bash
python -m pip install -r requirements.txt
python run_viewer.py --seed 7 --steps 600
```

For milestone 3, enable local social decisions and alignment colors:

```bash
python run_viewer.py --social --seed 7 --steps 600 --finite-fraction 0.5
```

Agents can perceive neighbors, remember recent encounters, choose movement
directions, accept or refuse exchanges, fight, flee, and consume finite remains.
See [the social model](SOCIAL_MODEL.md) for the precise assumptions and energy
accounting. Omit `--social` to run the earlier baseline controller.

The viewer opens a live pixel grid and starts playback. Trade is **enabled** by
default, using neighborhood trading. Without `--social`, its model and configuration defaults match `run_demo.py`.
The sidebar displays the actual seed, gamma and trade setting. For example:

```bash
python run_viewer.py --seed 7 --steps 600 --gamma 2 --trade-matching neighborhood
python run_viewer.py --seed 7 --steps 600 --gamma 2 --no-trade
```

- **Pause / Play** or **Space** stops or resumes time. **Step** or **N** pauses
  playback and advances exactly one timestep.
- **Steps / sec** changes the target playback rate (1–60). Slow machines or
  large populations may run below the target; simulation timesteps are never
  skipped. `--speed 10` sets the initial rate.
- Click an occupied cell to inspect an agent's energy, age, lineage, B/C
  production allocation, inventory provenance and latest action. Click the
  same cell again to cycle through its occupants. Selection follows that agent
  as it moves, and its final state remains visible if it dies.
- **Restart seed** or **R** recreates the initial world with the same seed and
  settings, paused at step zero. `--paused` starts a new viewer paused.
- **Colors** or **C** switches between alignment and production allocation.
  `--color-by alignment` or `--color-by production` sets the initial view.
- `--size 40` and `--agents 250` set the grid size and initial population.
  Omit `--steps` to continue until extinction or until the window is closed.

Green ground marks renewable sources; brown ground marks finite deposits.
Brighter ground means more environmental resource A, on a fixed scale
throughout the run. Empty finite cells stay dark brown so their type remains
visible. The readout below the grid reports remaining A and exhausted finite
cells; the agent inspector also identifies its current ground type.
In social mode, each of the nine alignments has its own color and legend.
The inspector shows lawfulness and concern as numerical tendencies. The
production view ranges from **cyan** (C production) through **white** (balanced
allocation) to **orange** (B production); this remains the baseline's default.
Multiple occupants are laid out as smaller squares inside their cell, without
changing their physical positions.

Green square outlines mark successful recipe consumption, pink rings mark
both participants' cells in actual exchanges, and red crosses mark deaths during the latest timestep.
Highlights remain visible while paused and are replaced on the next step.
In social mode, red diamonds mark attacks, blue arrows mark fleeing, pale
triangles mark eating remains, and small beige dots mark remaining carcasses.
Exchange uses mutual consent in social mode and automatic matching in the baseline.

The viewer uses the existing Matplotlib dependency and requires an interactive
desktop backend such as TkAgg or QtAgg. Run it as a script in a desktop terminal,
not with an inline notebook or headless `Agg` backend. On Linux, Tk support may
need the distribution's Python Tk package; a remote session needs a display.
The viewer reports a clear error if it detects a noninteractive backend.

Rendering, inspection and playback controls never consume simulation random
draws. Optional action recording keeps only the latest timestep and preserves
the same seeded trajectory as an ordinary `World` run. The viewer keeps the
model's usual in-memory history and trade records but does not export run
files; the experiment runners below provide saved measurements.

## Individual tendencies and local choices

Alignment labels describe two continuous heritable tendencies. They never
assign an action, ally, enemy, faction, route or territory. Higher lawfulness
means more consistent choices and stronger preferences for recent successful
trade contacts. Greater concern reduces attack utility and increases the value
assigned to a trade partner's benefit. Hunger, resources, effort, risk and
actual remembered encounters also affect decisions. Good agents can defend
themselves; evil agents can cooperate profitably.

| Alignment | Color |
| --- | --- |
| Lawful Good | Blue `#53b7ff` |
| Neutral Good | Mint `#5ce0b5` |
| Chaotic Good | Yellow `#e3e86a` |
| Lawful Neutral | Lavender `#b2a2ff` |
| True Neutral | Silver `#d4d9e0` |
| Chaotic Neutral | Orange `#ffb45c` |
| Lawful Evil | Magenta `#e075c4` |
| Neutral Evil | Brown `#ad8057` |
| Chaotic Evil | Red `#ff555f` |

Perception is local (radius two by default), with at most eight remembered
contacts for 30 timesteps. Interaction range is the same cell plus its four
cardinal neighbors. Each potential exchange still rechecks current stocks and
can involve multiple partners, but both participants must consent and pay the
configured energy cost. These are stochastic hand-written preferences; the
model is not trained and no collective outcome is guaranteed.

Combat costs energy and can kill. It gives the attacker no immediate food.
Eating is a separate action that consumes a finite corpse reserve with a
conversion loss and handling cost. Newborn bodies and energy are funded by the
parent, so reproduction cannot supply free prey. Natural deaths also leave
finite remains. Body reserves do not regenerate or decay in this milestone.

The same `--social` switch works in both experiment runners:

```bash
python experiments/compare.py --social --seeds 7 11 17 --gammas 1 --steps 600 --finite-fraction 0.5 --results-dir results/social_mixed
```

Keep the controller, seed and other parameters fixed when comparing renewable
and finite environments. Enabling social mode changes both policy and
interaction mechanics, so comparing it with the baseline does not isolate an
alignment effect. Full decision rules, limitations, configurable physical costs,
and conservation identities are documented in [SOCIAL_MODEL.md](SOCIAL_MODEL.md).

## Finite deposits and renewable patches

To watch a world with half its cells containing nonrenewable A:

```bash
python run_viewer.py --seed 7 --steps 600 --finite-fraction 0.5
```

`--finite-fraction` accepts a value between 0 and 1. Zero is the original,
all-renewable control and remains the default. One makes the whole world
finite. Intermediate values select that fraction of cells, rounded to the
nearest whole cell, in clustered patches. Every cell starts with its usual A
stock. Renewable cells replenish at `regen_rate`, capped by local capacity;
finite cells have zero regeneration. Once a finite cell is exhausted, it stays
empty for the rest of that run. Restarting the seed creates a new initial world.

The same resource settings are available in the viewer, the no-trade baseline
and the comparison runner:

| Option | Meaning |
| --- | --- |
| `--finite-fraction 0.5` | Fraction of cells assigned finite A deposits |
| `--resource-seed 23` | Change the source-type layout independently of starting agents |
| `--base-capacity 2` | Base initial A per cell, before the existing spatial contrast |
| `--regen-rate 0.01` | A restored per renewable cell per timestep |

The map has its own random stream. With the same simulation seed, changing
the finite fraction or resource seed preserves the initial agents, their
random-generator state, and every cell's initial A amount. For a fixed map
seed, increasing the finite fraction converts additional cells without moving
existing finite patches. If omitted, the resource seed derives from the
simulation seed. Layout generation smooths a seeded random field across the
periodic grid four times, then ranks cells to obtain the requested count; this
happens only at initialization and does not transport resources during a run.

For a controlled comparison, keep the simulation seed, policy and production
settings fixed and vary one resource setting. Repeat with several simulation
seeds. These commands save the renewable and mixed conditions separately:

```bash
python experiments/compare.py --gammas 1 --seeds 7 11 17 --steps 600 --finite-fraction 0 --results-dir results/scarcity/renewable
python experiments/compare.py --gammas 1 --seeds 7 11 17 --steps 600 --finite-fraction 0.5 --results-dir results/scarcity/mixed
```

Each also includes the existing no-trade control. Agent policies receive
available A amounts; the source-type mask is not a policy input.
No migration, settlement or cooperation outcome is prescribed.

## Implemented mechanics

- A periodic two-dimensional lattice with spatially heterogeneous resource A,
  configurable renewable patches and permanently exhaustible deposits.
- A heritable developmental genotype `g = (h_A, v)`, with nonnegative components and `||g||_2 = 1`, controlling A harvesting and movement cost.
- An independent, bounded production allocation `s` controlling manufactured B/C output.
- Separate self-produced and received B/C inventories.
- Local movement, harvesting, production, atomic recipe use, reproduction, and death.
- Reciprocal local exchange, with configurable efficiency and per-pair limits.
- Optional social tendencies, local perception and memory, voluntary exchange,
  directed movement, combat, fleeing and consumption of finite remains.
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

In the baseline, energy is obtained by atomically consuming the configured A/B/C recipe:

```text
energy change on successful use = recipe_energy_gain - use_cost
```

An incomplete recipe changes neither inventories nor energy. Self-produced
stock is consumed before received stock. B/C inventory alone gives no energy.
Every existing agent also pays metabolism each timestep, and agents die at
energy <= 0. Reproduction transfers child_energy from the parent and dissipates
reproduce_cost. Social mode also allows energy transfer from finite body
reserves and charges parents for each child's body; see the accounting rules above.

Environmental A regeneration applies only to renewable cells and is capped by
local carrying capacity. Finite cells use that capacity only as their initial
stock. Each harvest transfers at most `q_max * h_A` and at most the amount
remaining in the cell to the agent.

## Exchange and exploration

After existing agents act and dead agents are removed, the world builds
occupancy from final positions. The default `trade_matching="neighborhood"`
allows each agent to trade with agents in its own cell and the four immediately
adjacent cells (up, down, left and right). Boundaries wrap exactly as movement
does; diagonal and more distant cells are excluded.

The world shuffles agents, then shuffles each agent's as-yet-unvisited local
contacts, using the world random generator. Every unordered agent pair is
considered once per timestep, including on grids of side 1 or 2. An agent can
trade with multiple partners in that timestep. Each exchange rechecks current
surpluses and deficits, so earlier receipts or exports can limit later trades.
No persistent partner is assigned, and trading does not move either agent.

The original rules remain available explicitly for earlier experiment controls:

- `random`: pair consecutive shuffled occupants of each cell, including incompatible pairs.
- `compatible`: split the shuffled occupants into eligible B and C donors and
  pair the two queues. A B donor has self-produced B above its recipe reserve
  and a C deficit; a C donor satisfies the opposite conditions. Balanced or
  otherwise ineligible bystanders do not consume trading opportunities.

These two legacy rules form disjoint pairs within each cell and retain the
one-partner limit. Normal model runs, the viewer and the comparison runner now
default to `neighborhood`. `experiments/matching.py` remains the historical
comparison of the two legacy rules and a no-trade control.

In the baseline, all matching modes use automatic reciprocal exchange without
extra energy cost or partner memory. Social mode adds local search preferences,
bounded contact memory and bilateral consent before any exchange is committed.

For a B donor and C donor, exchange requires self-produced surplus above one
recipe and a deficit of the opposite good. Deficits include received inventory.
The common gross amount surrendered by each donor is

```text
q = min(B surplus, C surplus, C deficit / efficiency,
        B deficit / efficiency, max_trade_per_pair)
```

Each recipient gets `efficiency * q`; each good loses
`(1 - efficiency) * q`. Received goods cannot be re-exported, and exchange
does not alter energy in the baseline. Social mode charges `trade_energy_cost`
to each participant per accepted exchange. Configuration requires `0 < trade_efficiency <= 1`
and `max_trade_per_pair >= 0`. The cap is per pair per timestep, so an agent's
total exchange with several partners may exceed it while respecting available
self-produced surplus and remaining need.

In the baseline, when B or C is missing, the policy moves with `deficit_move_probability`
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
`--matchings`, and `--no-plots`.

To compare matching rules at fixed gamma, alongside one no-trade control:

```bash
python experiments/compare.py --gammas 2 --matchings neighborhood random compatible --seeds 7 11 17 --results-dir results/matching_world
```

The default matching list contains only `neighborhood`. Pass `--matchings random`
to reproduce the earlier comparison. Conditions differ only in trade_enabled and the selected matching
rule and start from the same seeded population and environment. Exchange consumes random draws,
so later action draws are not synchronized between conditions. Use multiple
seeds and compare outcomes across runs.

## Test whether specialist lineages can establish

```bash
python experiments/matching.py --no-plots
```

This diagnostic holds gamma at 2. Every condition starts with one pure B
specialist, one pure C specialist, and 18 balanced producers in one cell, all
with empty inventories and the same A-harvesting genotype. A supply exceeds
the maximum possible harvest; movement and mutation are disabled. Specialists
receive no free goods, extra energy, or exemption from ordinary action costs.

Each seed compares no trade, random matching, and compatible matching in two
phases:

- `survival`: 1,000 steps without reproduction. The policy energy target is
  raised so it does not throttle successful recipe use.
- `reproduction`: 200 steps with normal reproduction costs, probability,
  offspring energy, and policy energy target. Both genotype mutation rates
  remain zero so founder lineages retain their allocations.

This tests survival and establishment of seeded specialist types. It does not
test their spontaneous emergence through mutation or their success in a spatial
environment. `--generalists`, `--seeds`, `--survival-steps`,
`--reproduction-steps`, `--max-agents`, and `--results-dir` are configurable.
The default population cap is 1,000; check `capacity_reached_step` before
interpreting growth. Population share and lineage births/deaths matter alongside
absolute descendant counts.

Results live under `results/matching/PHASE/gamma_2/CONDITION/seed_S/`.
In addition to the standard files below, each run saves `founders.json` and
`lineage_history.csv` with separate B-specialist, C-specialist, and generalist
counts, births, deaths, and the combined specialist population share. The root
`summary.csv` records final measurements, specialist extinction time, and the
first step reaching the population cap.

## Saved measurements

Each run writes to
`RESULTS_DIR/gamma_G/CONDITION/seed_S/`, where CONDITION is `trade_off`,
`trade_neighborhood`, `trade_random`, or `trade_compatible`:

Social runs use `RESULTS_DIR/social/gamma_G/CONDITION/seed_S/`.

| File | Contents |
| --- | --- |
| `config.json` | Full configuration, seed, requested duration, and completed duration |
| `history.csv` | Population, energy, traits, inventories, cumulative trade and recipe-use measurements |
| `trades.csv` | Time, both donors' cells and IDs, gross transfers, net receipts, and losses per event |
| `allocation_snapshots.json` | Allocation samples at initialization, one-third, two-thirds, and the last simulated step |
| `resource_map.json` | Source-type mask, initial A map, final remaining A map, and final timestep; arrays indexed as `[y][x]` |
| `agent_traits.json` | Founders' and survivors' lawfulness, concern, alignment and lineage |
| `interactions.json` | Births with child traits, attacks, deaths, fleeing, food transfers and refused trades, with actual costs and losses |
| `carcasses.json` | Remaining finite body stocks and their positions |
| `figures/` | Population, energy, traits, allocation, stocks, trade, alignments and social activity |

The comparison additionally writes `summary.csv`, one final row per run.
Baseline output defaults to `results/`; comparison output defaults to
`results/comparison/`. Runs are separated by gamma, trade condition, and seed.
Use a distinct results directory for different durations, efficiencies, caps,
exploration probabilities, policies or resource settings; rerunning the same path replaces those files.

Resource history includes `renewable_A_remaining`, `finite_A_remaining`,
`initial_finite_A`, `finite_cells`, `exhausted_finite_cells` and
`cumulative_finite_A_harvested`. These remaining-stock values refer to the
environment, not agents' inventories. The finite budget obeys
`initial_finite_A = finite_A_remaining + cumulative_finite_A_harvested`, up to
floating-point roundoff. `environmental_resources_seed_S.png` plots remaining
renewable and finite ground stocks through time. The complete source map is
saved even for zero-step or early-extinction runs.

All history fields beginning with `cumulative_` accumulate since initialization.
Gross transfer equals net receipt plus transaction loss separately for each good.
In `trades.csv`, the C donor receives net_B and the B donor receives net_C.
`B_donor_x`, `B_donor_y`, `C_donor_x` and `C_donor_y` record both endpoints at
exchange time. The retained `cell_x` / `cell_y` columns identify the first
visited partner's cell, which need not contain both partners. In-memory events
include `partner_cells`, keyed by agent ID, for the same reason.
A no-trade run still writes the CSV header, with no event rows.
`energy_cost_per_agent` records the social exchange cost (zero in the baseline).
Social histories additionally record attacks, combat deaths, fleeing, predation,
refused trades, alignment populations and the finite-body energy budget.

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
the saved experiment controls and measurements. Compatible-matching tests cover
bystanders, unbalanced donor counts, locality, disjoint pairs, caps, losses,
receipt provenance, and seeded reproducibility. Diagnostic tests also check
founder setup and lineage birth/death accounting.

Neighborhood tests cover cardinal locality, periodic boundaries, multiple
partners, pair deduplication on tiny grids, updated inventory limits, material
balance, final positions, death removal and seeded event order. Viewer tests
also check both trade endpoints and independence from rendering and controls.
Resource tests cover permanent exhaustion, capped regeneration, competing
harvesters, exact initial controls, independent map seeds, finite-stock budgets,
saved maps and viewer reset/color behavior.
