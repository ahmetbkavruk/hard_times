# Passive environmental traces

This experiment records the spatial history of actual events. It is inspired
by persistent local state in [Reasoning with Neural Cellular Automata](https://arxiv.org/abs/2609.36126v1),
but it is a deterministic observer, not a trained neural cellular automaton.
No agent can sense these fields, and no field changes movement costs,
resources, energy, trade consent, combat, reproduction, or policy decisions.

## What is recorded

Enable recording with `Config(fields_enabled=True)` or `--fields` in the live
viewer or comparison runner. It is off by default and works with both the
baseline and social policies.

| Channel | Deposit | Exclusions |
| --- | --- | --- |
| `movement` | One unit at the destination of each executed cell change, including fleeing. | An action on a one-cell torus does not change cells and leaves no trace. Births and initial placement leave no trace. |
| `trade` | One unit per completed reciprocal exchange, split equally between the participants' cells. Co-located partners deposit one full unit in their shared cell. | Refusals, infeasible exchanges and unused trade opportunities leave no trace. Trade size does not change the weight. |
| `violence` | One unit at the victim's cell for each executed damaging attack. | Invalid attack requests, natural deaths and consumption of remains leave no trace. Damage size does not change the weight. |

Record events when their consequences occur, rather than reconstructing them
from the agents that survive a timestep. A mover or victim that dies later
still leaves the appropriate trace. Ordinary movement recording does not
require `record_actions=True` or the viewer.

Units are event weights. These fields are neither goods nor energy, and
diffusion is a chosen observation filter, not a claim that footprints,
markets, or danger physically diffuse. With `field_diffusion=0`, traces stay
where the events occurred. Local spreading is an explicit parameter to study,
not evidence that agents communicate.

## Update rule and timing

Let `F_c(x, t)` be the trace in channel `c` at cell `x` after timestep `t`.
Let `S_c(x, t)` be the deposits queued during the next timestep. Each channel
uses the same independent update:

```text
F_c(x, t+1) = (1-decay) * [(1-diffusion)*F_c(x, t)
                          + diffusion/4 * sum_four_neighbors(F_c(t))]
             + S_c(x, t)
```

Both parameters are finite numbers in `[0, 1]`; defaults are decay `0.02`
and diffusion `0.2`. The four directions are up, down, left and right on the
same periodic lattice as the agents. Directions retain their weights when
they coincide on grids of side one or two.

During actions and exchanges only the deposit buffer changes. After exchange,
the entire field advances once using a shared snapshot of the previous field;
new deposits are then added. No cell reads another cell's newly updated value.
A newly deposited trace first appears at its event location, and can spread
at most one cardinal cell per subsequent timestep. No noise or random update
mask is used, so there is no field RNG and no change to either simulation RNG.
Adding asynchronous or stochastic transport would be a separate experiment.

The mixing is a convex combination, so nonnegative fields stay nonnegative.
On the periodic grid it preserves total weight before decay. For each channel:

```text
total(t+1) = (1-decay) * total(t) + deposited(t+1)
```

Decay zero preserves past weight; decay one discards the old field completely.
Diffusion zero leaves old weight in its cell. For decay `0.02`, an isolated
deposit retains half its total weight after about 34.3 timesteps. With zero
decay and continuing deposits, total weight grows; it is not normalized or
clipped. Viewer saturation is display-only.

`World.fields` is `None` when disabled. Otherwise it owns the three fields,
pending deposits and a separate history, including an all-zero row at step
zero. `snapshot()` returns a detached, read-only `[channel, y, x]` array in
movement/trade/violence order; `layer(name)` returns a detached `[y, x]` array.
These arrays never enter `Observation`. Agent updates keep PR #12's existing
ordering; the shared snapshot applies only to field transport.

## Watching and exporting

```bash
python run_viewer.py --social --fields --seed 7 --steps 600 --finite-fraction 0.5
```

Press **F** or the **Field** button to cycle movement, trade, violence and no
overlay. Hiding an overlay does not stop recording. Ground resources and
agent colors remain visible under/above the tint. The legend specifies a
fixed saturation level, defaulting to five event units per cell. Compare runs
using the same `--field-scale` (a positive finite number); it changes only the
display. `--field-overlay trade` starts with trade traces. Restarting resets
all fields along with the world and reproduces the same seed.

For undiffused activity traces, add `--field-diffusion 0`. Use
`--field-decay 0` as well for cumulative event counts by cell.

```bash
python experiments/compare.py --social --fields --gammas 1 --seeds 7 11 17 --steps 600 --finite-fraction 0.5 --results-dir results/fields
```

The existing trade/no-trade comparisons export their usual files plus:

- `field_map.json`: final arrays, timestep, axis convention, decay and diffusion.
- `field_history.csv`: at every timestep, each channel's new deposits,
  cumulative deposits, remaining total, peak cell value and concentration.

Concentration is the sum of squared spatial shares. It is `1/(L*L)` for a
uniform nonzero field, one for a single occupied cell, and zero for an empty
field. The final values are also included in the comparison summary.
Field-enabled run directories include their decay and diffusion parameters,
keeping them separate from ordinary control runs. Use separate results roots
for separate experiments, as the comparison runner overwrites `summary.csv`.

## Interpretation and controls

An accumulated hotspot is not proof of a market, route, territory, memory
in an agent, or cooperation. Resource geography, repeated independent visits,
decay and smoothing can all create persistent-looking patterns. Interpret
the field alongside the original resource map, actual event counts and
multiple seeds. Compare undiffused fields when assessing concentration.

The first causal control is exact equality of physical trajectories with
recording on and off. Tests compare agents, resources, births and deaths,
inventories, encounters, trade/combat records, metrics and both RNG states.
Other tests check finite-speed propagation, wrapping, positivity, source and
decay accounting, excluded events, exports and viewer independence.

Adding agent responses, roads, movement discounts, learned rules or rewards
would change the model. Each should be a later, explicitly controlled
experiment against this passive baseline.
