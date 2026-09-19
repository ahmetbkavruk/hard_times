# Local social behavior (milestone 3)

This is an explicit, hand-written stochastic model for experiments. Its utility
weights are assumptions, not learned behavior or evidence that any society must
develop. The experiment is whether larger patterns arise under specified local
rules and environmental conditions. No rule reads an alignment label, assigns
allies or enemies by type, or creates a route, faction, territory or buffer zone.

## State and conventions

- The existing world is a periodic square grid. Contact means the same cell or
  one cardinal neighbor; all distances below use periodic Manhattan distance.
- Each agent has metabolic energy, the existing A/B/C inventories, ecological
  traits and production allocation. In social mode it also has a finite edible
  body reserve. Body energy cannot be spent by its living owner.
- Two heritable continuous tendencies lie in `[-1, 1]`: **lawfulness** controls
  consistency and preference for recent successful contacts; **concern**
  controls the weight given to another agent's benefit and the reluctance to
  inflict unprovoked harm. These are operational definitions for this model,
  not a claim to implement all D&D ethics. Contracts, promises and laws are not
  modeled in this milestone.
- Founders independently sample both tendencies uniformly. The nine display
  labels divide each axis at `-1/3` and `+1/3`, with boundary values neutral.
  Initial counts are random, not fixed quotas. Children inherit tendencies with
  Gaussian mutation reflected into the allowed interval; they start with empty
  memory. This uses a separate seeded random stream, preserving the previous
  founders and ecological random stream.
- At most eight contacts are remembered by default. A record contains recent
  successful trade count and the times of the last trade and received attack.
  Records expire after 30 timesteps; overflow discards the oldest record,
  breaking same-step ties by agent ID. Harm and trade information expire
  separately even when another event keeps the contact record alive.

## Information and decisions

Immutable observations expose the agent's own state, current-cell A, A in the
four adjacent cells, and agents/remains within radius two by default. Nearby
agents' energy, body reserve and B/C inventories are observable assumptions.
Observations contain relative positions and the observer's recent experiences
with those contacts. They expose no other agent's alignment or private memory,
no distant resources, and no resource-renewal mask. Population is retained only
for the existing reproduction cap. There is no memory of places or paths.

`SocialPolicy.action_distribution` is the inspectable decision rule. It scores
currently available actions from recipe deficits, useful production, hunger,
resource gradients, potential reciprocal exchange, remembered danger and food
remaining in nearby bodies. A stochastic choice uses

```text
probability(action) ∝ exp(utility(action) / temperature)
temperature = decision_temperature * (1.25 - 0.5 * lawfulness)
```

Utilities are heuristic dimensionless decision scores; they are not physical
energy. Their weights live in `ht/social_policy.py`. Physical transfers and
costs are applied only by the world, independently of these scores.

Lawfulness strengthens movement persistence and preference for recent
successful trade contacts, and lowers decision noise. Concern reduces the
utility of an attack and increases the value assigned to a partner's benefit.
Neither trait directly changes combat damage, production, metabolism or the
resource map. There is no lookup table of compulsory actions per alignment.

Attack candidates require contact and either hunger with an edible target or
a remembered attack by that target. Their score includes expected edible
return, estimated attack/metabolic effort, relative energy as a risk estimate,
concern and defensive value. A fed agent does not attack a stranger solely
because its label is evil. A good agent can fight a remembered aggressor.
Fleeing candidates are cardinal steps that increase distance from the nearest
remembered threat; ordinary movement can also avoid threats. Peaceful movement
can approach visible resource gradients, complementary traders or remains.

Exchange remains a separate contact opportunity after actions, allowing many
partners per step. Every feasible offer is submitted to both participants
using current stocks and memory. Each independently accepts or refuses. The
acceptance probability increases with the receiver's benefit and successful
past trades, and decreases for a remembered attacker; concern changes how the
partner's benefit is valued. Even a low-concern agent can profitably cooperate.
Refused offers transfer nothing and cost nothing. Only accepted trades create
positive memories. Custom policies used in social mode must implement
`willing_to_trade(observation, offer, rng)` to consent; absence means refusal.

This is an atomic bilateral exchange. There are no promises, deceptive offers,
loans, ownership claims, group enforcement or mandatory partner relationships.

## Resolution and accounting

Each existing agent receives one action opportunity in shuffled order and pays
metabolism after acting. Observations reflect changes made by earlier actors.
Newborns wait until the next step to act. An agent killed earlier in the step is
removed immediately and cannot act, eat, reproduce or trade later in that step.
Survivors then exchange at their final positions. Event timestamps identify the
completed step. The baseline mode retains the earlier resolution order.

**Movement:** one cardinal cell, including across the periodic boundary.
Fleeing pays the normal movement cost plus `flee_cost`. It does not teleport or
guarantee safety. Baseline movement still picks a random direction.

**Trade:** the previous gross-quantity cap, complementary deficits, own-stock
exports, receipt provenance and losses still apply. Social mode additionally
charges each consenting participant `trade_energy_cost` per completed trade.
An agent without enough energy to pay is ineligible. No goods or energy change
when either participant refuses.

**Attack:** an eligible local attacker pays `attack_cost`; the target loses

```text
damage = min(target energy, attack_damage * (0.5 + attacker harvesting trait))
```

The harvesting trait stands in for physical strength in this first model;
there is no additional combat genotype. Cost and damage dissipate metabolic
energy. No energy is awarded for hitting or killing. Defense is a later chosen
action, not automatic retaliation. Health is represented by metabolic energy;
there are no persistent wound or armor variables. Invalid/dead/distant targets
cause no attack or cost, although a wasted action still pays ordinary metabolism.

**Death and eating:** a dead agent's body reserve becomes one local carcass.
Its remaining A/B/C stocks are discarded and recorded; they are not duplicated
as loot or converted to food. A later `eat_agent` action consumes local remains,
including those of agents that died naturally. It cannot eat a living target.

```text
gross bite = min(bite_energy, remaining edible body energy)
body energy removed = gross bite
metabolic energy received = predation_efficiency * gross bite
conversion loss = (1 - predation_efficiency) * gross bite
eater energy change = metabolic energy received - eating_cost
```

Several eaters share the same finite reserve; every bite rechecks the remaining
stock. Exhausted carcasses are removed. In this milestone remains do not decay
or regenerate. Hunting therefore requires attacking and then consuming the
remains in separate actions; kills do not automatically feed the attacker.

**Birth:** founder body reserves are part of the initial condition. Every new
body is funded by the parent's metabolic energy:

```text
parent energy decrease = reproduce_cost + child_energy + body_energy
child metabolic energy = child_energy
child edible body reserve = body_energy
```

Consequently birth changes the total living metabolic-plus-body energy only by
the dissipated reproduction cost. Killing and eating offspring cannot generate
energy from unfunded bodies. The independent finite-body budget satisfies

```text
initial_body_energy + cumulative_body_energy_born
    = living_body_energy + carcass_energy + cumulative_body_energy_eaten
```

Here `body_energy_born` measures parent-funded transfer into new bodies, not an
external energy source. Received predation energy plus conversion loss equals
gross body energy eaten. The earlier finite-A budget remains independently
valid. The entire model is still an open ecological system: renewable A and
the existing production/recipe rules supply energy under their own assumptions.

## Observing and comparing

```bash
python run_viewer.py --social --seed 7 --steps 600 --finite-fraction 0.5
python experiments/compare.py --social --seeds 7 11 17 --gammas 1 --steps 600 --finite-fraction 0.5 --results-dir results/social_mixed
```

Social mode is opt-in (`social_enabled=True`). Without it the previous baseline
policy and exchange costs are preserved. Enabling social mode changes both
the controller and interaction mechanics; a social/baseline comparison does
not isolate an alignment effect. To test environmental conditions, keep social
mode, seeds and other parameters fixed and vary one environmental setting.
Use separate result directories for different settings and inspect several
seeds. Alignments are not reproductive fitness scores.

The viewer defaults to alignment colors in social mode. Its Colors button or
`C` switches to production allocation, without changing the world. The inspector
shows the actual numeric tendencies, current label, body reserve, remembered
contact count and latest action. Highlights report actual resolved events.

Saved runs include `agent_traits.json` (founders and survivors),
`interactions.json` (births, attacks, deaths, fleeing, food transfers and trade refusals),
`carcasses.json`, and the existing trades and resource files. `trades.csv` adds
the per-participant energy cost. Birth events include the child's tendencies and
funding, so intermediate generations can be analyzed even if they die before
the final snapshot. History records alignment populations, mean
tendencies, event totals and all body-budget terms. Separate plots show
alignment populations and social events. These measurements describe the run;
they do not assert that routes, territory or alliances have emerged.

Most physical parameters are available on `Config`; the viewer exposes the
social toggle, color mode and existing environment/trade controls. Rendering,
selection and speed changes do not consume either simulation random stream.
