"""Locality, voluntary decisions, finite biomass and reproducible social worlds."""

import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.runner import condition_directory, run_experiment
from ht import Action, ActionKind, Config, SocialPolicy, World
from ht.policies import Neighbor, NearbyBody, Observation, TradeOffer
from ht.social import ALIGNMENT_COLORS, Alignment, Encounter, alignment_for


class ConsentingPolicy:
    def __init__(self, reject=None):
        self.reject = reject
        self.offers = []

    def choose_action(self, observation, rng):
        return Action(ActionKind.USE)

    def willing_to_trade(self, observation, offer, rng):
        self.offers.append(offer)
        return offer.give_good != self.reject


class TestSocialModel(unittest.TestCase):
    def world(self, **overrides):
        settings = dict(L=7, initial_agents=3, social_enabled=True, metabolism=0)
        settings.update(overrides)
        world = World(Config(**settings), seed=7)
        for agent in world.agents.values():
            agent.x = agent.y = 0
        return world, list(world.agents.values())

    def test_nine_labels_colors_and_boundary_convention(self):
        labels = {alignment_for(order, care) for order in (-1, 0, 1) for care in (-1, 0, 1)}
        self.assertEqual(labels, set(Alignment))
        self.assertEqual(len(set(ALIGNMENT_COLORS.values())), 9)
        for value in (-1 / 3, 0, 1 / 3):
            self.assertIs(alignment_for(value, value), Alignment.TRUE_NEUTRAL)
        for value in (np.nan, np.inf, -np.inf, -1.01, 1.01):
            with self.assertRaises(ValueError):
                alignment_for(value, 0)

    def test_configuration_rejects_invalid_physical_parameters(self):
        for name in ("social_mutation_sigma", "trade_energy_cost", "attack_cost", "attack_damage",
                     "flee_cost", "body_energy", "bite_energy", "eating_cost", "reproduce_cost", "child_energy"):
            for value in (-0.01, np.inf, np.nan):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    Config(**{name: value})
        for name in ("perception_radius", "memory_capacity", "memory_duration"):
            for value in (0, -1, 0.5):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    Config(**{name: value})
        for value in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                Config(decision_temperature=value)
        for value in (0, -1, 1.01, np.nan):
            with self.assertRaises(ValueError):
                Config(predation_efficiency=value)

    def test_trait_initialization_preserves_founders_and_main_random_stream(self):
        cfg = Config(L=8, initial_agents=30)
        baseline = World(cfg, seed=11)
        social = World(replace(cfg, social_enabled=True), seed=11)
        self.assertEqual(baseline.rng.bit_generator.state, social.rng.bit_generator.state)
        self.assertEqual(baseline.founder_traits, social.founder_traits)
        for aid, agent in baseline.agents.items():
            for key, value in asdict(agent).items():
                if key != "body_energy":
                    np.testing.assert_equal(value, asdict(social.agents[aid])[key])
        changed_map = World(replace(cfg, social_enabled=True, resource_seed=19, finite_resource_fraction=0.5), seed=11)
        self.assertEqual(social.founder_traits, changed_map.founder_traits)

    def test_reproduction_funds_body_and_energy_and_inherits_traits(self):
        world, (parent,) = self.world(initial_agents=1, social_mutation_sigma=0)
        parent.energy = 50
        parent.lawfulness, parent.concern = -0.8, 0.7
        parent.memory[91] = Encounter(last_trade=0, trades=1)
        before = parent.energy + parent.body_energy
        world.do_reproduce(parent)
        child = world.agents[1]
        self.assertEqual((child.lawfulness, child.concern), (-0.8, 0.7))
        self.assertEqual(child.memory, {})
        self.assertAlmostEqual(before - sum(a.energy + a.body_energy for a in world.agents.values()), world.cfg.reproduce_cost)
        self.assertEqual(child.body_energy, world.cfg.body_energy)
        self.assertEqual(world.social_totals["body_energy_born"], child.body_energy)
        event = world.interactions[-1]
        self.assertEqual(event["target_id"], child.id)
        self.assertAlmostEqual(event["parent_energy_cost"],
                               event["child_energy"] + event["body_energy"] + event["dissipated_energy"])
        self.assertEqual(event["alignment"], child.alignment.value)
        # Meeting the reproduction threshold alone cannot fund an expensive body.
        poor, (agent,) = self.world(initial_agents=1, body_energy=100)
        agent.energy = poor.cfg.reproduce_threshold
        poor.do_reproduce(agent)
        self.assertEqual(len(poor.agents), 1)

    def test_observation_is_local_immutable_and_wraps(self):
        world, (observer, near, far) = self.world()
        near.x, near.y = 6, 0
        far.x, far.y = 3, 3
        before = copy.deepcopy(world.rng.bit_generator.state)
        observation = world.observe(observer)
        self.assertEqual([n.id for n in observation.neighbors], [near.id])
        self.assertEqual((observation.neighbors[0].dx, observation.neighbors[0].dy), (-1, 0))
        self.assertFalse(hasattr(observation.neighbors[0], "alignment"))
        self.assertFalse(hasattr(observation, "resources"))
        with self.assertRaises(AttributeError):
            observation.neighbors[0].energy = 1
        self.assertEqual(world.rng.bit_generator.state, before)
        near.energy = 1
        self.assertNotEqual(observation.neighbors[0].energy, near.energy)
        # Small periodic grids must not expose duplicate contacts.
        for length in (1, 2):
            tiny, (first, second, third) = self.world(L=length)
            self.assertEqual([n.id for n in tiny.observe(first).neighbors], [second.id, third.id])

    def test_memory_is_bounded_and_expires_without_remembering_places(self):
        world, (agent, _, _) = self.world(memory_capacity=2, memory_duration=3)
        for step, other in enumerate((10, 11, 12)):
            world.t = step
            world._remember(agent, other)
        self.assertEqual(set(agent.memory), {11, 12})
        world.t = 5
        world._prune_memory(agent)
        self.assertEqual(agent.memory, {})

    def test_movement_and_fleeing_are_one_cardinal_step_with_costs(self):
        world, (agent, _, _) = self.world()
        before = agent.energy
        world.do_move(agent, (-1, 0), fleeing=True)
        self.assertEqual((agent.x, agent.y), (6, 0))
        self.assertAlmostEqual(before - agent.energy, world.cfg.move_cost_scale * (1 - agent.v) + world.cfg.flee_cost)
        self.assertEqual(world.interactions[-1]["kind"], "flee")
        with self.assertRaises(ValueError):
            world.do_move(agent, (1, 1))
        with self.assertRaises(ValueError):
            world.do_move(agent, (3, 0))

    def test_attack_checks_range_identity_and_energy_before_charging(self):
        world, (actor, target, _) = self.world()
        target.x = 3
        before = [a.energy for a in world.agents.values()]
        for target_id in (target.id, actor.id, 999, None):
            self.assertFalse(world.do_attack(actor, target_id))
        self.assertEqual([a.energy for a in world.agents.values()], before)
        self.assertEqual(world.interactions, [])
        target.x = 6
        actor.energy = world.cfg.attack_cost
        self.assertFalse(world.do_attack(actor, target.id))

    def test_combat_dissipates_energy_records_threat_and_kill_creates_no_food(self):
        world, (actor, target, _) = self.world()
        before = actor.energy + target.energy
        self.assertTrue(world.do_attack(actor, target.id))
        attack = world.interactions[-1]
        self.assertAlmostEqual(before - actor.energy - target.energy, attack["damage"] + attack["energy_cost"])
        self.assertTrue(world.observe(target).neighbors[0].harmed_me)
        target.energy = 0.1
        target.self_produced_inventory[:] = (2, 3)
        prior_energy, prior_stock = actor.energy, actor.inventory.copy()
        self.assertTrue(world.do_attack(actor, target.id))
        self.assertNotIn(target.id, world.agents)
        self.assertAlmostEqual(prior_energy - actor.energy, world.cfg.attack_cost)
        np.testing.assert_equal(actor.inventory, prior_stock)
        self.assertEqual(world.carcasses[target.id].edible_energy, world.cfg.body_energy)
        self.assertEqual(world.interactions[-1]["discarded_inventory"], [0, 2, 3])
        self.assertEqual(world.total_deaths, 1)
        self.assertFalse(world.do_attack(actor, target.id))
        self.assertEqual(world.total_deaths, 1)

    def test_killed_agent_cannot_act_resurrect_or_trade_later_in_step(self):
        class NoShuffle:
            def shuffle(self, values):
                pass

        class AttackOrEat:
            def choose_action(self, observation, rng):
                return Action(ActionKind.ATTACK, target_id=1) if observation.energy > 5 else Action(ActionKind.USE)

        world, (actor, target) = self.world(initial_agents=2)
        world.record_actions = True
        world.rng = NoShuffle()
        world.policy = AttackOrEat()
        target.energy = 0.1
        target.inventory[0] = 1
        target.self_produced_inventory[:] = (0.5, 0.5)
        world.step()
        self.assertNotIn(target.id, world.last_actions)
        self.assertEqual(target.age, 0)
        self.assertEqual(world.usage_totals["recipe_uses"], 0)
        self.assertEqual(world.trades, [])
        self.assertEqual(world.total_deaths, 1)
        self.assertTrue(all(event["t"] == 1 for event in world.interactions))

    def test_competing_eaters_cannot_overdraw_body_or_gain_from_a_live_agent(self):
        world, (first, second, victim) = self.world(bite_energy=3)
        self.assertFalse(world.do_eat_agent(first, victim.id))
        victim.energy = 0
        world._remove_dead()
        before = first.energy + second.energy
        self.assertTrue(world.do_eat_agent(first, victim.id))
        self.assertTrue(world.do_eat_agent(second, victim.id))
        self.assertFalse(world.do_eat_agent(first, victim.id))
        self.assertEqual(world.carcasses, {})
        self.assertEqual(world.social_totals["body_energy_eaten"], 4)
        self.assertAlmostEqual(first.energy + second.energy - before,
                               4 * world.cfg.predation_efficiency - 2 * world.cfg.eating_cost)
        self.assertAlmostEqual(world.social_totals["predation_energy_received"]
                               + world.social_totals["predation_energy_loss"], 4)
        self.assertEqual([e["gross_energy"] for e in world.interactions if e["kind"] == "eat_agent"], [3, 1])

    def test_eating_remains_requires_contact_and_available_handling_energy(self):
        world, (actor, victim, _) = self.world()
        victim.x = 3
        victim.energy = 0
        world._remove_dead()
        self.assertFalse(world.do_eat_agent(actor, victim.id))
        actor.x = 2
        actor.energy = world.cfg.eating_cost
        self.assertFalse(world.do_eat_agent(actor, victim.id))
        self.assertEqual(world.carcasses[victim.id].edible_energy, world.cfg.body_energy)

    def test_trade_requires_both_consents_and_charges_only_accepted_offers(self):
        for reject in (None, 1, 2):
            with self.subTest(reject=reject):
                world, (first, second) = self.world(initial_agents=2, trade_efficiency=0.75)
                first.self_produced_inventory[:] = (2, 0)
                second.self_produced_inventory[:] = (0, 2)
                world.policy = ConsentingPolicy(reject)
                before = [first.energy, second.energy]
                stocks = np.array([first.self_produced_inventory.copy(), second.self_produced_inventory.copy()])
                world.exchange()
                self.assertEqual(len(world.policy.offers), 2)
                if reject is not None:
                    self.assertEqual(world.trades, [])
                    np.testing.assert_equal([first.self_produced_inventory, second.self_produced_inventory], stocks)
                    self.assertEqual([first.energy, second.energy], before)
                    self.assertEqual(first.memory, {})
                else:
                    self.assertEqual(len(world.trades), 1)
                    for agent, energy in zip((first, second), before):
                        self.assertAlmostEqual(energy - agent.energy, world.cfg.trade_energy_cost)
                        self.assertEqual(len(agent.memory), 1)
                    loss = world.trades[0]["transaction_loss"]
                    self.assertAlmostEqual(first.stock(1) + second.stock(1) + loss["B"], 2)
                    self.assertAlmostEqual(first.stock(2) + second.stock(2) + loss["C"], 2)

    def test_social_actions_are_disabled_in_baseline_control(self):
        world, (first, second, _) = self.world(social_enabled=False)
        self.assertFalse(world.do_attack(first, second.id))
        self.assertFalse(world.do_move(first, (1, 0), fleeing=True))
        self.assertEqual(world.interactions, [])

    def test_social_runs_preserve_finite_budgets_and_are_reproducible(self):
        cfg = Config(L=6, initial_agents=20, social_enabled=True, finite_resource_fraction=0.5)
        world = World(cfg, seed=17)
        for _ in range(80):
            world.step()
            self.assertAlmostEqual(world.initial_body_energy + world.social_totals["body_energy_born"],
                sum(a.body_energy for a in world.agents.values()) + sum(b.edible_energy for b in world.carcasses.values())
                + world.social_totals["body_energy_eaten"])
            self.assertAlmostEqual(world.initial_finite_A,
                world.resource_metrics()["finite_A_remaining"] + world.cumulative_finite_A_harvested)
            self.assertEqual(len(world.agents), cfg.initial_agents + world.total_births - world.total_deaths)
            for agent in world.agents.values():
                self.assertTrue(all(agent.stock(k) >= -1e-12 for k in range(3)))
                self.assertTrue(-1 <= agent.lawfulness <= 1 and -1 <= agent.concern <= 1)
                self.assertLessEqual(len(agent.memory), cfg.memory_capacity)
        repeat = World(cfg, seed=17)
        for _ in range(80):
            repeat.step()
        np.testing.assert_equal(world.history, repeat.history)
        self.assertEqual(world.interactions, repeat.interactions)
        self.assertEqual(world.trades, repeat.trades)
        self.assertEqual(world.rng.bit_generator.state, repeat.rng.bit_generator.state)

    def test_exports_preserve_traits_events_and_policy_condition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = Config(L=4, initial_agents=8, social_enabled=True)
            output = condition_directory(root, config, 7)
            world = run_experiment(config, 7, 12, output, make_plots=False)
            self.assertIn("social", output.parts)
            traits = json.loads((output / "agent_traits.json").read_text())
            self.assertEqual(traits["founders"], world.founder_traits)
            self.assertEqual(traits["survivors"], world.agent_traits())
            self.assertEqual(json.loads((output / "interactions.json").read_text()), json.loads(json.dumps(world.interactions)))
            self.assertTrue(json.loads((output / "config.json").read_text())["config"]["social_enabled"])


class TestSocialDecisions(unittest.TestCase):
    def setUp(self):
        self.policy = SocialPolicy(Config(social_enabled=True))
        self.neighbor = Neighbor(9, 1, 0, 1, 4, (0, 2), (0, 0))
        self.observation = Observation(
            energy=3, inventory=(0,), genotype=(0.8, 0.6), local_resources=(0,),
            population=2, neighbors=(self.neighbor,), grid_size=8,
        )

    def probability(self, observation, kind):
        return sum(p for action, p in self.policy.action_distribution(observation) if action.kind is kind)

    def test_concern_changes_attack_tendency_without_forcing_a_type(self):
        good = replace(self.observation, concern=1)
        evil = replace(self.observation, concern=-1)
        self.assertGreater(self.probability(evil, ActionKind.ATTACK), self.probability(good, ActionKind.ATTACK))
        for order in (-1, 0, 1):
            for concern in (-1, 0, 1):
                observation = replace(self.observation, lawfulness=order, concern=concern)
                probability = self.probability(observation, ActionKind.ATTACK)
                self.assertGreater(probability, 0)
                self.assertLess(probability, 1)
        fed = replace(evil, energy=20)
        self.assertEqual(self.probability(fed, ActionKind.ATTACK), 0)

    def test_actual_threat_enables_good_agent_defense_and_fleeing(self):
        safe = replace(self.observation, energy=12, concern=1)
        threatened = replace(safe, neighbors=(replace(self.neighbor, harmed_me=True),))
        self.assertEqual(self.probability(safe, ActionKind.FLEE), 0)
        self.assertGreater(self.probability(threatened, ActionKind.ATTACK), 0)
        self.assertGreater(self.probability(threatened, ActionKind.FLEE), 0)

    def test_evil_can_accept_profitable_trade_and_consistency_uses_experience(self):
        offer = TradeOffer(self.neighbor, 1, 2, 0.5, 0.5)
        evil = replace(self.observation, concern=-1, lawfulness=1)
        self.assertGreater(self.policy.trade_probability(evil, offer), 0.5)
        known = replace(offer, partner=replace(self.neighbor, recent_trades=3))
        hostile = replace(known, partner=replace(known.partner, harmed_me=True))
        self.assertGreater(self.policy.trade_probability(evil, known), self.policy.trade_probability(evil, offer))
        self.assertLess(self.policy.trade_probability(evil, hostile), self.policy.trade_probability(evil, offer))
        self.assertGreater(self.policy.trade_probability(evil, known),
                           self.policy.trade_probability(replace(evil, lawfulness=-1), known))

    def test_directed_search_prefers_visible_resources_and_never_targets_distant_agents(self):
        observation = replace(self.observation, neighbors=(replace(self.neighbor, dx=2),),
                              adjacent_resources=((1, 0, 8), (-1, 0, 0), (0, 1, 0), (0, -1, 0)))
        distribution = self.policy.action_distribution(observation)
        moves = {action.direction: p for action, p in distribution if action.kind is ActionKind.MOVE}
        self.assertGreater(moves[(1, 0)], moves[(-1, 0)])
        self.assertEqual(self.probability(observation, ActionKind.ATTACK), 0)
        self.assertAlmostEqual(sum(p for _, p in distribution), 1)

    def test_finite_carcass_can_be_chosen_for_food_but_is_not_a_live_prey(self):
        observation = replace(self.observation, neighbors=(), bodies=(NearbyBody(91, 0, 0, 1),))
        self.assertGreater(self.probability(observation, ActionKind.EAT_AGENT), 0)
        self.assertEqual(self.probability(observation, ActionKind.ATTACK), 0)


if __name__ == "__main__":
    unittest.main()
