import unittest
import numpy as np

from ht import Action, ActionKind, Agent, Config, Observation, RuleBasedPolicy, World


class AlwaysMovePolicy:
    def choose_action(self, observation, rng):
        return Action(ActionKind.MOVE)


class AlwaysProducePolicy:
    def choose_action(self, observation, rng):
        return Action(ActionKind.PRODUCE)


class InvalidHarvestPolicy:
    def __init__(self, resource):
        self.resource = resource

    def choose_action(self, observation, rng):
        return Action(ActionKind.HARVEST, resource=self.resource)


class TestEndogenousNicheWorld(unittest.TestCase):
    def _single_agent_world(self, **overrides):
        cfg = Config(initial_agents=1, **overrides)
        world = World(cfg, seed=7)
        return cfg, world, next(iter(world.agents.values()))

    def _trade_world(self, efficiency=1.0, maximum=10.0, agents=2, seed=7):
        cfg = Config(
            L=4,
            initial_agents=agents,
            trade_efficiency=efficiency,
            max_trade_per_pair=maximum,
        )
        world = World(cfg, seed=seed)
        traders = list(world.agents.values())
        for trader in traders:
            trader.x = trader.y = 0
            trader.self_produced_inventory[:] = 0.0
            trader.received_inventory[:] = 0.0
        return cfg, world, traders

    def _make_complementary(self, cfg, b_donor, c_donor, surplus=1.0):
        b_donor.self_produced_inventory[:] = (cfg.recipe[1] + surplus, 0.0)
        c_donor.self_produced_inventory[:] = (0.0, cfg.recipe[2] + surplus)

    def test_genotype_normalization(self):
        g = Agent.normalize_genotype(np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertAlmostEqual(np.linalg.norm(g), 1.0, places=12)
        self.assertTrue(np.all(g >= 0.0))

    def test_resources_never_exceed_capacity(self):
        cfg = Config(L=12, initial_agents=10)
        w = World(cfg, seed=1)
        for _ in range(20):
            w.regenerate_resources()
        self.assertTrue(np.all(w.resources <= w.capacity + 1e-12))

    def test_periodic_coordinates(self):
        cfg = Config(L=5, initial_agents=1)
        w = World(cfg, seed=2)
        a = next(iter(w.agents.values()))
        a.x = 0
        a.x = (a.x - 1) % cfg.L
        self.assertEqual(a.x, 4)

    def test_reproducibility_fixed_seed(self):
        cfg = Config(L=15, initial_agents=40, max_agents=500)
        w1 = World(cfg, seed=123)
        w2 = World(cfg, seed=123)
        w1.run(100)
        w2.run(100)
        self.assertEqual(w1.history, w2.history)

    def test_policy_can_be_replaced_without_changing_world(self):
        cfg = Config(L=5, initial_agents=1, metabolism=0.0, move_cost_scale=0.0)
        world = World(cfg, seed=12, policy=AlwaysMovePolicy())
        agent = next(iter(world.agents.values()))
        inventory_before = agent.inventory.copy()

        world.step()

        np.testing.assert_array_equal(agent.inventory, inventory_before)
        self.assertEqual(agent.age, 1)

    def test_observation_is_immutable_policy_input(self):
        observation = Observation(
            energy=1.0,
            inventory=(1.0, 1.0, 1.0),
            genotype=(1.0, 1.0, 1.0, 1.0),
            local_resources=(1.0, 1.0, 1.0),
            population=10,
        )
        with self.assertRaises(TypeError):
            observation.inventory[0] = 0.0
        with self.assertRaises(AttributeError):
            observation.energy = 2.0

    def test_invalid_harvest_resource_indices_are_rejected(self):
        cfg = Config(L=5, initial_agents=1)
        for resource in (-1, cfg.num_resources):
            with self.subTest(resource=resource):
                world = World(cfg, seed=12, policy=InvalidHarvestPolicy(resource))
                with self.assertRaisesRegex(ValueError, "invalid harvest resource index"):
                    world.step()

    def test_no_global_fitness_attribute(self):
        cfg = Config(L=10, initial_agents=1)
        w = World(cfg, seed=3)
        a = next(iter(w.agents.values()))
        self.assertFalse(hasattr(a, "fitness"))

    def test_reproduction_conserves_child_energy(self):
        cfg = Config(L=10, initial_agents=1)
        w = World(cfg, seed=5)
        parent = next(iter(w.agents.values()))
        parent.energy = 50.0
        before_total = sum(a.energy for a in w.agents.values())
        w.do_reproduce(parent)
        after_total = sum(a.energy for a in w.agents.values())
        self.assertEqual(len(w.agents), 2)
        self.assertAlmostEqual(before_total - after_total, cfg.reproduce_cost, places=12)
        child = max(w.agents.values(), key=lambda a: a.id)
        self.assertEqual(child.generation, parent.generation + 1)

    def test_production_allocation_endpoints_and_midpoint(self):
        cases = ((0.0, (0.0, 1.0)), (0.5, (0.5, 0.5)), (1.0, (1.0, 0.0)))
        for s, expected in cases:
            with self.subTest(s=s):
                cfg, world, agent = self._single_agent_world(production_amount=1.0)
                agent.production_allocation = s
                agent.received_inventory[:] = (0.2, 0.3)
                before_a = agent.inventory.copy()
                before_received = agent.received_inventory.copy()
                world.do_produce(agent)
                np.testing.assert_allclose(agent.self_produced_inventory, expected)
                np.testing.assert_array_equal(agent.inventory, before_a)
                np.testing.assert_array_equal(agent.received_inventory, before_received)
                self.assertAlmostEqual(
                    agent.self_produced_inventory.sum(), cfg.production_amount
                )

    def test_production_exactly_conserves_configured_output(self):
        cfg, world, agent = self._single_agent_world(production_amount=2.75)
        before = agent.self_produced_inventory.copy()
        world.do_produce(agent)
        produced = agent.self_produced_inventory - before
        self.assertEqual(float(produced.sum()), cfg.production_amount)

    def test_nonlinear_production_keeps_endpoints_and_reduces_mixed_output(self):
        cases = (
            (0.0, (0.0, 1.0)),
            (0.25, (0.0625, 0.5625)),
            (0.5, (0.25, 0.25)),
            (1.0, (1.0, 0.0)),
        )
        for allocation, expected in cases:
            with self.subTest(allocation=allocation):
                cfg, world, agent = self._single_agent_world(
                    production_specialization_exponent=2.0
                )
                agent.production_allocation = allocation
                agent.received_inventory[:] = (0.2, 0.3)
                before_energy = agent.energy
                world.do_produce(agent)
                np.testing.assert_allclose(agent.self_produced_inventory, expected)
                np.testing.assert_array_equal(agent.received_inventory, (0.2, 0.3))
                self.assertAlmostEqual(before_energy - agent.energy, cfg.production_cost)

    def test_production_accepts_fractional_gamma(self):
        _, world, agent = self._single_agent_world(production_specialization_exponent=1.5)
        agent.production_allocation = 0.5
        world.do_produce(agent)
        np.testing.assert_allclose(agent.self_produced_inventory, (2**-1.5, 2**-1.5))

    def test_production_and_exploration_configuration_bounds(self):
        for gamma in (0.0, 0.99, np.nan, np.inf, -np.inf):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                Config(production_specialization_exponent=gamma)
        for probability in (-0.1, 1.1, np.nan, np.inf):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                Config(deficit_move_probability=probability)

    def test_deficient_agents_can_explore_in_both_trade_conditions(self):
        for trade_enabled in (False, True):
            for allocation in (0.0, 0.5, 1.0):
                observation = Observation(
                    energy=12.0, inventory=(0.0,), genotype=(1.0, 0.0),
                    local_resources=(8.0,), population=1,
                    production_allocation=allocation,
                )
                policy = RuleBasedPolicy(Config(
                    trade_enabled=trade_enabled, deficit_move_probability=1.0
                ))
                self.assertEqual(
                    policy.choose_action(observation, np.random.default_rng(7)).kind,
                    ActionKind.MOVE,
                )

    def test_zero_deficit_exploration_preserves_production_and_rng_state(self):
        observation = Observation(
            energy=12.0, inventory=(0.0,), genotype=(1.0, 0.0),
            local_resources=(8.0,), population=1,
        )
        rng = np.random.default_rng(7)
        expected_rng = np.random.default_rng(7)
        policy = RuleBasedPolicy(Config(deficit_move_probability=0.0))
        self.assertEqual(policy.choose_action(observation, rng).kind, ActionKind.PRODUCE)
        self.assertEqual(rng.random(), expected_rng.random())

    def test_recipe_use_takes_priority_over_exploration(self):
        observation = Observation(
            energy=4.0, inventory=(1.0,), genotype=(1.0, 0.0),
            local_resources=(8.0,), population=1,
            self_produced_inventory=(0.5, 0.0), received_inventory=(0.0, 0.5),
        )
        policy = RuleBasedPolicy(Config(deficit_move_probability=1.0))
        self.assertEqual(
            policy.choose_action(observation, np.random.default_rng(7)).kind, ActionKind.USE
        )

    def test_bc_inventory_alone_provides_no_energy(self):
        cfg, world, agent = self._single_agent_world()
        agent.inventory[0] = 0.0
        agent.self_produced_inventory[:] = cfg.recipe[1:]
        before = agent.energy
        self.assertFalse(world.do_use(agent))
        self.assertEqual(agent.energy, before)

    def test_recipe_use_is_atomic(self):
        cfg, world, agent = self._single_agent_world()
        agent.inventory[0] = cfg.recipe[0]
        agent.self_produced_inventory[:] = (cfg.recipe[1], 0.0)
        before_a = agent.inventory.copy()
        before_bc = agent.self_produced_inventory.copy()
        self.assertFalse(world.do_use(agent))
        np.testing.assert_array_equal(agent.inventory, before_a)
        np.testing.assert_array_equal(agent.self_produced_inventory, before_bc)

        agent.received_inventory[1] = cfg.recipe[2]
        agent.inventory[0] += 0.25
        agent.self_produced_inventory[0] += 0.25
        agent.received_inventory[1] += 0.25
        before_energy = agent.energy
        self.assertTrue(world.do_use(agent))
        self.assertAlmostEqual(
            agent.energy, before_energy + cfg.recipe_energy_gain - cfg.use_cost
        )
        self.assertAlmostEqual(agent.stock(0), 0.25)
        self.assertAlmostEqual(agent.stock(1), 0.25)
        self.assertAlmostEqual(agent.stock(2), 0.25)
        self.assertTrue(np.all(agent.self_produced_inventory >= 0.0))
        self.assertTrue(np.all(agent.received_inventory >= 0.0))

    def test_recipe_use_rejects_tiny_shortfall_without_any_changes(self):
        cfg, world, agent = self._single_agent_world()
        agent.inventory[0] = cfg.recipe[0]
        agent.self_produced_inventory[0] = cfg.recipe[1] - 1e-13
        agent.received_inventory[1] = cfg.recipe[2]

        before_a = agent.inventory.copy()
        before_self = agent.self_produced_inventory.copy()
        before_received = agent.received_inventory.copy()
        before_energy = agent.energy

        self.assertFalse(world.do_use(agent))
        np.testing.assert_array_equal(agent.inventory, before_a)
        np.testing.assert_array_equal(agent.self_produced_inventory, before_self)
        np.testing.assert_array_equal(agent.received_inventory, before_received)
        self.assertEqual(agent.energy, before_energy)
        self.assertTrue(np.all(agent.self_produced_inventory >= 0.0))
        self.assertTrue(np.all(agent.received_inventory >= 0.0))

    def test_allocation_mutation_stays_bounded(self):
        cfg, world, parent = self._single_agent_world(
            allocation_mutation_sigma=100.0, max_agents=1000
        )
        for endpoint in (0.0, 1.0):
            parent.production_allocation = endpoint
            for _ in range(100):
                parent.energy = 50.0
                world.do_reproduce(parent)
                child = world.agents[world.next_agent_id - 1]
                self.assertGreater(child.production_allocation, 0.0)
                self.assertLess(child.production_allocation, 1.0)

    def test_allocation_reflection_handles_arbitrary_overshoots(self):
        cases = {
            -0.08: 0.08,
            1.07: 0.93,
            -2.08: 0.08,
            3.07: 0.93,
            10.25: 0.25,
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertAlmostEqual(
                    Agent.reflect_production_allocation(value), expected
                )

    def test_production_allocation_is_independent_of_harvest_and_movement(self):
        cfg, world, agent = self._single_agent_world()
        genotype = agent.genotype.copy()
        h_a, movement = agent.hA, agent.v

        harvest_amounts = []
        movement_costs = []
        for allocation in (0.0, 1.0):
            agent.production_allocation = allocation
            agent.inventory[0] = 0.0
            world.resources[0, agent.y, agent.x] = world.capacity[0, agent.y, agent.x]
            before_resource = world.resources[0, agent.y, agent.x]
            world.do_harvest(agent, 0)
            harvest_amounts.append(
                before_resource - world.resources[0, agent.y, agent.x]
            )

            agent.energy = 10.0
            world.do_move(agent)
            movement_costs.append(10.0 - agent.energy)

            np.testing.assert_array_equal(agent.genotype, genotype)
            self.assertEqual(agent.hA, h_a)
            self.assertEqual(agent.v, movement)

        self.assertAlmostEqual(harvest_amounts[0], harvest_amounts[1])
        self.assertAlmostEqual(movement_costs[0], movement_costs[1])

    def test_baseline_has_turnover(self):
        cfg = Config(L=20, initial_agents=80, max_agents=800)
        w = World(cfg, seed=11)
        w.run(150)
        final = w.history[-1]
        self.assertGreater(final["births"], 0)
        self.assertGreater(final["deaths"], 0)
        self.assertGreater(final["max_generation"], 0)

    def test_history_records_manufactured_inventory_totals(self):
        cfg, world, agent = self._single_agent_world()
        agent.self_produced_inventory[:] = (1.25, 2.5)
        agent.received_inventory[:] = (0.75, 0.5)

        world.record_metrics()

        self.assertEqual(world.history[-1]["total_B_stock"], 2.0)
        self.assertEqual(world.history[-1]["total_C_stock"], 3.0)

    def test_recipe_metrics_count_actual_receipt_consumption_only(self):
        _, world, agent = self._single_agent_world()
        agent.self_produced_inventory[:] = (0.25, 0.5)
        agent.received_inventory[:] = (0.25, 0.0)
        before = world.usage_totals.copy()
        self.assertFalse(world.do_use(agent))  # A is missing.
        self.assertEqual(world.usage_totals, before)
        agent.inventory[0] = 2.0
        self.assertTrue(world.do_use(agent))
        agent.self_produced_inventory[:] = (0.5, 0.5)
        self.assertTrue(world.do_use(agent))
        world.record_metrics()
        final = world.history[-1]
        self.assertEqual(final["cumulative_recipe_uses"], 2)
        self.assertEqual(final["cumulative_recipe_uses_with_received_stock"], 1)
        self.assertEqual(final["cumulative_received_B_used"], 0.25)
        self.assertEqual(final["cumulative_received_C_used"], 0.0)

    def test_exchange_follows_death_removal_and_precedes_metrics(self):
        cfg = Config(
            initial_agents=3, production_amount=0.0, production_cost=0.0,
            metabolism=0.2, trade_efficiency=0.75,
        )
        world = World(cfg, seed=7, policy=AlwaysProducePolicy())
        first, second, doomed = world.agents.values()
        for agent in (first, second, doomed):
            agent.x = agent.y = 0
        self._make_complementary(cfg, first, second)
        doomed.energy = 0.1
        world.step()
        self.assertNotIn(doomed.id, world.agents)
        self.assertEqual(set(world.trades[0]["partner_ids"]), {first.id, second.id})
        self.assertEqual(world.trades[0]["t"], 1)
        self.assertEqual(world.history[-1]["cumulative_trade_events"], 1)
        for good in ("B", "C"):
            self.assertAlmostEqual(world.history[-1][f"cumulative_trade_gross_{good}"], 2 / 3)
            self.assertAlmostEqual(world.history[-1][f"cumulative_trade_received_{good}"], 0.5)
            self.assertAlmostEqual(world.history[-1][f"cumulative_trade_loss_{good}"], 1 / 6)
        before = world.history[-1].copy()
        world.record_metrics()
        self.assertEqual(world.history[-1], before)

    def test_no_colocation_means_no_trade(self):
        cfg, world, agents = self._trade_world()
        self._make_complementary(cfg, *agents)
        agents[1].x = 1
        before = [a.self_produced_inventory.copy() for a in agents]
        world.exchange()
        self.assertEqual(world.trades, [])
        for agent, stock in zip(agents, before):
            np.testing.assert_array_equal(agent.self_produced_inventory, stock)

    def test_trade_configuration_bounds_are_required(self):
        for efficiency in (0.0, -0.1, 1.1, np.nan):
            with self.subTest(efficiency=efficiency):
                with self.assertRaises(ValueError):
                    Config(trade_efficiency=efficiency)
        for maximum in (-0.1, np.nan):
            with self.subTest(maximum=maximum):
                with self.assertRaises(ValueError):
                    Config(max_trade_per_pair=maximum)

    def test_complementary_colocation_trades_exactly(self):
        cfg, world, agents = self._trade_world()
        self._make_complementary(cfg, *agents, surplus=0.4)
        world.exchange()
        np.testing.assert_allclose(agents[0].self_produced_inventory, (0.5, 0.0))
        np.testing.assert_allclose(agents[1].self_produced_inventory, (0.0, 0.5))
        np.testing.assert_allclose(agents[0].received_inventory, (0.0, 0.4))
        np.testing.assert_allclose(agents[1].received_inventory, (0.4, 0.0))
        self.assertEqual(len(world.trades), 1)

    def test_same_sided_producers_do_not_trade(self):
        cfg, world, agents = self._trade_world()
        for agent in agents:
            agent.self_produced_inventory[0] = cfg.recipe[1] + 1.0
        world.exchange()
        self.assertEqual(world.trades, [])

    def test_trade_cap_applies_in_both_directions(self):
        cfg, world, agents = self._trade_world(maximum=0.2)
        self._make_complementary(cfg, *agents)
        world.exchange()
        event = world.trades[0]
        self.assertEqual({v["gross"] for v in event["surrendered"].values()}, {0.2})

    def test_trade_efficiency_controls_receipts_and_loss(self):
        cfg, world, agents = self._trade_world(efficiency=0.5)
        self._make_complementary(cfg, *agents)
        world.exchange()
        np.testing.assert_allclose(agents[0].received_inventory, (0.0, 0.5))
        np.testing.assert_allclose(agents[1].received_inventory, (0.5, 0.0))
        self.assertEqual(world.trades[0]["transaction_loss"], {"B": 0.5, "C": 0.5})

    def test_odd_population_leaves_one_agent_unpaired(self):
        cfg, world, agents = self._trade_world(agents=3)
        self._make_complementary(cfg, agents[0], agents[1])
        agents[2].self_produced_inventory[0] = cfg.recipe[1] + 1.0
        world.exchange()
        paired = set(world.trades[0]["partner_ids"])
        self.assertEqual(len(paired), 2)
        self.assertEqual(len(set(world.agents) - paired), 1)

    def test_agent_has_at_most_one_trade_per_timestep(self):
        cfg, world, agents = self._trade_world(agents=4)
        self._make_complementary(cfg, agents[0], agents[1])
        self._make_complementary(cfg, agents[2], agents[3])
        world.exchange()
        participant_ids = [aid for event in world.trades for aid in event["partner_ids"]]
        self.assertEqual(len(participant_ids), len(set(participant_ids)))

    def test_disabled_trade_changes_nothing(self):
        cfg = Config(L=4, initial_agents=2, trade_enabled=False)
        world = World(cfg, seed=7)
        agents = list(world.agents.values())
        for agent in agents:
            agent.x = agent.y = 0
        self._make_complementary(cfg, *agents)
        before = [a.self_produced_inventory.copy() for a in agents]
        world.exchange()
        self.assertEqual(world.trades, [])
        for agent, stock in zip(agents, before):
            np.testing.assert_array_equal(agent.self_produced_inventory, stock)

    def test_received_inventory_cannot_be_reexported(self):
        cfg, world, agents = self._trade_world()
        agents[0].received_inventory[0] = cfg.recipe[1] + 10.0
        agents[1].self_produced_inventory[1] = cfg.recipe[2] + 1.0
        world.exchange()
        self.assertEqual(world.trades, [])

    def test_exchange_does_not_modify_energy(self):
        cfg, world, agents = self._trade_world()
        self._make_complementary(cfg, *agents)
        energy = [a.energy for a in agents]
        world.exchange()
        self.assertEqual([a.energy for a in agents], energy)

    def test_exchange_stock_conservation_accounts_for_loss_by_good(self):
        cfg, world, agents = self._trade_world(efficiency=0.75)
        self._make_complementary(cfg, *agents)
        before = [sum(a.stock(k) for a in agents) for k in (1, 2)]
        world.exchange()
        after = [sum(a.stock(k) for a in agents) for k in (1, 2)]
        loss = world.trades[0]["transaction_loss"]
        self.assertAlmostEqual(before[0], after[0] + loss["B"])
        self.assertAlmostEqual(before[1], after[1] + loss["C"])


if __name__ == "__main__":
    unittest.main()
