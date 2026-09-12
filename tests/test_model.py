import unittest
import numpy as np

from ht import Action, ActionKind, Agent, Config, Observation, World


class AlwaysMovePolicy:
    def choose_action(self, observation, rng):
        return Action(ActionKind.MOVE)


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


if __name__ == "__main__":
    unittest.main()
