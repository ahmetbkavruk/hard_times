import unittest
import numpy as np

from ht import Action, ActionKind, Agent, Config, Observation, World


class AlwaysMovePolicy:
    def choose_action(self, observation, rng):
        return Action(ActionKind.MOVE)


class RecordingPolicy:
    def __init__(self):
        self.observation = None

    def choose_action(self, observation, rng):
        self.observation = observation
        return Action(ActionKind.MOVE)


class InvalidHarvestPolicy:
    def __init__(self, resource):
        self.resource = resource

    def choose_action(self, observation, rng):
        return Action(ActionKind.HARVEST, self.resource)


class TestEndogenousNicheWorld(unittest.TestCase):
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

    def test_policy_receives_only_read_only_observation_and_rng(self):
        policy = RecordingPolicy()
        world = World(Config(L=5, initial_agents=1), seed=13, policy=policy)
        agent = next(iter(world.agents.values()))
        expected_energy = agent.energy
        expected_inventory = tuple(agent.inventory)
        expected_genotype = tuple(agent.genotype)

        world.step()

        observation = policy.observation
        self.assertIsInstance(observation, Observation)
        self.assertEqual(observation.energy, expected_energy)
        self.assertEqual(observation.inventory, expected_inventory)
        self.assertEqual(observation.genotype, expected_genotype)
        self.assertIsInstance(observation.local_resources, tuple)
        self.assertFalse(hasattr(observation, "do_harvest"))
        with self.assertRaises(AttributeError):
            observation.energy = 0.0
        with self.assertRaises(TypeError):
            observation.inventory[0] = 0.0

    def test_harvest_resource_index_must_be_in_range(self):
        cfg = Config(L=5, initial_agents=1)
        for invalid_index in (-1, cfg.num_resources):
            with self.subTest(resource=invalid_index):
                world = World(
                    cfg, seed=14, policy=InvalidHarvestPolicy(invalid_index)
                )
                with self.assertRaisesRegex(ValueError, "resource index"):
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

    def test_baseline_has_turnover(self):
        cfg = Config(L=20, initial_agents=80, max_agents=800)
        w = World(cfg, seed=11)
        w.run(150)
        final = w.history[-1]
        self.assertGreater(final["births"], 0)
        self.assertGreater(final["deaths"], 0)
        self.assertGreater(final["max_generation"], 0)


if __name__ == "__main__":
    unittest.main()
