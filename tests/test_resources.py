from dataclasses import fields
import unittest

import numpy as np

from ht import Agent, Config, World


class TestFiniteResources(unittest.TestCase):
    def assert_same_agents(self, first, second):
        self.assertEqual(list(first.agents), list(second.agents))
        for aid in first.agents:
            for field in fields(Agent):
                np.testing.assert_equal(
                    getattr(first.agents[aid], field.name), getattr(second.agents[aid], field.name)
                )
        self.assertEqual(first.rng.bit_generator.state, second.rng.bit_generator.state)

    def test_resource_configuration_bounds(self):
        self.assertEqual(Config().finite_resource_fraction, 0.0)
        for fraction in (-0.1, 1.1, np.nan, np.inf):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                Config(finite_resource_fraction=fraction)
        for seed in (-1, 2.5, np.nan):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                Config(resource_seed=seed)
        for name in ("base_capacity", "regen_rate", "q_max"):
            for value in (-1, np.nan, np.inf):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    Config(**{name: value})
        for contrast in (np.nan, np.inf, -9):
            with self.subTest(contrast=contrast), self.assertRaises(ValueError):
                Config(capacity_contrast=contrast)

    def test_source_fraction_changes_only_renewability_at_initialization(self):
        reference = World(Config(L=8, initial_agents=12), seed=7)
        previous_finite = np.zeros((8, 8), dtype=bool)
        for fraction in (0.0, 0.25, 0.5, 1.0):
            world = World(Config(L=8, initial_agents=12, finite_resource_fraction=fraction), seed=7)
            self.assert_same_agents(reference, world)
            np.testing.assert_array_equal(reference.resources, world.resources)
            np.testing.assert_array_equal(reference.capacity, world.capacity)
            finite = ~world.renewable_mask
            self.assertEqual(int(finite.sum()), round(64 * fraction))
            self.assertTrue(np.all(finite[previous_finite]))
            self.assertFalse(world.renewable_mask.flags.writeable)
            previous_finite = finite

    def test_map_seed_changes_distribution_without_changing_agents_or_initial_stock(self):
        first = World(Config(L=12, initial_agents=8, finite_resource_fraction=0.5, resource_seed=11), seed=7)
        second = World(Config(L=12, initial_agents=8, finite_resource_fraction=0.5, resource_seed=12), seed=7)
        repeated_map = World(Config(L=12, initial_agents=8, finite_resource_fraction=0.5, resource_seed=11), seed=19)
        self.assert_same_agents(first, second)
        np.testing.assert_array_equal(first.resources, second.resources)
        self.assertFalse(np.array_equal(first.renewable_mask, second.renewable_mask))
        np.testing.assert_array_equal(first.renewable_mask, repeated_map.renewable_mask)

    def test_only_renewable_cells_regrow_and_finite_cells_never_refill(self):
        world = World(Config(L=4, initial_agents=0, base_capacity=2, capacity_contrast=0,
                             regen_rate=0.25, finite_resource_fraction=0.5), seed=7)
        world.resources[:] = 0.5
        mask = world.renewable_mask.copy()
        for _ in range(40):
            world.regenerate_resources()
        np.testing.assert_array_equal(world.resources[0, mask], 2)
        np.testing.assert_array_equal(world.resources[0, ~mask], 0.5)
        world.resources[0, ~mask] = 0
        for _ in range(40):
            world.regenerate_resources()
        np.testing.assert_array_equal(world.resources[0, ~mask], 0)
        np.testing.assert_array_equal(world.renewable_mask, mask)

    def test_harvesting_an_exhausted_deposit_cannot_create_goods(self):
        world = World(Config(L=1, initial_agents=1, base_capacity=0.25, capacity_contrast=0,
                             finite_resource_fraction=1), seed=7)
        agent = world.agents[0]
        agent.genotype[:] = (1, 0)
        world.do_harvest(agent, 0)
        self.assertEqual(agent.inventory[0], 0.25)
        self.assertEqual(world.resources[0, 0, 0], 0)
        for _ in range(10):
            world.regenerate_resources()
            world.do_harvest(agent, 0)
        self.assertEqual(agent.inventory[0], 0.25)
        self.assertEqual(world.resources[0, 0, 0], 0)
        self.assertEqual(world.cumulative_finite_A_harvested, 0.25)
        self.assertEqual(world.resource_metrics()["exhausted_finite_cells"], 1)
        self.assertAlmostEqual(agent.energy, world.cfg.initial_energy - 11 * world.cfg.harvest_cost)

    def test_agents_competing_for_the_last_stock_transfer_it_exactly_once(self):
        world = World(Config(L=1, initial_agents=2, base_capacity=1, capacity_contrast=0,
                             q_max=0.75, finite_resource_fraction=1), seed=7)
        first, second = world.agents.values()
        first.genotype[:] = second.genotype[:] = (1, 0)
        world.do_harvest(first, 0)
        world.regenerate_resources()
        world.do_harvest(second, 0)
        world.regenerate_resources()
        world.do_harvest(first, 0)
        self.assertEqual(first.inventory[0], 0.75)
        self.assertEqual(second.inventory[0], 0.25)
        self.assertEqual(world.resources[0, 0, 0], 0)
        self.assertEqual(world.initial_finite_A, world.cumulative_finite_A_harvested)

    def test_finite_budget_is_monotone_and_reconciles_with_harvesting_during_a_run(self):
        for fraction in (0.5, 1.0):
            with self.subTest(fraction=fraction):
                world = World(Config(L=8, initial_agents=20, base_capacity=0.5, capacity_contrast=0,
                                     finite_resource_fraction=fraction), seed=7)
                world.run(100)
                remaining = [row["finite_A_remaining"] for row in world.history]
                exhausted = [row["exhausted_finite_cells"] for row in world.history]
                self.assertTrue(all(after <= before for before, after in zip(remaining, remaining[1:])))
                self.assertTrue(all(after >= before for before, after in zip(exhausted, exhausted[1:])))
                self.assertGreater(world.cumulative_finite_A_harvested, 0)
                for row in world.history:
                    self.assertAlmostEqual(
                        row["initial_finite_A"], row["finite_A_remaining"] + row["cumulative_finite_A_harvested"],
                        places=10,
                    )
                self.assertTrue(np.all(world.resources >= 0))
                self.assertTrue(np.all(world.resources <= world.capacity))

    def test_empty_world_and_zero_stock_have_defined_resource_metrics(self):
        for fraction in (0, 1):
            world = World(Config(L=2, initial_agents=0, base_capacity=0, capacity_contrast=0,
                                 finite_resource_fraction=fraction), seed=7)
            metrics = world.resource_metrics()
            self.assertEqual(metrics["renewable_A_remaining"], 0)
            self.assertEqual(metrics["finite_A_remaining"], 0)
            self.assertEqual(metrics["finite_cells"], 4 * fraction)
            self.assertEqual(metrics["exhausted_finite_cells"], 4 * fraction)
            world.step()
            self.assertEqual(world.resource_metrics(), metrics)


if __name__ == "__main__":
    unittest.main()
