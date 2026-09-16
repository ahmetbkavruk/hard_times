import unittest

import numpy as np

from ht import Config, World


class TestCompatibleMatching(unittest.TestCase):
    def _world(self, agents=2, **overrides):
        config = Config(L=4, initial_agents=agents, trade_matching="compatible", **overrides)
        world = World(config, seed=7)
        agents = list(world.agents.values())
        for agent in agents:
            agent.x = agent.y = 0
            agent.self_produced_inventory[:] = config.recipe[1:]
        return world, agents

    def _complementary(self, first, second):
        first.self_produced_inventory[:] = (1.5, 0.0)
        second.self_produced_inventory[:] = (0.0, 1.5)

    def test_matching_configuration_is_validated_and_neighborhood_is_default(self):
        self.assertEqual(Config().trade_matching, "neighborhood")
        for value in ("unknown", "", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Config(trade_matching=value)

    def test_compatible_pair_is_found_among_balanced_bystanders(self):
        world, agents = self._world(20)
        self._complementary(agents[0], agents[1])
        # Matching must use stock and needs, not an allocation label.
        agents[0].production_allocation = 0.0
        agents[1].production_allocation = 1.0
        world.exchange()
        self.assertEqual(len(world.trades), 1)
        self.assertEqual(set(world.trades[0]["partner_ids"]), {agents[0].id, agents[1].id})
        for agent in agents[2:]:
            np.testing.assert_array_equal(agent.self_produced_inventory, (0.5, 0.5))
            np.testing.assert_array_equal(agent.received_inventory, (0.0, 0.0))

    def test_unbalanced_sides_form_disjoint_pairs_with_caps_and_loss(self):
        world, agents = self._world(11, trade_efficiency=0.75, max_trade_per_pair=0.2)
        for agent in agents[:7]:
            agent.self_produced_inventory[:] = (1.5, 0.0)
        for agent in agents[7:10]:
            agent.self_produced_inventory[:] = (0.0, 1.5)
        before = [sum(agent.stock(k) for agent in agents) for k in (1, 2)]
        energies = [agent.energy for agent in agents]
        world.exchange()
        self.assertEqual(len(world.trades), 3)
        participants = [agent_id for event in world.trades for agent_id in event["partner_ids"]]
        self.assertEqual(len(participants), len(set(participants)))
        self.assertEqual([agent.energy for agent in agents], energies)
        for event in world.trades:
            self.assertEqual({entry["gross"] for entry in event["surrendered"].values()}, {0.2})
            for entry in event["received"].values():
                self.assertAlmostEqual(entry["net"], 0.15)
        for index, good in enumerate(("B", "C")):
            remaining = sum(agent.stock(index + 1) for agent in agents)
            loss = sum(event["transaction_loss"][good] for event in world.trades)
            self.assertAlmostEqual(before[index], remaining + loss)

    def test_compatible_matching_stays_within_the_cell(self):
        world, agents = self._world(3)
        self._complementary(agents[0], agents[1])
        agents[1].x = 1
        world.exchange()
        self.assertEqual(world.trades, [])

    def test_received_surplus_is_not_an_export(self):
        world, agents = self._world()
        self._complementary(*agents)
        agents[0].self_produced_inventory[0] = 0.0
        agents[0].received_inventory[0] = 2.0
        world.exchange()
        self.assertEqual(world.trades, [])

    def test_received_stock_reduces_or_eliminates_the_deficit(self):
        for received, expected_events in ((0.5, 0), (0.25, 1)):
            with self.subTest(received=received):
                world, agents = self._world()
                self._complementary(*agents)
                agents[0].received_inventory[1] = received
                world.exchange()
                self.assertEqual(len(world.trades), expected_events)
                if expected_events:
                    self.assertAlmostEqual(agents[0].received_inventory[1], 0.5)
                    self.assertAlmostEqual(agents[1].received_inventory[0], 0.25)

    def test_disabled_trade_and_zero_cap_prevent_compatible_exchanges(self):
        for options in ({"trade_enabled": False}, {"max_trade_per_pair": 0.0}):
            with self.subTest(options=options):
                world, agents = self._world(**options)
                self._complementary(*agents)
                before = [agent.self_produced_inventory.copy() for agent in agents]
                world.exchange()
                self.assertEqual(world.trades, [])
                for agent, stock in zip(agents, before):
                    np.testing.assert_array_equal(agent.self_produced_inventory, stock)

    def test_compatible_world_and_events_are_reproducible(self):
        config = Config(L=2, initial_agents=20, trade_matching="compatible")
        first, second = World(config, seed=17), World(config, seed=17)
        first.run(30)
        second.run(30)
        self.assertGreater(len(first.trades), 0)
        self.assertEqual(first.history, second.history)
        self.assertEqual(first.trades, second.trades)


if __name__ == "__main__":
    unittest.main()
