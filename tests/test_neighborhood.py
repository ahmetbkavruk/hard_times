import unittest
from unittest.mock import patch

import numpy as np

from ht import Action, ActionKind, Config, World


class MovePolicy:
    def choose_action(self, observation, rng):
        return Action(ActionKind.MOVE)


class TestNeighborhoodTrading(unittest.TestCase):
    def make_world(self, cells, **options):
        config = Config(initial_agents=len(cells), **{"L": 5, **options})
        world = World(config, seed=7)
        agents = list(world.agents.values())
        for index, (agent, cell) in enumerate(zip(agents, cells)):
            agent.x, agent.y = cell
            agent.self_produced_inventory[:] = (1.5, 0) if index == 0 else (0, 1.5)
        return world, agents

    def test_same_cell_and_four_neighbors_trade_but_diagonals_and_distant_cells_do_not(self):
        for cell in ((2, 2), (1, 2), (3, 2), (2, 1), (2, 3), (1, 1), (3, 3), (0, 2), (2, 4)):
            with self.subTest(cell=cell):
                world, agents = self.make_world([(2, 2), cell])
                world.exchange()
                expected = int(abs(cell[0] - 2) + abs(cell[1] - 2) <= 1)
                self.assertEqual(len(world.trades), expected)
                if expected:
                    self.assertAlmostEqual(agents[0].received_inventory[1], 0.5)

    def test_neighborhood_wraps_across_each_periodic_boundary(self):
        for first, second in (
            ((0, 2), (4, 2)), ((4, 2), (0, 2)),
            ((2, 0), (2, 4)), ((2, 4), (2, 0)),
        ):
            with self.subTest(first=first, second=second):
                world, agents = self.make_world([first, second])
                world.exchange()
                self.assertEqual(len(world.trades), 1)
                self.assertEqual(world.trades[0]["partner_cells"], {0: first, 1: second})
        world, _ = self.make_world([(0, 0), (4, 4)])
        world.exchange()
        self.assertEqual(world.trades, [])  # Wrapping does not add diagonal contacts.

    def test_one_agent_can_trade_with_every_local_partner_in_the_same_step(self):
        cells = [(2, 2), (2, 2), (1, 2), (3, 2), (2, 1), (2, 3)]
        world, agents = self.make_world(cells, max_trade_per_pair=0.0625)
        world.exchange()
        self.assertEqual(len(world.trades), 5)
        self.assertEqual(
            {tuple(sorted(event["partner_ids"])) for event in world.trades},
            {(0, partner) for partner in range(1, 6)},
        )
        self.assertEqual(agents[0].received_inventory[1], 5 * 0.0625)
        for partner in agents[1:]:
            self.assertEqual(partner.received_inventory[0], 0.0625)

    def test_each_pair_trades_only_once_even_on_tiny_periodic_grids(self):
        for size, cells in (
            (1, [(0, 0)] * 4),
            (2, [(0, 0), (1, 1), (0, 1), (1, 0)]),
        ):
            with self.subTest(size=size):
                world, agents = self.make_world(cells, L=size, max_trade_per_pair=0.0625)
                agents[1].self_produced_inventory[:] = (1.5, 0)
                world.exchange()
                pairs = [tuple(sorted(event["partner_ids"])) for event in world.trades]
                self.assertCountEqual(pairs, [(0, 2), (0, 3), (1, 2), (1, 3)])
                self.assertTrue(all(first != second for first, second in pairs))
                # Needs remain unfilled, so duplicate visits would cause extra trades.
                for agent in agents:
                    self.assertEqual(sum(agent.received_inventory), 0.125)

    def test_every_exchange_rechecks_surplus_and_deficit(self):
        for limited_by in ("surplus", "deficit"):
            with self.subTest(limited_by=limited_by):
                world, agents = self.make_world(
                    [(2, 2), (1, 2), (3, 2), (2, 1)],
                    trade_efficiency=0.5, max_trade_per_pair=0.125,
                )
                if limited_by == "surplus":
                    agents[0].self_produced_inventory[0] = 0.5 + 0.1875
                else:
                    agents[0].received_inventory[1] = 0.5 - 0.09375
                world.exchange()
                self.assertEqual(len(world.trades), 2)
                self.assertEqual(sum(event["surrendered"][0]["gross"] for event in world.trades), 0.1875)
                if limited_by == "surplus":
                    self.assertEqual(agents[0].self_produced_inventory[0], 0.5)
                else:
                    self.assertEqual(agents[0].stock(2), 0.5)
                for event in world.trades:
                    self.assertLessEqual(event["surrendered"][0]["gross"], 0.125)

    def test_multiple_trades_conserve_each_good_including_losses_and_do_not_change_energy(self):
        world, agents = self.make_world(
            [(2, 2), (2, 2), (1, 2), (3, 2), (2, 1), (2, 3)],
            trade_efficiency=0.75, max_trade_per_pair=0.125,
        )
        before = [sum(agent.stock(k) for agent in agents) for k in range(3)]
        energies = [agent.energy for agent in agents]
        world.exchange()
        self.assertEqual(len(world.trades), 5)
        self.assertEqual([agent.energy for agent in agents], energies)
        self.assertEqual(sum(agent.stock(0) for agent in agents), before[0])
        for index, good in enumerate(("B", "C"), start=1):
            after = sum(agent.stock(index) for agent in agents)
            losses = sum(event["transaction_loss"][good] for event in world.trades)
            self.assertEqual(before[index], after + losses)
            self.assertEqual(losses, world.trade_totals[f"trade_loss_{good}"])
        for agent in agents:
            self.assertTrue(np.all(agent.self_produced_inventory >= 0))
            self.assertTrue(np.all(agent.received_inventory >= 0))

    def test_trading_uses_final_positions_and_excludes_agents_that_died_this_step(self):
        world, agents = self.make_world(
            [(0, 0), (3, 3), (0, 0)], metabolism=0.2, max_trade_per_pair=0.0625,
        )
        world.policy = MovePolicy()
        agents[2].energy = 0.1

        def move(agent):
            agent.x, agent.y = ((2, 2), (2, 3), (2, 2))[agent.id]

        with patch.object(world, "do_move", side_effect=move):
            world.step()
        self.assertEqual(set(world.agents), {0, 1})
        self.assertEqual(len(world.trades), 1)
        self.assertEqual(world.trades[0]["partner_cells"], {0: (2, 2), 1: (2, 3)})
        self.assertEqual(world.history[-1]["cumulative_trade_events"], 1)

    def test_neighborhood_trajectories_and_contact_order_are_seeded(self):
        config = Config(L=4, initial_agents=24, max_trade_per_pair=0.125)
        first, second = World(config, seed=19), World(config, seed=19)
        first.run(30)
        second.run(30)
        self.assertGreater(len(first.trades), 0)
        self.assertEqual(first.history, second.history)
        self.assertEqual(first.trades, second.trades)
        self.assertEqual(first.rng.bit_generator.state, second.rng.bit_generator.state)


if __name__ == "__main__":
    unittest.main()
