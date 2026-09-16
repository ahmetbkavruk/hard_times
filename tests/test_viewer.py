"""Check observer independence and controls without needing a display server."""

import copy
from dataclasses import fields
from types import SimpleNamespace
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent
import numpy as np

from ht.actions import Action, ActionKind
from ht.agent import Agent
from ht.config import Config
from ht.viewer import LiveViewer
from ht.world import World


class FixedPolicy:
    def __init__(self, kind):
        self.kind = kind

    def choose_action(self, observation, rng):
        return Action(self.kind)


class TestLiveViewer(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def assert_same_world(self, left, right):
        np.testing.assert_array_equal(left.resources, right.resources)
        np.testing.assert_array_equal(left.capacity, right.capacity)
        self.assertEqual(left.rng.bit_generator.state, right.rng.bit_generator.state)
        for attribute in ("t", "next_agent_id", "total_births", "total_deaths", "trades", "trade_totals", "usage_totals"):
            self.assertEqual(getattr(left, attribute), getattr(right, attribute))
        np.testing.assert_equal(left.history, right.history)
        self.assertEqual(list(left.agents), list(right.agents))
        for aid in left.agents:
            for field in fields(Agent):
                np.testing.assert_equal(getattr(left.agents[aid], field.name), getattr(right.agents[aid], field.name))

    def test_optional_action_recording_preserves_full_seeded_trajectories(self):
        for enabled, matching in ((False, "random"), (True, "random"), (True, "compatible")):
            with self.subTest(trade=enabled, matching=matching):
                config = Config(L=6, initial_agents=24, max_agents=100, trade_enabled=enabled, trade_matching=matching)
                plain = World(config, seed=17)
                observed = World(config, seed=17, record_actions=True)
                for _ in range(100):
                    plain.step()
                    observed.step()
                self.assert_same_world(plain, observed)
                self.assertEqual(plain.last_actions, {})
                self.assertEqual(plain.last_recipe_uses, set())

    def test_redraw_inspection_speed_and_pause_do_not_change_trajectory(self):
        config = Config(L=6, initial_agents=12)
        viewer = LiveViewer(config, seed=11, paused=True)
        plain = World(config, seed=11)
        for _ in range(8):
            viewer.refresh()
            agent = next(iter(viewer.world.agents.values()))
            viewer.on_click(SimpleNamespace(inaxes=viewer.grid, button=1, xdata=agent.x, ydata=agent.y))
            viewer.speed_slider.set_val(37)
            viewer.toggle_play()
            viewer.toggle_play()
            viewer.tick()  # A paused timer callback must not advance.
            self.assert_same_world(plain, viewer.world)
            plain.step()
            viewer.single_step()
            self.assert_same_world(plain, viewer.world)

    def test_real_canvas_events_and_restart_replay_the_same_seed(self):
        config = Config(L=5, initial_agents=3)
        viewer = LiveViewer(config, seed=7, steps=2, paused=True)
        for agent in viewer.world.agents.values():
            agent.x = agent.y = 2
        viewer.refresh()
        viewer.figure.canvas.draw()
        self.assertEqual(len(set(viewer.agent_positions.values())), 3)
        x, y = viewer.grid.transData.transform((2, 2))
        for expected in (0, 1, 2, 0):
            event = MouseEvent("button_press_event", viewer.figure.canvas, x, y, button=1)
            viewer.figure.canvas.callbacks.process("button_press_event", event)
            self.assertEqual(viewer.selected_agent.id, expected)
        # Click the actual Step button through Matplotlib's event dispatcher.
        x, y = viewer.step_button.ax.transAxes.transform((0.5, 0.5))
        for name in ("button_press_event", "button_release_event"):
            viewer.figure.canvas.callbacks.process(name, MouseEvent(name, viewer.figure.canvas, x, y, button=1))
        self.assertEqual(viewer.world.t, 1)
        self.assertTrue(viewer.paused)
        viewer.figure.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", viewer.figure.canvas, key=" "))
        viewer.tick()
        viewer.tick()
        self.assertEqual(viewer.world.t, 2)
        self.assertTrue(viewer.finished)
        viewer.single_step()
        self.assertEqual(viewer.world.t, 2)
        viewer.figure.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", viewer.figure.canvas, key="r"))
        self.assertTrue(viewer.paused)
        self.assertIsNone(viewer.selected_agent)
        self.assert_same_world(viewer.world, World(config, seed=7))

    def test_eating_highlights_only_successful_consumption_and_expire_next_step(self):
        viewer = LiveViewer(Config(L=3, initial_agents=1, metabolism=0, trade_enabled=False), paused=True)
        viewer.world.policy = FixedPolicy(ActionKind.USE)
        agent = viewer.world.agents[0]
        viewer.selected_agent = agent
        viewer.single_step()
        self.assertEqual(viewer.eating_cells, [])
        self.assertIn("incomplete recipe", viewer.agent_text.get_text())
        agent.inventory[0] = 1.0
        agent.self_produced_inventory[:] = (0.5, 0.5)
        viewer.single_step()
        self.assertEqual(viewer.world.last_recipe_uses, {agent.id})
        self.assertEqual(viewer.eating_cells, [(agent.x, agent.y)])
        self.assertIn("Eat recipe", viewer.agent_text.get_text())
        viewer.single_step()
        self.assertEqual(viewer.eating_cells, [])

    def test_trade_highlight_uses_real_latest_events(self):
        viewer = LiveViewer(Config(L=3, initial_agents=2, metabolism=0), paused=True)
        viewer.world.policy = FixedPolicy(ActionKind.USE)
        first, second = viewer.world.agents.values()
        for agent in (first, second):
            agent.x = agent.y = 1
        first.self_produced_inventory[:] = (1.5, 0)
        second.self_produced_inventory[:] = (0, 1.5)
        viewer.selected_agent = first
        viewer.single_step()
        self.assertEqual(len(viewer.recent_trades), 1)
        np.testing.assert_array_equal(viewer.trade_marks.get_offsets(), [(1, 1)])
        self.assertIn("Traded with #1", viewer.agent_text.get_text())
        viewer.single_step()
        self.assertEqual(viewer.recent_trades, [])
        self.assertEqual(len(viewer.trade_marks.get_offsets()), 0)

    def test_selected_death_and_extinction_leave_a_readable_final_frame(self):
        viewer = LiveViewer(Config(L=3, initial_agents=1, initial_energy=0.01, metabolism=1), paused=True)
        viewer.selected_agent = viewer.world.agents[0]
        viewer.single_step()
        self.assertTrue(viewer.finished)
        self.assertEqual(viewer.status.get_text(), "EXTINCT")
        self.assertIn("Died at step 1", viewer.agent_text.get_text())
        self.assertEqual(len(viewer.death_cells), 1)
        self.assertEqual(len(viewer.dots.get_offsets()), 0)
        state = copy.deepcopy(viewer.world.rng.bit_generator.state)
        viewer.toggle_play()
        viewer.tick()
        viewer.single_step()
        self.assertEqual(viewer.world.t, 1)
        self.assertEqual(viewer.world.rng.bit_generator.state, state)


if __name__ == "__main__":
    unittest.main()
