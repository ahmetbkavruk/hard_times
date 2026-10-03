"""Check observer independence and controls without needing a display server."""

import copy
from dataclasses import fields
from types import SimpleNamespace
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.colors import to_hex
import numpy as np

from ht.actions import Action, ActionKind
from ht.agent import Agent
from ht.config import Config
from ht.viewer import FINITE_RESOURCE_COLORS, LiveViewer, RESOURCE_COLORS
from ht.world import World
from ht.social import ALIGNMENT_COLORS


class FixedPolicy:
    def __init__(self, kind):
        self.kind = kind

    def choose_action(self, observation, rng):
        return Action(self.kind)


class TestLiveViewer(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_field_overlays_keys_and_restart_are_passive(self):
        config = Config(L=5, initial_agents=12, social_enabled=True, fields_enabled=True)
        viewer = LiveViewer(config, seed=11, paused=True, field_scale=2)
        plain = World(config, seed=11)
        for _ in range(3):
            plain.step()
            viewer.single_step()
        before = viewer.world.fields.snapshot()
        self.assertEqual(viewer.field_overlay, "movement")
        for expected in ("trade", "violence", "none", "movement"):
            viewer.figure.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", viewer.figure.canvas, key="f"))
            self.assertEqual(viewer.field_overlay, expected)
            self.assertEqual(viewer.field_image.get_visible(), expected != "none")
            if expected != "none":
                self.assertIn("2 events/cell", viewer.field_text.get_text())
            self.assert_same_world(plain, viewer.world)
            np.testing.assert_array_equal(viewer.world.fields.snapshot(), before)
        viewer.reset()
        self.assertEqual(viewer.world.fields.snapshot().sum(), 0)
        self.assertEqual(len(viewer.world.fields.history), 1)
        self.assert_same_world(viewer.world, World(config, seed=11))

    def test_field_overlay_uses_fixed_scale_and_disabled_control_is_harmless(self):
        viewer = LiveViewer(Config(L=4, initial_agents=1, fields_enabled=True), paused=True, field_scale=2)
        viewer.world.fields.deposit("movement", (0, 0), 1)
        viewer.world.fields.deposit("movement", (1, 0), 20)
        viewer.world.fields.advance()
        viewer.refresh()
        alpha = viewer.field_image.get_array()[:, :, 3]
        self.assertAlmostEqual(alpha[0, 0], 0.425)
        self.assertAlmostEqual(alpha[0, 1], 0.85)
        self.assertEqual(alpha[3, 3], 0)
        disabled = LiveViewer(Config(L=4, initial_agents=1), paused=True)
        disabled.toggle_field()
        self.assertEqual(disabled.field_overlay, "none")
        self.assertFalse(disabled.field_image.get_visible())
        for value in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                LiveViewer(viewer.config, field_scale=value)
        with self.assertRaises(ValueError):
            LiveViewer(disabled.config, field_overlay="trade")

    def assert_same_world(self, left, right):
        np.testing.assert_array_equal(left.resources, right.resources)
        np.testing.assert_array_equal(left.capacity, right.capacity)
        np.testing.assert_array_equal(left.renewable_mask, right.renewable_mask)
        self.assertEqual(left.rng.bit_generator.state, right.rng.bit_generator.state)
        self.assertEqual(left.trait_rng.bit_generator.state, right.trait_rng.bit_generator.state)
        for attribute in ("t", "next_agent_id", "total_births", "total_deaths", "trades", "trade_totals", "usage_totals",
                          "social_totals", "interactions", "carcasses"):
            self.assertEqual(getattr(left, attribute), getattr(right, attribute))
        np.testing.assert_equal(left.history, right.history)
        self.assertEqual(list(left.agents), list(right.agents))
        for aid in left.agents:
            for field in fields(Agent):
                np.testing.assert_equal(getattr(left.agents[aid], field.name), getattr(right.agents[aid], field.name))

    def test_optional_action_recording_preserves_full_seeded_trajectories(self):
        for enabled, matching in (
            (False, "neighborhood"), (True, "neighborhood"),
            (True, "random"), (True, "compatible"),
        ):
            with self.subTest(trade=enabled, matching=matching):
                config = Config(L=6, initial_agents=24, max_agents=100, trade_enabled=enabled,
                                trade_matching=matching, finite_resource_fraction=0.5)
                plain = World(config, seed=17)
                observed = World(config, seed=17, record_actions=True)
                for _ in range(100):
                    plain.step()
                    observed.step()
                self.assert_same_world(plain, observed)
                self.assertEqual(plain.last_actions, {})
                self.assertEqual(plain.last_recipe_uses, set())

    def test_redraw_inspection_speed_and_pause_do_not_change_trajectory(self):
        config = Config(L=6, initial_agents=12, finite_resource_fraction=0.5, social_enabled=True)
        viewer = LiveViewer(config, seed=11, paused=True)
        plain = World(config, seed=11)
        for _ in range(8):
            viewer.refresh()
            viewer.toggle_colors()
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
        second.x = 2
        first.self_produced_inventory[:] = (1.5, 0)
        second.self_produced_inventory[:] = (0, 1.5)
        viewer.selected_agent = first
        viewer.single_step()
        self.assertEqual(len(viewer.recent_trades), 1)
        np.testing.assert_array_equal(viewer.trade_marks.get_offsets(), [(1, 1), (2, 1)])
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

    def test_finite_ground_stays_identifiable_when_empty_and_restart_restores_it(self):
        config = Config(L=4, initial_agents=1, finite_resource_fraction=0.5)
        viewer = LiveViewer(config, seed=7, paused=True)
        original_mask = viewer.world.renewable_mask.copy()
        fy, fx = np.argwhere(~original_mask)[0]
        ry, rx = np.argwhere(original_mask)[0]
        agent = viewer.world.agents[0]
        agent.x, agent.y = int(fx), int(fy)
        viewer.selected_agent = agent
        viewer.world.resources[0, fy, fx] = 0
        viewer.world.resources[0, ry, rx] = 0
        viewer.refresh()
        image = viewer.resource_image.get_array()
        np.testing.assert_array_equal(image[fy, fx], FINITE_RESOURCE_COLORS(0.0))
        np.testing.assert_array_equal(image[ry, rx], RESOURCE_COLORS(0.0))
        self.assertIn("Empty finite cells: 1 / 8", viewer.resource_text.get_text())
        self.assertIn("Ground: finite", viewer.agent_text.get_text())
        viewer.reset()
        np.testing.assert_array_equal(viewer.world.renewable_mask, original_mask)
        np.testing.assert_array_equal(viewer.world.resources, viewer.world.capacity)
        self.assertIn("Empty finite cells: 0 / 8", viewer.resource_text.get_text())

    def test_all_alignments_have_distinct_colors_and_inspection_does_not_change_state(self):
        viewer = LiveViewer(Config(L=4, initial_agents=9, social_enabled=True), seed=7, paused=True)
        for agent, (order, care) in zip(viewer.world.agents.values(),
                                       ((o, c) for o in (-1, 0, 1) for c in (-1, 0, 1))):
            agent.lawfulness, agent.concern = order, care
        viewer.selected_agent = viewer.world.agents[0]
        before = copy.deepcopy(viewer.world.rng.bit_generator.state)
        viewer.refresh()
        self.assertEqual({to_hex(color) for color in viewer.dots.get_facecolors()}, set(ALIGNMENT_COLORS.values()))
        self.assertIn(viewer.selected_agent.alignment.label, viewer.agent_text.get_text())
        self.assertIn("Policy      social", viewer.world_text.get_text())
        viewer.toggle_colors()
        self.assertFalse(viewer.alignment_legend.get_visible())
        self.assertIsNotNone(viewer.dots.get_array())
        viewer.toggle_colors()
        self.assertTrue(viewer.alignment_legend.get_visible())
        self.assertEqual(viewer.world.rng.bit_generator.state, before)
        viewer.reset()
        self.assert_same_world(viewer.world, World(viewer.config, seed=7))

    def test_social_highlights_are_actual_events_and_clear_on_restart(self):
        class AttackPolicy:
            def choose_action(self, observation, rng):
                return Action(ActionKind.ATTACK, target_id=1)

        config = Config(L=3, initial_agents=2, social_enabled=True, metabolism=0, trade_enabled=False)
        viewer = LiveViewer(config, seed=7, paused=True)
        first, target = viewer.world.agents.values()
        first.x = first.y = target.x = target.y = 1
        target.energy = 0.1
        viewer.world.policy = AttackPolicy()
        viewer.selected_agent = first
        viewer.single_step()
        self.assertEqual(len(viewer.attack_marks.get_offsets()), 1)
        self.assertEqual(len(viewer.body_marks.get_offsets()), 1)
        self.assertIn("Attack #1", viewer.agent_text.get_text())

        class EatPolicy:
            def choose_action(self, observation, rng):
                return Action(ActionKind.EAT_AGENT, target_id=1)

        viewer.world.policy = EatPolicy()
        viewer.single_step()
        self.assertEqual(len(viewer.attack_marks.get_offsets()), 0)
        self.assertEqual(len(viewer.predation_marks.get_offsets()), 1)
        self.assertIn("Ate remains of #1", viewer.agent_text.get_text())
        viewer.reset()
        self.assertEqual(viewer.recent_interactions, [])
        self.assertEqual(len(viewer.body_marks.get_offsets()), 0)
        self.assertEqual(len(viewer.predation_marks.get_offsets()), 0)


if __name__ == "__main__":
    unittest.main()
