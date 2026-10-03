"""Check passive recording, event budgets and finite-speed local propagation."""

from contextlib import redirect_stdout
from dataclasses import fields, replace
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.compare import run as run_comparison
from experiments.runner import condition_directory, run_experiment
from ht import Action, ActionKind, Config, World
from ht.agent import Agent
from ht.fields import FIELD_CHANNELS, FieldSystem


class ConsentPolicy:
    def __init__(self, consent=True):
        self.consent = consent

    def choose_action(self, observation, rng):
        return Action(ActionKind.USE)

    def willing_to_trade(self, observation, offer, rng):
        return self.consent


class TestFieldSystem(unittest.TestCase):
    def test_parameters_and_deposits_are_validated(self):
        for name in ("field_decay", "field_diffusion"):
            for value in (-0.1, 1.1, np.inf, np.nan):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    Config(**{name: value})
                with self.assertRaises(ValueError):
                    FieldSystem(5, **{name.removeprefix("field_"): value})
        for size in (0, -1, 2.5):
            with self.assertRaises(ValueError):
                FieldSystem(size)
        field = FieldSystem(5)
        for value in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                field.deposit("movement", (0, 0), value)
        for cell in ((-1, 0), (0, 5), (0.5, 1)):
            with self.assertRaises(ValueError):
                field.deposit("movement", cell)
        with self.assertRaises(ValueError):
            field.deposit("unknown", (0, 0))

    def test_sources_wait_for_commit_and_spread_one_cell_per_later_step(self):
        field = FieldSystem(9, decay=0, diffusion=1)
        field.deposit("movement", (4, 4), 4)
        self.assertEqual(field.snapshot().sum(), 0)
        field.advance()
        self.assertEqual(np.count_nonzero(field.layer("movement")), 1)
        self.assertEqual(field.layer("movement")[4, 4], 4)
        for distance in (1, 2, 3):
            field.advance()
            values = field.layer("movement")
            y, x = np.nonzero(values)
            self.assertTrue(np.all(abs(x - 4) + abs(y - 4) <= distance))
            self.assertAlmostEqual(values.sum(), 4)
        self.assertEqual(field.layer("trade").sum(), 0)
        self.assertEqual(field.layer("violence").sum(), 0)

    def test_wrapped_neighbors_and_small_grids_preserve_event_weight(self):
        field = FieldSystem(5, decay=0, diffusion=1)
        field.deposit("trade", (0, 0), 4)
        field.advance()
        field.advance()
        expected = np.zeros((5, 5))
        expected[0, 1] = expected[0, 4] = expected[1, 0] = expected[4, 0] = 1
        np.testing.assert_array_equal(field.layer("trade"), expected)
        for size in (1, 2):
            tiny = FieldSystem(size, decay=0, diffusion=1)
            tiny.deposit("trade", (0, 0), 4)
            tiny.advance()
            tiny.advance()
            self.assertEqual(tiny.layer("trade").sum(), 4)
            if size == 1:
                self.assertEqual(tiny.layer("trade")[0, 0], 4)
            else:
                np.testing.assert_array_equal(tiny.layer("trade"), [[0, 2], [2, 0]])

    def test_decay_source_balance_positivity_and_limiting_cases(self):
        for decay in (0, 0.25, 1):
            for diffusion in (0, 0.4, 1):
                field = FieldSystem(7, decay=decay, diffusion=diffusion)
                for step in range(12):
                    previous = field.snapshot().sum(axis=(1, 2))
                    field.deposit("movement", (step % 7, 2), 2)
                    field.deposit("trade", (1, step % 7), 0.5)
                    field.advance()
                    np.testing.assert_allclose(field.snapshot().sum(axis=(1, 2)), (1 - decay) * previous + [2, 0.5, 0])
                    self.assertTrue(np.all(field.snapshot() >= 0))
                self.assertEqual(field.history[-1]["movement_cumulative_deposited"], 24)
                self.assertEqual(field.history[-1]["trade_cumulative_deposited"], 6)
        local = FieldSystem(5, decay=0, diffusion=0)
        local.deposit("violence", (2, 3))
        for _ in range(4):
            local.advance()
        self.assertEqual(local.layer("violence")[3, 2], 1)
        self.assertEqual(np.count_nonzero(local.snapshot()), 1)

    def test_concentration_and_detached_observer_snapshots(self):
        field = FieldSystem(3, decay=0, diffusion=0)
        self.assertEqual(field.history[0]["trade_concentration"], 0)
        field.deposit("trade", (0, 0))
        field.advance()
        self.assertEqual(field.history[-1]["trade_concentration"], 1)
        for y in range(3):
            for x in range(3):
                if (x, y) != (0, 0):
                    field.deposit("trade", (x, y))
        field.advance()
        self.assertAlmostEqual(field.history[-1]["trade_concentration"], 1 / 9)
        for snapshot in (field.snapshot(), field.layer("trade")):
            with self.assertRaises(ValueError):
                snapshot.flat[0] = 100
            snapshot.setflags(write=True)
            snapshot[:] = 100
        self.assertEqual(field.layer("trade").sum(), 9)


class TestFieldWorld(unittest.TestCase):
    def world(self, **overrides):
        settings = dict(L=5, initial_agents=2, social_enabled=True, fields_enabled=True, metabolism=0, field_decay=0, field_diffusion=0)
        settings.update(overrides)
        world = World(Config(**settings), seed=7, policy=ConsentPolicy())
        for agent in world.agents.values():
            agent.x = agent.y = 0
        return world, list(world.agents.values())

    def test_only_real_displacements_including_fleeing_leave_movement(self):
        world, (agent, _) = self.world()
        world.do_move(agent, (-1, 0))
        world.do_move(agent, (0, 1), fleeing=True)
        self.assertEqual(world.fields.snapshot().sum(), 0)
        world.fields.advance()
        layer = world.fields.layer("movement")
        self.assertEqual(layer[0, 4], 1)
        self.assertEqual(layer[1, 4], 1)
        self.assertEqual(layer.sum(), 2)
        tiny, (agent,) = self.world(L=1, initial_agents=1)
        tiny.do_move(agent, (1, 0))
        tiny.fields.advance()
        self.assertEqual(tiny.fields.snapshot().sum(), 0)

    def test_completed_trade_counts_once_and_refusals_leave_no_trace(self):
        for same_cell in (False, True):
            world, (first, second) = self.world()
            second.x = 0 if same_cell else 1
            first.self_produced_inventory[:] = (1.5, 0)
            second.self_produced_inventory[:] = (0, 1.5)
            world.policy = ConsentPolicy(False)
            world.step()
            self.assertEqual(world.fields.snapshot().sum(), 0)
            world.policy = ConsentPolicy(True)
            world.step()
            layer = world.fields.layer("trade")
            self.assertEqual(layer.sum(), 1)
            self.assertEqual(layer[0, 0], 1 if same_cell else 0.5)
            if not same_cell:
                self.assertEqual(layer[0, 1], 0.5)
            world.step()  # No remaining deficit, therefore no second event.
            self.assertEqual(world.fields.history[-1]["trade_cumulative_deposited"], 1)

    def test_attack_uses_victim_cell_and_survives_victim_removal(self):
        world, (attacker, victim) = self.world(trade_enabled=False)
        victim.x, victim.y = 1, 0
        victim.energy = 0.1
        self.assertFalse(world.do_attack(attacker, 12345))
        self.assertTrue(world.do_attack(attacker, victim.id))
        self.assertNotIn(victim.id, world.agents)
        self.assertFalse(world.do_attack(attacker, victim.id))
        world.step()
        self.assertEqual(world.fields.layer("violence")[0, 1], 1)
        self.assertEqual(world.fields.history[-1]["violence_cumulative_deposited"], world.social_totals["attacks"])

    def test_all_actions_see_the_same_old_field_and_no_field_in_observation(self):
        world, agents = self.world(trade_enabled=False)
        snapshots = []

        class MovingPolicy:
            def choose_action(self, observation, rng):
                self_outer.assertFalse(any("field" in name for name in vars(observation)))
                snapshots.append(world.fields.snapshot())
                return Action(ActionKind.MOVE, direction=(1, 0))

        self_outer = self
        world.policy = MovingPolicy()
        world.step()
        self.assertEqual(len(snapshots), len(agents))
        for snapshot in snapshots:
            self.assertEqual(snapshot.sum(), 0)
        self.assertEqual(world.fields.layer("movement")[0, 1], 2)
        self.assertEqual(world.fields.t, world.t)

    def test_recording_preserves_complete_seeded_trajectories(self):
        for social in (False, True):
            for matching in ("neighborhood", "random", "compatible"):
                for seed in (7, 11):
                    with self.subTest(social=social, matching=matching, seed=seed):
                        config = Config(L=6, initial_agents=20, max_agents=80, social_enabled=social, trade_matching=matching, finite_resource_fraction=0.5)
                        plain = World(config, seed=seed, record_actions=True)
                        observed = World(replace(config, fields_enabled=True), seed=seed, record_actions=True)
                        for _ in range(60):
                            plain.step()
                            observed.step()
                        self.assertIsNone(plain.fields)
                        for name in vars(plain):
                            if name not in ("cfg", "fields", "policy", "rng", "trait_rng", "agents"):
                                np.testing.assert_equal(getattr(plain, name), getattr(observed, name))
                        for name in ("rng", "trait_rng"):
                            self.assertEqual(getattr(plain, name).bit_generator.state, getattr(observed, name).bit_generator.state)
                        self.assertEqual(list(plain.agents), list(observed.agents))
                        for aid in plain.agents:
                            for field in fields(Agent):
                                np.testing.assert_equal(getattr(plain.agents[aid], field.name), getattr(observed.agents[aid], field.name))
                        totals = observed.fields.history[-1]
                        self.assertEqual(totals["trade_cumulative_deposited"], len(observed.trades))
                        self.assertEqual(totals["violence_cumulative_deposited"], observed.social_totals["attacks"])

    def test_exported_fields_reconcile_and_controls_have_distinct_paths(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = Config(L=5, initial_agents=12, social_enabled=True, fields_enabled=True)
            output = condition_directory(root, config, 7)
            world = run_experiment(config, 7, 8, output, make_plots=False)
            exported = json.loads((output / "field_map.json").read_text())
            self.assertEqual(exported["t"], world.t)
            for channel in FIELD_CHANNELS:
                np.testing.assert_array_equal(exported["fields"][channel], world.fields.layer(channel))
            self.assertEqual(len((output / "field_history.csv").read_text().splitlines()), world.t + 2)
            control = replace(config, fields_enabled=False)
            self.assertNotEqual(output, condition_directory(root, control, 7))
            self.assertNotEqual(output, condition_directory(root, replace(config, field_decay=0.5), 7))
            run_experiment(control, 7, 0, output, make_plots=False)
            self.assertFalse((output / "field_map.json").exists())
            self.assertFalse((output / "field_history.csv").exists())
            rows = run_comparison(seeds=(7,), gammas=(1,), steps=2, results_dir=root, fields_enabled=True, make_plots=False)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["fields_enabled"] and row["t"] == 2 for row in rows))
            self.assertEqual(rows[0]["trade_total"], 0)


if __name__ == "__main__":
    unittest.main()
