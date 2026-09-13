import csv
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.baseline import run as run_baseline
from experiments.compare import run as run_comparison
from experiments.matching import run as run_matching
from experiments.runner import run_experiment
from ht import Config


class TestExperiments(unittest.TestCase):
    def test_matching_modes_have_distinct_outputs_and_one_no_trade_control(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            rows = run_comparison(
                seeds=(7,), gammas=(2.0,), steps=2, results_dir=root,
                matchings=("random", "compatible"), make_plots=False,
            )
            self.assertEqual(len(rows), 3)
            self.assertEqual(sum(not row["trade_enabled"] for row in rows), 1)
            for condition, enabled, matching in (
                ("trade_off", False, "random"),
                ("trade_random", True, "random"),
                ("trade_compatible", True, "compatible"),
            ):
                output = root / "gamma_2" / condition / "seed_7"
                config = json.loads((output / "config.json").read_text())["config"]
                self.assertEqual(config["trade_enabled"], enabled)
                self.assertEqual(config["trade_matching"], matching)
                self.assertEqual(config["production_specialization_exponent"], 2.0)

    def test_fixed_gamma_diagnostic_records_founders_and_lineage_bookkeeping(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            rows = run_matching(
                seeds=(7,), survival_steps=100, reproduction_steps=80,
                max_agents=100, results_dir=root, make_plots=False,
            )
            self.assertEqual(len(rows), 6)
            survival = {row["trade_matching"]: row for row in rows
                        if row["phase"] == "survival" and row["trade_enabled"]}
            self.assertEqual(survival["random"]["specialists_alive"], 0)
            self.assertEqual(survival["compatible"]["specialists_alive"], 2)
            reproduction = next(row for row in rows if row["phase"] == "reproduction"
                                and row["trade_enabled"] and row["trade_matching"] == "compatible")
            self.assertGreater(reproduction["births"], 0)
            for row in rows:
                self.assertEqual(row["production_specialization_exponent"], 2.0)
                self.assertEqual(
                    sum(row[f"{group}_births"] for group in ("B_specialist", "C_specialist", "generalist")),
                    row["births"],
                )
                self.assertEqual(
                    sum(row[f"{group}_deaths"] for group in ("B_specialist", "C_specialist", "generalist")),
                    row["deaths"],
                )
            output = root / "reproduction" / "gamma_2" / "trade_compatible" / "seed_7"
            founders = json.loads((output / "founders.json").read_text())
            self.assertEqual([agent["production_allocation"] for agent in founders], [0.0, 1.0] + [0.5] * 18)
            config = json.loads((output / "config.json").read_text())["config"]
            self.assertEqual(config["allocation_mutation_sigma"], 0.0)
            self.assertEqual(config["mutation_sigma"], 0.0)
            with (output / "history.csv").open() as stream:
                initial = next(csv.DictReader(stream))
            self.assertAlmostEqual(float(initial["mean_specialization"]), 0.1)
            with (output / "lineage_history.csv").open() as stream:
                final = list(csv.DictReader(stream))[-1]
            self.assertEqual(int(final["B_specialist_births"]), reproduction["B_specialist_births"])
            self.assertEqual(int(final["C_specialist_births"]), reproduction["C_specialist_births"])

    def test_baseline_explicitly_disables_trade_and_exports_its_config(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            world = run_baseline(steps=10, results_dir=Path(directory), make_plots=False)
            output = Path(directory) / "gamma_1" / "trade_off" / "seed_7"
            metadata = json.loads((output / "config.json").read_text())
            self.assertFalse(world.cfg.trade_enabled)
            self.assertEqual(world.trades, [])
            self.assertFalse(metadata["config"]["trade_enabled"])
            self.assertEqual(metadata["seed"], 7)
            self.assertEqual(metadata["completed_steps"], 10)
            with (output / "trades.csv").open() as stream:
                reader = csv.DictReader(stream)
                self.assertIn("B_donor_id", reader.fieldnames)
                self.assertEqual(list(reader), [])

    def test_exports_reconcile_event_losses_with_history(self):
        config = Config(trade_enabled=True, trade_efficiency=0.75)
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)
            world = run_experiment(config, 7, 10, output, make_plots=False)
            with (output / "trades.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            with (output / "history.csv").open() as stream:
                final = list(csv.DictReader(stream))[-1]
            self.assertGreater(len(rows), 0)
            self.assertEqual(len(rows), len(world.trades))
            self.assertEqual(int(final["cumulative_trade_events"]), len(rows))
            for good in ("B", "C"):
                for row in rows:
                    self.assertAlmostEqual(
                        float(row[f"gross_{good}"]),
                        float(row[f"net_{good}"]) + float(row[f"loss_{good}"]),
                    )
                self.assertAlmostEqual(
                    float(final[f"cumulative_trade_loss_{good}"]),
                    sum(float(row[f"loss_{good}"]) for row in rows),
                )
            for event, row in zip(world.trades, rows):
                self.assertEqual(
                    {int(row["B_donor_id"]), int(row["C_donor_id"])},
                    set(event["partner_ids"]),
                )

    def test_comparison_pairs_conditions_with_identical_initial_states(self):
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)
            summaries = run_comparison(
                seeds=(7,), gammas=(1.0, 2.0), steps=2,
                results_dir=output, deficit_move_probability=0.25, make_plots=False,
            )
            self.assertEqual(len(summaries), 4)
            for gamma in (1, 2):
                configs = []
                starts = []
                for condition in ("trade_off", "trade_random"):
                    run_dir = output / f"gamma_{gamma}" / condition / "seed_7"
                    configs.append(json.loads((run_dir / "config.json").read_text())["config"])
                    starts.append(json.loads((run_dir / "allocation_snapshots.json").read_text())["0"])
                self.assertFalse(configs[0].pop("trade_enabled"))
                self.assertTrue(configs[1].pop("trade_enabled"))
                self.assertEqual(configs[0], configs[1])
                self.assertEqual(configs[0]["production_specialization_exponent"], gamma)
                self.assertEqual(configs[0]["deficit_move_probability"], 0.25)
                self.assertEqual(starts[0], starts[1])
            self.assertTrue((output / "summary.csv").is_file())

    def test_early_extinction_and_zero_steps_are_saved(self):
        config = Config(initial_agents=1, initial_energy=0.01, metabolism=1.0)
        for steps, expected_time in ((0, 0), (25, 1)):
            with self.subTest(steps=steps), TemporaryDirectory() as directory:
                with redirect_stdout(io.StringIO()):
                    world = run_experiment(config, 7, steps, Path(directory), make_plots=False)
                metadata = json.loads((Path(directory) / "config.json").read_text())
                snapshots = json.loads((Path(directory) / "allocation_snapshots.json").read_text())
                self.assertEqual(world.t, expected_time)
                self.assertEqual(metadata["requested_steps"], steps)
                self.assertEqual(metadata["completed_steps"], expected_time)
                self.assertEqual(metadata["config"], json.loads(json.dumps(asdict(config))))
                self.assertIn(str(expected_time), snapshots)


if __name__ == "__main__":
    unittest.main()
