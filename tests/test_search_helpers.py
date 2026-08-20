import unittest
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from bwb_pipeline.cma_search import (
    CMAConfig,
    build_repeat_start_plan,
    run_cma_island,
    stable_child_seed,
)
from bwb_pipeline.evaluator import (
    BWBEvaluator,
    DESIGN_COLUMNS,
    DesignSpace,
    make_antithetic_sobol_noise,
)
from bwb_pipeline.optimization import (
    ConvergenceConfig,
    OptimizationConfig,
    _save_round_checkpoint,
    build_convergence_diagnostics,
    confirm_and_select,
    run_case_search,
)

from test_evaluator import ConstantStress, FakeForward, FakeLD, MISSION, midpoint_design


class SearchHelperTests(unittest.TestCase):
    def setUp(self):
        bank = make_antithetic_sobol_noise(8, 18, 99)
        self.evaluator = BWBEvaluator(
            FakeForward(), ConstantStress(), FakeLD(), bank, bank
        )

    def test_named_seed_is_stable_and_keyed(self):
        first = stable_child_seed(123, "cma", 1, 0, (4, 3, 3), 0)
        second = stable_child_seed(123, "cma", 1, 0, (4, 3, 3), 0)
        other = stable_child_seed(123, "cma", 1, 0, (5, 3, 3), 0)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_later_round_has_two_warm_and_one_cold_starts(self):
        archive = self.evaluator.evaluate(midpoint_design(), MISSION)
        archive["repeat"] = 0
        for name in self.evaluator.design_space.continuous_columns:
            archive[f"unit__{name}"] = self.evaluator.design_space.frame_to_unit(archive)[0][
                list(self.evaluator.design_space.continuous_columns).index(name)
            ]
        plan = build_repeat_start_plan(
            self.evaluator, (4, 3, 3), 1, 3, 2, archive, 123, 1
        )
        modes = [item[1] for item in plan]
        self.assertEqual(modes[0], "warm_same_topology")
        self.assertEqual(modes[1], "warm_global_transfer")
        self.assertEqual(modes[2], "cold_uniform")

    def test_confirmation_returns_exactly_one_official_minimum(self):
        designs = pd.concat([midpoint_design(0.70), midpoint_design(0.60)])
        archive = self.evaluator.evaluate(designs, MISSION)
        confirmed, final = confirm_and_select(self.evaluator, MISSION, archive, 10)
        self.assertEqual(len(final), 1)
        self.assertTrue(bool(final.loc[0, "hard_accepted"]))
        self.assertAlmostEqual(
            float(final.loc[0, "official_loss"]),
            float(confirmed.loc[confirmed["hard_accepted"], "official_loss"].min()),
        )
        self.assertIn("search__official_loss", confirmed.columns)
        self.assertEqual(float(final.loc[0, "Altitude"]), MISSION["Altitude"])

    def test_public_convergence_helper(self):
        archive = self.evaluator.evaluate(midpoint_design(), MISSION)
        copies = []
        for repeat in range(3):
            item = archive.copy()
            item["repeat"] = repeat
            item["start_mode"] = (
                "cold_uniform" if repeat == 2 else "warm_same_topology"
            )
            copies.append(item)
        round_archive = pd.concat(copies, ignore_index=True)
        diagnostics = build_convergence_diagnostics(
            [round_archive, round_archive], ConvergenceConfig(), case_id=1
        )
        self.assertEqual(len(diagnostics), 2)
        self.assertTrue(bool(diagnostics.loc[1, "converged_diagnostic"]))

    def test_infeasible_cold_repeat_cannot_supply_convergence_support(self):
        warm_a = self.evaluator.evaluate(midpoint_design(), MISSION)
        warm_a["repeat"] = 0
        warm_a["start_mode"] = "warm_same_topology"
        warm_b_design = midpoint_design()
        warm_b_design["# of Ribs"] = 5
        warm_b = self.evaluator.evaluate(warm_b_design, MISSION)
        warm_b["repeat"] = 1
        warm_b["start_mode"] = "warm_global_transfer"
        cold_infeasible = warm_a.copy()
        cold_infeasible["repeat"] = 2
        cold_infeasible["start_mode"] = "cold_uniform"
        cold_infeasible["hard_accepted"] = False
        cold_infeasible["constraint_tier"] = 2
        cold_infeasible["search_merit"] = 4.1
        archive = pd.concat([warm_a, warm_b, cold_infeasible], ignore_index=True)
        diagnostics = build_convergence_diagnostics(
            [archive, archive], ConvergenceConfig(), case_id=1
        )
        self.assertFalse(bool(diagnostics.loc[1, "converged_diagnostic"]))
        self.assertFalse(
            bool(diagnostics.loc[1, "cold_repeat_supports_round_best_topology"])
        )

    def test_default_shaped_optimization_mapping_loads(self):
        config = OptimizationConfig.from_mapping(
            {
                "repeats": 3,
                "rounds": [
                    {"total_evaluations_per_case": 4800, "prior_weight": 1.0}
                ],
                "local_refine": {"enabled": False},
                "convergence": {},
            }
        )
        self.assertEqual(config.repeats, 3)
        self.assertEqual(config.rounds[0].total_evaluations_per_case, 4800)

    def test_internal_cma_restart_is_deterministic_and_exhausts_budget(self):
        class StopEveryGenerationStrategy:
            def __init__(self, x0, sigma, options):
                self.mean = np.asarray(x0, dtype=float)
                self.sigma = float(sigma)
                self.D = np.ones_like(self.mean)
                self.options = options
                self.iteration = 0
                self.rng = np.random.default_rng(int(options["seed"]))

            def ask(self):
                return np.clip(
                    self.mean[None, :]
                    + self.rng.normal(
                        0.0,
                        0.01,
                        size=(int(self.options["popsize"]), len(self.mean)),
                    ),
                    0.0,
                    1.0,
                ).tolist()

            def tell(self, values, fitness):
                values = np.asarray(values, dtype=float)
                self.mean = values[int(np.argmin(fitness))]
                self.iteration += 1

            def stop(self):
                return {"forced_numerical_stop": True} if self.iteration >= 1 else {}

        stub = types.SimpleNamespace(CMAEvolutionStrategy=StopEveryGenerationStrategy)

        def run_once():
            with patch.dict(sys.modules, {"cma": stub}):
                return run_cma_island(
                    evaluator=self.evaluator,
                    mission=MISSION,
                    topology=(4, 3, 3),
                    start_unit=np.full(18, 0.5),
                    generations=5,
                    seed=123,
                    config=CMAConfig(population=4, archive_per_run=4),
                    case_id=1,
                    round_index=0,
                    repeat=0,
                    start_mode="cold_uniform",
                )

        first = run_once()
        second = run_once()
        self.assertEqual(first.actual_evaluations, 20)
        self.assertEqual(first.trace["internal_restart_count"].tolist(), [0, 1, 2, 3, 4])
        pd.testing.assert_frame_equal(first.trace, second.trace)
        pd.testing.assert_frame_equal(first.archive, second.archive)

    def test_round_checkpoint_is_immutable(self):
        table = pd.DataFrame({"value": [1.0, 2.0]})
        kwargs = dict(
            case_id=1,
            round_index=0,
            round_archive=table,
            allocation=table,
            cma_trace=table,
            local_trace=pd.DataFrame(),
            run_summary=table,
            convergence_row={"case_id": 1, "round": 0, "ok": True},
            master_seed=123,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _save_round_checkpoint(root, **kwargs)
            _save_round_checkpoint(root, **kwargs)
            changed = dict(kwargs)
            changed["round_archive"] = pd.DataFrame({"value": [9.0]})
            with self.assertRaises(RuntimeError):
                _save_round_checkpoint(root, **changed)

    def test_high_level_case_search_contract_with_cma_stub(self):
        class FakeStrategy:
            def __init__(self, x0, sigma, options):
                self.mean = np.asarray(x0, dtype=float)
                self.sigma = float(sigma)
                self.options = options
                self.D = np.ones_like(self.mean)
                self.iteration = 0
                self.rng = np.random.default_rng(int(options["seed"]))

            def ask(self):
                return np.clip(
                    self.mean[None, :]
                    + self.rng.normal(
                        0.0, 0.01,
                        size=(int(self.options["popsize"]), len(self.mean)),
                    ),
                    0.0,
                    1.0,
                ).tolist()

            def tell(self, values, fitness):
                values = np.asarray(values, dtype=float)
                self.mean = values[int(np.argmin(fitness))]
                self.iteration += 1

            def stop(self):
                if self.iteration >= int(self.options["maxiter"]):
                    return {"maxiter": self.iteration}
                return {}

        cma_stub = types.SimpleNamespace(CMAEvolutionStrategy=FakeStrategy)
        prior = pd.DataFrame(
            {
                "case_id": [1],
                "prior_rank": [0],
                "# of Ribs": [4],
                "# of Fuselage Ribs": [3],
                "# of Fuselage Spars": [3],
            }
        )
        config = OptimizationConfig.from_mapping(
            {
                "repeats": 3,
                "rounds": [
                    {"total_evaluations_per_case": 12, "prior_weight": 1.0}
                ],
                "minimum_generations_per_topology_repeat": 1,
                "cma_population": 4,
                "cma_archive_per_run": 4,
                "warm_repeats_after_round_zero": 2,
                "confirmation_shortlist_per_case": 8,
                "local_refine": {"enabled": False},
            }
        )
        with patch.dict(sys.modules, {"cma": cma_stub}):
            result = run_case_search(
                self.evaluator, MISSION, prior, config, seed_source=123
            )
        self.assertEqual(len(result.final_design), 1)
        self.assertTrue(bool(result.final_design.loc[0, "hard_accepted"]))
        self.assertEqual(int(result.run_summary["actual_evaluations"].sum()), 12)
        self.assertEqual(
            int(result.cma_trace["generation_hard_accepted_count"].sum()), 12
        )
        self.assertAlmostEqual(
            float(result.convergence.loc[0, "cma_candidate_hard_feasible_fraction"]),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
