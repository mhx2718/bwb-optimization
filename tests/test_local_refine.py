import importlib.util
import unittest

import numpy as np

from bwb_pipeline.evaluator import (
    BWBEvaluator,
    DESIGN_COLUMNS,
    make_antithetic_sobol_noise,
)
from bwb_pipeline.local_refine import LocalRefineConfig, refine_one_start

from test_evaluator import ConstantStress, FakeForward, FakeLD, MISSION, midpoint_design


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is an optional integration dependency here")
class LocalRefineTests(unittest.TestCase):
    @staticmethod
    def differentiable_forward():
        import torch

        class DifferentiableForward(FakeForward):
            def predict_tensor(self, values, input_is_scaled=False):
                c2_index = list(DESIGN_COLUMNS).index("C2/C1")
                c2 = values[:, c2_index]
                weight = 35.0 + 100.0 * (c2 - 0.60) ** 2
                return torch.stack(
                    [
                        weight,
                        torch.full_like(weight, 1.2e9),
                        torch.full_like(weight, 0.8e9),
                    ],
                    dim=1,
                )

        return DifferentiableForward()

    def test_projected_adam_is_monotone_and_preserves_topology(self):
        bank = make_antithetic_sobol_noise(8, 18, 77)
        evaluator = BWBEvaluator(
            self.differentiable_forward(), ConstantStress(), FakeLD(), bank, bank
        )
        start_eval = evaluator.evaluate(midpoint_design(0.70), MISSION).iloc[0]
        config = LocalRefineConfig(
            starts_per_case_per_round=1,
            steps=10,
            learning_rate=0.02,
            trust_radius_fraction=0.05,
            exact_check_every=1,
        )
        result = refine_one_start(
            evaluator, MISSION, start_eval, config, seed=123,
            round_index=0, start_rank=0,
        )
        champion = result.champion.iloc[0]
        self.assertLessEqual(
            float(champion["search_merit"]), float(start_eval["search_merit"])
        )
        for name in ("# of Ribs", "# of Fuselage Ribs", "# of Fuselage Spars"):
            self.assertEqual(int(champion[name]), int(start_eval[name]))

    def test_infeasible_to_infeasible_proposal_never_replaces_incumbent(self):
        bank = make_antithetic_sobol_noise(8, 18, 77)
        evaluator = BWBEvaluator(
            self.differentiable_forward(),
            ConstantStress(0.85),
            FakeLD(),
            bank,
            bank,
        )
        start_eval = evaluator.evaluate(midpoint_design(0.70), MISSION).iloc[0]
        self.assertFalse(bool(start_eval["hard_accepted"]))
        result = refine_one_start(
            evaluator,
            MISSION,
            start_eval,
            LocalRefineConfig(
                starts_per_case_per_round=1,
                steps=5,
                learning_rate=0.02,
                trust_radius_fraction=0.05,
                exact_check_every=1,
            ),
            seed=123,
            round_index=0,
            start_rank=0,
        )
        champion = result.champion.iloc[0]
        self.assertFalse(bool(champion["hard_accepted"]))
        self.assertAlmostEqual(
            float(champion["C2/C1"]), float(start_eval["C2/C1"])
        )
        self.assertFalse(
            bool(result.trace.loc[result.trace["local_step"] > 0, "accepted_update"].any())
        )


if __name__ == "__main__":
    unittest.main()
