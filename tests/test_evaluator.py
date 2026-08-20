import json
import unittest

import numpy as np
import pandas as pd

from bwb_pipeline.evaluator import (
    BWBEvaluator,
    DESIGN_COLUMNS,
    DesignSpace,
    EvaluationThresholds,
    make_antithetic_sobol_noise,
)


class FakeForward:
    feature_names = DESIGN_COLUMNS

    def predict(self, frame):
        c2 = frame["C2/C1"].to_numpy(dtype=float)
        weight = 35.0 + 100.0 * (c2 - 0.60) ** 2
        return np.column_stack(
            [weight, np.full(len(frame), 1.2e9), np.full(len(frame), 0.8e9)]
        )


class ConstantStress:
    feature_names = DESIGN_COLUMNS + ("Altitude", "KCAS", "AOA")

    def __init__(self, probability=0.95):
        self.probability = float(probability)

    def predict_proba(self, frame):
        return np.full(len(frame), self.probability)


class FakeLD:
    def predict_many(self, designs, mission):
        warnings = [
            ["geometry extrapolation"] if value > 0.80 else []
            for value in designs["C2/C1"].to_numpy(dtype=float)
        ]
        return pd.DataFrame(
            {
                "LD": np.full(len(designs), 20.0),
                "CL": np.full(len(designs), 0.2),
                "CD": np.full(len(designs), 0.01),
                "warnings": warnings,
                "ld_in_domain": [not value for value in warnings],
            }
        )


MISSION = {
    "case_id": 1,
    "LD_target": 6.0,
    "Payload_target_m3": 0.75,
    "Fuel_target_m3": 0.45,
    "Altitude": 15.0,
    "KCAS": 120.0,
    "AOA": 1.0,
}


def midpoint_design(c2=0.70):
    space = DesignSpace()
    row = {name: np.mean(space.bounds[name]) for name in DESIGN_COLUMNS}
    row["C2/C1"] = c2
    row["# of Ribs"] = 4
    row["# of Fuselage Ribs"] = 3
    row["# of Fuselage Spars"] = 3
    return pd.DataFrame([row]).loc[:, DESIGN_COLUMNS]


class EvaluatorTests(unittest.TestCase):
    def make_evaluator(self, stress_probability=0.95):
        bank = make_antithetic_sobol_noise(8, 18, 123)
        return BWBEvaluator(
            FakeForward(),
            ConstantStress(stress_probability),
            FakeLD(),
            bank,
            bank,
            thresholds=EvaluationThresholds(0.90, 0.80),
        )

    def test_hard_accepted_and_finite_merit(self):
        result = self.make_evaluator().evaluate(midpoint_design(), MISSION)
        self.assertTrue(bool(result.loc[0, "hard_accepted"]))
        self.assertEqual(int(result.loc[0, "constraint_tier"]), 0)
        self.assertTrue(np.isfinite(result.loc[0, "search_merit"]))
        self.assertGreaterEqual(result.loc[0, "worst_robust_median"], 0.80)

    def test_ld_warning_is_hard_rejection(self):
        result = self.make_evaluator().evaluate(midpoint_design(0.82), MISSION)
        self.assertFalse(bool(result.loc[0, "ld_in_domain"]))
        self.assertFalse(bool(result.loc[0, "hard_accepted"]))
        self.assertEqual(int(result.loc[0, "constraint_tier"]), 3)
        self.assertEqual(json.loads(result.loc[0, "ld_warnings"]), ["geometry extrapolation"])

    def test_nominal_failure_is_finite_and_robust_not_nan(self):
        result = self.make_evaluator(0.85).evaluate(midpoint_design(), MISSION)
        self.assertEqual(int(result.loc[0, "constraint_tier"]), 2)
        self.assertFalse(bool(result.loc[0, "robustness_evaluated"]))
        self.assertEqual(float(result.loc[0, "worst_robust_median"]), 0.0)
        self.assertTrue(np.isfinite(result.loc[0, "search_merit"]))

    def test_constraint_tiers_are_disjoint(self):
        feasible = self.make_evaluator().evaluate(midpoint_design(), MISSION)
        nominal_fail = self.make_evaluator(0.85).evaluate(midpoint_design(), MISSION)
        ld_fail = self.make_evaluator().evaluate(midpoint_design(0.82), MISSION)
        self.assertLess(feasible.loc[0, "search_merit"], 1.0)
        self.assertGreaterEqual(nominal_fail.loc[0, "search_merit"], 4.0)
        self.assertGreaterEqual(ld_fail.loc[0, "search_merit"], 6.0)


if __name__ == "__main__":
    unittest.main()

