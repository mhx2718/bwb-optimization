import unittest

import numpy as np
import pandas as pd

from bwb_pipeline.evaluator import DESIGN_COLUMNS, DesignSpace, TOPOLOGY_COLUMNS
from bwb_pipeline.topology_budget import (
    _performance_ranks,
    allocate_generation_blocks,
    build_empirical_topology_prior,
    build_round_weights,
    gamma_rank_weights,
)


class TopologyBudgetTests(unittest.TestCase):
    def test_gamma_rank_is_monotone_and_normalized(self):
        weight = gamma_rank_weights(np.arange(20), 0.70, 0.15)
        self.assertAlmostEqual(float(weight.sum()), 1.0)
        self.assertTrue(np.all(np.diff(weight) < 0.0))

    def test_exact_integer_budget_and_uniform_floor(self):
        prior = pd.DataFrame(
            {
                "case_id": [1] * 4,
                "prior_rank": np.arange(4),
                "# of Ribs": [3, 4, 5, 6],
                "# of Fuselage Ribs": [3] * 4,
                "# of Fuselage Spars": [3] * 4,
            }
        )
        weighted = build_round_weights(prior, 1.0, uniform_exploration_fraction=0.15)
        self.assertTrue(np.all(weighted["allocation_weight"] >= 0.15 / 4 - 1e-15))
        allocated = allocate_generation_blocks(
            weighted, total_evaluations=4800, repeats=3, population=16,
            minimum_generations_per_repeat=10,
        )
        self.assertEqual(int(allocated["allocated_evaluations"].sum()), 4800)
        self.assertTrue(np.all(allocated["generations_per_repeat"] >= 10))
        self.assertTrue(np.all(allocated["allocated_evaluations"] % 48 == 0))

    def test_empirical_prior_uses_unique_physical_designs(self):
        space = DesignSpace()
        rows = []
        for ribs in (3, 4):
            for index in range(3):
                row = {name: np.mean(space.bounds[name]) for name in DESIGN_COLUMNS}
                row["C1"] += index
                row["# of Ribs"] = ribs
                row["# of Fuselage Ribs"] = 3
                row["# of Fuselage Spars"] = 3
                row.update(
                    {
                        "Aircraft Empty Weight": 35 + ribs,
                        "Payload Volume": 0.9,
                        "Fuel Volume": 0.7,
                        "Max Hotspot Stress": 300.0,
                        "split": "train",
                    }
                )
                rows.extend([row.copy(), row.copy()])  # flight duplicate
        mission = {
            "case_id": 1,
            "Payload_target_m3": 0.75,
            "Fuel_target_m3": 0.45,
        }
        prior = build_empirical_topology_prior(
            pd.DataFrame(rows), [mission], minimum_feasible_unique_designs=3,
            volume_scale_to_m3=1.0,
        )
        self.assertEqual(len(prior), 2)
        self.assertTrue(np.all(prior["feasible_unique_designs"] == 3))
        self.assertEqual(int(prior.iloc[0]["# of Ribs"]), 3)

    def test_all_data_support_and_train_only_rank_are_reported_separately(self):
        space = DesignSpace()
        rows = []
        for index, split in enumerate(("train", "train", "test")):
            row = {name: np.mean(space.bounds[name]) for name in DESIGN_COLUMNS}
            row["C1"] += index
            row["# of Ribs"] = 4
            row["# of Fuselage Ribs"] = 3
            row["# of Fuselage Spars"] = 3
            row.update(
                {
                    "Aircraft Empty Weight": 40.0,
                    "Payload Volume": 0.9,
                    "Fuel Volume": 0.7,
                    "Max Hotspot Stress": 300.0,
                    "split": split,
                }
            )
            rows.append(row)
        prior = build_empirical_topology_prior(
            pd.DataFrame(rows),
            [{"case_id": 1, "Payload_target_m3": 0.75, "Fuel_target_m3": 0.45}],
            minimum_feasible_unique_designs=3,
            split="train",
            eligibility_split=None,
            volume_scale_to_m3=1.0,
        )
        self.assertEqual(int(prior.loc[0, "feasible_unique_designs"]), 3)
        self.assertEqual(
            int(prior.loc[0, "ranking_feasible_unique_designs"]), 2
        )
        self.assertEqual(prior.loc[0, "eligibility_split"], "all")
        self.assertEqual(prior.loc[0, "ranking_split"], "train")

    def test_one_lucky_repeat_does_not_control_evidence_rank(self):
        rows = []
        for topology, accepted, merits in (
            ((4, 3, 3), (True, False, False), (0.2, 4.0, 4.0)),
            ((5, 3, 3), (False, False, False), (2.0, 2.0, 2.0)),
        ):
            for repeat in range(3):
                rows.append(
                    {
                        **dict(zip(TOPOLOGY_COLUMNS, topology)),
                        "repeat": repeat,
                        "hard_accepted": accepted[repeat],
                        "official_loss": 0.2 if accepted[repeat] else 99.0,
                        "search_merit": merits[repeat],
                    }
                )
        ranks = _performance_ranks(
            pd.DataFrame(rows), [(4, 3, 3), (5, 3, 3)]
        )
        self.assertEqual(ranks.tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
