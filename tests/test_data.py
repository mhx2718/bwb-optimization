from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from bwb_pipeline.config import DataConfig
from bwb_pipeline.data import DataValidationError, prepare_data
from bwb_pipeline.reproducibility import SeedRegistry
from bwb_pipeline.schema import (
    ALL_COLUMNS,
    DESIGN_COLUMNS,
    FLIGHT_COLUMNS,
    FORWARD_TARGET_COLUMNS,
    STRESS_TARGET_COLUMN,
)


def synthetic_frame(n_designs: int = 40) -> pd.DataFrame:
    rows = []
    flights = ((5.0, 80.0, 2.0), (15.0, 120.0, 6.0), (25.0, 200.0, 10.0))
    for design_index in range(n_designs):
        design = {
            name: float(0.01 * (column + 1) + design_index)
            for column, name in enumerate(DESIGN_COLUMNS)
        }
        # Preserve valid-looking topology integers; exact values do not affect tests.
        design["# of Ribs"] = float(3 + design_index % 12)
        design["# of Fuselage Ribs"] = float(3 + 2 * (design_index % 5))
        design["# of Fuselage Spars"] = float(3 + design_index % 10)
        targets = {
            "Aircraft Empty Weight": 100.0 + design_index,
            "Payload Volume": 8.0e8 + 1.0e6 * design_index,
            "Fuel Volume": 5.0e8 + 2.0e6 * design_index,
        }
        for flight_index, flight in enumerate(flights):
            row = dict(design)
            row.update(dict(zip(FLIGHT_COLUMNS, flight)))
            row.update(targets)
            # Every design contains one feasible and two infeasible/feasible rows in
            # an alternating pattern, giving all persisted splits both classes.
            feasible = (design_index + flight_index) % 3 != 0
            row[STRESS_TARGET_COLUMN] = 300.0 if feasible else 400.0
            rows.append(row)
    return pd.DataFrame(rows).loc[:, ALL_COLUMNS]


class DataPipelineTests(unittest.TestCase):
    def test_group_split_is_exact_disjoint_and_order_invariant(self) -> None:
        frame = synthetic_frame()
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        config = DataConfig()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.csv"
            split_path = root / "split.json"
            frame.to_csv(data_path, index=False)
            bundle = prepare_data(
                data_path,
                config,
                SeedRegistry(123),
                split_manifest_path=split_path,
            )
            self.assertEqual(bundle.exact_duplicate_rows_removed, 1)
            self.assertEqual(len(bundle.structural_designs), 40)
            self.assertEqual(
                bundle.split_manifest.group_counts,
                {"train": 28, "validation": 6, "test": 6},
            )
            group_sets = {
                split: set(bundle.rows_for_split(split)["design_id"])
                for split in ("train", "validation", "test")
            }
            self.assertTrue(group_sets["train"].isdisjoint(group_sets["validation"]))
            self.assertTrue(group_sets["train"].isdisjoint(group_sets["test"]))
            self.assertTrue(group_sets["validation"].isdisjoint(group_sets["test"]))

            # Canonical fingerprint and persisted assignment do not depend on CSV order.
            shuffled_path = root / "shuffled.csv"
            frame.sample(frac=1.0, random_state=99).to_csv(shuffled_path, index=False)
            repeated = prepare_data(
                shuffled_path,
                config,
                SeedRegistry(123),
                split_manifest_path=split_path,
            )
            self.assertEqual(bundle.dataset_fingerprint, repeated.dataset_fingerprint)
            self.assertEqual(
                bundle.split_manifest.design_assignments,
                repeated.split_manifest.design_assignments,
            )

    def test_forward_target_variation_within_design_fails_loudly(self) -> None:
        frame = synthetic_frame(20)
        frame.loc[1, FORWARD_TARGET_COLUMNS[0]] += 1.0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "bad.csv"
            frame.to_csv(data_path, index=False)
            with self.assertRaises(DataValidationError):
                prepare_data(
                    data_path,
                    DataConfig(),
                    SeedRegistry(1),
                    split_manifest_path=root / "split.json",
                )

    def test_non_integer_topology_fails_loudly(self) -> None:
        frame = synthetic_frame(20)
        frame.loc[0, "# of Ribs"] = 3.5
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "bad_topology.csv"
            frame.to_csv(data_path, index=False)
            with self.assertRaises(DataValidationError):
                prepare_data(
                    data_path,
                    DataConfig(),
                    SeedRegistry(1),
                    split_manifest_path=root / "split.json",
                )


if __name__ == "__main__":
    unittest.main()
