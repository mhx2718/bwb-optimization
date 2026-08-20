from __future__ import annotations

import unittest
from pathlib import Path

from bwb_pipeline.config import DataConfig, load_pipeline_config
from bwb_pipeline.schema import (
    ALL_COLUMNS,
    ALL_INPUT_COLUMNS,
    CONTINUOUS_DESIGN_COLUMNS,
    DESIGN_COLUMNS,
    FLIGHT_COLUMNS,
    FORWARD_TARGET_COLUMNS,
    TOPOLOGY_COLUMNS,
    assert_exact_schema,
)


class SchemaConfigTests(unittest.TestCase):
    def test_authoritative_dimensions_and_disjoint_roles(self) -> None:
        self.assertEqual(len(DESIGN_COLUMNS), 21)
        self.assertEqual(len(CONTINUOUS_DESIGN_COLUMNS), 18)
        self.assertEqual(len(ALL_INPUT_COLUMNS), 24)
        self.assertEqual(len(ALL_COLUMNS), 28)
        self.assertEqual(len(FORWARD_TARGET_COLUMNS), 3)
        self.assertTrue(set(TOPOLOGY_COLUMNS).isdisjoint(CONTINUOUS_DESIGN_COLUMNS))
        self.assertTrue(set(FLIGHT_COLUMNS).isdisjoint(DESIGN_COLUMNS))

    def test_schema_accepts_trailing_whitespace_but_rejects_missing(self) -> None:
        columns = list(ALL_COLUMNS)
        columns[-1] += " "
        assert_exact_schema(columns)
        with self.assertRaises(ValueError):
            assert_exact_schema(columns[:-1])

    def test_split_fractions_must_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            DataConfig(train_fraction=0.7, validation_fraction=0.2, test_fraction=0.2)

    def test_repository_default_config_loads(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_pipeline_config(root / "configs" / "default.yaml")
        self.assertEqual(config.master_seed, 20260819)
        self.assertEqual(config.forward_model.epochs, 2000)
        self.assertAlmostEqual(config.stress_classifier.probability_threshold, 0.90)


if __name__ == "__main__":
    unittest.main()

