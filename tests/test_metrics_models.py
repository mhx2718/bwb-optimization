from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from bwb_pipeline.metrics import (
    expected_calibration_error,
    probability_metrics,
    regression_metrics,
    threshold_metrics,
)
from bwb_pipeline.models.stress import PlattCalibrator


class MetricModelTests(unittest.TestCase):
    def test_regression_metrics_perfect_prediction(self) -> None:
        values = np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 8.0]])
        table = regression_metrics(values, values.copy(), ("a", "b"))
        self.assertTrue(np.allclose(table["RMSE"], 0.0))
        self.assertTrue(np.allclose(table["R2"], 1.0))

    def test_probability_and_threshold_metrics(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        probability = np.asarray([0.1, 0.2, 0.8, 0.9])
        self.assertLess(probability_metrics(labels, probability)["brier_score"], 0.1)
        at_half = threshold_metrics(labels, probability, 0.5)
        self.assertEqual(at_half["false_feasible"], 0)
        self.assertEqual(at_half["false_infeasible"], 0)
        self.assertGreaterEqual(expected_calibration_error(labels, probability), 0.0)

    def test_platt_calibration_is_deterministic_and_monotone(self) -> None:
        scores = np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        first = PlattCalibrator.fit(scores, labels)
        second = PlattCalibrator.fit(scores, labels)
        self.assertEqual(first, second)
        probability = first.predict(scores)
        self.assertTrue(np.all(np.diff(probability) > 0.0))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_forward_model_shape_and_gradients(self) -> None:
        import torch

        from bwb_pipeline.models.forward import ResidualForwardNet

        model = ResidualForwardNet(hidden_width=16, residual_blocks=2)
        values = torch.zeros((5, 21), requires_grad=True)
        outputs = model(values)
        self.assertEqual(tuple(outputs.shape), (5, 3))
        self.assertTrue(bool((outputs > 0.0).all()))
        outputs.sum().backward()
        self.assertIsNotNone(values.grad)


if __name__ == "__main__":
    unittest.main()

