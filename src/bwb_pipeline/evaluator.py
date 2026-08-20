"""Authoritative, deterministic evaluation for BWB optimization.

The optimizer never calls the forward, stress, or L/D models directly.  This
module is the single contract that combines their predictions, applies the
official weighted score, and enforces the three requested hard constraints:

* no warning (or error) from the official L/D predictor;
* nominal stress-feasibility probability >= 0.90;
* median stress-feasibility probability >= 0.80 at *each* configured noise
  level (0.1% and 0.5% of the official continuous-variable ranges by default).

All constraint tiers are mapped to finite, disjoint objective intervals.  This
is intentional: optimizers such as CMA-ES cannot make progress reliably when
given NaN/Inf values or when a soft penalty lets a lower official score buy its
way through a hard feasibility constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .schema import (
    CONTINUOUS_DESIGN_COLUMNS,
    DESIGN_COLUMNS,
    FLIGHT_COLUMNS,
    GEOMETRY_COLUMNS,
    OFFICIAL_DESIGN_BOUNDS,
    TOPOLOGY_COLUMNS,
    VOLUME_TO_CUBIC_METERS,
)

# Backward-compatible public alias; the values themselves live in schema.py.
OFFICIAL_BOUNDS = OFFICIAL_DESIGN_BOUNDS


def _noise_slug(fraction: float) -> str:
    percent = 100.0 * float(fraction)
    return (f"{percent:g}pct").replace(".", "p")


def reflect_unit(values: np.ndarray) -> np.ndarray:
    """Reflect arbitrary coordinates into [0, 1] without boundary pile-up."""

    array = np.asarray(values, dtype=np.float64)
    return 1.0 - np.abs(np.mod(array, 2.0) - 1.0)


def make_antithetic_sobol_noise(
    n_samples: int,
    dimension: int,
    seed: int,
) -> np.ndarray:
    """Create a deterministic common-random-number bank in [-1, 1].

    Antithetic pairing prevents a finite bank from acquiring a systematic
    positive or negative displacement.  Search and confirmation must use
    separately seeded instances of this function.
    """

    n_samples = int(n_samples)
    dimension = int(dimension)
    if n_samples < 2 or n_samples % 2:
        raise ValueError("An antithetic noise bank needs a positive even size.")
    if dimension < 1:
        raise ValueError("Noise dimension must be positive.")

    half = n_samples // 2
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
    if half & (half - 1) == 0:
        unit = sampler.random_base2(int(math.log2(half)))
    else:
        unit = sampler.random(half)
    directions = 2.0 * np.asarray(unit, dtype=np.float64) - 1.0
    return np.vstack([directions, -directions])


@dataclass(frozen=True)
class EvaluationThresholds:
    nominal_probability: float = 0.90
    robust_probability: float = 0.80

    def __post_init__(self) -> None:
        if not (0.0 < self.robust_probability < self.nominal_probability <= 1.0):
            raise ValueError(
                "Require 0 < robust_probability < nominal_probability <= 1."
            )


@dataclass(frozen=True)
class DesignSpace:
    """The fixed topology/continuous representation used by CMA-ES."""

    design_columns: Tuple[str, ...] = DESIGN_COLUMNS
    topology_columns: Tuple[str, ...] = TOPOLOGY_COLUMNS
    continuous_columns: Tuple[str, ...] = CONTINUOUS_DESIGN_COLUMNS
    bounds: Mapping[str, Tuple[float, float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.bounds is None:
            object.__setattr__(self, "bounds", dict(OFFICIAL_BOUNDS))
        missing = set(self.design_columns).difference(self.bounds)
        if missing:
            raise KeyError(f"Missing official design bounds: {sorted(missing)}")
        if set(self.topology_columns) & set(self.continuous_columns):
            raise ValueError("Topology and continuous columns must be disjoint.")
        if set(self.topology_columns) | set(self.continuous_columns) != set(
            self.design_columns
        ):
            raise ValueError("Topology + continuous columns must cover the design.")

    @property
    def continuous_lower(self) -> np.ndarray:
        return np.asarray([self.bounds[name][0] for name in self.continuous_columns])

    @property
    def continuous_upper(self) -> np.ndarray:
        return np.asarray([self.bounds[name][1] for name in self.continuous_columns])

    @property
    def continuous_width(self) -> np.ndarray:
        return self.continuous_upper - self.continuous_lower

    def unit_to_continuous(self, unit_values: np.ndarray) -> np.ndarray:
        unit = np.atleast_2d(np.asarray(unit_values, dtype=np.float64))
        if unit.shape[1] != len(self.continuous_columns):
            raise ValueError(
                f"Expected {len(self.continuous_columns)} continuous coordinates, "
                f"got {unit.shape}."
            )
        unit = np.clip(unit, 0.0, 1.0)
        return self.continuous_lower[None, :] + unit * self.continuous_width[None, :]

    def continuous_to_unit(self, values: np.ndarray) -> np.ndarray:
        continuous = np.atleast_2d(np.asarray(values, dtype=np.float64))
        return np.clip(
            (continuous - self.continuous_lower[None, :])
            / self.continuous_width[None, :],
            0.0,
            1.0,
        )

    def assemble(
        self,
        continuous_values: np.ndarray,
        topology: Sequence[float],
    ) -> pd.DataFrame:
        continuous = np.atleast_2d(
            np.asarray(continuous_values, dtype=np.float64)
        )
        topology_values = np.asarray(topology, dtype=np.float64).reshape(-1)
        if len(topology_values) != len(self.topology_columns):
            raise ValueError("Topology must contain exactly three integer counts.")
        if not np.isfinite(topology_values).all():
            raise ValueError("Topology contains NaN/Inf.")
        allowed = (
            np.arange(3, 15, dtype=np.int64),
            np.arange(3, 12, 2, dtype=np.int64),
            np.arange(3, 13, dtype=np.int64),
        )
        for name, value, legal in zip(self.topology_columns, topology_values, allowed):
            if not np.isclose(value, np.rint(value), rtol=0.0, atol=1.0e-12):
                raise ValueError(f"Topology value {name}={value} is not an integer.")
            if int(np.rint(value)) not in set(legal.tolist()):
                raise ValueError(
                    f"Topology value {name}={value} is outside the legal set."
                )
        frame = pd.DataFrame(index=np.arange(len(continuous)))
        for index, name in enumerate(self.continuous_columns):
            lower, upper = self.bounds[name]
            frame[name] = np.clip(continuous[:, index], lower, upper)
        for value, name in zip(topology_values, self.topology_columns):
            frame[name] = float(value)
        frame = frame.loc[:, self.design_columns]
        frame.loc[:, "# of Ribs"] = np.rint(frame["# of Ribs"])
        frame.loc[:, "# of Fuselage Spars"] = np.rint(
            frame["# of Fuselage Spars"]
        )
        fuselage_ribs = frame["# of Fuselage Ribs"].to_numpy(dtype=np.float64)
        frame.loc[:, "# of Fuselage Ribs"] = 3.0 + 2.0 * np.rint(
            (fuselage_ribs - 3.0) / 2.0
        )
        return frame

    def frame_to_unit(self, designs: pd.DataFrame) -> np.ndarray:
        return self.continuous_to_unit(
            designs.loc[:, self.continuous_columns].to_numpy(dtype=np.float64)
        )


def _as_warning_list(value: Any) -> Sequence[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped == "[]":
            return []
        try:
            decoded = json.loads(stripped)
        except (TypeError, ValueError, json.JSONDecodeError):
            return [stripped]
        if isinstance(decoded, list):
            return [str(item) for item in decoded]
        return [str(decoded)]
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _coerce_forward_predictions(prediction: Any, n_rows: int) -> np.ndarray:
    if isinstance(prediction, pd.DataFrame):
        named_candidates = (
            ("Aircraft Empty Weight", "Payload Volume", "Fuel Volume"),
            ("weight", "payload", "fuel"),
            ("weight_kg", "payload_m3", "fuel_m3"),
        )
        for names in named_candidates:
            if all(name in prediction.columns for name in names):
                array = prediction.loc[:, names].to_numpy(dtype=np.float64)
                break
        else:
            array = prediction.to_numpy(dtype=np.float64)
    elif isinstance(prediction, Mapping):
        keys = (
            ("weight", "payload", "fuel"),
            ("weight_kg", "payload_m3", "fuel_m3"),
        )
        for names in keys:
            if all(name in prediction for name in names):
                array = np.column_stack([prediction[name] for name in names])
                break
        else:
            raise KeyError("Forward prediction mapping has no recognized W/P/F keys.")
    else:
        array = np.asarray(prediction, dtype=np.float64)
    array = np.asarray(array, dtype=np.float64)
    if array.shape != (n_rows, 3):
        raise ValueError(f"Forward predictor returned {array.shape}; expected {(n_rows, 3)}.")
    return array


class BWBEvaluator:
    """Combine all authoritative models behind one reproducible interface."""

    def __init__(
        self,
        forward_model: Any,
        stress_model: Any,
        ld_model: Any,
        search_noise_bank: np.ndarray,
        confirmation_noise_bank: Optional[np.ndarray] = None,
        noise_fractions: Sequence[float] = (0.001, 0.005),
        thresholds: EvaluationThresholds = EvaluationThresholds(),
        design_space: Optional[DesignSpace] = None,
        forward_output_scales: Sequence[float] = (
            1.0,
            VOLUME_TO_CUBIC_METERS,
            VOLUME_TO_CUBIC_METERS,
        ),
        skip_robust_when_nominal_fails: bool = True,
    ) -> None:
        self.forward_model = forward_model
        self.stress_model = stress_model
        self.ld_model = ld_model
        self.design_space = design_space or DesignSpace()
        self.thresholds = thresholds
        self.noise_fractions = tuple(float(value) for value in noise_fractions)
        if self.noise_fractions != (0.001, 0.005):
            # The class supports deliberate sensitivity studies, but the default
            # production contract is explicitly the two requested levels.
            if not self.noise_fractions or any(value <= 0 for value in self.noise_fractions):
                raise ValueError("Noise fractions must be positive.")
        self.forward_output_scales = np.asarray(
            forward_output_scales, dtype=np.float64
        )
        if self.forward_output_scales.shape != (3,):
            raise ValueError("forward_output_scales must have length three.")
        self.skip_robust_when_nominal_fails = bool(
            skip_robust_when_nominal_fails
        )
        self.noise_banks = {
            "search": self._validate_bank(search_noise_bank),
            "confirmation": self._validate_bank(
                confirmation_noise_bank
                if confirmation_noise_bank is not None
                else search_noise_bank
            ),
        }

    def _validate_bank(self, bank: np.ndarray) -> np.ndarray:
        values = np.asarray(bank, dtype=np.float64)
        expected_dimension = len(self.design_space.continuous_columns)
        if values.ndim != 2 or values.shape[1] != expected_dimension:
            raise ValueError(
                f"Noise bank must have shape (N, {expected_dimension}); "
                f"got {values.shape}."
            )
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("Noise bank must be finite and non-empty.")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("Noise-bank directions must be inside [-1, 1].")
        return values.copy()

    @staticmethod
    def _mission_value(mission: Mapping[str, Any], key: str) -> float:
        aliases = {
            "Altitude": ("Altitude", "alt_kft"),
            "KCAS": ("KCAS", "kcas"),
            "AOA": ("AOA", "aoa"),
        }
        for candidate in aliases[key]:
            if candidate in mission:
                return float(mission[candidate])
        raise KeyError(f"Mission is missing {key!r}.")

    def _stress_inputs(
        self, designs: pd.DataFrame, mission: Mapping[str, Any]
    ) -> pd.DataFrame:
        inputs = designs.loc[:, self.design_space.design_columns].copy()
        for name in FLIGHT_COLUMNS:
            inputs[name] = self._mission_value(mission, name)
        feature_names = getattr(self.stress_model, "feature_names", None)
        if feature_names is not None:
            missing = set(feature_names).difference(inputs.columns)
            if missing:
                raise KeyError(f"Stress model features are missing: {sorted(missing)}")
            inputs = inputs.loc[:, list(feature_names)]
        return inputs

    def _predict_forward(self, designs: pd.DataFrame) -> np.ndarray:
        feature_names = getattr(
            self.forward_model, "feature_names", self.design_space.design_columns
        )
        frame = designs.loc[:, list(feature_names)]
        if hasattr(self.forward_model, "predict"):
            prediction = self.forward_model.predict(frame)
        elif callable(self.forward_model):
            prediction = self.forward_model(frame)
        else:
            raise TypeError("Forward model must be callable or expose predict().")
        return _coerce_forward_predictions(prediction, len(frame)) * (
            self.forward_output_scales[None, :]
        )

    def _predict_stress_probability(
        self, designs: pd.DataFrame, mission: Mapping[str, Any]
    ) -> np.ndarray:
        inputs = self._stress_inputs(designs, mission)
        if hasattr(self.stress_model, "predict_proba"):
            probability = self.stress_model.predict_proba(inputs)
        elif callable(self.stress_model):
            probability = self.stress_model(inputs)
        else:
            raise TypeError("Stress model must be callable or expose predict_proba().")
        probability = np.asarray(probability, dtype=np.float64)
        if probability.ndim == 2:
            if probability.shape[1] != 2:
                raise ValueError("2-D stress probabilities must have two columns.")
            probability = probability[:, 1]
        probability = probability.reshape(-1)
        if len(probability) != len(designs):
            raise ValueError("Stress predictor changed the row count.")
        if not np.isfinite(probability).all():
            raise FloatingPointError("Stress predictor returned NaN/Inf.")
        return np.clip(probability, 0.0, 1.0)

    def _predict_ld(
        self, designs: pd.DataFrame, mission: Mapping[str, Any]
    ) -> pd.DataFrame:
        if hasattr(self.ld_model, "predict_many"):
            raw = self.ld_model.predict_many(designs, mission)
        elif hasattr(self.ld_model, "predict_batch"):
            raw = self.ld_model.predict_batch(designs, mission)
        else:
            rows = []
            predictor = getattr(self.ld_model, "evaluate", self.ld_model)
            if not callable(predictor):
                raise TypeError("L/D model must expose predict_many/evaluate or be callable.")
            for _, design in designs.iterrows():
                geometry = {
                    name: float(design[name]) for name in GEOMETRY_COLUMNS
                }
                rows.append(
                    predictor(
                        geometry,
                        alt_kft=self._mission_value(mission, "Altitude"),
                        kcas=self._mission_value(mission, "KCAS"),
                        aoa=self._mission_value(mission, "AOA"),
                    )
                )
            raw = rows

        if isinstance(raw, pd.DataFrame):
            result = raw.reset_index(drop=True).copy()
        else:
            converted = []
            for row in raw:
                if hasattr(row, "__dict__") and not isinstance(row, Mapping):
                    converted.append(vars(row))
                else:
                    converted.append(dict(row))
            result = pd.DataFrame(converted)
        if len(result) != len(designs):
            raise ValueError("L/D adapter changed the row count.")

        for required in ("LD", "CL", "CD"):
            if required not in result:
                result[required] = np.nan
        warning_source = (
            result["warnings"]
            if "warnings" in result
            else result.get("ld_warnings", pd.Series([[]] * len(result)))
        )
        warnings = [_as_warning_list(value) for value in warning_source]
        result["ld_warnings"] = [
            json.dumps(value, ensure_ascii=False) for value in warnings
        ]
        result["ld_warning_count"] = [len(value) for value in warnings]
        if "ld_evaluation_error" not in result:
            result["ld_evaluation_error"] = ""
        finite = np.isfinite(
            result.loc[:, ["LD", "CL", "CD"]].to_numpy(dtype=np.float64)
        ).all(axis=1)
        positive_cd = result["CD"].to_numpy(dtype=np.float64) > 0.0
        if "ld_in_domain" in result:
            adapter_valid = result["ld_in_domain"].to_numpy(dtype=bool)
        elif "ld_valid" in result:
            adapter_valid = result["ld_valid"].to_numpy(dtype=bool)
        else:
            adapter_valid = np.ones(len(result), dtype=bool)
        no_error = result["ld_evaluation_error"].fillna("").astype(str).eq("")
        result["ld_in_domain"] = (
            adapter_valid
            & finite
            & positive_cd
            & (result["ld_warning_count"].to_numpy(dtype=np.int64) == 0)
            & no_error.to_numpy(dtype=bool)
        )
        result["ld_valid"] = result["ld_in_domain"]
        return result

    def _robust_medians(
        self,
        designs: pd.DataFrame,
        mission: Mapping[str, Any],
        eligible: np.ndarray,
        bank_name: str,
    ) -> Tuple[Dict[float, np.ndarray], np.ndarray]:
        bank = self.noise_banks[bank_name]
        n_rows = len(designs)
        evaluated = np.zeros(n_rows, dtype=bool)
        medians = {
            fraction: np.zeros(n_rows, dtype=np.float64)
            for fraction in self.noise_fractions
        }
        if not np.any(eligible):
            return medians, evaluated

        base_unit = self.design_space.frame_to_unit(designs.loc[eligible])
        topology_values = designs.loc[
            eligible, self.design_space.topology_columns
        ].to_numpy(dtype=np.float64)
        target_indices = np.flatnonzero(eligible)
        for fraction in self.noise_fractions:
            perturbed_unit = reflect_unit(
                base_unit[:, None, :] + float(fraction) * bank[None, :, :]
            )
            flat_continuous = self.design_space.unit_to_continuous(
                perturbed_unit.reshape(-1, perturbed_unit.shape[-1])
            )
            repeated_topology = np.repeat(topology_values, len(bank), axis=0)
            noisy_frame = pd.DataFrame(index=np.arange(len(flat_continuous)))
            for j, name in enumerate(self.design_space.continuous_columns):
                noisy_frame[name] = flat_continuous[:, j]
            for j, name in enumerate(self.design_space.topology_columns):
                noisy_frame[name] = repeated_topology[:, j]
            noisy_frame = noisy_frame.loc[:, self.design_space.design_columns]
            probability = self._predict_stress_probability(noisy_frame, mission)
            probability = probability.reshape(len(target_indices), len(bank))
            medians[fraction][target_indices] = np.median(probability, axis=1)
        evaluated[target_indices] = True
        return medians, evaluated

    def evaluate_unit(
        self,
        unit_values: np.ndarray,
        topology: Sequence[float],
        mission: Mapping[str, Any],
        bank_name: str = "search",
    ) -> pd.DataFrame:
        continuous = self.design_space.unit_to_continuous(unit_values)
        designs = self.design_space.assemble(continuous, topology)
        return self.evaluate(designs, mission, bank_name=bank_name)

    def evaluate(
        self,
        designs: pd.DataFrame,
        mission: Mapping[str, Any],
        bank_name: str = "search",
    ) -> pd.DataFrame:
        """Evaluate designs and return a finite optimizer-ready table."""

        if bank_name not in self.noise_banks:
            raise KeyError(f"Unknown robustness bank {bank_name!r}.")
        frame = designs.loc[:, self.design_space.design_columns].copy()
        if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
            raise ValueError("Design table contains NaN/Inf.")

        forward = self._predict_forward(frame)
        probability = self._predict_stress_probability(frame, mission)
        ld = self._predict_ld(frame, mission)
        ld_in_domain = ld["ld_in_domain"].to_numpy(dtype=bool)
        nominal_pass = probability >= self.thresholds.nominal_probability
        robust_eligible = (
            nominal_pass & ld_in_domain
            if self.skip_robust_when_nominal_fails
            else np.ones(len(frame), dtype=bool)
        )
        robust_medians, robustness_evaluated = self._robust_medians(
            frame, mission, robust_eligible, bank_name
        )
        worst_robust = np.min(
            np.column_stack(
                [robust_medians[fraction] for fraction in self.noise_fractions]
            ),
            axis=1,
        )

        weight_kg = forward[:, 0]
        payload_m3 = forward[:, 1]
        fuel_m3 = forward[:, 2]
        ld_ratio = ld["LD"].to_numpy(dtype=np.float64)
        mass_objective = weight_kg / 50.0
        ld_target = float(mission["LD_target"])
        payload_target = float(mission["Payload_target_m3"])
        fuel_target = float(mission["Fuel_target_m3"])
        ld_shortfall = np.maximum(0.0, (ld_target - ld_ratio) / ld_target)
        payload_shortfall = np.maximum(
            0.0, (payload_target - payload_m3) / payload_target
        )
        fuel_shortfall = np.maximum(
            0.0, (fuel_target - fuel_m3) / fuel_target
        )
        objective_matrix = np.column_stack(
            [mass_objective, ld_shortfall, payload_shortfall, fuel_shortfall]
        )
        forward_physical_valid = (
            np.isfinite(forward).all(axis=1) & (forward >= 0.0).all(axis=1)
        )
        finite_objectives = (
            np.isfinite(objective_matrix).all(axis=1) & forward_physical_valid
        )
        official_loss = (
            0.4 * mass_objective
            + 0.2 * ld_shortfall
            + 0.2 * payload_shortfall
            + 0.2 * fuel_shortfall
        )
        robust_pass = (
            robustness_evaluated
            & (worst_robust >= self.thresholds.robust_probability)
        )
        hard_accepted = ld_in_domain & nominal_pass & robust_pass & finite_objectives

        nominal_violation = np.maximum(
            0.0, self.thresholds.nominal_probability - probability
        ) / self.thresholds.nominal_probability
        robust_violation = np.maximum(
            0.0, self.thresholds.robust_probability - worst_robust
        ) / self.thresholds.robust_probability
        constraint_tier = np.where(
            ~ld_in_domain | ~finite_objectives,
            3,
            np.where(~nominal_pass, 2, np.where(~robust_pass, 1, 0)),
        ).astype(np.int64)
        active_violation = np.where(
            constraint_tier == 0,
            0.0,
            np.where(
                constraint_tier == 1,
                robust_violation,
                np.where(constraint_tier == 2, nominal_violation, 1.0),
            ),
        )
        safe_loss = np.where(np.isfinite(official_loss), official_loss, 1.0e12)
        bounded_loss = np.maximum(safe_loss, 0.0) / (
            1.0 + np.maximum(safe_loss, 0.0)
        )
        search_merit = np.where(
            constraint_tier == 0,
            bounded_loss,
            2.0 * constraint_tier + np.minimum(active_violation, 0.999999),
        )
        if not np.isfinite(search_merit).all():
            raise AssertionError("Finite tier construction produced NaN/Inf.")

        result = frame.reset_index(drop=True)
        result["weight_kg"] = weight_kg
        result["payload_m3"] = payload_m3
        result["fuel_m3"] = fuel_m3
        result["forward_physical_valid"] = forward_physical_valid
        result["LD"] = ld_ratio
        result["CL"] = ld["CL"].to_numpy(dtype=np.float64)
        result["CD"] = ld["CD"].to_numpy(dtype=np.float64)
        result["ld_in_domain"] = ld_in_domain
        result["ld_valid"] = ld_in_domain
        result["ld_warnings"] = ld["ld_warnings"].to_numpy()
        result["ld_warning_count"] = ld["ld_warning_count"].to_numpy()
        result["ld_evaluation_error"] = ld["ld_evaluation_error"].to_numpy()
        result["probability_feasible"] = probability
        result["robustness_evaluated"] = robustness_evaluated
        for fraction in self.noise_fractions:
            result[f"robust_median_{_noise_slug(fraction)}"] = robust_medians[
                fraction
            ]
        result["worst_robust_median"] = worst_robust
        result["objective_mass_over_50"] = mass_objective
        result["objective_ld_shortfall"] = ld_shortfall
        result["objective_payload_shortfall"] = payload_shortfall
        result["objective_fuel_shortfall"] = fuel_shortfall
        result["official_loss"] = official_loss
        result["nominal_probability_violation"] = nominal_violation
        result["robust_probability_violation"] = robust_violation
        result["constraint_tier"] = constraint_tier
        result["active_constraint_violation"] = active_violation
        result["hard_accepted"] = hard_accepted
        result["search_merit"] = search_merit
        result["robustness_bank"] = str(bank_name)
        result["robustness_samples"] = len(self.noise_banks[bank_name])
        return result


EVALUATION_OBJECTIVE_COLUMNS: Tuple[str, ...] = (
    "objective_mass_over_50",
    "objective_ld_shortfall",
    "objective_payload_shortfall",
    "objective_fuel_shortfall",
)


def build_evaluator_from_config(
    forward_model: Any,
    stress_model: Any,
    ld_model: Any,
    optimization_config: Mapping[str, Any],
    seed_registry: Any,
    design_space: Optional[DesignSpace] = None,
) -> BWBEvaluator:
    """Construct disjoint search/confirmation banks from the master seed tree."""

    if hasattr(seed_registry, "derive"):
        search_seed = int(seed_registry.derive("robustness", "search_bank"))
        confirmation_seed = int(
            seed_registry.derive("robustness", "confirmation_bank")
        )
    else:
        # Keep this helper usable with a plain master integer in small scripts.
        import hashlib

        def derive(label: str) -> int:
            payload = f"{int(seed_registry)}\0{label}".encode("utf-8")
            return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")

        search_seed = derive("robustness/search_bank")
        confirmation_seed = derive("robustness/confirmation_bank")
    space = design_space or DesignSpace()
    search_bank = make_antithetic_sobol_noise(
        int(optimization_config["search_noise_samples"]),
        len(space.continuous_columns),
        search_seed,
    )
    confirmation_bank = make_antithetic_sobol_noise(
        int(optimization_config["confirmation_noise_samples"]),
        len(space.continuous_columns),
        confirmation_seed,
    )
    return BWBEvaluator(
        forward_model=forward_model,
        stress_model=stress_model,
        ld_model=ld_model,
        search_noise_bank=search_bank,
        confirmation_noise_bank=confirmation_bank,
        noise_fractions=tuple(
            float(value)
            for value in optimization_config["noise_fractions_of_bound_width"]
        ),
        thresholds=EvaluationThresholds(
            nominal_probability=float(
                optimization_config["nominal_probability_threshold"]
            ),
            robust_probability=float(
                optimization_config["robust_probability_threshold"]
            ),
        ),
        design_space=space,
    )


def rank_evaluations(table: pd.DataFrame) -> pd.DataFrame:
    """Stable feasible-first ordering used by every search stage."""

    required = {
        "constraint_tier",
        "active_constraint_violation",
        "official_loss",
        "search_merit",
    }
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"Cannot rank evaluations; missing {sorted(missing)}")
    return table.sort_values(
        [
            "constraint_tier",
            "active_constraint_violation",
            "official_loss",
            "search_merit",
        ],
        kind="stable",
    ).reset_index(drop=True)
