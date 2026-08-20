"""Typed configuration for the deterministic data and model pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    data_path: Path = Path("bwb_structures_dataset.csv")
    output_root: Path = Path("final_reproducible_pipeline_results")
    ld_model_dir: Path = Path("models/ld_surrogate")


@dataclass(frozen=True)
class DeterminismConfig:
    strict: bool = True
    device: str = "cpu"
    torch_num_threads: int = 1

    def __post_init__(self) -> None:
        if self.torch_num_threads < 1:
            raise ValueError("torch_num_threads must be positive.")


@dataclass(frozen=True)
class DataConfig:
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    split_manifest: Path = Path("split_manifest.json")
    forward_duplicate_target_rtol: float = 1.0e-6
    forward_duplicate_target_atol: float = 1.0e-12
    drop_exact_duplicates: bool = True

    def __post_init__(self) -> None:
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if any(value <= 0.0 for value in fractions):
            raise ValueError("All split fractions must be positive.")
        if abs(sum(fractions) - 1.0) > 1.0e-12:
            raise ValueError("Train/validation/test fractions must sum to one.")
        if self.forward_duplicate_target_rtol < 0.0:
            raise ValueError("forward_duplicate_target_rtol cannot be negative.")
        if self.forward_duplicate_target_atol < 0.0:
            raise ValueError("forward_duplicate_target_atol cannot be negative.")


@dataclass(frozen=True)
class ForwardModelConfig:
    output_dir: Path = Path("final_forwardmodel_results")
    epochs: int = 2000
    patience: int = 300
    lr_patience: int = 120
    learning_rate: float = 1.0e-3
    print_every: int = 25
    batch_size: int = 256
    hidden_width: int = 384
    residual_blocks: int = 4
    weight_decay: float = 1.0e-5
    huber_delta: float = 0.5
    min_learning_rate: float = 1.0e-6
    gradient_clip_norm: float = 1.0
    force_retrain: bool = False

    def __post_init__(self) -> None:
        integer_positive = {
            "epochs": self.epochs,
            "patience": self.patience,
            "lr_patience": self.lr_patience,
            "print_every": self.print_every,
            "batch_size": self.batch_size,
            "hidden_width": self.hidden_width,
            "residual_blocks": self.residual_blocks,
        }
        invalid = [name for name, value in integer_positive.items() if value < 1]
        if invalid:
            raise ValueError(f"Forward-model fields must be positive: {invalid}.")
        if self.learning_rate <= 0.0 or self.min_learning_rate <= 0.0:
            raise ValueError("Learning rates must be positive.")
        if self.weight_decay < 0.0 or self.huber_delta <= 0.0:
            raise ValueError("Invalid forward loss/regularization configuration.")


DEFAULT_STRESS_CANDIDATES = (
    {"depth": 6, "learning_rate": 0.030, "l2_leaf_reg": 3.0},
    {"depth": 7, "learning_rate": 0.030, "l2_leaf_reg": 5.0},
    {"depth": 8, "learning_rate": 0.025, "l2_leaf_reg": 5.0},
    {"depth": 9, "learning_rate": 0.020, "l2_leaf_reg": 8.0},
    {"depth": 10, "learning_rate": 0.020, "l2_leaf_reg": 12.0},
)


@dataclass(frozen=True)
class StressClassifierConfig:
    output_dir: Path = Path("final_stress_classifier_results")
    stress_limit_mpa: float = 335.0
    probability_threshold: float = 0.90
    noise_median_probability_threshold: float = 0.80
    iterations: int = 4000
    early_stopping_rounds: int = 300
    thread_count: int = 1
    cv_folds: int = 5
    calibration_regularization: float = 1.0e-6
    force_retrain: bool = False
    candidates: tuple[Mapping[str, float], ...] = field(
        default_factory=lambda: DEFAULT_STRESS_CANDIDATES
    )

    def __post_init__(self) -> None:
        if self.stress_limit_mpa != 335.0:
            raise ValueError(
                "This published split/classification contract fixes the stress "
                "limit at exactly 335 MPa. Create a versioned schema before "
                "changing the label definition."
            )
        for name, value in (
            ("probability_threshold", self.probability_threshold),
            (
                "noise_median_probability_threshold",
                self.noise_median_probability_threshold,
            ),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one.")
        if self.iterations < 1 or self.early_stopping_rounds < 1:
            raise ValueError("CatBoost iteration settings must be positive.")
        if self.thread_count < 1 or self.cv_folds < 2:
            raise ValueError("thread_count >= 1 and cv_folds >= 2 are required.")
        if not self.candidates:
            raise ValueError("At least one deterministic CatBoost candidate is required.")


@dataclass(frozen=True)
class PipelineConfig:
    master_seed: int = 20260819
    project: ProjectConfig = field(default_factory=ProjectConfig)
    determinism: DeterminismConfig = field(default_factory=DeterminismConfig)
    data: DataConfig = field(default_factory=DataConfig)
    forward_model: ForwardModelConfig = field(default_factory=ForwardModelConfig)
    stress_classifier: StressClassifierConfig = field(
        default_factory=StressClassifierConfig
    )

    def __post_init__(self) -> None:
        if not 0 <= self.master_seed < 2**63:
            raise ValueError("master_seed must be a non-negative signed 64-bit integer.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration section {name!r} must be a mapping.")
    return value


def _path_fields(values: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    converted = dict(values)
    for name in names:
        if name in converted:
            converted[name] = Path(converted[name])
    return converted


def pipeline_config_from_mapping(values: Mapping[str, Any]) -> PipelineConfig:
    """Create a typed config while ignoring unrelated optimizer/plot sections."""

    project_values = _path_fields(
        _mapping(values.get("project"), "project"),
        ("data_path", "output_root", "ld_model_dir"),
    )
    data_values = _path_fields(
        _mapping(values.get("data"), "data"), ("split_manifest",)
    )
    forward_values = _path_fields(
        _mapping(values.get("forward_model"), "forward_model"), ("output_dir",)
    )
    stress_values = _path_fields(
        _mapping(values.get("stress_classifier"), "stress_classifier"),
        ("output_dir",),
    )
    if "candidates" in stress_values:
        stress_values["candidates"] = tuple(
            dict(candidate) for candidate in stress_values["candidates"]
        )

    return PipelineConfig(
        master_seed=int(values.get("master_seed", 20260819)),
        project=ProjectConfig(**project_values),
        determinism=DeterminismConfig(
            **_mapping(values.get("determinism"), "determinism")
        ),
        data=DataConfig(**data_values),
        forward_model=ForwardModelConfig(**forward_values),
        stress_classifier=StressClassifierConfig(**stress_values),
    )


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, Mapping):
        raise TypeError(f"Top-level YAML document must be a mapping: {path}.")
    return pipeline_config_from_mapping(values)
