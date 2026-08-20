"""Deterministic CatBoost stress-feasibility classifier with calibration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

from ..artifacts import (
    ArtifactManifest,
    assert_artifact_compatible,
    assert_artifact_payloads,
    atomic_output_path,
    atomic_write_json,
    installed_versions,
    load_artifact_manifest,
    load_json,
    require_complete_artifact,
    save_artifact_manifest,
    source_fingerprint,
)
from ..config import StressClassifierConfig
from ..data import DataBundle, SPLIT_NAMES
from ..metrics import probability_metrics, threshold_metrics
from ..reproducibility import SeedRegistry, set_global_determinism
from ..schema import ALL_INPUT_COLUMNS, STRESS_TARGET_COLUMN

try:
    from catboost import CatBoostClassifier
except ImportError as exc:  # pragma: no cover - depends on optional runtime package
    CatBoostClassifier = None
    _CATBOOST_IMPORT_ERROR = exc
else:
    _CATBOOST_IMPORT_ERROR = None


STRESS_ARTIFACT_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "catboost",
)

ENSEMBLE_MEMBER_FLAG = "retain_as_ensemble_member"


def _declared_ensemble_candidates(
    config: StressClassifierConfig,
) -> tuple[Mapping[str, float], ...]:
    declared = tuple(
        candidate
        for candidate in config.candidates
        if bool(candidate.get(ENSEMBLE_MEMBER_FLAG, False))
    )
    if len(declared) == 1:
        raise ValueError(
            "An ensemble declaration needs at least two retained candidates."
        )
    return declared


def _require_catboost() -> None:
    if CatBoostClassifier is None:
        raise ImportError(
            "CatBoost is required for the stress classifier. Install the locked "
            "project dependencies before training or loading this model."
        ) from _CATBOOST_IMPORT_ERROR


@dataclass(frozen=True)
class PlattCalibrator:
    """Two-parameter sigmoid calibration applied to CatBoost raw scores."""

    slope: float = 1.0
    intercept: float = 0.0

    @classmethod
    def fit(
        cls,
        raw_scores: np.ndarray,
        labels: np.ndarray,
        *,
        regularization: float = 1.0e-6,
    ) -> "PlattCalibrator":
        scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
        outcomes = np.asarray(labels, dtype=np.float64).reshape(-1)
        if scores.shape != outcomes.shape or not len(scores):
            raise ValueError("Calibration scores and labels must be non-empty and aligned.")
        if not np.isfinite(scores).all() or not np.isin(outcomes, [0.0, 1.0]).all():
            raise ValueError("Calibration data must contain finite scores and binary labels.")
        if len(np.unique(outcomes)) != 2:
            raise ValueError("Platt calibration requires both classes in validation data.")
        if regularization < 0.0:
            raise ValueError("Calibration regularization cannot be negative.")

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            slope, intercept = parameters
            logits = slope * scores + intercept
            losses = np.logaddexp(0.0, logits) - outcomes * logits
            residual = expit(logits) - outcomes
            penalty = regularization * ((slope - 1.0) ** 2 + intercept**2)
            gradient = np.asarray(
                [
                    np.mean(residual * scores)
                    + 2.0 * regularization * (slope - 1.0),
                    np.mean(residual) + 2.0 * regularization * intercept,
                ]
            )
            return float(np.mean(losses) + penalty), gradient

        result = minimize(
            objective,
            x0=np.asarray([1.0, 0.0]),
            method="L-BFGS-B",
            jac=True,
            options={"ftol": 1.0e-14, "gtol": 1.0e-10, "maxiter": 1000},
        )
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"Platt calibration failed: {result.message}.")
        return cls(slope=float(result.x[0]), intercept=float(result.x[1]))

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(raw_scores, dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("Raw classifier scores must be finite.")
        return expit(self.slope * scores + self.intercept)

    def to_dict(self) -> dict[str, float]:
        return {"slope": self.slope, "intercept": self.intercept}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PlattCalibrator":
        return cls(slope=float(values["slope"]), intercept=float(values["intercept"]))


class CalibratedStressPredictor:
    """Authoritative calibrated P(Max Hotspot Stress <= limit)."""

    feature_names = ALL_INPUT_COLUMNS

    def __init__(
        self,
        model: Any,
        calibrator: PlattCalibrator,
        *,
        stress_limit_mpa: float,
        probability_threshold: float,
        artifact_id: str | None = None,
    ) -> None:
        self.model = model
        self.calibrator = calibrator
        self.stress_limit_mpa = float(stress_limit_mpa)
        self.probability_threshold = float(probability_threshold)
        self.artifact_id = artifact_id

    def _array(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(values, pd.DataFrame):
            missing = [name for name in self.feature_names if name not in values.columns]
            if missing:
                raise ValueError(f"Missing stress-classifier columns: {missing}.")
            array = values.loc[:, self.feature_names].to_numpy(dtype=np.float64)
        else:
            array = np.asarray(values, dtype=np.float64)
            if array.ndim == 1:
                array = array[None, :]
        if array.ndim != 2 or array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected stress input shape (N, {len(self.feature_names)}), "
                f"received {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError("Stress-classifier inputs must be finite.")
        return array

    def raw_score(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        array = self._array(values)
        if len(array) == 0:
            return np.empty(0, dtype=np.float64)
        raw = self.model.predict(array, prediction_type="RawFormulaVal")
        return np.asarray(raw, dtype=np.float64).reshape(-1)

    def predict_proba(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.calibrator.predict(self.raw_score(values))

    def predict(
        self,
        values: pd.DataFrame | np.ndarray,
        *,
        threshold: float | None = None,
    ) -> np.ndarray:
        selected = self.probability_threshold if threshold is None else float(threshold)
        if not 0.0 <= selected <= 1.0:
            raise ValueError("Probability threshold must lie in [0, 1].")
        return self.predict_proba(values) >= selected


class ConservativeStressEnsemble:
    """Minimum-across-members calibrated stress-feasibility score.

    Each member is independently Platt calibrated.  The returned one-dimensional
    score is the row-wise minimum member probability.  It is intentionally a
    conservative feasibility score rather than a claim that the minimum itself
    is a calibrated probability.
    """

    feature_names = ALL_INPUT_COLUMNS

    def __init__(
        self,
        members: Sequence[CalibratedStressPredictor],
        *,
        probability_threshold: float,
        artifact_id: str | None = None,
        aggregation: str = "minimum",
    ) -> None:
        self.members = tuple(members)
        if not self.members:
            raise ValueError("A stress ensemble needs at least one member.")
        if aggregation != "minimum":
            raise ValueError("Only minimum aggregation is supported.")
        if any(
            tuple(member.feature_names) != tuple(self.feature_names)
            for member in self.members
        ):
            raise ValueError("Stress-ensemble members use inconsistent features.")
        self.aggregation = aggregation
        self.probability_threshold = float(probability_threshold)
        self.stress_limit_mpa = float(self.members[0].stress_limit_mpa)
        self.artifact_id = artifact_id

    @property
    def calibrators(self) -> tuple[PlattCalibrator, ...]:
        return tuple(member.calibrator for member in self.members)

    def member_raw_scores(
        self, values: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        return np.column_stack([member.raw_score(values) for member in self.members])

    def member_probabilities(
        self, values: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        return np.column_stack(
            [member.predict_proba(values) for member in self.members]
        )

    def limiting_member(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.argmin(self.member_probabilities(values), axis=1)

    def raw_score(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return the raw score of the member limiting each row.

        Call ``member_raw_scores`` for the unambiguous full matrix.  A single
        raw score cannot be passed through one common calibrator because every
        member owns a separate Platt mapping.
        """

        raw = self.member_raw_scores(values)
        limiting = self.limiting_member(values)
        return raw[np.arange(len(raw)), limiting]

    def predict_proba(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.min(self.member_probabilities(values), axis=1)

    def predict(
        self,
        values: pd.DataFrame | np.ndarray,
        *,
        threshold: float | None = None,
    ) -> np.ndarray:
        selected = self.probability_threshold if threshold is None else float(threshold)
        if not 0.0 <= selected <= 1.0:
            raise ValueError("Probability threshold must lie in [0, 1].")
        return self.predict_proba(values) >= selected


def _training_config(config: StressClassifierConfig) -> dict[str, Any]:
    values = asdict(config)
    values.pop("output_dir", None)
    values.pop("force_retrain", None)
    return values


def _expected_manifest(
    bundle: DataBundle,
    config: StressClassifierConfig,
    seed_registry: SeedRegistry,
) -> ArtifactManifest:
    module_root = Path(__file__).resolve().parents[1]
    code_hash = source_fingerprint(
        (
            Path(__file__),
            module_root / "schema.py",
            module_root / "config.py",
            module_root / "data.py",
            module_root / "metrics.py",
            module_root / "reproducibility.py",
            module_root / "artifacts.py",
        )
    )
    ensemble = bool(_declared_ensemble_candidates(config))
    return ArtifactManifest(
        artifact_type=(
            "bwb-conservative-stress-ensemble-24d-v2"
            if ensemble
            else "bwb-calibrated-stress-classifier-24d-v1"
        ),
        dataset_fingerprint=bundle.dataset_fingerprint,
        split_fingerprint=bundle.split_manifest.split_fingerprint,
        feature_names=ALL_INPUT_COLUMNS,
        target_names=(f"P({STRESS_TARGET_COLUMN}<={config.stress_limit_mpa:g})",),
        master_seed=seed_registry.master_seed,
        model_config=_training_config(config),
        unit_transforms={
            "label": "1=feasible",
            "calibration": (
                "memberwise-platt-on-held-out-validation-partition"
                if ensemble
                else "platt-on-validation-raw-score"
            ),
            "ensemble_aggregation": (
                "minimum" if ensemble else "not-applicable"
            ),
        },
        package_versions=installed_versions(STRESS_ARTIFACT_PACKAGES),
        code_version=f"0.1.0+{code_hash}",
    ).with_artifact_id()


def _build_catboost(
    config: StressClassifierConfig,
    candidate: Mapping[str, float],
    *,
    seed: int,
    iterations: int | None = None,
) -> Any:
    _require_catboost()
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=int(config.iterations if iterations is None else iterations),
        depth=int(candidate["depth"]),
        learning_rate=float(candidate["learning_rate"]),
        l2_leaf_reg=float(candidate["l2_leaf_reg"]),
        boosting_type="Ordered",
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        random_strength=0.5,
        random_seed=int(seed),
        task_type="CPU",
        thread_count=int(config.thread_count),
        allow_writing_files=False,
        verbose=False,
    )


def _progress(enabled: bool, message: str) -> None:
    """Emit an immediately visible training update without affecting the model."""

    if enabled:
        print(message, flush=True)


def _catboost_verbose_value(show_progress: bool, progress_interval: int) -> bool | int:
    """Translate the public progress controls to CatBoost's ``verbose`` API."""

    if progress_interval < 1:
        raise ValueError("progress_interval must be a positive integer.")
    return int(progress_interval) if show_progress else False


def _select_catboost_configuration(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    config: StressClassifierConfig,
    seed_registry: SeedRegistry,
    *,
    show_progress: bool,
    progress_interval: int,
) -> tuple[Mapping[str, float], int, pd.DataFrame]:
    if len(np.unique(y_train)) != 2:
        raise ValueError("Stress training requires both feasible and infeasible examples.")
    unique_groups = len(np.unique(groups))
    maximum_splits = min(config.cv_folds, unique_groups)
    if maximum_splits < 2:
        raise ValueError("Stress CV requires at least two unique design groups.")
    folds = None
    split_seed = seed_registry.derive("stress/cv/split", upper_bound=2**32)
    for n_splits in range(maximum_splits, 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=split_seed,
        )
        candidate_folds = list(splitter.split(x_train, y_train, groups=groups))
        if all(
            len(np.unique(y_train[fit_indices])) == 2
            and len(np.unique(y_train[validation_indices])) == 2
            for fit_indices, validation_indices in candidate_folds
        ):
            folds = candidate_folds
            break
    if folds is None:
        raise ValueError(
            "No deterministic stratified group CV partition contains both stress "
            "classes in every fit/validation fold. Inspect group support."
        )
    rows: list[dict[str, Any]] = []
    candidate_summaries: list[tuple[float, float, int, Mapping[str, float], int]] = []
    total_fits = len(config.candidates) * len(folds)
    completed_fits = 0
    fit_verbose = _catboost_verbose_value(show_progress, progress_interval)

    _progress(
        show_progress,
        "Stress model selection: "
        f"{len(config.candidates)} candidates x {len(folds)} group folds = "
        f"{total_fits} CatBoost fits; up to {config.iterations} iterations/fit, "
        f"early stopping patience={config.early_stopping_rounds}.",
    )

    for candidate_index, candidate in enumerate(config.candidates):
        _progress(
            show_progress,
            "Stress CV candidate "
            f"{candidate_index + 1}/{len(config.candidates)}: "
            f"depth={int(candidate['depth'])}, "
            f"learning_rate={float(candidate['learning_rate']):g}, "
            f"l2_leaf_reg={float(candidate['l2_leaf_reg']):g}.",
        )
        fold_losses = []
        fold_aps = []
        best_iterations = []
        for fold_index, (fit_indices, validation_indices) in enumerate(folds):
            _progress(
                show_progress,
                f"  [fit {completed_fits + 1}/{total_fits}] "
                f"fold {fold_index + 1}/{len(folds)}; "
                f"train rows={len(fit_indices)}, "
                f"validation rows={len(validation_indices)}.",
            )
            model = _build_catboost(
                config,
                candidate,
                seed=seed_registry.derive(f"stress/cv/fold/{fold_index}"),
            )
            model.fit(
                x_train[fit_indices],
                y_train[fit_indices],
                eval_set=(x_train[validation_indices], y_train[validation_indices]),
                early_stopping_rounds=config.early_stopping_rounds,
                use_best_model=True,
                verbose=fit_verbose,
            )
            probability = np.asarray(
                model.predict_proba(x_train[validation_indices])[:, 1],
                dtype=np.float64,
            )
            loss = float(
                log_loss(y_train[validation_indices], probability, labels=[0, 1])
            )
            ap = float(
                average_precision_score(y_train[validation_indices], probability)
            )
            best_iteration = max(1, int(model.get_best_iteration()) + 1)
            fold_losses.append(loss)
            fold_aps.append(ap)
            best_iterations.append(best_iteration)
            completed_fits += 1
            _progress(
                show_progress,
                f"  [fit {completed_fits}/{total_fits} complete] "
                f"best_iteration={best_iteration}, log_loss={loss:.6g}, "
                f"average_precision={ap:.6g}.",
            )
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "fold": fold_index,
                    "depth": int(candidate["depth"]),
                    "learning_rate": float(candidate["learning_rate"]),
                    "l2_leaf_reg": float(candidate["l2_leaf_reg"]),
                    "log_loss": loss,
                    "average_precision": ap,
                    "best_iteration": best_iteration,
                }
            )
        selected_iterations = max(1, int(np.median(best_iterations)))
        _progress(
            show_progress,
            f"Candidate {candidate_index + 1} summary: "
            f"mean_log_loss={float(np.mean(fold_losses)):.6g}, "
            f"mean_average_precision={float(np.mean(fold_aps)):.6g}, "
            f"median_best_iteration={selected_iterations}.",
        )
        candidate_summaries.append(
            (
                float(np.mean(fold_losses)),
                -float(np.mean(fold_aps)),
                candidate_index,
                candidate,
                selected_iterations,
            )
        )
    _, _, _, best_candidate, final_iterations = min(candidate_summaries)
    _progress(
        show_progress,
        "Selected stress configuration: "
        f"depth={int(best_candidate['depth'])}, "
        f"learning_rate={float(best_candidate['learning_rate']):g}, "
        f"l2_leaf_reg={float(best_candidate['l2_leaf_reg']):g}, "
        f"final_iterations={final_iterations}.",
    )
    return best_candidate, final_iterations, pd.DataFrame(rows)


def _load_predictor(
    model_path: Path,
    calibrator_path: Path,
    config: StressClassifierConfig,
    artifact_id: str,
) -> CalibratedStressPredictor:
    _require_catboost()
    model = CatBoostClassifier()
    model.load_model(str(model_path), format="cbm")
    calibrator = PlattCalibrator.from_dict(load_json(calibrator_path))
    return CalibratedStressPredictor(
        model,
        calibrator,
        stress_limit_mpa=config.stress_limit_mpa,
        probability_threshold=config.probability_threshold,
        artifact_id=artifact_id,
    )


def _load_ensemble(
    model_paths: Sequence[Path],
    calibrators_path: Path,
    config: StressClassifierConfig,
    artifact_id: str,
) -> ConservativeStressEnsemble:
    _require_catboost()
    payload = load_json(calibrators_path)
    records = payload.get("members", [])
    if len(records) != len(model_paths):
        raise ValueError(
            "Persisted stress-ensemble member count does not match configuration."
        )
    members: list[CalibratedStressPredictor] = []
    for index, (model_path, record) in enumerate(zip(model_paths, records)):
        if int(record["member_index"]) != index:
            raise ValueError("Stress-ensemble calibrator member order is invalid.")
        model = CatBoostClassifier()
        model.load_model(str(model_path), format="cbm")
        members.append(
            CalibratedStressPredictor(
                model,
                PlattCalibrator.from_dict(record["calibrator"]),
                stress_limit_mpa=config.stress_limit_mpa,
                probability_threshold=config.probability_threshold,
                artifact_id=artifact_id,
            )
        )
    return ConservativeStressEnsemble(
        members,
        probability_threshold=config.probability_threshold,
        artifact_id=artifact_id,
        aggregation="minimum",
    )


def _validation_partition(
    labels: np.ndarray,
    groups: np.ndarray,
    seed_registry: SeedRegistry,
) -> tuple[np.ndarray, np.ndarray]:
    """Split validation groups once into early-stop and calibration halves."""

    splitter = StratifiedGroupKFold(
        n_splits=2,
        shuffle=True,
        random_state=seed_registry.derive(
            "stress/ensemble/validation_partition", upper_bound=2**32
        ),
    )
    dummy = np.zeros((len(labels), 1), dtype=np.float64)
    for early_indices, calibration_indices in splitter.split(
        dummy, labels, groups=groups
    ):
        if (
            len(np.unique(labels[early_indices])) == 2
            and len(np.unique(labels[calibration_indices])) == 2
        ):
            return (
                np.asarray(early_indices, dtype=np.int64),
                np.asarray(calibration_indices, dtype=np.int64),
            )
    raise ValueError(
        "The validation split cannot be partitioned into group-disjoint "
        "early-stopping and calibration subsets containing both classes."
    )


def train_or_load_stress_classifier(
    bundle: DataBundle,
    config: StressClassifierConfig,
    seed_registry: SeedRegistry,
    *,
    artifact_dir: str | Path | None = None,
    show_progress: bool = True,
    progress_interval: int = 250,
) -> CalibratedStressPredictor | ConservativeStressEnsemble:
    """Load or train one classifier or a conservative calibrated ensemble.

    ``show_progress`` and ``progress_interval`` control console/notebook logging
    only. They do not enter the artifact identity and do not alter any random
    stream, fitted parameter, or model-selection rule.
    """

    _require_catboost()
    _catboost_verbose_value(show_progress, progress_interval)
    output_dir = Path(artifact_dir) if artifact_dir is not None else config.output_dir
    declared_ensemble_candidates = _declared_ensemble_candidates(config)
    ensemble_mode = bool(declared_ensemble_candidates)
    ensemble_members = (
        len(declared_ensemble_candidates) if ensemble_mode else 1
    )
    model_path = output_dir / "stress_classifier.cbm"
    calibrator_path = output_dir / "calibrator.json"
    member_model_paths = tuple(
        output_dir / f"stress_classifier_member_{index:02d}.cbm"
        for index in range(ensemble_members)
    )
    calibrators_path = output_dir / "calibrators.json"
    manifest_path = output_dir / "manifest.json"
    metrics_path = output_dir / "metrics.json"
    cv_path = output_dir / "cv_results.csv"
    member_results_path = output_dir / "member_results.csv"
    test_predictions_path = output_dir / "test_predictions.csv"
    expected = _expected_manifest(bundle, config, seed_registry)

    required_artifacts = (
        (*member_model_paths, calibrators_path, manifest_path)
        if ensemble_mode
        else (model_path, calibrator_path, manifest_path)
    )
    if not config.force_retrain and require_complete_artifact(required_artifacts):
        actual = load_artifact_manifest(manifest_path)
        assert_artifact_compatible(actual, expected)
        assert_artifact_payloads(actual, output_dir)
        print(f"Loaded compatible stress artifact {actual.artifact_id}.")
        if ensemble_mode:
            return _load_ensemble(
                member_model_paths,
                calibrators_path,
                config,
                artifact_id=actual.artifact_id,
            )
        return _load_predictor(
            model_path, calibrator_path, config, artifact_id=actual.artifact_id
        )

    set_global_determinism(
        seed_registry.derive("stress/training", upper_bound=2**32),
        strict=True,
        torch_num_threads=1,
    )
    rows = bundle.rows
    split_frames = {split: rows.loc[rows["split"] == split] for split in SPLIT_NAMES}
    if any(frame.empty for frame in split_frames.values()):
        raise ValueError("Stress training requires non-empty train/validation/test splits.")

    def arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = frame.loc[:, ALL_INPUT_COLUMNS].to_numpy(dtype=np.float64)
        labels = (
            frame[STRESS_TARGET_COLUMN].to_numpy(dtype=np.float64)
            <= config.stress_limit_mpa
        ).astype(np.int64)
        groups = frame["design_id"].to_numpy()
        return features, labels, groups

    x_train, y_train, train_groups = arrays(split_frames["train"])
    x_validation, y_validation, validation_groups = arrays(
        split_frames["validation"]
    )
    x_test, y_test, _ = arrays(split_frames["test"])
    if len(np.unique(y_validation)) != 2 or len(np.unique(y_test)) != 2:
        raise ValueError(
            "Validation and test splits must each contain both stress classes. "
            "Inspect the persisted group split instead of silently changing it."
        )

    _progress(
        show_progress,
        "Stress training data: "
        f"train={len(x_train)} rows ({float(np.mean(y_train)):.1%} feasible), "
        f"validation={len(x_validation)} rows "
        f"({float(np.mean(y_validation)):.1%} feasible), "
        f"test={len(x_test)} rows ({float(np.mean(y_test)):.1%} feasible).",
    )

    cv_results: pd.DataFrame | None = None
    member_models: list[Any] = []
    member_predictors: list[CalibratedStressPredictor] = []
    member_records: list[dict[str, Any]] = []

    if not ensemble_mode:
        candidate, iterations, cv_results = _select_catboost_configuration(
            x_train,
            y_train,
            train_groups,
            config,
            seed_registry,
            show_progress=show_progress,
            progress_interval=progress_interval,
        )
        member_candidates = [candidate]
        member_iterations = [iterations]
        early_indices = None
        calibration_indices = np.arange(len(x_validation), dtype=np.int64)
    else:
        member_candidates = list(declared_ensemble_candidates)
        member_iterations = [config.iterations] * ensemble_members
        early_indices, calibration_indices = _validation_partition(
            y_validation,
            validation_groups,
            seed_registry,
        )
        _progress(
            show_progress,
            "Stress ensemble validation partition: "
            f"early-stop rows={len(early_indices)}, "
            f"calibration rows={len(calibration_indices)}.",
        )

    fit_verbose = _catboost_verbose_value(show_progress, progress_interval)
    for member_index, (candidate, requested_iterations) in enumerate(
        zip(member_candidates, member_iterations)
    ):
        member_seed = seed_registry.derive(
            "stress", "final_model", "member", member_index
        )
        _progress(
            show_progress,
            f"Stress member {member_index + 1}/{len(member_candidates)}: "
            f"depth={int(candidate['depth'])}, "
            f"learning_rate={float(candidate['learning_rate']):g}, "
            f"l2_leaf_reg={float(candidate['l2_leaf_reg']):g}, "
            f"seed={member_seed}.",
        )
        model = _build_catboost(
            config,
            candidate,
            seed=member_seed,
            iterations=requested_iterations,
        )
        fit_kwargs: dict[str, Any] = {"verbose": fit_verbose}
        if early_indices is not None:
            fit_kwargs.update(
                {
                    "eval_set": (
                        x_validation[early_indices],
                        y_validation[early_indices],
                    ),
                    "early_stopping_rounds": config.early_stopping_rounds,
                    "use_best_model": True,
                }
            )
        model.fit(x_train, y_train, **fit_kwargs)
        if early_indices is None:
            fitted_iterations = int(requested_iterations)
        else:
            fitted_iterations = max(1, int(model.get_best_iteration()) + 1)
        calibration_raw = np.asarray(
            model.predict(
                x_validation[calibration_indices],
                prediction_type="RawFormulaVal",
            ),
            dtype=np.float64,
        ).reshape(-1)
        calibrator = PlattCalibrator.fit(
            calibration_raw,
            y_validation[calibration_indices],
            regularization=config.calibration_regularization,
        )
        member = CalibratedStressPredictor(
            model,
            calibrator,
            stress_limit_mpa=config.stress_limit_mpa,
            probability_threshold=config.probability_threshold,
            artifact_id=expected.artifact_id,
        )
        calibration_probability = member.predict_proba(
            x_validation[calibration_indices]
        )
        member_models.append(model)
        member_predictors.append(member)
        member_records.append(
            {
                "member_index": member_index,
                "seed": int(member_seed),
                "candidate": dict(candidate),
                "requested_iterations": int(requested_iterations),
                "fitted_iterations": int(fitted_iterations),
                "calibrator": calibrator.to_dict(),
                "calibration_rows": int(len(calibration_indices)),
                "calibration_log_loss": float(
                    log_loss(
                        y_validation[calibration_indices],
                        calibration_probability,
                        labels=[0, 1],
                    )
                ),
                "calibration_average_precision": float(
                    average_precision_score(
                        y_validation[calibration_indices],
                        calibration_probability,
                    )
                ),
            }
        )
        _progress(
            show_progress,
            f"Stress member {member_index + 1} complete: "
            f"fitted_iterations={fitted_iterations}, "
            f"calibration_log_loss="
            f"{member_records[-1]['calibration_log_loss']:.6g}.",
        )

    if ensemble_mode:
        predictor: CalibratedStressPredictor | ConservativeStressEnsemble = (
            ConservativeStressEnsemble(
                member_predictors,
                probability_threshold=config.probability_threshold,
                artifact_id=expected.artifact_id,
                aggregation="minimum",
            )
        )
    else:
        predictor = member_predictors[0]

    metrics: dict[str, Any] = {
        "ensemble": {
            "members": int(ensemble_members),
            "aggregation": (
                "minimum" if ensemble_mode else "single"
            ),
            "cross_validate_candidates": not ensemble_mode,
            "score_interpretation": (
                "minimum individually calibrated member probability; "
                "a conservative feasibility score, not an aggregate "
                "calibrated probability"
                if ensemble_mode
                else "individually calibrated probability"
            ),
        },
        "member_records": member_records,
        "label_definition": (
            f"1 = {STRESS_TARGET_COLUMN} <= {config.stress_limit_mpa:g} MPa"
        ),
        "splits": {},
    }
    for split, frame in split_frames.items():
        features, labels, _ = arrays(frame)
        probability = predictor.predict_proba(features)
        split_metrics: dict[str, Any] = {
            "probability": probability_metrics(labels, probability),
            "thresholds": {},
        }
        if ensemble_mode:
            member_probability = predictor.member_probabilities(features)
            split_metrics["members"] = [
                probability_metrics(labels, member_probability[:, index])
                for index in range(member_probability.shape[1])
            ]
        for threshold in (0.5, 0.8, 0.9):
            split_metrics["thresholds"][f"{threshold:.1f}"] = threshold_metrics(
                labels, probability, threshold
            )
        metrics["splits"][split] = split_metrics

    test_probability = predictor.predict_proba(x_test)
    test_predictions = split_frames["test"].loc[
        :, ["design_id", "stress_input_id", STRESS_TARGET_COLUMN]
    ].copy()
    test_predictions["true_feasible"] = y_test
    test_predictions["probability_feasible"] = test_probability
    if ensemble_mode:
        test_raw_members = predictor.member_raw_scores(x_test)
        test_probability_members = predictor.member_probabilities(x_test)
        test_predictions["limiting_member"] = np.argmin(
            test_probability_members, axis=1
        )
        for member_index in range(ensemble_members):
            test_predictions[f"raw_score_member_{member_index}"] = (
                test_raw_members[:, member_index]
            )
            test_predictions[f"probability_member_{member_index}"] = (
                test_probability_members[:, member_index]
            )
    else:
        test_predictions["raw_score"] = predictor.raw_score(x_test)
    for threshold in (0.5, 0.8, 0.9):
        test_predictions[f"predicted_feasible_at_{threshold:.1f}"] = (
            test_probability >= threshold
        ).astype(np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)
    _progress(show_progress, f"Saving stress artifact to {output_dir.resolve()}.")
    payload_names: list[str] = []
    if ensemble_mode:
        for model, path in zip(member_models, member_model_paths):
            with atomic_output_path(path) as temporary:
                model.save_model(str(temporary), format="cbm")
            payload_names.append(path.name)
        atomic_write_json(
            calibrators_path,
            {
                "aggregation": "minimum",
                "members": member_records,
            },
        )
        payload_names.append(calibrators_path.name)
        with atomic_output_path(member_results_path) as temporary:
            pd.json_normalize(member_records, sep=".").to_csv(
                temporary, index=False
            )
        payload_names.append(member_results_path.name)
    else:
        with atomic_output_path(model_path) as temporary:
            member_models[0].save_model(str(temporary), format="cbm")
        atomic_write_json(
            calibrator_path,
            member_predictors[0].calibrator.to_dict(),
        )
        payload_names.extend((model_path.name, calibrator_path.name))
    if cv_results is not None:
        with atomic_output_path(cv_path) as temporary:
            cv_results.to_csv(temporary, index=False)
        payload_names.append(cv_path.name)
    with atomic_output_path(test_predictions_path) as temporary:
        test_predictions.to_csv(temporary, index=False)
    # JSON round-trip converts NumPy scalars and represents any unavailable metric
    # consistently with pandas' null convention.
    atomic_write_json(metrics_path, json.loads(json.dumps(metrics, allow_nan=False)))
    payload_names.extend((metrics_path.name, test_predictions_path.name))
    persisted = expected.with_payload_hashes(
        output_dir,
        tuple(payload_names),
    )
    save_artifact_manifest(manifest_path, persisted)
    print(
        f"Trained stress {'ensemble ' if ensemble_mode else ''}artifact "
        f"{expected.artifact_id}."
    )
    return predictor
