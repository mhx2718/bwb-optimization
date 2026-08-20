"""Leakage-safe BWB dataset preparation and persistent design-group splits."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .artifacts import atomic_write_json, canonical_json_bytes, load_json, sha256_json
from .config import DataConfig
from .reproducibility import SeedRegistry
from .schema import (
    ALL_COLUMNS,
    ALL_INPUT_COLUMNS,
    DESIGN_COLUMNS,
    FORWARD_TARGET_COLUMNS,
    STRESS_LIMIT_MPA,
    STRESS_TARGET_COLUMN,
    TOPOLOGY_COLUMNS,
    assert_exact_schema,
    canonicalize_column_names,
)


SPLIT_NAMES = ("train", "validation", "test")
SPLIT_MANIFEST_SCHEMA_VERSION = 1


class DataValidationError(ValueError):
    """Raised when the dataset violates the documented modeling assumptions."""


class IncompatibleSplitError(RuntimeError):
    """Raised rather than silently changing an existing train/test split."""


@dataclass(frozen=True)
class SplitManifest:
    master_seed: int
    dataset_fingerprint: str
    fractions: Mapping[str, float]
    design_assignments: Mapping[str, str]
    group_counts: Mapping[str, int]
    row_counts: Mapping[str, int]
    feasible_row_fractions: Mapping[str, float]
    split_fingerprint: str = ""
    schema_version: int = SPLIT_MANIFEST_SCHEMA_VERSION

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "master_seed": int(self.master_seed),
            "dataset_fingerprint": self.dataset_fingerprint,
            "fractions": dict(self.fractions),
            "design_assignments": dict(sorted(self.design_assignments.items())),
            "group_counts": dict(self.group_counts),
            "row_counts": dict(self.row_counts),
            "feasible_row_fractions": dict(self.feasible_row_fractions),
        }

    def with_fingerprint(self) -> "SplitManifest":
        payload = self.payload()
        return SplitManifest(
            **{key: value for key, value in payload.items() if key != "schema_version"},
            schema_version=int(payload["schema_version"]),
            split_fingerprint=sha256_json(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        manifest = self if self.split_fingerprint else self.with_fingerprint()
        return asdict(manifest)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "SplitManifest":
        return cls(**dict(values))


@dataclass(frozen=True)
class DataBundle:
    """Both row-level and unique-structural-design views of one fixed split."""

    rows: pd.DataFrame
    structural_designs: pd.DataFrame
    split_manifest: SplitManifest
    dataset_fingerprint: str
    source_file_sha256: str
    source_path: Path
    exact_duplicate_rows_removed: int
    conflicting_stress_input_groups: tuple[str, ...]

    def rows_for_split(self, split: str) -> pd.DataFrame:
        _validate_split_name(split)
        return self.rows.loc[self.rows["split"] == split].copy()

    def structural_for_split(self, split: str) -> pd.DataFrame:
        _validate_split_name(split)
        return self.structural_designs.loc[
            self.structural_designs["split"] == split
        ].copy()


def _validate_split_name(split: str) -> None:
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLIT_NAMES}.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_numeric_fingerprint(frame: pd.DataFrame) -> str:
    array = frame.loc[:, ALL_COLUMNS].to_numpy(dtype=np.float64)
    keys = tuple(array[:, column] for column in range(array.shape[1] - 1, -1, -1))
    order = np.lexsort(keys)
    canonical = np.ascontiguousarray(array[order], dtype="<f8")
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes({"columns": ALL_COLUMNS, "dtype": "<f8"}))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _row_hashes(frame: pd.DataFrame, columns: tuple[str, ...], prefix: str) -> list[str]:
    array = np.ascontiguousarray(
        frame.loc[:, columns].to_numpy(dtype=np.float64), dtype="<f8"
    )
    header = canonical_json_bytes({"prefix": prefix, "columns": columns})
    return [
        hashlib.sha256(header + row.tobytes(order="C")).hexdigest()
        for row in array
    ]


def _largest_remainder_counts(total: int, fractions: np.ndarray) -> np.ndarray:
    raw = total * fractions
    counts = np.floor(raw).astype(np.int64)
    remainder = int(total - counts.sum())
    priorities = sorted(
        range(len(fractions)), key=lambda index: (-(raw[index] - counts[index]), index)
    )
    for index in priorities[:remainder]:
        counts[index] += 1
    return counts


def _group_strata(rows: pd.DataFrame, stress_limit: float) -> pd.DataFrame:
    labels = (rows[STRESS_TARGET_COLUMN].to_numpy(dtype=np.float64) <= stress_limit).astype(
        np.int64
    )
    temporary = pd.DataFrame(
        {"design_id": rows["design_id"].to_numpy(), "feasible": labels}
    )
    grouped = temporary.groupby("design_id", sort=True)["feasible"].agg(["sum", "count"])
    grouped["stratum"] = np.where(
        grouped["sum"] == 0,
        "all_infeasible",
        np.where(grouped["sum"] == grouped["count"], "all_feasible", "mixed"),
    )
    return grouped.reset_index()


def _rebalance_assignments(
    assignment: dict[str, str],
    group_strata: Mapping[str, str],
    target_counts: Mapping[str, int],
    stratum_totals: Mapping[str, int],
    fractions: Mapping[str, float],
) -> None:
    """Reach exact global counts while minimally perturbing stratum balance."""

    def split_counts() -> dict[str, int]:
        return {
            split: sum(value == split for value in assignment.values())
            for split in SPLIT_NAMES
        }

    def stratum_count(split: str, stratum: str) -> int:
        return sum(
            assigned == split and group_strata[design_id] == stratum
            for design_id, assigned in assignment.items()
        )

    while True:
        current = split_counts()
        donors = [
            split for split in SPLIT_NAMES if current[split] > target_counts[split]
        ]
        receivers = [
            split for split in SPLIT_NAMES if current[split] < target_counts[split]
        ]
        if not donors and not receivers:
            return
        if not donors or not receivers:
            raise RuntimeError("Unable to rebalance deterministic group split.")
        donor = donors[0]
        receiver = receivers[0]
        candidates: list[tuple[float, str]] = []
        for design_id, assigned in assignment.items():
            if assigned != donor:
                continue
            stratum = group_strata[design_id]
            ideal_donor = stratum_totals[stratum] * fractions[donor]
            ideal_receiver = stratum_totals[stratum] * fractions[receiver]
            before = (
                (stratum_count(donor, stratum) - ideal_donor) ** 2
                + (stratum_count(receiver, stratum) - ideal_receiver) ** 2
            )
            after = (
                (stratum_count(donor, stratum) - 1 - ideal_donor) ** 2
                + (stratum_count(receiver, stratum) + 1 - ideal_receiver) ** 2
            )
            candidates.append((after - before, design_id))
        _, selected = min(candidates)
        assignment[selected] = receiver


def _create_design_assignments(
    group_table: pd.DataFrame,
    config: DataConfig,
    seed_registry: SeedRegistry,
) -> dict[str, str]:
    fractions = {
        "train": float(config.train_fraction),
        "validation": float(config.validation_fraction),
        "test": float(config.test_fraction),
    }
    fraction_array = np.asarray([fractions[name] for name in SPLIT_NAMES])
    assignment: dict[str, str] = {}
    group_strata = dict(zip(group_table["design_id"], group_table["stratum"]))

    for stratum, subset in group_table.groupby("stratum", sort=True):
        design_ids = np.asarray(sorted(subset["design_id"].tolist()), dtype=object)
        rng = seed_registry.numpy_rng(f"data/split/stratum/{stratum}")
        design_ids = design_ids[rng.permutation(len(design_ids))]
        counts = _largest_remainder_counts(len(design_ids), fraction_array)
        start = 0
        for split, count in zip(SPLIT_NAMES, counts):
            for design_id in design_ids[start : start + int(count)]:
                assignment[str(design_id)] = split
            start += int(count)

    global_counts = _largest_remainder_counts(len(group_table), fraction_array)
    target_counts = dict(zip(SPLIT_NAMES, map(int, global_counts)))
    stratum_totals = group_table["stratum"].value_counts().to_dict()
    _rebalance_assignments(
        assignment, group_strata, target_counts, stratum_totals, fractions
    )
    if set(assignment) != set(group_table["design_id"]):
        raise RuntimeError("Not every design group received exactly one split assignment.")
    return assignment


def _build_split_manifest(
    rows: pd.DataFrame,
    dataset_fingerprint: str,
    config: DataConfig,
    seed_registry: SeedRegistry,
) -> SplitManifest:
    group_table = _group_strata(rows, stress_limit=STRESS_LIMIT_MPA)
    assignments = _create_design_assignments(group_table, config, seed_registry)
    split_series = rows["design_id"].map(assignments)
    if split_series.isna().any():
        raise RuntimeError("Split assignment failed for at least one row.")
    feasible = rows[STRESS_TARGET_COLUMN].to_numpy(dtype=np.float64) <= 335.0
    fractions = {
        "train": float(config.train_fraction),
        "validation": float(config.validation_fraction),
        "test": float(config.test_fraction),
    }
    group_counts = {
        split: int(sum(value == split for value in assignments.values()))
        for split in SPLIT_NAMES
    }
    row_counts = {split: int((split_series == split).sum()) for split in SPLIT_NAMES}
    feasible_fractions: dict[str, float] = {}
    for split in SPLIT_NAMES:
        mask = split_series.to_numpy() == split
        feasible_fractions[split] = float(feasible[mask].mean())
    return SplitManifest(
        master_seed=seed_registry.master_seed,
        dataset_fingerprint=dataset_fingerprint,
        fractions=fractions,
        design_assignments=assignments,
        group_counts=group_counts,
        row_counts=row_counts,
        feasible_row_fractions=feasible_fractions,
    ).with_fingerprint()


def _load_or_create_split_manifest(
    path: Path | None,
    rows: pd.DataFrame,
    dataset_fingerprint: str,
    config: DataConfig,
    seed_registry: SeedRegistry,
) -> SplitManifest:
    expected_ids = set(rows["design_id"])
    expected_fractions = {
        "train": float(config.train_fraction),
        "validation": float(config.validation_fraction),
        "test": float(config.test_fraction),
    }
    if path is not None and path.exists():
        manifest = SplitManifest.from_dict(load_json(path))
        problems = []
        if manifest.schema_version != SPLIT_MANIFEST_SCHEMA_VERSION:
            problems.append("schema_version")
        if manifest.master_seed != seed_registry.master_seed:
            problems.append("master_seed")
        if manifest.dataset_fingerprint != dataset_fingerprint:
            problems.append("dataset_fingerprint")
        if canonical_json_bytes(manifest.fractions) != canonical_json_bytes(
            expected_fractions
        ):
            problems.append("fractions")
        if set(manifest.design_assignments) != expected_ids:
            problems.append("design_ids")
        computed = SplitManifest(
            **{
                key: value
                for key, value in asdict(manifest).items()
                if key not in {"split_fingerprint"}
            },
            split_fingerprint="",
        ).with_fingerprint()
        if manifest.split_fingerprint != computed.split_fingerprint:
            problems.append("split_fingerprint")
        if problems:
            raise IncompatibleSplitError(
                "Existing split manifest is incompatible: "
                f"{sorted(set(problems))}. Refusing to change a published split."
            )
        return manifest

    manifest = _build_split_manifest(rows, dataset_fingerprint, config, seed_registry)
    if path is not None:
        atomic_write_json(path, manifest.to_dict())
    return manifest


def _validate_forward_target_invariance(
    rows: pd.DataFrame,
    rtol: float,
    atol: float,
) -> None:
    grouped = rows.groupby("design_id", sort=True)
    violations: list[tuple[str, str, float, float]] = []
    for design_id, subset in grouped:
        values = subset.loc[:, FORWARD_TARGET_COLUMNS].to_numpy(dtype=np.float64)
        minimum = values.min(axis=0)
        maximum = values.max(axis=0)
        reference = np.maximum(np.abs(np.median(values, axis=0)), atol)
        relative_span = (maximum - minimum) / reference
        allowed = atol + rtol * np.maximum(np.abs(minimum), np.abs(maximum))
        invalid = (maximum - minimum) > allowed
        for column_index in np.flatnonzero(invalid):
            violations.append(
                (
                    str(design_id),
                    FORWARD_TARGET_COLUMNS[int(column_index)],
                    float(maximum[column_index] - minimum[column_index]),
                    float(relative_span[column_index]),
                )
            )
    if violations:
        preview = violations[:10]
        raise DataValidationError(
            "The structural targets are not invariant across rows sharing the "
            "same 21-D design, so a flight-free forward model is not valid. "
            f"First violations (design_id, target, absolute span, relative span): {preview}."
        )


def _build_structural_designs(rows: pd.DataFrame) -> pd.DataFrame:
    aggregation = {
        **{column: "first" for column in DESIGN_COLUMNS},
        **{column: "median" for column in FORWARD_TARGET_COLUMNS},
        "split": "first",
    }
    structural = (
        rows.groupby("design_id", sort=True, as_index=False)
        .agg(aggregation)
        .loc[:, ("design_id",) + DESIGN_COLUMNS + FORWARD_TARGET_COLUMNS + ("split",)]
    )
    return structural.reset_index(drop=True)


def prepare_data(
    data_path: str | Path,
    config: DataConfig,
    seed_registry: SeedRegistry,
    split_manifest_path: str | Path | None = None,
) -> DataBundle:
    """Validate, canonicalize, group-split, and expose both model data views."""

    source_path = Path(data_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {source_path}.")
    source_sha256 = _sha256_file(source_path)
    frame = pd.read_csv(source_path)
    assert_exact_schema(frame.columns)
    frame.columns = canonicalize_column_names(frame.columns)
    frame = frame.loc[:, ALL_COLUMNS].apply(pd.to_numeric, errors="raise")
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        locations = np.argwhere(~np.isfinite(values))[:10].tolist()
        raise DataValidationError(
            f"Dataset contains missing or non-finite values at {locations}."
        )
    if (frame.loc[:, FORWARD_TARGET_COLUMNS] <= 0.0).any().any():
        raise DataValidationError("Weight and volume targets must be strictly positive.")
    if (frame[STRESS_TARGET_COLUMN] < 0.0).any():
        raise DataValidationError("Stress values cannot be negative.")
    for column in TOPOLOGY_COLUMNS:
        topology_values = frame[column].to_numpy(dtype=np.float64)
        if not np.allclose(topology_values, np.rint(topology_values), atol=1.0e-10):
            raise DataValidationError(
                f"Topology column {column!r} contains non-integer values."
            )
    fuselage_ribs = np.rint(frame["# of Fuselage Ribs"]).astype(np.int64)
    if np.any(fuselage_ribs % 2 != 1):
        raise DataValidationError("# of Fuselage Ribs must contain odd integers.")

    before = len(frame)
    if config.drop_exact_duplicates:
        frame = frame.drop_duplicates(subset=ALL_COLUMNS, keep="first")
    duplicates_removed = before - len(frame)
    frame = frame.reset_index(drop=True)
    dataset_fingerprint = _canonical_numeric_fingerprint(frame)

    frame["design_id"] = _row_hashes(frame, DESIGN_COLUMNS, "design-v1")
    frame["stress_input_id"] = _row_hashes(frame, ALL_INPUT_COLUMNS, "stress-v1")
    frame = frame.sort_values(
        ["design_id", "stress_input_id", *FLIGHT_SORT_COLUMNS], kind="mergesort"
    ).reset_index(drop=True)

    _validate_forward_target_invariance(
        frame,
        rtol=config.forward_duplicate_target_rtol,
        atol=config.forward_duplicate_target_atol,
    )

    labels = (frame[STRESS_TARGET_COLUMN] <= STRESS_LIMIT_MPA).astype(np.int64)
    conflict_counts = (
        pd.DataFrame({"stress_input_id": frame["stress_input_id"], "label": labels})
        .groupby("stress_input_id", sort=True)["label"]
        .nunique()
    )
    conflicts = tuple(conflict_counts.index[conflict_counts > 1].tolist())

    if split_manifest_path is None:
        split_path = Path(config.split_manifest)
    else:
        split_path = Path(split_manifest_path)
    manifest = _load_or_create_split_manifest(
        split_path, frame, dataset_fingerprint, config, seed_registry
    )
    frame["split"] = frame["design_id"].map(manifest.design_assignments)
    if frame["split"].isna().any():
        raise RuntimeError("At least one dataset row is absent from split manifest.")

    structural = _build_structural_designs(frame)
    return DataBundle(
        rows=frame,
        structural_designs=structural,
        split_manifest=manifest,
        dataset_fingerprint=dataset_fingerprint,
        source_file_sha256=source_sha256,
        source_path=source_path.resolve(),
        exact_duplicate_rows_removed=int(duplicates_removed),
        conflicting_stress_input_groups=conflicts,
    )


# Explicitly ordered only for deterministic row presentation; model behavior does
# not depend on row order. These names are kept separate from the grouping key.
FLIGHT_SORT_COLUMNS = ("Altitude", "KCAS", "AOA")
