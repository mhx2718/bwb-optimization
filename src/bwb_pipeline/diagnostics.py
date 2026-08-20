"""Deterministic design-support and empirical Pareto diagnostics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .schema import DESIGN_COLUMNS, OFFICIAL_DESIGN_BOUNDS, TOPOLOGY_COLUMNS

DEFAULT_OBJECTIVE_COLUMNS: Tuple[str, ...] = (
    "objective_mass_over_50",
    "objective_ld_shortfall",
    "objective_payload_shortfall",
    "objective_fuel_shortfall",
)


def _require_columns(table: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _validated_design_values(
    table: pd.DataFrame,
    design_columns: Sequence[str],
) -> np.ndarray:
    if not isinstance(table, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    _require_columns(table, design_columns)
    try:
        values = table.loc[:, design_columns].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Design columns must be numeric.") from exc
    if values.ndim != 2 or values.shape[1] != len(design_columns):
        raise ValueError("Design matrix has an unexpected shape.")
    if not np.isfinite(values).all():
        raise ValueError("Design columns contain missing or non-finite values.")
    values = values.copy()
    values[values == 0.0] = 0.0  # Canonicalize signed zero.
    return values


def _design_ids(values: np.ndarray, design_columns: Sequence[str]) -> np.ndarray:
    prefix = ("\x1f".join(design_columns) + "\x1e").encode("utf-8")
    identifiers = []
    for row in np.asarray(values, dtype="<f8"):
        digest = hashlib.sha256(prefix + row.tobytes()).hexdigest()
        identifiers.append(digest)
    return np.asarray(identifiers, dtype=object)


def _bounds_arrays(
    design_columns: Sequence[str],
    bounds: Mapping[str, Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    missing = [column for column in design_columns if column not in bounds]
    if missing:
        raise KeyError(f"Missing design bounds: {missing}")
    lower = np.asarray([bounds[column][0] for column in design_columns], dtype=float)
    upper = np.asarray([bounds[column][1] for column in design_columns], dtype=float)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Design bounds must be finite.")
    if np.any(upper <= lower):
        raise ValueError("Every design bound must have positive width.")
    return lower, upper


def _scale_design_values(
    values: np.ndarray,
    design_columns: Sequence[str],
    bounds: Mapping[str, Sequence[float]],
) -> np.ndarray:
    lower, upper = _bounds_arrays(design_columns, bounds)
    return (values - lower[None, :]) / (upper - lower)[None, :]


def deduplicate_designs(
    table: pd.DataFrame,
    *,
    design_columns: Sequence[str] = DESIGN_COLUMNS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate exact 21D designs and retain a row-to-design mapping.

    Repeated flight-condition rows often share exactly the same geometry and
    structure.  They must be treated as one reference design before computing
    leave-one-design-out distances; otherwise their zero distances create a
    misleading support distribution.

    Returns
    -------
    unique_designs:
        One row per exact design, sorted by a content-derived ``design_id``.
    row_mapping:
        One row per input position, preserving input order and index labels.
    """

    columns = tuple(design_columns)
    values = _validated_design_values(table, columns)
    identifiers = _design_ids(values, columns)

    working = pd.DataFrame(values, columns=columns)
    working.insert(0, "dataset_row_number", np.arange(len(working), dtype=int))
    working.insert(1, "source_index", [str(value) for value in table.index])
    working.insert(2, "design_id", identifiers)

    # A SHA-256 collision is practically impossible, but detecting it prevents
    # the identifier from ever becoming a silent deduplication criterion.
    for _, group in working.groupby("design_id", sort=False):
        group_values = group.loc[:, columns].to_numpy(dtype=np.float64)
        if not np.all(group_values == group_values[[0]]):
            raise RuntimeError("Design identifier collision detected.")

    counts = working.groupby("design_id", sort=False).size().rename(
        "source_row_count"
    )
    unique = (
        working.sort_values(
            ["design_id", "dataset_row_number"], kind="stable"
        )
        .drop_duplicates("design_id", keep="first")
        .drop(columns=["dataset_row_number", "source_index"])
        .merge(counts, left_on="design_id", right_index=True, how="left")
        .sort_values("design_id", kind="stable")
        .reset_index(drop=True)
    )
    unique["source_row_count"] = unique["source_row_count"].astype(int)

    mapping = working.loc[
        :,
        ["dataset_row_number", "source_index", "design_id"] + list(columns),
    ].copy()
    mapping["source_row_count"] = mapping["design_id"].map(counts).astype(int)
    return unique, mapping


def _top_k_neighbors(
    query: np.ndarray,
    reference: np.ndarray,
    reference_ids: np.ndarray,
    *,
    k: int,
    excluded_reference_positions: Optional[np.ndarray] = None,
    chunk_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact Euclidean neighbors with deterministic content-ID tie breaks."""

    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    reference_ids = np.asarray(reference_ids, dtype=object)
    k = int(k)
    chunk_size = int(chunk_size)
    if query.ndim != 2 or reference.ndim != 2:
        raise ValueError("Query and reference arrays must be two-dimensional.")
    if query.shape[1] != reference.shape[1]:
        raise ValueError("Query and reference dimensions do not match.")
    if k < 1 or k > len(reference):
        raise ValueError("k must be between 1 and the reference size.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if not np.all(reference_ids[:-1] <= reference_ids[1:]):
        raise ValueError("Reference designs must be sorted by design_id.")
    if excluded_reference_positions is not None:
        excluded_reference_positions = np.asarray(
            excluded_reference_positions, dtype=np.int64
        )
        if excluded_reference_positions.shape != (len(query),):
            raise ValueError("Excluded positions have the wrong shape.")

    all_distances = np.empty((len(query), k), dtype=np.float64)
    all_indices = np.empty((len(query), k), dtype=np.int64)
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        distances = cdist(
            query[start:stop], reference, metric="euclidean"
        )
        if excluded_reference_positions is not None:
            local_rows = np.arange(stop - start, dtype=np.int64)
            excluded = excluded_reference_positions[start:stop]
            if np.any(excluded < 0) or np.any(excluded >= len(reference)):
                raise ValueError("Excluded reference position is out of range.")
            distances[local_rows, excluded] = np.inf

        # reference is content-ID sorted.  Stable distance sorting therefore
        # resolves exact ties by design_id, independently of source row order.
        order = np.argsort(distances, axis=1, kind="stable")[:, :k]
        selected = np.take_along_axis(distances, order, axis=1)
        if not np.isfinite(selected).all():
            raise RuntimeError("Could not find the requested finite neighbors.")
        all_distances[start:stop] = selected
        all_indices[start:stop] = order
    return all_distances, all_indices


@dataclass(frozen=True)
class KNNDiagnostics:
    """Unique-design and all-row LOO k-NN outputs."""

    unique_designs: pd.DataFrame
    dataset_rows: pd.DataFrame


def compute_loo_knn_diagnostics(
    dataset: pd.DataFrame,
    *,
    k: int = 5,
    design_columns: Sequence[str] = DESIGN_COLUMNS,
    bounds: Mapping[str, Sequence[float]] = OFFICIAL_DESIGN_BOUNDS,
    chunk_size: int = 512,
) -> KNNDiagnostics:
    """Compute deterministic leave-one-unique-design-out k-NN distances."""

    k = int(k)
    unique, row_mapping = deduplicate_designs(
        dataset, design_columns=design_columns
    )
    if len(unique) <= k:
        raise ValueError(
            f"LOO k={k} requires at least {k + 1} unique designs; "
            f"found {len(unique)}."
        )

    values = unique.loc[:, design_columns].to_numpy(dtype=np.float64)
    scaled = _scale_design_values(values, design_columns, bounds)
    identifiers = unique["design_id"].to_numpy(dtype=object)
    distances, indices = _top_k_neighbors(
        scaled,
        scaled,
        identifiers,
        k=k,
        excluded_reference_positions=np.arange(len(unique), dtype=np.int64),
        chunk_size=chunk_size,
    )

    result = unique.copy()
    result["loo_knn_k"] = k
    result["loo_nn1_distance"] = distances[:, 0]
    result["loo_knn_mean_distance"] = distances.mean(axis=1)
    result["loo_knn_radius"] = distances[:, -1]
    result["nearest_design_id"] = identifiers[indices[:, 0]]

    metric_columns = [
        "design_id",
        "loo_knn_k",
        "loo_nn1_distance",
        "loo_knn_mean_distance",
        "loo_knn_radius",
        "nearest_design_id",
    ]
    row_result = row_mapping.merge(
        result.loc[:, metric_columns], on="design_id", how="left", validate="m:1"
    ).sort_values("dataset_row_number", kind="stable")
    return KNNDiagnostics(
        unique_designs=result.reset_index(drop=True),
        dataset_rows=row_result.reset_index(drop=True),
    )


def _support_band(value: float, q95: float, q99: float) -> str:
    if value <= q95:
        return "within_loo_q95"
    if value <= q99:
        return "between_loo_q95_q99"
    return "beyond_loo_q99"


def compute_query_knn_support(
    reference_dataset: pd.DataFrame,
    query_designs: pd.DataFrame,
    *,
    k: int = 5,
    design_columns: Sequence[str] = DESIGN_COLUMNS,
    topology_columns: Sequence[str] = TOPOLOGY_COLUMNS,
    bounds: Mapping[str, Sequence[float]] = OFFICIAL_DESIGN_BOUNDS,
    reference_loo: Optional[pd.DataFrame] = None,
    chunk_size: int = 512,
) -> pd.DataFrame:
    """Locate optimized designs relative to a unique-design reference set."""

    reference_unique, _ = deduplicate_designs(
        reference_dataset, design_columns=design_columns
    )
    query_values = _validated_design_values(query_designs, design_columns)
    if len(reference_unique) < int(k):
        raise ValueError(
            f"Query k={k} requires at least {k} unique reference designs."
        )

    if reference_loo is None:
        reference_loo = compute_loo_knn_diagnostics(
            reference_dataset,
            k=k,
            design_columns=design_columns,
            bounds=bounds,
            chunk_size=chunk_size,
        ).unique_designs
    _require_columns(reference_loo, ["loo_knn_mean_distance"])
    loo_values = reference_loo["loo_knn_mean_distance"].to_numpy(dtype=float)
    if not np.isfinite(loo_values).all() or len(loo_values) == 0:
        raise ValueError("Reference LOO distances must be finite and non-empty.")
    q95, q99 = np.quantile(loo_values, [0.95, 0.99])

    ref_values = reference_unique.loc[:, design_columns].to_numpy(dtype=float)
    scaled_reference = _scale_design_values(ref_values, design_columns, bounds)
    scaled_query = _scale_design_values(query_values, design_columns, bounds)
    ref_ids = reference_unique["design_id"].to_numpy(dtype=object)
    distances, indices = _top_k_neighbors(
        scaled_query,
        scaled_reference,
        ref_ids,
        k=int(k),
        chunk_size=chunk_size,
    )

    query_ids = _design_ids(query_values, design_columns)
    result = query_designs.copy().reset_index(drop=True)
    source_indices = [str(value) for value in query_designs.index]
    if "query_source_index" in result:
        result["query_source_index"] = source_indices
    else:
        result.insert(0, "query_source_index", source_indices)
    if "design_id" in result:
        result["design_id"] = query_ids
    else:
        result.insert(1, "design_id", query_ids)
    result["reference_knn_k"] = int(k)
    result["reference_nn1_distance"] = distances[:, 0]
    result["reference_knn_mean_distance"] = distances.mean(axis=1)
    result["reference_knn_radius"] = distances[:, -1]
    result["nearest_reference_design_id"] = ref_ids[indices[:, 0]]
    result["reference_loo_percentile"] = [
        100.0 * float(np.mean(loo_values <= value))
        for value in result["reference_knn_mean_distance"]
    ]
    result["reference_loo_q95"] = float(q95)
    result["reference_loo_q99"] = float(q99)
    result["support_band"] = [
        _support_band(float(value), float(q95), float(q99))
        for value in result["reference_knn_mean_distance"]
    ]

    topology_columns = tuple(topology_columns)
    _require_columns(reference_unique, topology_columns)
    _require_columns(result, topology_columns)
    topology_set = {
        tuple(row)
        for row in reference_unique.loc[:, topology_columns].to_numpy()
    }
    result["topology_seen"] = [
        tuple(row) in topology_set
        for row in result.loc[:, topology_columns].to_numpy()
    ]
    return result


def _coerce_acceptance(values: pd.Series) -> np.ndarray:
    accepted = np.zeros(len(values), dtype=bool)
    not_missing = values.notna().to_numpy(dtype=bool)
    observed = values.loc[not_missing]
    valid = observed.isin([True, False, 0, 1])
    if not bool(valid.all()):
        bad = observed.loc[~valid].astype(str).unique().tolist()
        raise ValueError(f"Acceptance column contains non-boolean values: {bad}")
    accepted[not_missing] = observed.astype(bool).to_numpy()
    return accepted


def _verify_duplicate_evaluations(
    table: pd.DataFrame,
    objective_columns: Sequence[str],
    accepted_column: str,
    tolerance: float,
) -> None:
    check_columns = list(objective_columns) + [accepted_column]
    for design_id, group in table.groupby("design_id", sort=False):
        if len(group) == 1:
            continue
        first = group.iloc[0]
        first_objectives = first.loc[list(objective_columns)].to_numpy(dtype=float)
        group_objectives = group.loc[:, objective_columns].to_numpy(dtype=float)
        if not np.allclose(
            group_objectives,
            first_objectives[None, :],
            rtol=0.0,
            atol=tolerance,
            equal_nan=True,
        ):
            raise ValueError(
                "Duplicate design has inconsistent common-stage objectives: "
                f"{design_id}"
            )
        acceptance_values = _coerce_acceptance(group[accepted_column])
        if np.any(acceptance_values != acceptance_values[0]):
            raise ValueError(
                "Duplicate design has inconsistent common-stage acceptance: "
                f"{design_id}"
            )
        _require_columns(group, check_columns)


def _nondominated_mask(
    objectives: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    n_points = len(objectives)
    nondominated = np.ones(n_points, dtype=bool)
    for index in range(n_points):
        no_worse = np.all(
            objectives <= objectives[index][None, :] + tolerance,
            axis=1,
        )
        strictly_better = np.any(
            objectives < objectives[index][None, :] - tolerance,
            axis=1,
        )
        dominates = no_worse & strictly_better
        dominates[index] = False
        if np.any(dominates):
            nondominated[index] = False
    return nondominated


def annotate_empirical_pareto_archive(
    candidates: pd.DataFrame,
    *,
    objective_columns: Sequence[str] = DEFAULT_OBJECTIVE_COLUMNS,
    accepted_column: str = "hard_accepted",
    design_columns: Sequence[str] = DESIGN_COLUMNS,
    duplicate_tolerance: float = 1.0e-12,
    dominance_tolerance: float = 0.0,
) -> pd.DataFrame:
    """Deduplicate and annotate a common-stage evaluated candidate archive.

    All objectives are interpreted as minimization objectives.  Missing or
    non-finite objectives are fail-closed and cannot enter the empirical front.
    Repeated designs must have matching common-stage objectives and acceptance;
    an inconsistency raises instead of selecting a favorable duplicate.
    """

    objective_columns = tuple(objective_columns)
    design_columns = tuple(design_columns)
    _require_columns(
        candidates,
        list(design_columns) + list(objective_columns) + [accepted_column],
    )
    if float(duplicate_tolerance) < 0.0 or float(dominance_tolerance) < 0.0:
        raise ValueError("Pareto tolerances must be non-negative.")

    frame = candidates.copy().reset_index(drop=True)
    frame.insert(
        0,
        "archive_source_index",
        [str(value) for value in candidates.index],
    )
    frame.insert(1, "archive_row_number", np.arange(len(frame), dtype=int))
    values = _validated_design_values(frame, design_columns)
    frame["design_id"] = _design_ids(values, design_columns)
    frame["hard_accepted_normalized"] = _coerce_acceptance(
        frame[accepted_column]
    )
    _verify_duplicate_evaluations(
        frame,
        objective_columns,
        "hard_accepted_normalized",
        float(duplicate_tolerance),
    )

    frame = (
        frame.sort_values(
            ["design_id", "archive_row_number"], kind="stable"
        )
        .drop_duplicates("design_id", keep="first")
        .reset_index(drop=True)
    )
    objectives = frame.loc[:, objective_columns].to_numpy(dtype=np.float64)
    finite = np.isfinite(objectives).all(axis=1)
    eligible = frame["hard_accepted_normalized"].to_numpy(dtype=bool) & finite
    frame["finite_objectives"] = finite
    frame["pareto_eligible"] = eligible
    frame["is_empirical_pareto"] = False
    if np.any(eligible):
        eligible_positions = np.flatnonzero(eligible)
        front = _nondominated_mask(
            objectives[eligible], float(dominance_tolerance)
        )
        frame.loc[
            eligible_positions[front], "is_empirical_pareto"
        ] = True
    return frame


def empirical_pareto_archive(
    candidates: pd.DataFrame,
    *,
    objective_columns: Sequence[str] = DEFAULT_OBJECTIVE_COLUMNS,
    accepted_column: str = "hard_accepted",
    design_columns: Sequence[str] = DESIGN_COLUMNS,
    score_column: Optional[str] = "official_loss",
    duplicate_tolerance: float = 1.0e-12,
    dominance_tolerance: float = 0.0,
) -> pd.DataFrame:
    """Return the stable empirical nondominated front of the archive."""

    annotated = annotate_empirical_pareto_archive(
        candidates,
        objective_columns=objective_columns,
        accepted_column=accepted_column,
        design_columns=design_columns,
        duplicate_tolerance=duplicate_tolerance,
        dominance_tolerance=dominance_tolerance,
    )
    front = annotated.loc[annotated["is_empirical_pareto"]].copy()
    sort_columns = []
    if score_column is not None:
        if score_column not in front.columns:
            raise KeyError(f"Missing score column: {score_column!r}")
        scores = front[score_column].to_numpy(dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError("Empirical Pareto scores must be finite.")
        sort_columns.append(score_column)
    sort_columns.extend(list(objective_columns))
    sort_columns.append("design_id")
    return front.sort_values(sort_columns, kind="stable").reset_index(drop=True)
