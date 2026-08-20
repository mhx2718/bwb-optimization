"""Data-derived topology ranking and exact gamma-like budget allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .evaluator import DESIGN_COLUMNS, TOPOLOGY_COLUMNS
from .schema import VOLUME_TO_CUBIC_METERS


@dataclass(frozen=True)
class TopologyBudgetConfig:
    minimum_feasible_unique_designs: int = 21
    gamma_shape: float = 0.70
    gamma_scale_fraction: float = 0.15
    uniform_exploration_fraction: float = 0.15
    repeats: int = 3
    population: int = 16
    minimum_generations_per_repeat: int = 10

    def __post_init__(self) -> None:
        if self.minimum_feasible_unique_designs < 1:
            raise ValueError("Minimum topology support must be positive.")
        if not (0.0 < self.gamma_shape <= 1.0):
            raise ValueError("A decreasing gamma-rank kernel needs 0 < shape <= 1.")
        if self.gamma_scale_fraction <= 0.0:
            raise ValueError("Gamma scale fraction must be positive.")
        if not (0.0 <= self.uniform_exploration_fraction < 1.0):
            raise ValueError("Exploration fraction must lie in [0, 1).")
        if self.repeats < 1 or self.population < 2:
            raise ValueError("CMA repeats/population are invalid.")
        if self.minimum_generations_per_repeat < 1:
            raise ValueError("Every topology repeat needs at least one generation.")


def topology_label(values: Sequence[Any]) -> str:
    return "".join(str(int(value)) for value in values)


def _resolve_column(
    frame: pd.DataFrame,
    explicit: Optional[str],
    candidates: Sequence[str],
) -> str:
    if explicit is not None:
        if explicit not in frame:
            raise KeyError(f"Required column {explicit!r} is absent.")
        return explicit
    for name in candidates:
        if name in frame:
            return name
    raise KeyError(f"None of the candidate columns exist: {list(candidates)}")


def build_empirical_topology_prior(
    rows: pd.DataFrame,
    missions: Sequence[Mapping[str, Any]],
    stress_limit_mpa: float = 335.0,
    minimum_feasible_unique_designs: int = 21,
    split: Optional[str] = "train",
    eligibility_split: Optional[str] = None,
    split_column: str = "split",
    weight_column: Optional[str] = None,
    payload_column: Optional[str] = None,
    fuel_column: Optional[str] = None,
    stress_column: Optional[str] = None,
    volume_scale_to_m3: float = VOLUME_TO_CUBIC_METERS,
) -> pd.DataFrame:
    """Rank supported observed topologies by median label-based official score.

    Eligibility follows the user's dataset-wide ``>20`` support rule by
    default, while rank/score statistics use only ``split`` (normally train).
    Thus validation/test target values never influence the rank ordering.
    Flight-condition duplicates are collapsed by the 21 physical design
    variables. L/D is deliberately treated as fully satisfied, so the prior
    cannot peek at the external aerodynamic surrogate.
    """

    required_design = set(DESIGN_COLUMNS)
    missing = required_design.difference(rows.columns)
    if missing:
        raise KeyError(f"Topology prior is missing design columns: {sorted(missing)}")
    all_data = rows.copy()

    weight = _resolve_column(
        all_data,
        weight_column,
        ("Aircraft Empty Weight", "Aircraft Empty Weight_kg", "weight_kg"),
    )
    payload = _resolve_column(
        all_data,
        payload_column,
        ("Payload Volume", "Payload Volume_m3", "payload_m3"),
    )
    fuel = _resolve_column(
        all_data,
        fuel_column,
        ("Fuel Volume", "Fuel Volume_m3", "fuel_m3"),
    )
    stress = _resolve_column(
        all_data,
        stress_column,
        ("Max Hotspot Stress", "Max Hotspot Stress_MPa", "stress_mpa", "stress"),
    )
    numeric_columns = list(DESIGN_COLUMNS) + [weight, payload, fuel, stress]
    numeric = all_data.loc[:, numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Topology-prior inputs contain NaN/Inf.")
    all_data.loc[:, numeric_columns] = numeric

    def select_split(frame: pd.DataFrame, selected: Optional[str]) -> pd.DataFrame:
        if selected is None:
            return frame.copy()
        if split_column not in frame:
            raise KeyError(
                f"Requested split {selected!r}, but {split_column!r} is absent."
            )
        return frame.loc[frame[split_column].astype(str) == str(selected)].copy()

    def feasible_physical(frame: pd.DataFrame) -> pd.DataFrame:
        observations = frame.loc[
            frame[stress].to_numpy(dtype=np.float64) <= float(stress_limit_mpa)
        ].copy()
        if observations.empty:
            return pd.DataFrame(columns=[*DESIGN_COLUMNS, weight, payload, fuel])
        # A physical design can be repeated at several flight conditions. Count
        # it once and aggregate only tiny verified target discrepancies.
        return (
            observations.groupby(list(DESIGN_COLUMNS), as_index=False, sort=True)
            .agg({weight: "median", payload: "median", fuel: "median"})
            .reset_index(drop=True)
        )

    ranking_data = select_split(all_data, split)
    eligibility_data = select_split(all_data, eligibility_split)
    physical = feasible_physical(ranking_data)
    eligibility_physical = feasible_physical(eligibility_data)
    if physical.empty:
        raise ValueError("Ranking split has no stress-feasible physical designs.")
    if eligibility_physical.empty:
        raise ValueError("Eligibility population has no stress-feasible designs.")

    eligibility_counts = (
        eligibility_physical.groupby(list(TOPOLOGY_COLUMNS), sort=True)
        .size()
        .rename("feasible_unique_designs")
        .reset_index()
    )
    physical["payload_m3_for_score"] = (
        physical[payload].to_numpy(dtype=np.float64) * float(volume_scale_to_m3)
    )
    physical["fuel_m3_for_score"] = (
        physical[fuel].to_numpy(dtype=np.float64) * float(volume_scale_to_m3)
    )
    rows_out = []
    for mission in missions:
        case_id = int(mission["case_id"])
        payload_target = float(mission["Payload_target_m3"])
        fuel_target = float(mission["Fuel_target_m3"])
        score = (
            0.4 * physical[weight].to_numpy(dtype=np.float64) / 50.0
            + 0.2
            * np.maximum(
                0.0,
                (payload_target - physical["payload_m3_for_score"].to_numpy())
                / payload_target,
            )
            + 0.2
            * np.maximum(
                0.0,
                (fuel_target - physical["fuel_m3_for_score"].to_numpy())
                / fuel_target,
            )
            # L/D is treated as full score by construction (zero shortfall).
        )
        scored = physical.loc[:, list(TOPOLOGY_COLUMNS)].copy()
        scored["label_score_ld_full"] = score
        grouped = scored.groupby(list(TOPOLOGY_COLUMNS), sort=True).agg(
            ranking_feasible_unique_designs=("label_score_ld_full", "size"),
            median_label_score_ld_full=("label_score_ld_full", "median"),
            q25_label_score_ld_full=("label_score_ld_full", lambda x: x.quantile(0.25)),
            q75_label_score_ld_full=("label_score_ld_full", lambda x: x.quantile(0.75)),
        )
        grouped = grouped.reset_index().merge(
            eligibility_counts,
            on=list(TOPOLOGY_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
        grouped = grouped.loc[
            grouped["feasible_unique_designs"].to_numpy(dtype=np.int64)
            >= int(minimum_feasible_unique_designs)
        ].copy()
        if grouped.empty:
            maximum = int(eligibility_counts["feasible_unique_designs"].max())
            raise RuntimeError(
                f"Case {case_id}: no topology has at least "
                f"{minimum_feasible_unique_designs} feasible unique designs in "
                f"the eligibility population (maximum observed={maximum}) and "
                "at least one feasible design in the ranking split."
            )
        grouped = grouped.sort_values(
            ["median_label_score_ld_full", *TOPOLOGY_COLUMNS], kind="stable"
        ).reset_index(drop=True)
        grouped.insert(0, "prior_rank", np.arange(len(grouped), dtype=np.int64))
        grouped.insert(0, "case_id", case_id)
        grouped["eligibility_split"] = (
            "all" if eligibility_split is None else str(eligibility_split)
        )
        grouped["ranking_split"] = "all" if split is None else str(split)
        grouped["topology"] = list(
            zip(*(grouped[name].astype(int) for name in TOPOLOGY_COLUMNS))
        )
        grouped["topology_label"] = [
            topology_label(values) for values in grouped["topology"]
        ]
        rows_out.append(grouped)
    return pd.concat(rows_out, ignore_index=True)


def gamma_rank_weights(
    ranks: Sequence[int],
    shape: float = 0.70,
    scale_fraction: float = 0.15,
) -> np.ndarray:
    """Return the normalized decreasing Gamma-density rank kernel.

    The half-rank offset avoids the singularity of shape < 1 at zero.  Shape
    0.70 and scale 0.15*K strongly favor the head while retaining a long tail.
    """

    rank = np.asarray(ranks, dtype=np.float64)
    if rank.ndim != 1 or not len(rank) or np.any(rank < 0):
        raise ValueError("Ranks must be a non-empty nonnegative vector.")
    if not (0.0 < shape <= 1.0) or scale_fraction <= 0.0:
        raise ValueError("Invalid Gamma-rank parameters.")
    scale = max(1.0, float(scale_fraction) * len(rank))
    shifted = rank + 0.5
    log_weight = (float(shape) - 1.0) * np.log(shifted) - shifted / scale
    log_weight -= np.max(log_weight)
    weight = np.exp(log_weight)
    return weight / np.sum(weight)


def _performance_ranks(
    previous_archive: pd.DataFrame,
    topology_values: Sequence[Tuple[int, int, int]],
) -> np.ndarray:
    """Rank topologies using the median of repeat champions, not one lucky run."""

    if previous_archive.empty:
        return np.arange(len(topology_values), dtype=np.int64)
    required = set(TOPOLOGY_COLUMNS) | {"repeat", "search_merit", "official_loss"}
    missing = required.difference(previous_archive.columns)
    if missing:
        raise KeyError(f"Previous archive lacks {sorted(missing)}")
    data = previous_archive.copy()
    if "hard_accepted" not in data:
        data["hard_accepted"] = False
    rows = []
    for topology in topology_values:
        mask = np.ones(len(data), dtype=bool)
        for name, value in zip(TOPOLOGY_COLUMNS, topology):
            mask &= data[name].to_numpy(dtype=np.int64) == int(value)
        group = data.loc[mask]
        if group.empty:
            rows.append((2, float("inf"), 0, topology))
            continue
        repeat_champions = []
        for repeat, repeat_rows in group.groupby("repeat", sort=True):
            if int(repeat) < 0:
                # Local refinement is a deterministic proposal stage, not an
                # additional independent CMA repeat for rank aggregation.
                continue
            accepted = repeat_rows.loc[repeat_rows["hard_accepted"].to_numpy(dtype=bool)]
            if not accepted.empty:
                repeat_champions.append(
                    (
                        True,
                        float(accepted["official_loss"].min()),
                        float(accepted["search_merit"].min()),
                    )
                )
            else:
                repeat_champions.append(
                    (
                        False,
                        float("nan"),
                        float(repeat_rows["search_merit"].min()),
                    )
                )
        if not repeat_champions:
            rows.append((2, float(group["search_merit"].min()), 0, topology))
            continue
        success = np.asarray([item[0] for item in repeat_champions], dtype=bool)
        official = np.asarray([item[1] for item in repeat_champions], dtype=np.float64)
        merits = np.asarray([item[2] for item in repeat_champions], dtype=np.float64)
        # A topology is trusted only if at least two of the three repeats found
        # a feasible design.  This avoids reallocating a whole round to a single
        # lucky trajectory.
        successful = int(np.sum(success))
        tier = 0 if successful >= min(2, len(repeat_champions)) else 1
        # Trusted topologies compare the median successful official loss. An
        # untrusted topology compares the median tier-separated merit across
        # *all* repeats, so one lucky feasible trajectory cannot erase two
        # failed trajectories.
        value = (
            float(np.median(official[success]))
            if tier == 0
            else float(np.median(merits))
        )
        rows.append((tier, value, -successful, topology))
    order = sorted(
        range(len(rows)),
        key=lambda i: (rows[i][0], rows[i][1], rows[i][2], rows[i][3]),
    )
    ranks = np.empty(len(rows), dtype=np.int64)
    ranks[np.asarray(order, dtype=np.int64)] = np.arange(len(rows), dtype=np.int64)
    return ranks


def build_round_weights(
    case_prior: pd.DataFrame,
    prior_weight: float,
    gamma_shape: float = 0.70,
    gamma_scale_fraction: float = 0.15,
    uniform_exploration_fraction: float = 0.15,
    previous_archive: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Blend the immutable data prior, previous-round evidence, and uniform tail."""

    if not (0.0 <= prior_weight <= 1.0):
        raise ValueError("prior_weight must lie in [0, 1].")
    if not (0.0 <= uniform_exploration_fraction < 1.0):
        raise ValueError("uniform exploration must lie in [0, 1).")
    required = set(TOPOLOGY_COLUMNS) | {"prior_rank"}
    missing = required.difference(case_prior.columns)
    if missing:
        raise KeyError(f"Topology prior lacks {sorted(missing)}")
    ordered = case_prior.sort_values("prior_rank", kind="stable").reset_index(drop=True)
    prior = gamma_rank_weights(
        ordered["prior_rank"].to_numpy(dtype=np.int64),
        gamma_shape,
        gamma_scale_fraction,
    )
    topology_values = [
        tuple(int(row[name]) for name in TOPOLOGY_COLUMNS)
        for _, row in ordered.iterrows()
    ]
    if previous_archive is None or previous_archive.empty or prior_weight >= 1.0:
        evidence_rank = ordered["prior_rank"].to_numpy(dtype=np.int64)
        evidence = prior.copy()
    else:
        evidence_rank = _performance_ranks(previous_archive, topology_values)
        evidence = gamma_rank_weights(
            evidence_rank, gamma_shape, gamma_scale_fraction
        )
    exploitation = float(prior_weight) * prior + (1.0 - float(prior_weight)) * evidence
    uniform = np.full(len(ordered), 1.0 / len(ordered), dtype=np.float64)
    final = (
        (1.0 - float(uniform_exploration_fraction)) * exploitation
        + float(uniform_exploration_fraction) * uniform
    )
    final /= np.sum(final)
    result = ordered.copy()
    result["evidence_rank"] = evidence_rank
    result["prior_gamma_weight"] = prior
    result["evidence_gamma_weight"] = evidence
    result["allocation_weight"] = final
    return result


def allocate_generation_blocks(
    weighted_topologies: pd.DataFrame,
    total_evaluations: int,
    repeats: int = 3,
    population: int = 32,
    minimum_generations_per_repeat: int = 10,
) -> pd.DataFrame:
    """Allocate an exact number of whole, equal three-repeat CMA generations."""

    if weighted_topologies.empty:
        raise ValueError("Cannot allocate a budget to an empty topology set.")
    if "allocation_weight" not in weighted_topologies:
        raise KeyError("weighted_topologies needs allocation_weight.")
    repeats = int(repeats)
    population = int(population)
    minimum_generations_per_repeat = int(minimum_generations_per_repeat)
    block = repeats * population
    total_evaluations = int(total_evaluations)
    if total_evaluations % block:
        raise ValueError(
            f"Total budget {total_evaluations} must be divisible by "
            f"repeats*population={block}."
        )
    total_blocks = total_evaluations // block
    n_topologies = len(weighted_topologies)
    minimum_blocks = n_topologies * minimum_generations_per_repeat
    if total_blocks < minimum_blocks:
        required = minimum_blocks * block
        raise ValueError(
            f"Budget is too small for the topology floor; need at least {required}."
        )
    # Pandas copy-on-write can expose a read-only NumPy view.  Allocation only
    # needs local normalized values, so always materialize a writable copy and
    # avoid mutating data owned by the caller.
    weights = weighted_topologies["allocation_weight"].to_numpy(
        dtype=np.float64,
        copy=True,
    )
    if not np.isfinite(weights).all() or np.any(weights < 0.0) or np.sum(weights) <= 0.0:
        raise ValueError("Allocation weights must be finite, nonnegative, and nonzero.")
    weights = weights / np.sum(weights)
    remaining = total_blocks - minimum_blocks
    raw = remaining * weights
    extra = np.floor(raw).astype(np.int64)
    leftover = int(remaining - np.sum(extra))
    # Largest-remainder allocation with prior_rank/topology order as a stable tie-break.
    remainder_order = np.argsort(-(raw - extra), kind="stable")
    extra[remainder_order[:leftover]] += 1
    generations = minimum_generations_per_repeat + extra
    result = weighted_topologies.copy().reset_index(drop=True)
    result["generations_per_repeat"] = generations
    result["evaluations_per_repeat"] = generations * population
    result["allocated_evaluations"] = generations * block
    result["allocated_fraction"] = result["allocated_evaluations"] / total_evaluations
    if int(result["allocated_evaluations"].sum()) != total_evaluations:
        raise AssertionError("Integer allocation did not conserve the evaluation budget.")
    return result
