"""Multi-round topology-aware CMA-ES + local-refinement orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .artifacts import atomic_output_path, atomic_write_json
from .cma_search import CMAConfig, run_cma_round
from .evaluator import BWBEvaluator, DESIGN_COLUMNS, TOPOLOGY_COLUMNS, rank_evaluations
from .local_refine import LocalRefineConfig, refine_round_champions
from .topology_budget import (
    allocate_generation_blocks,
    build_round_weights,
)


class NoConfirmedDesignError(RuntimeError):
    """Raised instead of silently promoting a constraint-violating design."""


@dataclass(frozen=True)
class RoundConfig:
    total_evaluations_per_case: int
    prior_weight: float

    def __post_init__(self) -> None:
        if self.total_evaluations_per_case < 1:
            raise ValueError("Each round needs a positive evaluation budget.")
        if not 0.0 <= self.prior_weight <= 1.0:
            raise ValueError("Round prior_weight must lie in [0, 1].")


@dataclass(frozen=True)
class ConvergenceConfig:
    relative_loss_tolerance: float = 1.0e-3
    repeat_loss_spread_tolerance: float = 5.0e-3
    required_repeats_same_topology: int = 2

    def __post_init__(self) -> None:
        if self.relative_loss_tolerance < 0.0:
            raise ValueError("relative_loss_tolerance cannot be negative.")
        if self.repeat_loss_spread_tolerance < 0.0:
            raise ValueError("repeat_loss_spread_tolerance cannot be negative.")
        if self.required_repeats_same_topology < 1:
            raise ValueError("At least one repeat is required for consensus.")


@dataclass(frozen=True)
class OptimizationConfig:
    rounds: Tuple[RoundConfig, ...]
    repeats: int = 3
    gamma_shape: float = 0.50
    gamma_scale_fraction: float = 0.08
    uniform_exploration_fraction: float = 0.15
    minimum_generations_per_topology_repeat: int = 10
    cma_population: int = 32
    cma_initial_sigma: float = 0.22
    cma_archive_per_run: int = 64
    warm_repeats_after_round_zero: int = 2
    confirmation_shortlist_per_case: int = 128
    local_refine: LocalRefineConfig = LocalRefineConfig()
    convergence: ConvergenceConfig = ConvergenceConfig()

    def __post_init__(self) -> None:
        if not self.rounds:
            raise ValueError("At least one optimization round is required.")
        if self.repeats < 1:
            raise ValueError("repeats must be positive.")
        if self.confirmation_shortlist_per_case < 1:
            raise ValueError("The confirmation shortlist cannot be empty.")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "OptimizationConfig":
        rounds = tuple(
            RoundConfig(
                total_evaluations_per_case=int(item["total_evaluations_per_case"]),
                prior_weight=float(item["prior_weight"]),
            )
            for item in mapping["rounds"]
        )
        local_values = dict(mapping.get("local_refine", {}))
        convergence_values = dict(mapping.get("convergence", {}))
        return cls(
            rounds=rounds,
            repeats=int(mapping.get("repeats", 3)),
            gamma_shape=float(mapping.get("gamma_shape", 0.70)),
            gamma_scale_fraction=float(mapping.get("gamma_scale_fraction", 0.15)),
            uniform_exploration_fraction=float(
                mapping.get("uniform_exploration_fraction", 0.15)
            ),
            minimum_generations_per_topology_repeat=int(
                mapping.get("minimum_generations_per_topology_repeat", 10)
            ),
            cma_population=int(mapping.get("cma_population", 16)),
            cma_initial_sigma=float(mapping.get("cma_initial_sigma", 0.22)),
            cma_archive_per_run=int(mapping.get("cma_archive_per_run", 64)),
            warm_repeats_after_round_zero=int(
                mapping.get("warm_repeats_after_round_zero", 2)
            ),
            confirmation_shortlist_per_case=int(
                mapping.get("confirmation_shortlist_per_case", 128)
            ),
            local_refine=LocalRefineConfig(**local_values),
            convergence=ConvergenceConfig(**convergence_values),
        )

    def cma_config(self) -> CMAConfig:
        return CMAConfig(
            population=self.cma_population,
            initial_sigma=self.cma_initial_sigma,
            archive_per_run=self.cma_archive_per_run,
            repeats=self.repeats,
            warm_repeats_after_round_zero=self.warm_repeats_after_round_zero,
        )


@dataclass
class CaseOptimizationResult:
    case_id: int
    final_design: pd.DataFrame
    confirmed_shortlist: pd.DataFrame
    search_archive: pd.DataFrame
    convergence: pd.DataFrame
    allocations: pd.DataFrame
    cma_trace: pd.DataFrame
    local_trace: pd.DataFrame
    run_summary: pd.DataFrame

    def save(self, output_directory: Path) -> None:
        root = Path(output_directory) / f"case_{self.case_id}"
        root.mkdir(parents=True, exist_ok=True)
        tables = {
            "final_design.csv": self.final_design,
            "confirmed_shortlist.csv": self.confirmed_shortlist,
            "search_archive.csv": self.search_archive,
            "convergence.csv": self.convergence,
            "topology_allocations.csv": self.allocations,
            "cma_trace.csv": self.cma_trace,
            "local_trace.csv": self.local_trace,
            "run_summary.csv": self.run_summary,
        }
        for name, table in tables.items():
            table.to_csv(root / name, index=False)


@dataclass
class MultiCaseOptimizationResult:
    cases: Mapping[int, CaseOptimizationResult]

    @property
    def final_designs(self) -> pd.DataFrame:
        return pd.concat(
            [self.cases[key].final_design for key in sorted(self.cases)],
            ignore_index=True,
        )

    def save(self, output_directory: Path) -> None:
        root = Path(output_directory)
        root.mkdir(parents=True, exist_ok=True)
        for result in self.cases.values():
            result.save(root)
        self.final_designs.to_csv(root / "final_designs_all_cases.csv", index=False)


def _write_immutable_csv(path: Path, table: pd.DataFrame) -> str:
    """Write once, or prove that a deterministic rerun produced identical bytes."""

    payload = table.to_csv(index=False).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(
                f"Immutable checkpoint differs from the existing file: {path}"
            )
        return digest
    with atomic_output_path(path) as temporary:
        temporary.write_bytes(payload)
    return digest


def _save_round_checkpoint(
    checkpoint_root: Path,
    *,
    case_id: int,
    round_index: int,
    round_archive: pd.DataFrame,
    allocation: pd.DataFrame,
    cma_trace: pd.DataFrame,
    local_trace: pd.DataFrame,
    run_summary: pd.DataFrame,
    convergence_row: Mapping[str, Any],
    master_seed: int | None,
) -> None:
    """Persist a complete immutable boundary before the next round begins."""

    directory = (
        Path(checkpoint_root)
        / f"case_{int(case_id)}"
        / "round_checkpoints"
        / f"round_{int(round_index)}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "archive.csv": round_archive,
        "allocation.csv": allocation,
        "cma_trace.csv": cma_trace,
        "local_trace.csv": local_trace,
        "run_summary.csv": run_summary,
        "convergence.csv": pd.DataFrame([convergence_row]),
    }
    hashes = {
        name: _write_immutable_csv(directory / name, table)
        for name, table in tables.items()
    }
    marker = {
        "checkpoint_schema": "bwb-round-boundary-v1",
        "case_id": int(case_id),
        "round": int(round_index),
        "master_seed": None if master_seed is None else int(master_seed),
        "files_sha256": hashes,
        "rows": {name: int(len(table)) for name, table in tables.items()},
    }
    marker_path = directory / "checkpoint_manifest.json"
    if marker_path.exists():
        existing = json.loads(marker_path.read_text(encoding="utf-8"))
        if existing != marker:
            raise RuntimeError(
                f"Immutable checkpoint manifest differs: {marker_path}"
            )
    else:
        atomic_write_json(marker_path, marker)


def _topology_tuple(row: pd.Series) -> Tuple[int, int, int]:
    return tuple(int(row[name]) for name in TOPOLOGY_COLUMNS)  # type: ignore[return-value]


def _design_digest(row: pd.Series, design_columns: Sequence[str]) -> str:
    values = row.loc[list(design_columns)].to_numpy(dtype="<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _round_convergence_row(
    round_index: int,
    round_archive: pd.DataFrame,
    previous_best_loss: float,
    config: ConvergenceConfig,
) -> Mapping[str, Any]:
    accepted = round_archive.loc[
        round_archive["hard_accepted"].to_numpy(dtype=bool)
    ]
    if accepted.empty:
        best = rank_evaluations(round_archive).iloc[0]
        best_loss = float("nan")
        best_topology = _topology_tuple(best)
    else:
        best = accepted.sort_values("official_loss", kind="stable").iloc[0]
        best_loss = float(best["official_loss"])
        best_topology = _topology_tuple(best)
    if np.isfinite(previous_best_loss) and np.isfinite(best_loss):
        # Keep the sign.  A worse new round must not be converted to zero and
        # mislabeled as stagnation/convergence.
        relative_improvement = (
            previous_best_loss - best_loss
        ) / max(abs(previous_best_loss), 1.0e-12)
        relative_change = abs(relative_improvement)
    else:
        relative_improvement = float("nan")
        relative_change = float("nan")

    repeat_best = []
    champion_topologies = []
    champion_modes = []
    repeat_champion_units = []
    unit_columns = sorted(
        column for column in round_archive.columns if column.startswith("unit__")
    )
    for repeat, group in round_archive.groupby("repeat", sort=True):
        if int(repeat) < 0:
            continue
        ranked = rank_evaluations(group)
        feasible = ranked.loc[ranked["hard_accepted"].to_numpy(dtype=bool)]
        if feasible.empty:
            repeat_best.append(np.nan)
            continue
        champion = feasible.sort_values("official_loss", kind="stable").iloc[0]
        champion_topologies.append(_topology_tuple(champion))
        champion_modes.append(str(champion.get("start_mode", "unknown")))
        if unit_columns:
            repeat_champion_units.append(
                champion.loc[unit_columns].to_numpy(dtype=np.float64)
            )
        repeat_best.append(float(champion["official_loss"]))
    finite_repeat = np.asarray(repeat_best, dtype=np.float64)
    finite_repeat = finite_repeat[np.isfinite(finite_repeat)]
    if len(finite_repeat) >= 2:
        repeat_spread = float(
            (np.max(finite_repeat) - np.min(finite_repeat))
            / max(abs(np.median(finite_repeat)), 1.0e-12)
        )
    else:
        repeat_spread = float("nan")
    agreement = champion_topologies.count(best_topology)
    cold_repeat_support = any(
        topology == best_topology and mode.startswith("cold")
        for topology, mode in zip(champion_topologies, champion_modes)
    )
    pairwise_rms = []
    for left in range(len(repeat_champion_units)):
        for right in range(left + 1, len(repeat_champion_units)):
            pairwise_rms.append(
                float(
                    np.sqrt(
                        np.mean(
                            (
                                repeat_champion_units[left]
                                - repeat_champion_units[right]
                            )
                            ** 2
                        )
                    )
                )
            )
    converged = bool(
        np.isfinite(relative_change)
        and relative_change <= config.relative_loss_tolerance
        and np.isfinite(repeat_spread)
        and repeat_spread <= config.repeat_loss_spread_tolerance
        and agreement >= config.required_repeats_same_topology
        and cold_repeat_support
    )
    return {
        "round": int(round_index),
        "best_accepted_official_loss": best_loss,
        "best_topology": json.dumps(best_topology),
        "best_design_sha256": _design_digest(best, DESIGN_COLUMNS),
        "relative_loss_improvement": relative_improvement,
        "absolute_relative_loss_change": relative_change,
        "repeat_loss_spread": repeat_spread,
        "repeats_supporting_round_best_topology": int(agreement),
        "cold_repeat_supports_round_best_topology": bool(cold_repeat_support),
        "repeat_champion_pairwise_unit_rms_median": (
            float(np.median(pairwise_rms)) if pairwise_rms else float("nan")
        ),
        "repeat_champion_pairwise_unit_rms_max": (
            float(np.max(pairwise_rms)) if pairwise_rms else float("nan")
        ),
        # This is intentionally labeled as an archive statistic. The bounded
        # archive is feasibility-selected and cannot estimate the raw search
        # population's feasibility rate; that rate lives in the CMA trace.
        "retained_archive_accepted_fraction": float(
            round_archive["hard_accepted"].mean()
        ),
        "converged_diagnostic": converged,
        "convergence_status": (
            "converged_by_repeat_consensus"
            if converged
            else "budget_limited_no_consensus"
        ),
    }


def build_convergence_diagnostics(
    round_archives: Sequence[pd.DataFrame],
    config: ConvergenceConfig = ConvergenceConfig(),
    case_id: Optional[int] = None,
) -> pd.DataFrame:
    """Build the public round-by-round convergence table.

    This pure helper is useful when reloading archived rounds without rerunning
    CMA-ES.  A convergence flag is diagnostic only; it never changes the fixed
    configured evaluation budget.
    """

    rows = []
    previous_best = float("inf")
    for round_index, archive in enumerate(round_archives):
        if archive.empty:
            raise ValueError(f"Round {round_index} archive is empty.")
        row = dict(
            _round_convergence_row(round_index, archive, previous_best, config)
        )
        if case_id is not None:
            row = {"case_id": int(case_id), **row}
        current = float(row["best_accepted_official_loss"])
        if np.isfinite(current):
            previous_best = min(previous_best, current)
        rows.append(row)
    return pd.DataFrame(rows)


def confirm_and_select(
    evaluator: BWBEvaluator,
    mission: Mapping[str, Any],
    search_archive: pd.DataFrame,
    shortlist_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Use a held-out common bank and select exactly one feasible official minimum."""

    if search_archive.empty:
        raise ValueError("Cannot confirm an empty search archive.")
    ranked = rank_evaluations(search_archive).drop_duplicates(
        subset=list(evaluator.design_space.design_columns), keep="first"
    )
    # Freeze a deterministic, auditable shortlist.  Besides the global head,
    # retain each independent repeat champion and each searched-topology
    # champion so adaptive multiplicity cannot silently erase a whole run.
    def safety_head(group: pd.DataFrame, n: int = 1) -> pd.DataFrame:
        return group.sort_values(
            ["hard_accepted", "worst_robust_median", "probability_feasible", "official_loss"],
            ascending=[False, False, False, True],
            kind="stable",
        ).head(n)

    mandatory_parts = [ranked.head(1), safety_head(ranked, n=min(8, len(ranked)))]
    if "repeat" in ranked:
        mandatory_parts.extend(
            group.head(1) for _, group in ranked.groupby("repeat", sort=True)
        )
        mandatory_parts.extend(
            safety_head(group)
            for _, group in ranked.groupby("repeat", sort=True)
        )
    mandatory_parts.extend(
        group.head(1)
        for _, group in ranked.groupby(list(TOPOLOGY_COLUMNS), sort=True)
    )
    mandatory_parts.extend(
        safety_head(group)
        for _, group in ranked.groupby(list(TOPOLOGY_COLUMNS), sort=True)
    )
    mandatory = pd.concat(mandatory_parts, ignore_index=True).drop_duplicates(
        subset=list(evaluator.design_space.design_columns), keep="first"
    )
    target_size = max(int(shortlist_size), len(mandatory))
    shortlist = pd.concat([mandatory, ranked], ignore_index=True).drop_duplicates(
        subset=list(evaluator.design_space.design_columns), keep="first"
    ).head(target_size).reset_index(drop=True)
    confirmed = evaluator.evaluate(
        shortlist.loc[:, evaluator.design_space.design_columns],
        mission,
        bank_name="confirmation",
    )
    confirmed["case_id"] = int(mission["case_id"])
    for key, value in mission.items():
        if key != "case_id":
            confirmed[str(key)] = value
    confirmed["confirmation_rank_input"] = np.arange(len(confirmed), dtype=np.int64)
    search_columns = [
        name
        for name in shortlist.columns
        if name not in evaluator.design_space.design_columns
    ]
    for name in search_columns:
        confirmed[f"search__{name}"] = shortlist[name].to_numpy()
    confirmed["search_accepted"] = shortlist["hard_accepted"].to_numpy(dtype=bool)
    confirmed["confirmation_accepted"] = confirmed["hard_accepted"].to_numpy(
        dtype=bool
    )
    confirmed["selected_final"] = False
    accepted = confirmed.loc[confirmed["hard_accepted"].to_numpy(dtype=bool)].copy()
    if accepted.empty:
        diagnostic = rank_evaluations(confirmed).iloc[0]
        raise NoConfirmedDesignError(
            f"Case {mission['case_id']}: no held-out confirmed design. "
            f"Best tier={int(diagnostic['constraint_tier'])}, "
            f"P_nominal={float(diagnostic['probability_feasible']):.4f}, "
            f"worst median={float(diagnostic['worst_robust_median']):.4f}, "
            f"L/D in-domain={bool(diagnostic['ld_in_domain'])}."
        )
    final = accepted.sort_values("official_loss", kind="stable").head(1).copy()
    final["selected"] = True
    final["selected_final"] = True
    final["selection_rule"] = (
        "minimum official loss among held-out-confirmed hard-feasible candidates"
    )
    selected_input_rank = int(final.iloc[0]["confirmation_rank_input"])
    confirmed.loc[
        confirmed["confirmation_rank_input"] == selected_input_rank,
        "selected_final",
    ] = True
    return confirmed.reset_index(drop=True), final.reset_index(drop=True)


def optimize_case(
    evaluator: BWBEvaluator,
    mission: Mapping[str, Any],
    topology_prior: pd.DataFrame,
    config: OptimizationConfig,
    seed_source: Any,
    progress_callback: Optional[Any] = None,
    checkpoint_directory: Optional[str | Path] = None,
) -> CaseOptimizationResult:
    """Run all configured rounds, keeping warm evidence and a cold repeat."""

    case_id = int(mission["case_id"])
    case_prior = topology_prior.loc[
        topology_prior["case_id"].to_numpy(dtype=np.int64) == case_id
    ].copy()
    if case_prior.empty:
        raise ValueError(f"Topology prior has no rows for case {case_id}.")

    cumulative_archive: Optional[pd.DataFrame] = None
    last_round_archive: Optional[pd.DataFrame] = None
    allocations = []
    cma_traces = []
    local_traces = []
    summaries = []
    convergence_rows = []
    previous_best_loss = float("inf")
    case_evaluation_offset = 0

    for round_index, round_config in enumerate(config.rounds):
        weights = build_round_weights(
            case_prior,
            prior_weight=round_config.prior_weight,
            gamma_shape=config.gamma_shape,
            gamma_scale_fraction=config.gamma_scale_fraction,
            uniform_exploration_fraction=config.uniform_exploration_fraction,
            previous_archive=last_round_archive,
        )
        allocation = allocate_generation_blocks(
            weights,
            total_evaluations=round_config.total_evaluations_per_case,
            repeats=config.repeats,
            population=config.cma_population,
            minimum_generations_per_repeat=(
                config.minimum_generations_per_topology_repeat
            ),
        )
        allocation["case_id"] = case_id
        allocation["round"] = int(round_index)
        allocation["round_prior_weight"] = float(round_config.prior_weight)
        allocations.append(allocation)

        cma_result = run_cma_round(
            evaluator=evaluator,
            mission=mission,
            allocation=allocation,
            round_index=round_index,
            config=config.cma_config(),
            seed_source=seed_source,
            previous_archive=cumulative_archive,
            progress_callback=progress_callback,
        )
        round_archive = cma_result.archive
        cma_trace = cma_result.trace.copy()
        if not cma_trace.empty:
            cma_trace["case_cumulative_cma_evaluations"] = (
                case_evaluation_offset
                + cma_trace["case_round_cumulative_evaluations"].to_numpy(
                    dtype=np.int64
                )
            )
        cma_traces.append(cma_trace)
        summaries.append(cma_result.run_summary)
        case_evaluation_offset += int(
            cma_result.run_summary["actual_evaluations"].sum()
        )

        if config.local_refine.enabled:
            local_result = refine_round_champions(
                evaluator,
                mission,
                round_archive,
                config.local_refine,
                seed_source,
                round_index,
            )
            round_archive = pd.concat(
                [round_archive, local_result.archive], ignore_index=True
            )
            local_traces.append(local_result.trace)
        convergence_row = _round_convergence_row(
            round_index, round_archive, previous_best_loss, config.convergence
        )
        if cma_trace.empty:
            raw_feasible_fraction = float("nan")
        else:
            raw_feasible_fraction = float(
                cma_trace["generation_hard_accepted_count"].sum()
                / cma_trace["generation_evaluations"].sum()
            )
        convergence_row["cma_candidate_hard_feasible_fraction"] = (
            raw_feasible_fraction
        )
        convergence_row = {"case_id": case_id, **convergence_row}
        convergence_rows.append(convergence_row)
        current_best = convergence_row["best_accepted_official_loss"]
        if np.isfinite(current_best):
            previous_best_loss = min(previous_best_loss, float(current_best))

        if checkpoint_directory is not None:
            _save_round_checkpoint(
                Path(checkpoint_directory),
                case_id=case_id,
                round_index=round_index,
                round_archive=round_archive,
                allocation=allocation,
                cma_trace=cma_trace,
                local_trace=(
                    local_result.trace
                    if config.local_refine.enabled
                    else pd.DataFrame()
                ),
                run_summary=cma_result.run_summary,
                convergence_row=convergence_row,
                master_seed=getattr(seed_source, "master_seed", None),
            )

        last_round_archive = round_archive.copy()
        cumulative_archive = (
            round_archive.copy()
            if cumulative_archive is None
            else pd.concat([cumulative_archive, round_archive], ignore_index=True)
        )

    if cumulative_archive is None or cumulative_archive.empty:
        raise RuntimeError("Optimization produced no evaluated candidate.")
    cumulative_archive = rank_evaluations(cumulative_archive).drop_duplicates(
        subset=list(evaluator.design_space.design_columns), keep="first"
    )
    confirmed, final = confirm_and_select(
        evaluator,
        mission,
        cumulative_archive,
        config.confirmation_shortlist_per_case,
    )
    selected_topology = _topology_tuple(final.iloc[0])
    cumulative_archive["search_accepted"] = cumulative_archive[
        "hard_accepted"
    ].astype("boolean")
    status = confirmed.loc[
        :,
        list(evaluator.design_space.design_columns)
        + ["confirmation_accepted", "selected_final"],
    ]
    cumulative_archive = cumulative_archive.merge(
        status,
        on=list(evaluator.design_space.design_columns),
        how="left",
        validate="1:1",
        sort=False,
    )
    cumulative_archive["confirmation_accepted"] = cumulative_archive[
        "confirmation_accepted"
    ].astype("boolean")
    cumulative_archive["selected_final"] = cumulative_archive[
        "selected_final"
    ].astype("boolean")
    convergence_table = pd.DataFrame(convergence_rows)
    convergence_table["heldout_selected_topology"] = None
    convergence_table["heldout_selected_official_loss"] = np.nan
    convergence_table["heldout_selection_changed_topology"] = pd.NA
    convergence_table["heldout_selection_changed_design"] = pd.NA
    convergence_table["heldout_selected_matches_last_round_champion"] = pd.NA
    convergence_table["final_convergence_status"] = pd.NA
    if not convergence_table.empty:
        last = convergence_table.index[-1]
        search_topology = tuple(
            int(value)
            for value in json.loads(convergence_table.loc[last, "best_topology"])
        )
        changed = selected_topology != search_topology
        selected_input_rank = int(final.iloc[0]["confirmation_rank_input"])
        changed_design = selected_input_rank != 0
        selected_digest = _design_digest(
            final.iloc[0], evaluator.design_space.design_columns
        )
        matches_last_round = (
            selected_digest
            == str(convergence_table.loc[last, "best_design_sha256"])
        )
        base_converged = bool(
            convergence_table.loc[last, "converged_diagnostic"]
        )
        convergence_table.loc[last, "heldout_selected_topology"] = json.dumps(
            selected_topology
        )
        convergence_table.loc[last, "heldout_selected_official_loss"] = float(
            final.iloc[0]["official_loss"]
        )
        convergence_table.loc[
            last, "heldout_selection_changed_topology"
        ] = bool(changed)
        convergence_table.loc[
            last, "heldout_selection_changed_design"
        ] = bool(changed_design)
        convergence_table.loc[
            last, "heldout_selected_matches_last_round_champion"
        ] = bool(matches_last_round)
        convergence_table.loc[last, "final_convergence_status"] = (
            "heldout_selection_changed_topology"
            if changed
            else (
                "heldout_selection_changed_design"
                if changed_design
                else (
                    "heldout_selected_not_last_round_champion"
                    if not matches_last_round
                    else (
                        "converged_by_repeat_consensus"
                        if base_converged
                        else "budget_limited_no_consensus"
                    )
                )
            )
        )
    return CaseOptimizationResult(
        case_id=case_id,
        final_design=final,
        confirmed_shortlist=confirmed,
        search_archive=cumulative_archive,
        convergence=convergence_table,
        allocations=pd.concat(allocations, ignore_index=True),
        cma_trace=(
            pd.concat(cma_traces, ignore_index=True) if cma_traces else pd.DataFrame()
        ),
        local_trace=(
            pd.concat(local_traces, ignore_index=True)
            if local_traces
            else pd.DataFrame()
        ),
        run_summary=(
            pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
        ),
    )


def run_case_search(
    evaluator: BWBEvaluator,
    mission: Mapping[str, Any],
    topology_prior: pd.DataFrame,
    config: OptimizationConfig,
    seed_source: Any,
    progress_callback: Optional[Any] = None,
    checkpoint_directory: Optional[str | Path] = None,
) -> CaseOptimizationResult:
    """Public high-level alias with an explicit search-oriented name."""

    return optimize_case(
        evaluator=evaluator,
        mission=mission,
        topology_prior=topology_prior,
        config=config,
        seed_source=seed_source,
        progress_callback=progress_callback,
        checkpoint_directory=checkpoint_directory,
    )


def optimize_all_cases(
    evaluator: BWBEvaluator,
    missions: Sequence[Mapping[str, Any]],
    topology_prior: pd.DataFrame,
    config: OptimizationConfig,
    seed_source: Any,
    progress_callback: Optional[Any] = None,
    checkpoint_directory: Optional[str | Path] = None,
) -> MultiCaseOptimizationResult:
    results: Dict[int, CaseOptimizationResult] = {}
    for mission in missions:
        result = optimize_case(
            evaluator,
            mission,
            topology_prior,
            config,
            seed_source,
            progress_callback=progress_callback,
            checkpoint_directory=checkpoint_directory,
        )
        results[result.case_id] = result
        if checkpoint_directory is not None:
            result.save(Path(checkpoint_directory))
    return MultiCaseOptimizationResult(results)
