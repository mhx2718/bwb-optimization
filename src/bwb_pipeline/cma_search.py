"""Deterministic ask/tell active CMA-ES islands for fixed BWB topologies."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .evaluator import BWBEvaluator, TOPOLOGY_COLUMNS, rank_evaluations


def stable_child_seed(master_seed: int, *keys: Any) -> int:
    """Order-independent named child seed; never uses Python's salted hash()."""

    payload = json.dumps([int(master_seed), *keys], separators=(",", ":"), default=str)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1) or 1


def _child_seed(seed_source: Any, *keys: Any) -> int:
    if hasattr(seed_source, "derive"):
        return int(seed_source.derive(*keys))
    if hasattr(seed_source, "seed") and callable(seed_source.seed):
        return int(seed_source.seed(*keys))
    if callable(seed_source):
        return int(seed_source(*keys))
    return stable_child_seed(int(seed_source), *keys)


@dataclass(frozen=True)
class CMAConfig:
    population: int = 16
    initial_sigma: float = 0.22
    archive_per_run: int = 64
    repeats: int = 3
    warm_repeats_after_round_zero: int = 2

    def __post_init__(self) -> None:
        if self.population < 4:
            raise ValueError("CMA population must be at least four.")
        if not (0.0 < self.initial_sigma <= 1.0):
            raise ValueError("CMA initial sigma must lie in (0, 1].")
        if self.archive_per_run < 1 or self.repeats < 1:
            raise ValueError("CMA archive/repeat counts must be positive.")
        if not (0 <= self.warm_repeats_after_round_zero < self.repeats):
            raise ValueError("At least one repeat must remain cold after round zero.")


@dataclass
class CMAIslandResult:
    archive: pd.DataFrame
    trace: pd.DataFrame
    actual_evaluations: int
    stop_reasons: Mapping[str, Any]
    final_mean_unit: np.ndarray


def _bounded_archive(
    current: Optional[pd.DataFrame],
    new_rows: pd.DataFrame,
    limit: int,
    continuous_columns: Sequence[str],
) -> pd.DataFrame:
    combined = new_rows.copy() if current is None else pd.concat(
        [current, new_rows], ignore_index=True
    )
    ranked = rank_evaluations(combined)
    if continuous_columns:
        ranked = ranked.drop_duplicates(
            subset=list(continuous_columns), keep="first"
        )
    return ranked.head(int(limit)).reset_index(drop=True)


def run_cma_island(
    evaluator: BWBEvaluator,
    mission: Mapping[str, Any],
    topology: Sequence[int],
    start_unit: np.ndarray,
    generations: int,
    seed: int,
    config: CMAConfig,
    case_id: int,
    round_index: int,
    repeat: int,
    start_mode: str,
) -> CMAIslandResult:
    """Run one fixed-topology CMA island with an exact ask/tell audit trace."""

    try:
        import cma
    except ImportError as exc:  # pragma: no cover - exercised in integration
        raise ImportError(
            "CMA search requires pycma. Install with `python -m pip install cma>=4.4`."
        ) from exc

    generations = int(generations)
    if generations < 1:
        raise ValueError("CMA island needs at least one generation.")
    dimension = len(evaluator.design_space.continuous_columns)
    x0 = np.asarray(start_unit, dtype=np.float64).reshape(-1)
    if x0.shape != (dimension,):
        raise ValueError(f"CMA start has shape {x0.shape}; expected {(dimension,)}.")
    x0 = np.clip(x0, 0.0, 1.0)
    def make_strategy(
        center: np.ndarray,
        sigma: float,
        strategy_seed: int,
        remaining_generations: int,
    ) -> Any:
        options = {
            "seed": int(strategy_seed),
            "bounds": [0.0, 1.0],
            "popsize": int(config.population),
            "CMA_active": True,
            "CMA_diagonal": min(20, int(remaining_generations)),
            "maxiter": int(remaining_generations),
            # The externally allocated generation budget controls normal
            # termination. If a numerical stop still occurs, the island is
            # deterministically restarted for the exact remaining budget.
            "tolfun": 0.0,
            "tolfunhist": 0.0,
            "tolflatfitness": int(remaining_generations) + 1,
            "tolstagnation": int(remaining_generations) + 1,
            "tolx": 1.0e-12,
            "verbose": -9,
            "verb_disp": 0,
            "verb_log": 0,
            "verb_time": False,
        }
        return cma.CMAEvolutionStrategy(
            np.clip(center, 0.0, 1.0).tolist(), float(sigma), options
        )

    strategy_seed = int(seed)
    strategy = make_strategy(
        x0, float(config.initial_sigma), strategy_seed, generations
    )
    archive: Optional[pd.DataFrame] = None
    trace_rows = []
    actual = 0
    best_merit = float("inf")
    best_loss = float("inf")
    stop_reasons: Mapping[str, Any] = {}
    internal_restart_count = 0
    internal_stop_history = []
    for generation in range(generations):
        stop_reasons = strategy.stop()
        if stop_reasons:
            internal_stop_history.append(
                {
                    "before_generation": int(generation),
                    "reasons": dict(stop_reasons),
                }
            )
            internal_restart_count += 1
            if archive is not None and not archive.empty:
                center = archive.iloc[0].loc[
                    [
                        f"unit__{name}"
                        for name in evaluator.design_space.continuous_columns
                    ]
                ].to_numpy(dtype=np.float64)
            else:
                center = np.asarray(getattr(strategy, "mean", x0), dtype=np.float64)
            strategy_seed = stable_child_seed(
                int(seed), "internal_restart", internal_restart_count
            )
            sigma = max(
                0.02,
                min(float(config.initial_sigma), float(strategy.sigma)),
            )
            strategy = make_strategy(
                center,
                sigma,
                strategy_seed,
                generations - generation,
            )
        unit_values = np.asarray(strategy.ask(), dtype=np.float64)
        evaluated = evaluator.evaluate_unit(
            unit_values, topology, mission, bank_name="search"
        )
        evaluated["case_id"] = int(case_id)
        evaluated["round"] = int(round_index)
        evaluated["repeat"] = int(repeat)
        evaluated["generation"] = int(generation)
        evaluated["optimizer_seed"] = int(seed)
        evaluated["start_mode"] = str(start_mode)
        for j, name in enumerate(evaluator.design_space.continuous_columns):
            evaluated[f"unit__{name}"] = unit_values[:, j]
        fitness = evaluated["search_merit"].to_numpy(dtype=np.float64)
        if not np.isfinite(fitness).all():
            raise FloatingPointError("CMA received a non-finite search merit.")
        strategy.tell(unit_values.tolist(), fitness.tolist())
        actual += len(unit_values)
        archive = _bounded_archive(
            archive,
            evaluated,
            config.archive_per_run,
            [f"unit__{name}" for name in evaluator.design_space.continuous_columns],
        )
        generation_best = rank_evaluations(evaluated).iloc[0]
        best_merit = min(best_merit, float(generation_best["search_merit"]))
        accepted = evaluated.loc[evaluated["hard_accepted"].to_numpy(dtype=bool)]
        if not accepted.empty:
            best_loss = min(best_loss, float(accepted["official_loss"].min()))
        axis_ratio = getattr(strategy, "D", np.ones(dimension))
        axis_ratio = float(np.max(axis_ratio) / max(np.min(axis_ratio), 1.0e-300))
        trace_rows.append(
            {
                "case_id": int(case_id),
                "round": int(round_index),
                "repeat": int(repeat),
                "generation": int(generation),
                "optimizer_seed": int(seed),
                "active_strategy_seed": int(strategy_seed),
                "internal_restart_count": int(internal_restart_count),
                "start_mode": str(start_mode),
                "generation_evaluations": int(len(unit_values)),
                "generation_hard_accepted_count": int(
                    evaluated["hard_accepted"].sum()
                ),
                "cumulative_evaluations": int(actual),
                "generation_accepted_fraction": float(
                    evaluated["hard_accepted"].mean()
                ),
                "best_search_merit": float(best_merit),
                "best_accepted_official_loss": (
                    float(best_loss) if np.isfinite(best_loss) else np.nan
                ),
                "cma_sigma": float(strategy.sigma),
                "cma_axis_ratio": axis_ratio,
            }
        )
    if archive is None:
        archive = pd.DataFrame()
    expected_evaluations = int(generations * config.population)
    if actual != expected_evaluations:
        raise AssertionError(
            "CMA island did not consume its exact allocated budget: "
            f"expected {expected_evaluations}, observed {actual}."
        )
    stop_reasons = {
        "internal_restart_history": internal_stop_history,
        "final_strategy_stop": dict(strategy.stop()),
    }
    mean = np.asarray(getattr(strategy, "mean", x0), dtype=np.float64)
    return CMAIslandResult(
        archive=archive,
        trace=pd.DataFrame(trace_rows),
        actual_evaluations=int(actual),
        stop_reasons=stop_reasons,
        final_mean_unit=np.clip(mean, 0.0, 1.0),
    )


def _row_unit(row: pd.Series, evaluator: BWBEvaluator) -> np.ndarray:
    unit_columns = [
        f"unit__{name}" for name in evaluator.design_space.continuous_columns
    ]
    if all(name in row.index for name in unit_columns):
        return row.loc[unit_columns].to_numpy(dtype=np.float64)
    frame = pd.DataFrame([row])
    return evaluator.design_space.frame_to_unit(frame)[0]


def build_repeat_start_plan(
    evaluator: BWBEvaluator,
    topology: Sequence[int],
    round_index: int,
    repeats: int,
    warm_repeats_after_round_zero: int,
    previous_archive: Optional[pd.DataFrame],
    seed_source: Any,
    case_id: int,
) -> Sequence[Tuple[np.ndarray, str, int]]:
    """Use same-topology/global warm starts while always retaining a cold run."""

    topology_tuple = tuple(int(value) for value in topology)
    same_best: Optional[pd.Series] = None
    global_best: Optional[pd.Series] = None
    if previous_archive is not None and not previous_archive.empty:
        ranked = rank_evaluations(previous_archive)
        global_best = ranked.iloc[0]
        mask = np.ones(len(previous_archive), dtype=bool)
        for name, value in zip(TOPOLOGY_COLUMNS, topology_tuple):
            mask &= previous_archive[name].to_numpy(dtype=np.int64) == value
        same = previous_archive.loc[mask]
        if not same.empty:
            same_best = rank_evaluations(same).iloc[0]

    plan = []
    for repeat in range(int(repeats)):
        seed = _child_seed(
            seed_source, "cma", int(case_id), int(round_index), topology_tuple, repeat
        )
        if round_index > 0 and repeat < warm_repeats_after_round_zero:
            if repeat == 0 and same_best is not None:
                start = _row_unit(same_best, evaluator)
                mode = "warm_same_topology"
            elif global_best is not None:
                start = _row_unit(global_best, evaluator)
                mode = "warm_global_transfer"
            elif same_best is not None:
                start = _row_unit(same_best, evaluator)
                mode = "warm_same_topology"
            else:
                start = np.random.default_rng(seed).uniform(
                    0.0, 1.0, len(evaluator.design_space.continuous_columns)
                )
                mode = "cold_uniform"
        else:
            start = np.random.default_rng(seed).uniform(
                0.0, 1.0, len(evaluator.design_space.continuous_columns)
            )
            mode = "cold_uniform"
        plan.append((np.asarray(start, dtype=np.float64), mode, int(seed)))
    return plan


@dataclass
class CMARoundResult:
    archive: pd.DataFrame
    trace: pd.DataFrame
    run_summary: pd.DataFrame


def run_cma_round(
    evaluator: BWBEvaluator,
    mission: Mapping[str, Any],
    allocation: pd.DataFrame,
    round_index: int,
    config: CMAConfig,
    seed_source: Any,
    previous_archive: Optional[pd.DataFrame] = None,
    progress_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> CMARoundResult:
    """Run all allocated topology islands in stable topology/repeat order."""

    required = set(TOPOLOGY_COLUMNS) | {"generations_per_repeat"}
    missing = required.difference(allocation.columns)
    if missing:
        raise KeyError(f"Round allocation lacks {sorted(missing)}")
    case_id = int(mission["case_id"])
    archives = []
    traces = []
    summaries = []
    global_evaluation_offset = 0
    ordered = allocation.sort_values(list(TOPOLOGY_COLUMNS), kind="stable")
    for _, row in ordered.iterrows():
        topology = tuple(int(row[name]) for name in TOPOLOGY_COLUMNS)
        plan = build_repeat_start_plan(
            evaluator,
            topology,
            round_index,
            config.repeats,
            config.warm_repeats_after_round_zero,
            previous_archive,
            seed_source,
            case_id,
        )
        for repeat, (start, mode, seed) in enumerate(plan):
            result = run_cma_island(
                evaluator=evaluator,
                mission=mission,
                topology=topology,
                start_unit=start,
                generations=int(row["generations_per_repeat"]),
                seed=seed,
                config=config,
                case_id=case_id,
                round_index=round_index,
                repeat=repeat,
                start_mode=mode,
            )
            archives.append(result.archive)
            trace = result.trace.copy()
            if not trace.empty:
                trace["case_round_cumulative_evaluations"] = (
                    global_evaluation_offset
                    + trace["cumulative_evaluations"].to_numpy(dtype=np.int64)
                )
            traces.append(trace)
            summaries.append(
                {
                    "case_id": case_id,
                    "round": int(round_index),
                    **dict(zip(TOPOLOGY_COLUMNS, topology)),
                    "repeat": int(repeat),
                    "optimizer_seed": int(seed),
                    "start_mode": mode,
                    "planned_evaluations": int(
                        row["generations_per_repeat"] * config.population
                    ),
                    "actual_evaluations": int(result.actual_evaluations),
                    "stop_reasons": json.dumps(result.stop_reasons, default=str),
                }
            )
            if progress_callback is not None:
                progress_callback(summaries[-1])
            global_evaluation_offset += int(result.actual_evaluations)
    archive = (
        pd.concat(archives, ignore_index=True) if archives else pd.DataFrame()
    )
    trace = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    if not trace.empty:
        observed = trace["best_accepted_official_loss"].to_numpy(dtype=np.float64)
        running = np.minimum.accumulate(np.where(np.isfinite(observed), observed, np.inf))
        trace["case_round_best_accepted_loss_so_far"] = np.where(
            np.isfinite(running), running, np.nan
        )
    return CMARoundResult(archive, trace, pd.DataFrame(summaries))
