"""Differentiable W/P/F proposal + authoritative certification refinement.

Projected AdamW differentiates the trained 21-D PyTorch forward model exactly.
The CatBoost stress classifier and official L/D model deliberately remain
outside the gradient graph: every ``exact_check_every`` steps the proposal is
checked by :class:`BWBEvaluator` and is accepted only when its authoritative
finite merit improves.  This proposal--certification split never pretends that
a tree model or warning flag has a useful derivative, and it can never replace
the incumbent by a worse or infeasible candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .cma_search import _child_seed
from .evaluator import BWBEvaluator, TOPOLOGY_COLUMNS, rank_evaluations


@dataclass(frozen=True)
class LocalRefineConfig:
    enabled: bool = True
    starts_per_case_per_round: int = 12
    steps: int = 300
    learning_rate: float = 0.02
    weight_decay: float = 1.0e-5
    trust_radius_fraction: float = 0.05
    exact_check_every: int = 5

    def __post_init__(self) -> None:
        if self.starts_per_case_per_round < 1 or self.steps < 1:
            raise ValueError("Local-refinement starts/steps must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("Invalid Adam learning-rate/weight-decay settings.")
        if not (0.0 < self.trust_radius_fraction <= 1.0):
            raise ValueError("Trust radius must lie in (0, 1].")
        if self.exact_check_every < 1:
            raise ValueError("exact_check_every must be positive.")


@dataclass
class LocalRefineResult:
    champion: pd.DataFrame
    archive: pd.DataFrame
    trace: pd.DataFrame
    exact_evaluations: int


class DifferentiableForwardProtocolError(TypeError):
    """The configured forward artifact cannot provide exact autograd proposals."""


def _unit_from_row(row: pd.Series, evaluator: BWBEvaluator) -> np.ndarray:
    names = [f"unit__{name}" for name in evaluator.design_space.continuous_columns]
    if all(name in row.index for name in names):
        return row.loc[names].to_numpy(dtype=np.float64)
    return evaluator.design_space.frame_to_unit(pd.DataFrame([row]))[0]


def _evaluate_one(
    evaluator: BWBEvaluator,
    unit: np.ndarray,
    topology: Sequence[int],
    mission: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    result = evaluator.evaluate_unit(
        np.asarray(unit, dtype=np.float64)[None, :],
        topology,
        mission,
        bank_name="search",
    )
    for key, value in metadata.items():
        result[key] = value
    for j, name in enumerate(evaluator.design_space.continuous_columns):
        result[f"unit__{name}"] = float(unit[j])
    return result


def _differentiable_wpf_loss(
    evaluator: BWBEvaluator,
    unit_parameter: Any,
    topology: Sequence[int],
    mission: Mapping[str, Any],
) -> Any:
    """Official W/P/F score terms through ``ForwardPredictor.predict_tensor``.

    L/D is intentionally absent here because the official aerodynamic adapter
    is NumPy/JSON based.  Its contribution and every hard constraint are still
    enforced by the authoritative acceptance check.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - integration dependency
        raise ImportError("Gradient refinement requires PyTorch.") from exc

    predictor = evaluator.forward_model
    if not hasattr(predictor, "predict_tensor"):
        raise DifferentiableForwardProtocolError(
            "Forward predictor must expose differentiable "
            "predict_tensor(values, input_is_scaled=False)."
        )
    lower = torch.as_tensor(
        evaluator.design_space.continuous_lower,
        dtype=unit_parameter.dtype,
        device=unit_parameter.device,
    )
    width = torch.as_tensor(
        evaluator.design_space.continuous_width,
        dtype=unit_parameter.dtype,
        device=unit_parameter.device,
    )
    continuous = lower + unit_parameter * width
    topology_map = {
        name: float(value)
        for name, value in zip(evaluator.design_space.topology_columns, topology)
    }
    continuous_map = {
        name: continuous[index]
        for index, name in enumerate(evaluator.design_space.continuous_columns)
    }
    feature_names = tuple(
        getattr(predictor, "feature_names", evaluator.design_space.design_columns)
    )
    features = []
    for name in feature_names:
        if name in continuous_map:
            features.append(continuous_map[name])
        elif name in topology_map:
            features.append(
                torch.as_tensor(
                    topology_map[name],
                    dtype=unit_parameter.dtype,
                    device=unit_parameter.device,
                )
            )
        else:
            raise DifferentiableForwardProtocolError(
                f"Differentiable forward feature {name!r} is not a 21-D design input."
            )
    design_tensor = torch.stack(features, dim=0)[None, :]
    try:
        prediction = predictor.predict_tensor(
            design_tensor, input_is_scaled=False
        )
    except TypeError:
        prediction = predictor.predict_tensor(design_tensor)
    if not torch.is_tensor(prediction) or prediction.shape != (1, 3):
        raise DifferentiableForwardProtocolError(
            "predict_tensor must return a differentiable torch Tensor of shape (N, 3)."
        )
    scales = torch.as_tensor(
        evaluator.forward_output_scales,
        dtype=prediction.dtype,
        device=prediction.device,
    )
    physical = prediction * scales[None, :]
    weight, payload, fuel = physical[0]
    payload_target = torch.as_tensor(
        float(mission["Payload_target_m3"]),
        dtype=prediction.dtype,
        device=prediction.device,
    )
    fuel_target = torch.as_tensor(
        float(mission["Fuel_target_m3"]),
        dtype=prediction.dtype,
        device=prediction.device,
    )
    return (
        0.4 * weight / 50.0
        + 0.2 * torch.relu((payload_target - payload) / payload_target)
        + 0.2 * torch.relu((fuel_target - fuel) / fuel_target)
    )


def refine_one_start(
    evaluator: BWBEvaluator,
    mission: Mapping[str, Any],
    start: pd.Series,
    config: LocalRefineConfig,
    seed: int,
    round_index: int,
    start_rank: int,
) -> LocalRefineResult:
    """Refine one fixed-topology start and retain a monotone champion."""

    topology = tuple(int(start[name]) for name in TOPOLOGY_COLUMNS)
    x_start = _unit_from_row(start, evaluator)
    lower = np.maximum(0.0, x_start - config.trust_radius_fraction)
    upper = np.minimum(1.0, x_start + config.trust_radius_fraction)
    metadata = {
        "case_id": int(mission["case_id"]),
        "round": int(round_index),
        "repeat": -1,
        "generation": -1,
        "optimizer_seed": int(seed),
        "start_mode": "local_refine",
        "local_start_rank": int(start_rank),
    }
    champion = _evaluate_one(
        evaluator, x_start, topology, mission, dict(metadata, local_step=0)
    )
    exact_evaluations = 1
    archive_rows = [champion]
    trace_rows = [
        {
            **metadata,
            "local_step": 0,
            "accepted_update": True,
            "search_merit": float(champion.iloc[0]["search_merit"]),
            "official_loss": float(champion.iloc[0]["official_loss"]),
            "hard_accepted": bool(champion.iloc[0]["hard_accepted"]),
            "step_norm": 0.0,
        }
    ]

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - integration dependency
        raise ImportError("Gradient refinement requires PyTorch.") from exc
    # The child seed is recorded even though the deterministic full-batch
    # forward proposal currently has no random draw.  This keeps the module
    # reproducible if a future approved forward layer consumes RNG state.
    torch.manual_seed(int(seed))
    parameter = torch.nn.Parameter(
        torch.as_tensor(x_start, dtype=torch.float32).clone()
    )
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    lower_t = torch.as_tensor(lower, dtype=parameter.dtype)
    upper_t = torch.as_tensor(upper, dtype=parameter.dtype)
    champion_merit = float(champion.iloc[0]["search_merit"])
    champion_feasible = bool(champion.iloc[0]["hard_accepted"])
    champion_loss = float(champion.iloc[0]["official_loss"])

    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        proposal_loss = _differentiable_wpf_loss(
            evaluator, parameter, topology, mission
        )
        if not torch.isfinite(proposal_loss):
            raise FloatingPointError("Differentiable W/P/F proposal loss is NaN/Inf.")
        gradient = torch.autograd.grad(
            proposal_loss, parameter, retain_graph=False, create_graph=False
        )[0]
        parameter.grad = gradient
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("Differentiable W/P/F gradient is NaN/Inf.")
        torch.nn.utils.clip_grad_norm_([parameter], max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            parameter.copy_(torch.maximum(torch.minimum(parameter, upper_t), lower_t))
        proposal = parameter.detach().cpu().numpy().copy()
        if step % config.exact_check_every != 0 and step != config.steps:
            continue

        incumbent_unit = _unit_from_row(champion.iloc[0], evaluator)
        proposed_step_norm = float(np.linalg.norm(proposal - incumbent_unit))
        evaluated = _evaluate_one(
            evaluator,
            proposal,
            topology,
            mission,
            dict(metadata, local_step=int(step)),
        )
        exact_evaluations += 1
        archive_rows.append(evaluated)
        candidate_merit = float(evaluated.iloc[0]["search_merit"])
        candidate_feasible = bool(evaluated.iloc[0]["hard_accepted"])
        candidate_loss = float(evaluated.iloc[0]["official_loss"])
        # Gradient proposals never improve an infeasible incumbent with another
        # infeasible point. The first accepted move must cross every
        # authoritative gate; thereafter the unmodified official objective must
        # decrease strictly.
        accepted = candidate_feasible and (
            (not champion_feasible)
            or candidate_loss < champion_loss - 1.0e-14
        )
        if accepted:
            champion = evaluated
            champion_merit = candidate_merit
            champion_feasible = True
            champion_loss = candidate_loss
        else:
            # Reject against the authoritative evaluator and return to the last
            # monotone champion rather than allowing an infeasible drift.
            with torch.no_grad():
                parameter.copy_(
                    torch.as_tensor(
                        incumbent_unit,
                        dtype=parameter.dtype,
                        device=parameter.device,
                    )
                )
        trace_rows.append(
            {
                **metadata,
                "local_step": int(step),
                "accepted_update": bool(accepted),
                "search_merit": candidate_merit,
                "official_loss": float(evaluated.iloc[0]["official_loss"]),
                "hard_accepted": bool(evaluated.iloc[0]["hard_accepted"]),
                "step_norm": proposed_step_norm,
                "differentiable_wpf_proposal_loss": float(
                    proposal_loss.detach().cpu()
                ),
            }
        )

    archive = pd.concat(archive_rows, ignore_index=True)
    archive = rank_evaluations(archive).drop_duplicates(
        subset=[
            f"unit__{name}" for name in evaluator.design_space.continuous_columns
        ],
        keep="first",
    )
    return LocalRefineResult(
        champion=champion.reset_index(drop=True),
        archive=archive.reset_index(drop=True),
        trace=pd.DataFrame(trace_rows),
        exact_evaluations=int(exact_evaluations),
    )


def refine_round_champions(
    evaluator: BWBEvaluator,
    mission: Mapping[str, Any],
    round_archive: pd.DataFrame,
    config: LocalRefineConfig,
    seed_source: Any,
    round_index: int,
) -> LocalRefineResult:
    """Refine diverse best starts and combine their monotone archives."""

    if round_archive.empty:
        raise ValueError("Local refinement received an empty CMA archive.")
    ranked = rank_evaluations(round_archive)
    accepted = ranked.loc[ranked["hard_accepted"].to_numpy(dtype=bool)]
    source = accepted if not accepted.empty else ranked
    starts = source.drop_duplicates(subset=list(TOPOLOGY_COLUMNS), keep="first")
    if len(starts) < config.starts_per_case_per_round:
        starts = pd.concat([starts, source], ignore_index=True).drop_duplicates(
            subset=list(evaluator.design_space.design_columns), keep="first"
        )
    starts = starts.head(config.starts_per_case_per_round).reset_index(drop=True)

    champions = []
    archives = []
    traces = []
    total_evaluations = 0
    for rank, row in starts.iterrows():
        topology = tuple(int(row[name]) for name in TOPOLOGY_COLUMNS)
        seed = _child_seed(
            seed_source,
            "local_refine",
            int(mission["case_id"]),
            int(round_index),
            topology,
            int(rank),
        )
        result = refine_one_start(
            evaluator,
            mission,
            row,
            config,
            seed,
            round_index,
            int(rank),
        )
        champions.append(result.champion)
        archives.append(result.archive)
        traces.append(result.trace)
        total_evaluations += result.exact_evaluations
    all_champions = rank_evaluations(pd.concat(champions, ignore_index=True))
    return LocalRefineResult(
        champion=all_champions.head(1).reset_index(drop=True),
        archive=pd.concat(archives, ignore_index=True),
        trace=pd.concat(traces, ignore_index=True),
        exact_evaluations=int(total_evaluations),
    )
