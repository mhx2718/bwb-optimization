#!/usr/bin/env python3
"""Build the public notebook deterministically without requiring nbformat."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "01_reproducible_bwb_optimization.ipynb"


def _source(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def _id(kind: str, text: str) -> str:
    return hashlib.sha256((kind + "\0" + text).encode("utf-8")).hexdigest()[:16]


def markdown(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": _id("markdown", text),
        "metadata": {},
        "source": _source(text),
    }


def code(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _id("code", text),
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


CELLS = [
    markdown(
        r"""
# Reproducible topology-aware BWB optimization

This notebook is the readable orchestration layer for the public pipeline.
Implementation details live in `src/bwb_pipeline/`; the important model,
constraint, budget, optimization, and reporting calls remain visible here.

Scientific contract:

- one public `MASTER_SEED`, with stable named child streams recorded in the manifest;
- forward model: exactly 21 geometry/structure inputs → weight, payload, fuel;
- stress classifier: all 24 inputs, class 1 means stress ≤ 335 MPa;
- nominal calibrated probability ≥ 0.90;
- median probability ≥ 0.80 at both configured perturbation levels;
- any L/D warning, error, non-finite result, or non-positive CD is a hard rejection;
- three independently seeded CMA-ES trajectories, multi-round warm/cold search, then projected-gradient proposals;
- one held-out confirmation bank; no confirmation feedback into search;
- exactly one final design per mission case.

The reported Pareto set is an **empirical nondominated archive**, not a proof of
the complete Pareto front or global optimum.
"""
    ),
    markdown(
        """
## 0. Start the kernel reproducibly

`PYTHONHASHSEED` must be set before Python starts. Launch Jupyter from the
repository root with the same process-level variables used by
`scripts/run_reproducible.py`. The code below sets safe defaults for libraries
that have not yet initialized, but it cannot retroactively change Python's hash
seed.
"""
    ),
    code(
        """
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "configs").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
if not (PROJECT_ROOT / "src" / "bwb_pipeline").is_dir():
    raise RuntimeError("Start this notebook from the repository root or notebooks/.")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print("Project root:", PROJECT_ROOT)
print("PYTHONHASHSEED at process start:", os.environ.get("PYTHONHASHSEED"))
"""
    ),
    markdown(
        """
## 1. Load configuration and create all named random streams

`MASTER_SEED` is the only user-controlled random input. SHA-256 namespaces
derive independent streams, so the three CMA repeats are reproducible without
being artificially correlated.
"""
    ),
    code(
        """
import json
import numpy as np
import pandas as pd
import yaml

from bwb_pipeline.config import pipeline_config_from_mapping
from bwb_pipeline.reproducibility import SeedRegistry, set_global_determinism

CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
raw_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
config = pipeline_config_from_mapping(raw_config)
MASTER_SEED = config.master_seed
seedbook = SeedRegistry(MASTER_SEED)
determinism_report = set_global_determinism(
    seedbook.derive("pipeline/global", upper_bound=2**32),
    strict=config.determinism.strict,
    torch_num_threads=config.determinism.torch_num_threads,
)
display(pd.Series(determinism_report, name="value"))
"""
    ),
    markdown(
        """
## 2. Authoritative schema and leakage-resistant 70/15/15 split

The split unit is the exact 21D physical design, not a CSV row. Therefore the
same geometry/structure observed under multiple flying conditions cannot leak
across train, validation, and test. Forward targets are checked for invariance
within each 21D group before a flight-free model is allowed.
"""
    ),
    code(
        """
from bwb_pipeline.data import prepare_data
from bwb_pipeline.manifest import hash_existing_files, hash_mapping
from bwb_pipeline.pipeline import guard_run_identity, project_source_files
from bwb_pipeline.schema import (
    ALL_INPUT_COLUMNS,
    ALL_TARGET_COLUMNS,
    CONTINUOUS_DESIGN_COLUMNS,
    DESIGN_COLUMNS,
    FLIGHT_COLUMNS,
    FORWARD_TARGET_COLUMNS,
    STRESS_TARGET_COLUMN,
    TOPOLOGY_COLUMNS,
)

OUTPUT_ROOT = PROJECT_ROOT / config.project.output_root
OPTIMIZATION_ROOT = PROJECT_ROOT / raw_config["optimization"]["output_dir"]
SOURCE_FILES = project_source_files(PROJECT_ROOT)
source_hashes = hash_existing_files(SOURCE_FILES, base_directory=PROJECT_ROOT)
run_identity = {
    "schema": "bwb-run-identity-v2",
    "master_seed": MASTER_SEED,
    "config_sha256": hash_mapping(raw_config),
    "source_tree_sha256": hash_mapping(source_hashes),
}
guard_run_identity(OUTPUT_ROOT, run_identity)
guard_run_identity(OPTIMIZATION_ROOT, run_identity)
bundle = prepare_data(
    PROJECT_ROOT / config.project.data_path,
    config.data,
    seedbook,
    split_manifest_path=OUTPUT_ROOT / config.data.split_manifest,
)
BASE_INPUT_FILES = [CONFIG_PATH, bundle.source_path, *SOURCE_FILES]
BASE_INPUT_SNAPSHOT = hash_existing_files(
    BASE_INPUT_FILES, base_directory=PROJECT_ROOT
)

assert len(DESIGN_COLUMNS) == 21
assert len(ALL_INPUT_COLUMNS) == 24
assert set(FLIGHT_COLUMNS).isdisjoint(DESIGN_COLUMNS)

split_report = pd.DataFrame({
    "unique_designs": bundle.split_manifest.group_counts,
    "rows": bundle.split_manifest.row_counts,
    "feasible_row_fraction": bundle.split_manifest.feasible_row_fractions,
})
display(split_report)
print("Dataset fingerprint:", bundle.dataset_fingerprint)
print("Split fingerprint:  ", bundle.split_manifest.split_fingerprint)

# Preflight the user-requested topology support rule before expensive training.
from bwb_pipeline.optimization import OptimizationConfig
from bwb_pipeline.topology_budget import build_empirical_topology_prior

missions = [dict(case) for case in raw_config["test_cases"]]
opt_values = raw_config["optimization"]
opt_config = OptimizationConfig.from_mapping(opt_values)
assert opt_values["nominal_probability_threshold"] == config.stress_classifier.probability_threshold
assert opt_values["robust_probability_threshold"] == config.stress_classifier.noise_median_probability_threshold
topology_prior = build_empirical_topology_prior(
    bundle.rows,
    missions,
    stress_limit_mpa=config.stress_classifier.stress_limit_mpa,
    minimum_feasible_unique_designs=opt_values[
        "minimum_feasible_unique_designs_per_topology"
    ],
    split=opt_values["topology_prior_split"],
    eligibility_split=opt_values.get("topology_eligibility_split"),
)
topology_prior.to_csv(OUTPUT_ROOT / "topology_empirical_prior.csv", index=False)
display(topology_prior.groupby("case_id").size().rename("eligible_topologies"))
"""
    ),
    markdown(
        """
## 3. Train or load the 21→3 differentiable forward model

The residual MLP receives **only** `DESIGN_COLUMNS`. It uses AdamW, a validation
plateau scheduler, early stopping, and train-only input/target scaling. The
defaults are 2000 epochs, patience 300, LR patience 120, learning rate 1e-3,
and logging every 25 epochs. If an exact-compatible artifact exists it is
loaded; an incompatible old 24-input checkpoint is rejected.
"""
    ),
    code(
        """
from bwb_pipeline.models import ResidualForwardNet, train_or_load_forward

forward = train_or_load_forward(
    bundle,
    config.forward_model,
    seedbook,
    artifact_dir=PROJECT_ROOT / config.forward_model.output_dir,
    device=config.determinism.device,
)
assert tuple(forward.feature_names) == tuple(DESIGN_COLUMNS)

forward_metrics_path = PROJECT_ROOT / config.forward_model.output_dir / "metrics.json"
forward_metrics = json.loads(forward_metrics_path.read_text(encoding="utf-8"))
display(pd.DataFrame(forward_metrics["rows"]))
print("Forward artifact:", forward.artifact_id)
"""
    ),
    markdown(
        """
## 4. Train or load the 24-input calibrated stress classifier

A deterministic CPU Ordered CatBoost model is selected by group-aware CV
inside the training partition. Validation logits fit a two-parameter Platt
calibrator. The untouched test report includes log loss, ROC-AUC, PR-AUC,
Brier/ECE, and threshold diagnostics at 0.5, 0.8, and 0.9. Optimization always
uses calibrated `P(stress ≤ 335 MPa)` and keeps the 0.90 threshold fixed.
"""
    ),
    code(
        """
from bwb_pipeline.models import train_or_load_stress_classifier

stress = train_or_load_stress_classifier(
    bundle,
    config.stress_classifier,
    seedbook,
    artifact_dir=PROJECT_ROOT / config.stress_classifier.output_dir,
)
assert tuple(stress.feature_names) == tuple(ALL_INPUT_COLUMNS)

stress_metrics_path = PROJECT_ROOT / config.stress_classifier.output_dir / "metrics.json"
stress_metrics = json.loads(stress_metrics_path.read_text(encoding="utf-8"))
display(pd.json_normalize(stress_metrics["splits"], sep="."))
display(pd.DataFrame(stress_metrics["splits"]["test"]["thresholds"]).T)
print("Stress artifact:", stress.artifact_id)

MODEL_DIRECTORIES = (
    PROJECT_ROOT / config.forward_model.output_dir,
    PROJECT_ROOT / config.stress_classifier.output_dir,
)
MODEL_FILES = [
    path for directory in MODEL_DIRECTORIES
    for path in directory.rglob("*") if path.is_file()
]
MODEL_INPUT_SNAPSHOT = hash_existing_files(
    MODEL_FILES, base_directory=PROJECT_ROOT
)
"""
    ),
    markdown(
        """
## 5. Load the official L/D model and verify its domain flag

The adapter preserves `LD`, `CL`, `CD`, and every warning. It fails closed on
exceptions, missing keys, non-finite values, `CD <= 0`, or any warning. This is
different from merely clipping the 21 design variables to their official
bounds: the L/D surrogate has its own validity domain.
"""
    ),
    code(
        """
from bwb_pipeline.ld_adapter import REQUIRED_LD_FILES, make_ld_adapter

ld_values = raw_config["ld_model"]
if ld_values.get("reject_any_warning") is not True:
    raise ValueError("Published optimization requires reject_any_warning=true.")
if ld_values.get("require_finite_ld_cl_cd") is not True:
    raise ValueError("Published optimization requires require_finite_ld_cl_cd=true.")
LD_SOURCE_FILES = [
    PROJECT_ROOT / config.project.ld_model_dir / name
    for name in REQUIRED_LD_FILES
]
LD_INPUT_SNAPSHOT = hash_existing_files(
    LD_SOURCE_FILES, base_directory=PROJECT_ROOT
)
ld_adapter = make_ld_adapter(
    PROJECT_ROOT / config.project.ld_model_dir,
    cache=True,
    max_cache_entries=ld_values.get("cache_max_entries", 50000),
)
probe = bundle.structural_designs.loc[:, DESIGN_COLUMNS].head(1)
ld_probe = ld_adapter.predict_many(probe, missions[0])
display(ld_probe)
print("Probe accepted by L/D domain gate:", bool(ld_probe.iloc[0]["ld_in_domain"]))
"""
    ),
    markdown(
        """
## 6. Rank supported topologies and allocate a gamma-shaped budget

Eligibility follows the predeclared all-dataset support rule (>20 unique
stress-feasible designs), so it is explicitly transductive. Rank order uses
TRAIN W/P/F labels only and gives L/D full credit (zero shortfall). A monotone
gamma-rank kernel concentrates budget at the head; a uniform floor preserves
exploration, and largest-remainder allocation conserves the exact integer
evaluation budget.
"""
    ),
    code(
        """
from bwb_pipeline.topology_budget import (
    allocate_generation_blocks,
    build_round_weights,
)

display(topology_prior.groupby("case_id", sort=True).head(10))

first_case_id = int(missions[0]["case_id"])
case_one_prior = topology_prior.loc[topology_prior["case_id"] == first_case_id]
round_zero_weights = build_round_weights(
    case_one_prior,
    prior_weight=opt_config.rounds[0].prior_weight,
    gamma_shape=opt_config.gamma_shape,
    gamma_scale_fraction=opt_config.gamma_scale_fraction,
    uniform_exploration_fraction=opt_config.uniform_exploration_fraction,
)
round_zero_allocation = allocate_generation_blocks(
    round_zero_weights,
    total_evaluations=opt_config.rounds[0].total_evaluations_per_case,
    repeats=opt_config.repeats,
    population=opt_config.cma_population,
    minimum_generations_per_repeat=opt_config.minimum_generations_per_topology_repeat,
)
display(round_zero_allocation.head(15))
assert round_zero_allocation["allocated_evaluations"].sum() == opt_config.rounds[0].total_evaluations_per_case
"""
    ),
    markdown(
        """
## 7. Build the single authoritative evaluator

All optimizers call this evaluator. Only the 18 continuous variables receive
antithetic Sobol perturbations; the three topology counts and mission remain
fixed. Every design sees the same search bank. A separately seeded held-out
bank is loaded only after all rounds finish.
"""
    ),
    code(
        """
from bwb_pipeline.evaluator import (
    BWBEvaluator,
    EvaluationThresholds,
    make_antithetic_sobol_noise,
)
from bwb_pipeline.schema import CONTINUOUS_DESIGN_COLUMNS

search_bank = make_antithetic_sobol_noise(
    opt_values["search_noise_samples"],
    len(CONTINUOUS_DESIGN_COLUMNS),
    seedbook.derive("probability_noise/search"),
)
confirmation_bank = make_antithetic_sobol_noise(
    opt_values["confirmation_noise_samples"],
    len(CONTINUOUS_DESIGN_COLUMNS),
    seedbook.derive("probability_noise/confirmation"),
)
evaluator = BWBEvaluator(
    forward,
    stress,
    ld_adapter,
    search_noise_bank=search_bank,
    confirmation_noise_bank=confirmation_bank,
    noise_fractions=opt_values["noise_fractions_of_bound_width"],
    thresholds=EvaluationThresholds(
        nominal_probability=opt_values["nominal_probability_threshold"],
        robust_probability=opt_values["robust_probability_threshold"],
    ),
)
display(evaluator.evaluate(probe, missions[0], bank_name="search"))
"""
    ),
    markdown(
        """
## 8. Multi-round active CMA-ES + projected AdamW refinement

Each topology is optimized in 18D continuous unit space. Three independently
seeded CMA trajectories run per topology. Later rounds warm-start two repeats from
same-topology/global incumbents and retain one cold repeat. CMA receives a
finite, non-overlapping feasibility tier, never a soft mixture that allows a
low objective to buy a hard-constraint violation.

Projected AdamW differentiates only the 21-input forward model's W/P/F terms.
CatBoost and L/D are not differentiable; every exact checkpoint is accepted
only after the authoritative evaluator confirms all hard gates and an improved
official score.
"""
    ),
    code(
        """
from bwb_pipeline.optimization import optimize_all_cases

optimization_result = optimize_all_cases(
    evaluator,
    missions,
    topology_prior,
    opt_config,
    seedbook,
    checkpoint_directory=OPTIMIZATION_ROOT,
    progress_callback=lambda row: print(
        f"case={row['case_id']} round={row['round']} "
        f"topology=({row['# of Ribs']},{row['# of Fuselage Ribs']},{row['# of Fuselage Spars']}) "
        f"repeat={row['repeat']} mode={row['start_mode']} "
        f"evals={row['actual_evaluations']}"
    ),
)
optimization_result.save(OPTIMIZATION_ROOT)
"""
    ),
    markdown(
        """
## 9. Convergence diagnostics and one confirmed design per case

The fixed budget always runs to completion. Diagnostics report round-to-round
change, three-repeat topology agreement, accepted-loss spread, feasibility
rate, and warm/cold provenance. A held-out bank confirms the frozen shortlist;
if no candidate passes, production mode raises instead of silently promoting a
least-infeasible row.
"""
    ),
    code(
        """
convergence = pd.concat(
    [optimization_result.cases[key].convergence for key in sorted(optimization_result.cases)],
    ignore_index=True,
)
run_summary = pd.concat(
    [optimization_result.cases[key].run_summary for key in sorted(optimization_result.cases)],
    ignore_index=True,
)
display(convergence)
display(run_summary.groupby(["case_id", "round", "start_mode"], sort=True)[
    ["planned_evaluations", "actual_evaluations"]
].sum())

final_designs = optimization_result.final_designs
assert len(final_designs) == len(missions)
assert final_designs["case_id"].nunique() == len(missions)
assert final_designs["hard_accepted"].all()
display(final_designs)
"""
    ),
    markdown(
        """
## 10. Empirical nondominated archives

Each case's bounded held-out-confirmed shortlist is nondominated-sorted on mass,
L/D shortfall, payload shortfall, and fuel shortfall. This visualization is an
empirical archive of retained generated candidates, not a claim of exhaustive
Pareto coverage or a record of every dominated CMA evaluation.
"""
    ),
    code(
        """
from bwb_pipeline.diagnostics import empirical_pareto_archive
from bwb_pipeline.visualization import plot_empirical_pareto

ANALYSIS_ROOT = OUTPUT_ROOT / "analysis"
ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
pareto_tables = {}
for case_id, case_result in sorted(optimization_result.cases.items()):
    front = empirical_pareto_archive(case_result.confirmed_shortlist)
    front["case_id"] = case_id
    pareto_tables[case_id] = front
    front.to_csv(ANALYSIS_ROOT / f"case_{case_id}_empirical_pareto.csv", index=False)
    figure = plot_empirical_pareto(front, selected=case_result.final_design)
    figure.savefig(ANALYSIS_ROOT / f"case_{case_id}_empirical_pareto.png", dpi=220, bbox_inches="tight")
    display(figure)
"""
    ),
    markdown(
        """
## 11. Duplicate-aware LOO k-NN support

LOO distances are computed after collapsing exact 21D duplicates. Otherwise a
design repeated at several flying conditions would have an artificial zero
nearest-neighbor distance. Official bound widths scale every dimension. Final
designs are queried against unique **training** designs, while the requested
all-dataset LOO table is saved separately.
"""
    ),
    code(
        """
from bwb_pipeline.diagnostics import (
    compute_loo_knn_diagnostics,
    compute_query_knn_support,
)
from bwb_pipeline.visualization import plot_knn_support_ecdf

k = int(raw_config["analysis"]["knn_k"])
loo = compute_loo_knn_diagnostics(bundle.rows, k=k)
loo.unique_designs.to_csv(ANALYSIS_ROOT / "dataset_unique_design_loo_knn.csv", index=False)
loo.dataset_rows.to_csv(ANALYSIS_ROOT / "dataset_rows_with_loo_knn.csv", index=False)

training_rows = bundle.rows.query("split == 'train'")
training_loo = compute_loo_knn_diagnostics(training_rows, k=k)
training_loo.unique_designs.to_csv(ANALYSIS_ROOT / "training_unique_design_loo_knn.csv", index=False)
optimized_support = compute_query_knn_support(
    training_rows,
    final_designs,
    k=k,
    reference_loo=training_loo.unique_designs,
)
optimized_support.to_csv(ANALYSIS_ROOT / "optimized_design_knn_support.csv", index=False)
display(optimized_support[[
    "case_id", "reference_nn1_distance", "reference_knn_mean_distance",
    "reference_loo_percentile", "support_band", "topology_seen",
]])
figure = plot_knn_support_ecdf(training_loo.unique_designs, optimized_support)
figure.savefig(ANALYSIS_ROOT / "knn_distance_ecdf.png", dpi=220, bbox_inches="tight")
display(figure)
"""
    ),
    markdown(
        """
## 12. Design pairplots

Unique dataset designs are gray, accepted generated designs are colored by
case, and the selected champions are stars. The diagonal uses deterministic
histograms; discrete topology axes receive no random jitter. Plot inputs are
saved so every figure can be reproduced exactly.
"""
    ),
    code(
        """
from bwb_pipeline.visualization import (
    plot_grouped_design_pairplots,
    prepare_pairplot_data,
)

accepted_generated = pd.concat([
    result.confirmed_shortlist.loc[result.confirmed_shortlist["hard_accepted"]]
    for result in optimization_result.cases.values()
], ignore_index=True)
pairplot_data = prepare_pairplot_data(
    bundle.rows,
    accepted_generated,
    selected=final_designs,
    maximum_dataset_rows=raw_config["analysis"]["pairplot_max_dataset_rows"],
    sample_seed=seedbook.derive("plots/pairplot_sample"),
)
pairplot_data.dataset_scatter.to_csv(ANALYSIS_ROOT / "pairplot_dataset_sample.csv", index=False)
pairplot_figures = plot_grouped_design_pairplots(pairplot_data)
for name, figure in pairplot_figures.items():
    figure.savefig(ANALYSIS_ROOT / f"design_pairplot_{name}.png", dpi=180, bbox_inches="tight")
    display(figure)
"""
    ),
    markdown(
        """
## 13. Freeze the reproducibility manifest

The final manifest records data/config/LD/model/source hashes, exact schema and
split, named child seeds (plus trace-recorded internal CMA descendants),
software/hardware state, determinism flags, stage budgets, and output hashes.
Non-finite JSON values are converted to `null`; the writer uses strict
`allow_nan=False` semantics.
"""
    ),
    code(
        """
from bwb_pipeline.manifest import write_run_manifest

current_source_files = project_source_files(PROJECT_ROOT)
current_base_files = [CONFIG_PATH, bundle.source_path, *current_source_files]
if hash_existing_files(current_base_files, base_directory=PROJECT_ROOT) != BASE_INPUT_SNAPSHOT:
    raise RuntimeError("Source/config/data changed during notebook execution.")
if hash_existing_files(LD_SOURCE_FILES, base_directory=PROJECT_ROOT) != LD_INPUT_SNAPSHOT:
    raise RuntimeError("Official L/D files changed during notebook execution.")
current_model_files = [
    path for directory in MODEL_DIRECTORIES
    for path in directory.rglob("*") if path.is_file()
]
if hash_existing_files(current_model_files, base_directory=PROJECT_ROOT) != MODEL_INPUT_SNAPSHOT:
    raise RuntimeError("Model artifacts changed during notebook execution.")
input_files = [*current_base_files, *LD_SOURCE_FILES, *current_model_files]
output_files = [
    path for directory in (OUTPUT_ROOT, OPTIMIZATION_ROOT, *MODEL_DIRECTORIES)
    for path in directory.rglob("*")
    if path.is_file() and path.name != "run_manifest.json"
]
manifest = write_run_manifest(
    OUTPUT_ROOT / "run_manifest.json",
    project_root=PROJECT_ROOT,
    config=raw_config,
    seed_manifest=seedbook.manifest(),
    schema={
        "design_columns": DESIGN_COLUMNS,
        "continuous_design_columns": CONTINUOUS_DESIGN_COLUMNS,
        "topology_columns": TOPOLOGY_COLUMNS,
        "flight_columns": FLIGHT_COLUMNS,
        "forward_targets": FORWARD_TARGET_COLUMNS,
        "stress_target": STRESS_TARGET_COLUMN,
        "all_targets": ALL_TARGET_COLUMNS,
    },
    input_files=input_files,
    output_files=output_files,
    stages={
        "dataset_fingerprint": bundle.dataset_fingerprint,
        "split_fingerprint": bundle.split_manifest.split_fingerprint,
        "forward_artifact_id": forward.artifact_id,
        "stress_artifact_id": stress.artifact_id,
        "optimization_cases": sorted(optimization_result.cases),
        "optimization_rounds": len(opt_config.rounds),
        "optimization_repeats": opt_config.repeats,
    },
    execution={
        "mode": "full_notebook",
        "force_retrain": bool(
            config.forward_model.force_retrain
            or config.stress_classifier.force_retrain
        ),
        "models_only": False,
        "source_tree_sha256": run_identity["source_tree_sha256"],
    },
)
print("Manifest:", OUTPUT_ROOT / "run_manifest.json")
print("Manifest payload SHA-256:", manifest["manifest_payload_sha256"])
print("Manifest file SHA-256:", manifest["manifest_file_sha256"])
print("Recorded child seeds:", len(seedbook.manifest()["derived_seeds"]))
"""
    ),
    markdown(
        """
## Interpretation guardrails

- `converged_by_repeat_consensus` is numerical evidence under the stated
  budget, not a mathematical proof of the global optimum.
- An audit/confirmation failure never promotes a runner-up silently.
- L/D warnings are part of the feasibility result, not cosmetic log messages.
- Topology eligibility is a predeclared transductive all-data support filter;
  ranking uses TRAIN W/P/F only. The uniform floor and cold trajectory prevent
  the prior from becoming a hard hand-authored topology choice.
- For bitwise reproduction, preserve the manifest's OS, Python/package builds,
  CPU/GPU, driver, thread settings, and input artifact hashes.
"""
    ),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "bwb_pipeline": {
                "algorithm": "topology-budgeted-active-CMA-ES-plus-projected-AdamW",
                "master_seed_contract": "sha256-namespaced-v1",
                "schema_version": "bwb-reproducible-run-v1",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(TARGET)


if __name__ == "__main__":
    main()
