# Reproducible topology-aware BWB optimization

This repository is a pipeline for training surrogate models
and optimizing a blended-wing-body (BWB) design. It deliberately separates
proposal models from authoritative acceptance checks and records every random
stream, data split, configuration, and artifact fingerprint.

The final optimizer is a topology-budgeted, multi-round CMA-ES search with a
projected-gradient proposal step. The gradient step uses only the differentiable
21-input forward model; every proposed replacement is accepted only after the
calibrated stress classifier and the external L/D surrogate pass the same hard
checks used by CMA-ES. CatBoost and the L/D code are never treated as
differentiable.

## Scientific contract

- Forward model: exactly 21 geometry/structure inputs; outputs aircraft empty
  weight, payload volume, and fuel volume. `Altitude`, `KCAS`, and `AOA` are
  excluded.
- Stress model: all 24 inputs; class 1 means `Max Hotspot Stress <= 335 MPa`.
- Leakage-resistant split: exact 21D designs, rather than individual rows, are
  assigned to train/validation/test in a 70/15/15 split. Flight-condition rows
  for one design can never cross partitions.
- Nominal stress gate: calibrated feasibility probability `>= 0.90`.
- Local robustness gate: for each configured perturbation size, the median
  calibrated feasibility probability over a fixed common noise bank is
  `>= 0.80`. Only the 18 continuous design variables are perturbed.
- L/D domain gate: an exception, missing/non-finite output, non-positive drag,
  or any warning from `predict_ld` rejects the design. A warned L/D value never
  contributes a full objective score.
- Final selection: one frozen common confirmation bank is applied after search.
  Confirmation never feeds back into another optimization round.

The displayed "Pareto front" is therefore an **empirical nondominated archive**
of evaluated candidates, not a proof of the complete continuous Pareto set.
Likewise, a finite CMA-ES budget cannot prove a global minimum; the convergence
report states whether independently seeded trajectories (including a cold
start) agree or whether the run remains budget-limited.

## Reproducibility model

`master_seed` is the only user-selected seed. Stable SHA-256 namespaces derive
independent child streams for the split, model initialization, data loaders,
CatBoost, each case/round/repeat, Sobol banks, and plots. This avoids the strong
correlations caused by repeatedly resetting every component to the same integer
while keeping the complete run controlled by one value. Rare internal CMA
numerical restarts use deterministic descendants recorded directly in the
per-generation trace.

Strict mode uses deterministic PyTorch algorithms, one CPU thread by default,
fixed-order tables, deterministic tie breaking, and immutable round
checkpoints. `run_manifest.json` records the seed map, exact feature order,
dataset/split/config/code hashes, software and hardware versions, determinism
flags, and output hashes.

Bit-for-bit reproduction still requires the same operating system, Python and
package builds, and hardware. The manifest makes those assumptions explicit.
The dependency files specify supported ranges; after installing, preserve the
exact environment with `python -m pip freeze --all > environment.lock.txt`
alongside the run manifest. A cross-platform range-only install is not claimed
to be bitwise reproducible.

## Execution path

The complete reported workflow is executed through:

```text
bwb_optimization.ipynb
```

This notebook is the single user-facing entry point for data validation,
train-or-load behavior, topology-prior construction, CMA-ES optimization,
AdamW refinement, independent confirmation, final selection, diagnostics,
figures, and the reproducibility manifest. To reproduce the reported results,
start a fresh kernel and run the notebook from top to bottom without changing
the execution order.

The numerical settings are defined by the YAML file selected in the notebook's
configuration cell. The notebook prints the active configuration and output
directories before starting expensive computation.

## Required inputs

Place the following assets relative to the repository root:

```text
bwb_structures_dataset.csv
models/ld_surrogate/
  predict_ld.py
  regressor.py
  flight_conversion.py
  reg_full.json
```

If the paths differ, update `CONFIG_PATH` and the corresponding entries in the
configuration cell of `bwb_optimization.ipynb`. The pipeline fails on missing
files, reordered features, incompatible artifacts, or changed model contracts
rather than silently guessing.

## Installation

Python 3.10 or newer is required; the reference environment uses Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook]"
```

Launch the canonical notebook from the repository root:

```bash
python -m jupyter lab bwb_optimization.ipynb
```

Then use **Restart Kernel and Run All Cells**. The notebook performs the
following stages in order:

1. validate the repository, configuration, data, and L/D assets;
2. create or verify the design-group train/validation/test split;
3. train or load compatible forward and stress-model artifacts;
4. construct the empirical topology prior;
5. run the topology-budgeted CMA-ES and AdamW search;
6. reevaluate the final shortlist with the independent confirmation bank;
7. select one hard-feasible design per mission;
8. export diagnostics, checkpoints, and
   `run_manifest.json`.


## Reproducing the reported results

For the reported results, do not execute optimization modules or helper scripts
independently. Use the same configuration file, input assets, software
environment, and fresh top-to-bottom execution of `bwb_optimization.ipynb`.
The resulting `run_manifest.json` records the configuration, seed namespaces,
data and source hashes, model fingerprints, software versions, hardware
information, and final-output hashes.

Interactive notebook execution is deterministic under the recorded software
and hardware contract. Bit-for-bit reproduction additionally requires the same
operating system, Python and package builds, and hardware. Preserve the exact
environment alongside the manifest using:

```bash
python -m pip freeze --all > environment.lock.txt
```


These modules and tests support development and verification; they are not
separate execution paths for reproducing the reported optimization results.


## Train-or-load behavior

The first run trains each model and writes:

```text
final_forwardmodel_results/
final_stress_classifier_results/
```

Later runs load an artifact only when its schema version, dataset fingerprint,
split fingerprint, ordered features/targets, model configuration, master seed,
relevant software versions, source fingerprint, and every persisted payload
SHA-256 match. A mismatch raises an explicit
compatibility error; it never loads the old 24-input forward checkpoint and it
never silently overwrites an artifact. Use a new output directory or the
documented force-retrain option deliberately.

Forward training defaults match the requested run:

```text
epochs=2000, patience=300, lr_patience=120,
learning_rate=1e-3, print_every=25
```

The stress model is a deterministic CPU CatBoost classifier selected with
group-aware cross-validation and calibrated by Platt scaling on the validation
partition. Reports include MAE/RMSE/R2 for the forward targets and log loss,
ROC-AUC, PR-AUC, Brier score, calibration, confusion statistics, and
false-feasible rate for the classifier.

## Topology budget and search

Topology eligibility follows the requested descriptive rule: more than 20
unique feasible designs in the complete supplied dataset. This is explicitly a
**transductive support filter**: held-out stress labels can decide whether a
topology is searchable. Rank order and preliminary score use TRAIN designs
only, so held-out W/P/F values never tune the allocation. Their median score is
computed without using L/D (equivalently, zero L/D shortfall), then sorted.
A monotone gamma-rank decay allocates most CMA generations to the leading
topologies; a uniform exploration floor gives every eligible topology positive
budget. Integer largest-remainder allocation makes the planned and realized
totals match exactly.

Each case runs three independently seeded CMA-ES trajectories. In later rounds,
some repeats warm-start from previous incumbents while at least one remains a
cold global restart. The configured prior weight decays by round, so observed
search evidence increasingly controls allocation without erasing exploration.
The local projected-Adam step may propose improvements, but an incumbent changes
only after the authoritative L/D and stress/noise gates pass and the official
weighted score improves.

The default is intentionally a high-effort run: 1,080,000 CMA candidate
evaluations per case (3,240,000 across the three cases), before local proposals
and confirmation. With two 64-point perturbation levels, the conservative
upper bound is 129 stress-probability rows per candidate; nominal-failing or
L/D-invalid designs skip the perturbation stage, so realized work is lower.
Tune the round budgets and search-bank size only as a new recorded experiment.

## Outputs

The run produces, per case and in combined form:

- a bounded, deduplicated elite archive retained from every CMA trajectory,
  plus aggregate traces covering the full evaluation budget;
- topology allocation and realized-budget ledgers;
- per-generation CMA traces and warm/cold provenance;
- local-refinement attempts and accepted replacements;
- common-bank confirmed candidates and one selected design;
- convergence diagnostics across all three repeats;
- the empirical four-objective nondominated archive of the frozen,
  common-bank confirmation shortlist;
- dataset LOO 5-NN support distances and optimized-design support percentiles;
- pairplots, tabular convergence diagnostics, a kNN ECDF, and empirical Pareto
  plots;
- a strict JSON reproducibility manifest with artifact hashes.

LOO k-NN operates on unique 21D designs scaled by the fixed official bounds.
This prevents repeated flight-condition rows from generating meaningless zero
distances. Row-level tables retain a mapping back to the unique-design result.

## Tests

```bash
pytest -q
```

The suite covers schema ordering, design-group split isolation, named seeds,
artifact compatibility, L/D fail-closed behavior, perturbation invariants,
topology-budget arithmetic, CMA restart determinism, authoritative local-step
acceptance, empirical Pareto membership, duplicate-aware k-NN, and final-output
contracts. A lightweight synthetic smoke example is included; a true end-to-end
run requires the private dataset and L/D surrogate assets listed above.

Round-boundary checkpoints are immutable and survive interruption. Automatic
resume is intentionally not claimed in this version; inspect or archive those
checkpoints, then rerun the deterministic configuration into the same clean
run identity when needed.
