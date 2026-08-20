# Method specification

This document fixes the algorithm before results are inspected. Values are
read from `configs/default.yaml`; formulas and stage boundaries are part of the
reproducibility contract.

## 1. Data units and split

Let \(x\in\mathbb{R}^{21}\) be the ordered geometry/structure vector, let
\(c=(h,V,\alpha)\) be the three flight conditions, and let
\(z=(x,c)\in\mathbb{R}^{24}\).

A stable `design_id` is SHA-256 over the ordered feature names and canonical
IEEE-754 float64 bytes of \(x\). Unique design IDs are deterministically assigned
70%/15%/15% to train/validation/test. All rows sharing \(x\) share a partition.
The three forward targets must be invariant across rows sharing \(x\), within
the configured numerical tolerance; otherwise the pipeline stops because a
21-input forward model would be scientifically inconsistent.

## 2. Forward surrogate

The forward model maps \(x\) to empty weight \(W\), payload volume \(P\), and
fuel volume \(F\). Inputs are standardized from training data only. Positive
target means from training data scale the three losses. A residual MLP with
SiLU activations and three positive heads is optimized with AdamW and early
stopping on validation loss. The test set is evaluated once after checkpoint
selection.

The official scalar objective for mission \(m\) is

\[
J_m(x)=0.4\frac{W(x)}{50}
+0.2\left[\frac{L_m^*-L_m(x)}{L_m^*}\right]_+
+0.2\left[\frac{F_m^*-F(x)}{F_m^*}\right]_+
+0.2\left[\frac{P_m^*-P(x)}{P_m^*}\right]_+ .
\]

Dataset volumes are converted explicitly to cubic metres before the shortfall
terms are evaluated.

## 3. Stress feasibility

The label is \(y=1\) exactly when stress is at most 335 MPa. A deterministic,
CPU CatBoost classifier consumes \(z\). Candidate hyperparameters are assessed
by group-aware cross-validation within the training partition only. The final
model is calibrated by a one-dimensional Platt map on validation logits. The
test partition is touched only for the final report.

A design at mission \(m\) must satisfy

\[
\hat p(y=1\mid x,c_m)\ge 0.90.
\]

## 4. L/D domain gate

The external surrogate receives the first ten geometry variables and the
mission conditions. Its structured result must contain finite `LD`, `CL`, and
`CD`, with `CD>0`. Any exception, missing key, or non-empty warning list makes
the candidate ineligible. This rule prevents a numerically attractive
extrapolation from winning the search.

## 5. Local probability robustness

Only the 18 continuous design variables are perturbed; topology and mission are
fixed. At a bound, a truncated interval moves inward rather than clipping an
outward perturbation, avoiding artificial point mass. One deterministic common
Sobol bank is reused by all candidates within a search stage.

For every configured fraction \(\delta\) of each official bound width,

\[
\operatorname{median}_{u\in B}
\hat p(y=1\mid T_\delta(x,u),c_m)\ge 0.80.
\]

The default search fractions are 0.1% and 0.5%. Nominal probability, both
perturbed medians, and the L/D-domain flag are hard gates, not weighted terms in
\(J_m\).

## 6. Empirical topology prior

Eligibility uses the complete supplied dataset: a topology enters when more
than 20 unique physical designs have a measured stress-feasible observation.
This is a predeclared transductive support filter and therefore inspects
held-out stress labels for inclusion only. Preliminary score and rank use TRAIN
designs exclusively, with L/D shortfall set to zero (“L/D receives full
credit”); validation/test W/P/F targets never tune the ordering. Both all-data
eligibility support and train ranking support are reported.

If rank starts at zero, the exploitation weight is a monotone gamma-like decay

\[
w_r=(r+0.5)^{k-1}\exp[-(r+0.5)/(sN)],\quad 0<k<1,
\]

mixed with a uniform exploration floor. Largest-remainder allocation converts
weights to integer CMA generation blocks while preserving the exact round
budget and deterministic ties.

Changing to a fully inductive prior is supported by setting
`topology_eligibility_split: train`; that changes the eligible set and therefore
defines a different, versioned experiment.

## 7. Multi-round active CMA-ES

For each fixed topology CMA-ES searches the 18 continuous coordinates in unit
space. Each round allocates an exact number of population-sized generation
blocks. Three independently seeded trajectories are derived from the one
master seed. Round zero is cold. Later rounds keep at least one cold repeat and
warm-start the remainder from archived incumbents; a uniform budget component
keeps all eligible topologies explorable.

CMA receives a finite, tier-separated merit:

1. valid L/D + nominal pass + every noise-median pass: bounded \(J_m\);
2. valid L/D + nominal pass but robustness fail: robustness violation;
3. valid L/D but nominal fail: nominal-probability violation;
4. invalid/warned L/D: worst tier.

Tier intervals do not overlap, so no arbitrary penalty coefficient can make an
infeasible point outrank a feasible point. The unmodified scientific score is
retained separately for reporting.

## 8. Gradient proposal and authoritative acceptance

Projected AdamW refines only differentiable weight/payload/fuel terms through
the forward MLP and stays within a trust region. CatBoost and the external L/D
surrogate are not differentiated. At each exact checkpoint, the resulting
design is repaired and evaluated by all authoritative models. It replaces its
incumbent only if every hard gate passes and \(J_m\) strictly improves.

This proposal/certification separation prevents a differentiable auxiliary
model from certifying its own exploit.

## 9. Confirmation, selection, and convergence

After all rounds, a frozen shortlist is deduplicated and reevaluated once using
an independent common probability-noise bank. Mandatory entries include the
global/repeat/topology objective champions and robustness-margin champions, so
a small-bank winner is not the only representative of its topology. This bank
is never used to launch new optimization. The selected design is the feasible
candidate with minimum official score; exactly one row is produced per case.
If no confirmed design exists, production mode fails rather than relabeling a
least-infeasible point.

The convergence table includes best score, exact candidate-feasible fraction,
round-to-round change, topology agreement, accepted-score spread, normalized
design distance, and cold-start support. `converged_by_repeat_consensus` also
requires the held-out selected design to be the last-round search champion;
otherwise a specific held-out-change status is reported. This is numerical
evidence, not a proof of the global optimum.

## 10. Empirical archive and support diagnostics

Each CMA trajectory retains a fixed-size elite archive while its trace accounts
for the complete evaluation budget; storing millions of dominated rows would
add I/O without improving selection. The frozen shortlist is reevaluated on one
common confirmation bank, and its hard-feasible rows are nondominated-sorted on
the four objective components. This is reported as a bounded empirical
nondominated archive, with its coverage limit stated explicitly.

For support diagnostics, exact 21D duplicates are collapsed before leave-one-
out 5-NN distances are computed in fixed-bound unit space. Generated designs
are queried against unique training designs and reported with their distance
percentile, 95%/99% support flags, and whether the exact topology was seen in
the training reference.
