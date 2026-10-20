# Reproduction and evaluation protocols

## Read the saved study

`python scripts/summarize_results.py` recomputes Table III directly from the
included observations. It runs on the CPU and does not run an optimizer.
`results/README.md` describes the other exported data.

## Benchmark definitions and budgets

| Group | Cases | Dimensions | Equal-FE budget | Recorded deadline |
|---|---:|---|---:|---:|
| CEC-derived F1–F12 | 24 | 10, 20 | 24,800,000 | 60 s / 120 s |
| Canonical functions | 18 | 50, 100, 200 | 1,024,000 | 30 s |
| Shifted/rotated Ackley and Rastrigin | 6 | 50, 100, 200 | 1,024,000 | 300 s |
| Unrotated Ackley and Rastrigin | 6 | 500, 1000, 2000 | 1,024,000 | 300 s |

Canonical functions are Ackley, Griewank, Rastrigin, Rosenbrock, Schwefel and
Sphere. The ten-coordinate AutoPSO policy and its nine exemplar channels are
retained. CoRePSO uses 100 inner updates per round; AutoPSO uses 800. Population
sizes are `(lineages, particles) = (100, 100)` for CEC-derived cases and `(16, 40)`
otherwise. Each traditional solver has 100 particles.

The CEC-derived implementation uses corrected Schaffer rotation, complete hybrid
segments, and ordered float64 arithmetic. F4 retains the continuous Rastrigin
form of the reference implementation. Thus these results must be identified as
CEC-derived, not an exact evaluation of the official CEC2022 definition.

The six generated shift/rotation instances are supplied as `.npz` files.
All search variables are represented in `[-5, 5]^d` and mapped to physical bounds
inside the objectives. Transformations and optima are not supplied to the solver.

Replicate IDs are 960701–960711 for equal-FE runs and 970701–970711 for recorded
time. Seeds are `7010300 + 100003 * replicate + 7919 * case_index`; the 54-case
ordering is given in `configs/numerical_fe.json`. Full algorithm-specific random
streams differ even when seed identifiers are paired.

## Hardware and clocks

Primary CoRePSO/AutoPSO runs used Windows 11, an RTX 4060 Ti (8 GB), Python
3.12.10, PyTorch 2.11.0/CUDA 12.8, NumPy 2.3.3 and SciPy 1.17.1, with one-thread
CPU pools. Traditional equal-FE records include 1122 RTX 3090 runs and 1848
RTX 4060 Ti runs. All numerical recorded-time runs used the RTX 4060 Ti.

CoRePSO/AutoPSO clocks include initialization, local evaluation-graph preparation
and search, but exclude shared swarm-graph preparation. Traditional-solver clocks
also include per-run graph capture. Synchronized checks admit only batches
completed by the deadline. Independent final scalar verification is outside the
search budget and clock. A result obtained on another machine is a new timing
measurement, not the recorded number in the manuscript.

The learned series contains 345 RTX 4060 Ti and 315 RTX 3090 records. The final
136 server records used asynchronous CUDA launches following 15 separate
small-budget equivalence checks; these checks are not formal observations.
Execution metadata are retained where present in the original records.

Engineering measurements used an RTX 3090, physical GPU0, float64 and serial
trials. The 57-configuration cost study used the same GPU and one CPU thread
on a Xeon Gold 5218R host. It fixed 10 coordination rounds, with 100 inner updates
per round, and reported setup separately. This is a workload-cost study, not an
equal-FE accuracy comparison or a claim of identical CPU/GPU trajectories.

## Dependencies

After the core installation in the README:

```bash
python -m pip install -e ".[benchmark]"
python scripts/fetch_dependencies.py cec
python scripts/fetch_dependencies.py evox
```

The CEC fetcher obtains the upstream Python reference and transformation text
files. The original Python source is used for independent endpoint checks.
The benchmark objective used by all solvers is `src/cuda_objectives.py`.
Traditional runners import the pinned EvoX checkout under `baselines/evox/src`.

For learned baselines:

```bash
python scripts/fetch_dependencies.py metabox
```

Install dependencies required by the selected MetaBox methods using its pinned
`pyproject.toml`. The loader imports only the chosen agents, optimizers and their
dependencies; it does not load the unrelated benchmark suites. The fetcher
applies `patches/metabox_accounting_and_bounds.patch` to the exact pinned
revision. This patch corrects DEDQN sample-cost alignment and FE accounting,
RLEPSO FE accounting, and SYMBOL boundary repair.

## Explicit benchmark entry points

The commands below start substantial new computations. They are provided as
reproduction entry points; installation and summary commands do not execute them.
Use fresh output paths for a new realization. Completed-record resume checks
reject duplicate or incompatible observations.

```bash
# CoRePSO, AutoPSO and the no-splitting control, all 54 equal-FE cases.
python src/run_scope_cuda.py --output outputs/numerical_fe

# Five traditional solvers, all 54 equal-FE cases.
python src/run_external_fe.py --config configs/traditional_fe.json --output outputs/traditional_fe

# Recorded-clock protocols, with graph preparation conventions stated above.
python src/run_time_cuda.py --output outputs/numerical_time
python src/run_external_time.py --output outputs/traditional_time

# Component controls on the 12 CEC-derived 20D cases.
python src/run_component_ablation.py --output outputs/component_ablation
```

Partition and transfer runners retain the RTX 4060 Ti hardware guard used for
comparison with the saved adaptive time results:

```bash
python src/run_remaining_numerical.py --stage partition --output outputs/remaining
python src/run_remaining_numerical.py --stage sources --output outputs/remaining
python src/run_remaining_numerical.py --stage transfer --output outputs/remaining
python src/run_remaining_numerical.py --stage cross_baselines --output outputs/remaining
```

The source stage precedes transfer. A per-replicate policy is selected using its
source function; transfer freezes that policy while retaining state feedback and
local search. These are within-benchmark-family transfers.

For the archived engineering and cost protocols, use an idle RTX 3090 physical
GPU0. The engineering runner retains its POSIX file lock and GPU occupancy check.
On a suitable Linux host:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
python src/run_engineering_applications.py --mode formal --output outputs/engineering
python src/run_cost_scaling.py --output outputs/cost
```

Engineering comprises FM-6D, Radar-19D and Radar-20D, each with nine configurations,
11 seeds and exactly 1,024,000 evaluations. Radar implements the positive and
negative branches from the CEC2011 technical report. Scores cannot be pooled with
the older MATLAB implementation that omits the negative branches.

## Learned comparison

The recorded study used DEDQN, SYMBOL, GLEET, RLHPSDE and RLEPSO, trained on F1–F12
for 100 epochs with seed 3849 and final-checkpoint selection. Training stops after
complete upstream updates at a 20,000-FE threshold; actual queries, including
SYMBOL teacher queries, are counted. The per-oracle training safety ceiling is
100,000. Evaluation rejects a whole batch if it would exceed 24,800,000 FEs and
records unused evaluations.

The original checkpoint binaries are not bundled. The following example shows
how the archived runner produces its expected local checkpoint location:

```bash
python src/run_metabbo_train.py --method DEDQN --output results_metabbo_20260915/train/DEDQN
```

The corresponding commands for `SYMBOL`, `GLEET`, `RLHPSDE` and `RLEPSO` must
complete before `python src/run_metabbo_eval.py --output outputs/learned` can run
the full grid. This creates a new training/evaluation realization; saved endpoints
in this release remain the evidence for the manuscript. Pickle loading is limited
to checkpoints generated locally by this runner; do not load untrusted pickles.

## Statistical families

Errors below `1e-8` are scored as zero. Tables use arithmetic means and sample
standard deviations. Mean comparisons use the tolerance stated in the README.

* Primary numerical tests: two-sided paired Wilcoxon tests over 11 repetitions,
  one 54-case Holm family per comparator and resource protocol. Tolerance-zeroed
  differences are omitted by SciPy's default Wilcoxon handling; all-zero
  differences receive `p = 1`.
* Table III: one joint, post-hoc family of 72 tests (12 functions × six
  comparators). AutoPSO uses paired Wilcoxon tests; the learned methods use
  independent two-sided Mann–Whitney tests with tie-corrected asymptotic variance
  and continuity correction. Significance is `p_Holm < 0.05`.
* Component analysis: one 12-case paired-test Holm family per control.
* Engineering: one three-instance independent-test Holm family per comparator.

The family determines the adjusted result: the 12-function AutoPSO comparison
inside Table III is not interchangeable with the primary 54-case family.
Non-rejection does not demonstrate equivalence.
