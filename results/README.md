# Recorded data

These files are exports of the completed study. No packaging example, smoke run,
pilot or validation trial is included.

| File | Content |
|---|---|
| `fe_endpoints.json.gz` | 4752 compact endpoints: 54 cases × 11 repetitions × eight configurations (seven solvers plus no splitting) |
| `seven_solver_fe.csv` | Mean and sample SD for 378 case/solver combinations under fixed FEs |
| `seven_solver_recorded_time.csv` | Mean and sample SD for 378 combinations under recorded clocks |
| `recorded_time_curves.json.gz` | 378 aggregate curves, each averaging 11 trajectories; protocol, time grid, means and endpoint SD |
| `learned_endpoints.json.gz` | All 660 learned evaluation records, including cap, actual/unused FEs, endpoint, seed and available execution metadata |
| `learned_significance.csv` | All 72 Table III tests, raw and jointly Holm-adjusted probabilities, directions and decisions |
| `component_endpoints.json.gz` | 660 observations for the five component removals on 12 functions |
| `component_ablation.csv`, `component_ablation_tests.csv` | Component means, SDs and 12-case-family comparisons, including the reused no-splitting control |
| `engineering_endpoints.json.gz` | All 297 formal application observations, with best coordinates and scalar-verified values |
| `engineering_results.csv` | 27 instance/configuration summaries and three-instance-family probabilities |
| `cost_records.json.gz` | All 627 records from the 57-configuration, 11-repeat cost study |
| `cost_scaling.csv` | Configuration-level cost, setup, objective and FE summaries |

Internal identifiers map as follows: `cvbr_hybrid` = CoRePSO,
`original` = AutoPSO, and `cvbr_hybrid_no_split` = the no-splitting control.
`no_inheritance` is a separate component control. Other method names retain
their manuscript spelling.

Numerical error values in endpoint exports are the recorded scalar-reference
results before the reporting floor. Statistical scripts score values below
`1e-8` as zero. Engineering values are objective values, not shifted CEC errors.
Cost records describe fixed-round workloads, not equal-FE accuracy trials.

The FE cache retains case identifiers, seed, budget and verified scalar error.
It does not retain per-query traces or the full original verification log.
The aggregate convergence cache likewise cannot reconstruct individual
trajectories. Original multi-gigabyte execution traces and learned checkpoint
binaries are outside this compact release.

Hostnames have been removed from exported execution dictionaries; recorded GPU
model, software versions, launch settings and numerical fields remain unchanged.
The 345 early learned records predate per-record execution metadata and are
identified as RTX 4060 Ti runs in the study's migration record and manuscript;
their missing execution fields have not been backfilled. The remaining 315
records name the RTX 3090. Do not pool their runtimes as a same-hardware benchmark.

`python scripts/summarize_results.py` validates the Table III grids and matches
all 72 recorded statistical results without invoking training or optimization.
