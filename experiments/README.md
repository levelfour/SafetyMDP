# Paper experiment

The shipped profile is [paper.json](paper.json). It selects 24 cases from
[the canonical manifest](manifests/domains.json), all five methods,
one repetition, at most 50 refinement-loop iterations, and no run or Storm-call
timeout: 120 sequential jobs. The [root README](../README.md) lists precisions
and tolerances. With timeouts disabled there is no
finite wall-clock completion guarantee.

The manifests pair files in `benchmarks/models/` and `benchmarks/specs/`.
The expected-SAFE and expected-UNSAFE manifests are disjoint 12-case subsets.
`metadata/` holds model design and numerical calibration;
it contains no executable code.

## Commands

Run from the repository root after installing the Python package and Storm:

```sh
python scripts/run_study.py --dry-run
python scripts/run_study.py --run --output results/paper
python scripts/run_study.py --run --resume --output results/paper
```

A dry run validates the profile, manifests and referenced file existence.
It creates no output, invokes no model checker, and does not parse the PRISM
model or check its semantics. Storm availability and executable identity are
checked when a study is actually started or resumed.
Use `--storm /path/to/storm` for a non-PATH executable.
A custom profile uses `--profile path.json` and a fresh output directory.

The low-level CLIs do not inherit `paper.json`:

| Setting | `mdp-safety` / `mdp-safety-bench` defaults | Paper profile |
|---|---|---|
| Loop iterations | 100 | 50 |
| Minimum OVI precision | 1e-12 | 1e-11 |
| Per-Storm-call timeout | 300 seconds | Disabled |
| Whole-run timeout | No such option in `mdp-safety`; 300 seconds in `mdp-safety-bench` | Disabled |

The driver checkpoints CSVs and `benchmark_config.json` after each returned
job, then updates `full_report.md` when a suite returns. With the one-suite
paper profile, this report is not a live per-job report; the CSVs provide
intermediate results. UNKNOWN and ERROR are recorded outcomes, and later jobs
still run. A process-level interruption can stop the driver.

A compatible `--resume` skips every recorded job, including UNKNOWN, ERROR
and TIMEOUT outcomes; it does not automatically retry unsuccessful verdicts.
A job interrupted before its summary row was saved is run again.

## Fixed order and progress

The paper manifests specify this execution order:

1. Gridworld, then Robot Mission, then Network Routing.
2. Within each family: dimensions 4, 6, 8, 10.
3. At each dimension: the SAFE specification, then the UNSAFE specification.
4. For each case: `l2`, `linf`, `gauge`, `sandwich-lk`, `sandwich-lk-gauge`.

Thus all five methods finish on Gridworld d=4 SAFE before any method starts
on Gridworld d=4 UNSAFE. That pair finishes before Gridworld d=6 begins.
If repetitions are increased, repetitions of each method are consecutive.
Custom manifests and method lists are executed in their written order.

Before launching each verifier process, the runner prints and flushes a
`START` line identifying the planned job number, benchmark, method and repetition.
A `DONE` line follows with the verdict and subprocess elapsed time.
For example, the first start line of the paper study is:

```text
[1/120] START gridworld-d4-safe | method=l2 | repetition=1/1
```

The most recent `START` without a matching `DONE` identifies the active job.
Output is flushed for visibility even when redirected to a log file.
A compatible resume skips recorded jobs and keeps the original job numbers;
it does not print `START` for skipped jobs. Checkpoints from a different
execution-order policy cannot be appended to this study.

## Outputs and metric definitions

For `--output results/paper`, the current suite is saved under
`results/paper/domains/`; `study_provenance.json` and `full_report.md` are
at the `results/paper/` level. Each suite directory contains
`benchmark_config.json`, `summary.csv`, `aggregate.csv`,
`aggregate_by_expected_status.csv` and `aggregate_by_family.csv`.

With one repetition, per-run artifacts are under
`domains/<case>/<method>/`. With multiple repetitions, an additional
`repeat-001/`, `repeat-002/`, ... directory is used. Returned verifier
results include `result.json`, `iterations.csv`, `weight_selections.csv`,
`outer_vertices.csv`, `pareto_samples.csv` and
`inner_approximation_points.csv`. An execution failure may leave no result
artifact; its available diagnostic is in the summary's `error` field.
UNKNOWN explanations are in the per-run `result.json` field `reason`,
not in that summary error field.

A completed verifier result is not necessarily decisive: UNKNOWN results can
also contain measurements. `pareto_samples.csv` contains nondominated
numerical estimates of sampled schedulers, not the complete Pareto frontier.
Inner-approximation points are their componentwise lower bounds.

The study omits full scheduler action maps from results by default; this does
not skip scheduler synthesis or coordinate evaluation. The single-instance
CLI records them unless `--omit-scheduler-actions` is set; the benchmark CLI
can retain them with `--keep-scheduler-actions`.

| Summary field | Meaning |
|---|---|
| `iterations` | Refinement-loop iterations entered, including a terminal containment check or failed selection; not necessarily successful cuts |
| `oracle_calls`, `support_queries` | Completed weighted support queries; same-direction OVI precision retries each count again |
| `storm_calls` | One weighted optimization plus `d` fixed-scheduler coordinate evaluations per completed query: `(d+1)*oracle_calls` |
| `weight_selection_calls` | Number of stored selection records, including the LK terminal tolerance check |
| `weight_selection_seconds`, `weight_selection_cpu_seconds` | Sums of wall/CPU times in those stored selection records |
| `outer_vertex_enumeration_seconds` | Time in the verifier's outer-vertex enumeration wrapper, recorded separately |
| `elapsed_seconds` | Verifier-internal elapsed time, including model preparation and verification |
| `wall_seconds` | Elapsed time of the verifier subprocess, including startup and artifact writing; used for end-to-end comparisons |

`storm_calls` counts model-checking work associated with completed query
records, not every subprocess invocation: version probes and model export are
excluded. Each oracle attempt now uses one Stormpy worker for the optimization
and all coordinate evaluations, applying its native scheduler with
`apply_scheduler`. Thus `(d+1)` model-checking calls do not mean `(d+1)` process
launches. A tool failure without a result leaves query counts missing.

Results record the CLI version as `storm_version` and the Stormpy/linked-Storm
versions and native backend identifier in `storm_configuration`. Checkpoints
reject resuming results from a different oracle backend or Stormpy version;
use a fresh output directory for this backend. Existing measurements are not
rewritten. `--storm` selects the model-export CLI, not Stormpy's linked library.

A selection can issue several oracle queries because precision retries reuse
its direction. Conversely, the LK terminal tolerance check records selection
time without issuing a query. Never assume that iteration, selection and
oracle-call counts are equal.

### Timing scope and incomplete attempts

Per-query `support_seconds` includes weighted-model preparation, worker startup,
native model loading, weighted OVI and scheduler export. `witness_seconds`
includes native scheduler application, induced-DTMC export, all coordinate
checks and worker teardown. These are wall-clock phase measurements, not just
solver CPU times; their split differs from the former CLI-per-coordinate
backend, which excluded induced-model construction from witness time.
Query-selection timing still excludes all of this oracle work. The per-Storm
timeout applies separately to worker setup, optimization, scheduler application
and each coordinate; `none` disables the deadlines.

Successful selection records include candidate scoring/projection, direction
construction, safe-region support optimization and its numerical retry, and
direction audits. LK inner-facet construction and outer-support geometry LPs
are also selection work. This time excludes the later Storm optimization,
same-scheduler coordinate evaluations and OVI precision retries.

Vertex enumeration before selection is measured separately; cache access
during a selection is part of selection time. A SAFE containment check adds
no selection record. The LK all-facet check that ends with approximation
tolerance reached is recorded, with an empty weight.

A general `RegionError` during selection or its audit returns UNKNOWN without
appending a selection record for that failed attempt. Its cost is therefore
absent from the selection-time sum, although included in overall runtime.
Do not interpret this unrecorded cost as zero or assume the selection total
accounts for all time spent before a numerical failure.

Per-run `iterations.csv` contains completed query records, indexed by
`iteration` and `attempt`. `weight_selections.csv` contains stored selection
records indexed by iteration. Use these keys, not row position, to relate them.
For timing trajectories, distinguish successful selections, LK terminal checks
and missing failed attempts. A short final record is not by itself evidence
that later geometric problems are cheaper.

## Comparing methods

Aggregate expected-SAFE and expected-UNSAFE cases separately. Manifest
`expected_status` labels are used only for grouping and checking returned
verdicts; they are not supplied to the verifier as answers. Report decision
counts and UNKNOWN/ERROR reasons alongside query and timing statistics.

The automatic report includes separate expected-status tables, while its
family table combines both expected statuses. It does not compute the
intersection of cases decided by every method; form that matched subset from
`summary.csv` for a like-for-like comparison. Aggregate CSVs include
`errors` and `label_mismatches`, which are not separate columns in the
current Markdown tables.

Missing measurements must remain missing, not zero. Aggregate `completed`
means a row has an oracle-call measurement; it includes UNKNOWN and must not
be presented as the number of solved cases. Selection-time sums for numerically
failed runs have the limitation above. Oracle/selection means use available
returned measurements; wall-time means use all recorded jobs, including ERROR
and TIMEOUT. A sum over no available measurements can appear as zero in the
aggregate CSV, so consult `completed` and the raw rows before interpreting it.

One repetition does not estimate within-case timing variance or support
significance tests. Aggregate standard deviations across different cases are
not repeated-measurement uncertainty for an individual benchmark.

## Measured native-Storm rerun (2026-10-04)

Dataset identifier:
`all_methods_domains_native_storm_50refinements_no_timeout_1rep_20261004`.
The separate local run archive contains `snapshot/`, `study.log`,
`results/study_provenance.json`, `results/domains/summary.csv`,
`analysis.json` and the Japanese `completion_report.md`; these measurements
are not bundled into the source distribution. Paths in this paragraph are
relative to that archive, not the repository.

All 120 sequential jobs completed from 16:54:31 to 21:44:23 UTC on October 4:
4 h 49 min 52 s. The sum of subprocess wall times was 17,389.171 s; the
remaining approximately 3.12 s was between-job/reporting overhead. The profile
was one repetition, 50 loop iterations, no run/Storm deadlines, fixed
manifest-method order, and the precisions documented above. The snapshot used
Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1, Stormpy 1.13.2 and Storm 1.13.0
for both the CLI and linked library, on macOS arm64. The native backend was
`stormpy-native-scheduler-ovi-v1`.

The audit checked 120 unique jobs in the specified order, all 36 model/spec
files, 53 verifier Python source files, runner/profile/manifest/Storm hashes,
per-run backend/options, and agreement among result JSON, query/selection CSVs
and summary rows. All 45 independently recomputed Q/W/T totals in the overall
and expected-status aggregate CSVs matched. All runs returned measurements.
The 99 decisive outcomes matched the expected labels; 21 were UNKNOWN, with
zero ERROR or TIMEOUT outcomes.

Q is the sum of completed weighted oracle calls; W is recorded selection wall
time; T is subprocess wall time. All times below are seconds, and totals
include UNKNOWN runs. They are not averages over only successfully decided
cases. W retains the failed-attempt limitation described above.

### Expected SAFE: 12 cases

| Method | SAFE | UNSAFE | UNKNOWN | Q | W (s) | T (s) |
|---|---:|---:|---:|---:|---:|---:|
| L2 | 10 | 0 | 2 | 247 | 3044.879 | 3316.605 |
| L-infinity | 9 | 0 | 3 | 284 | 606.262 | 893.748 |
| Gauge | 12 | 0 | 0 | 78 | 39.046 | 105.316 |
| LK-epsilon | 5 | 0 | 7 | 441 | 4242.446 | 4880.660 |
| LK-gauge | 9 | 0 | 3 | 312 | 3608.969 | 4161.018 |

### Expected UNSAFE: 12 cases

| Method | SAFE | UNSAFE | UNKNOWN | Q | W (s) | T (s) |
|---|---:|---:|---:|---:|---:|---:|
| L2 | 0 | 12 | 0 | 58 | 36.112 | 88.742 |
| L-infinity | 0 | 12 | 0 | 58 | 11.148 | 63.094 |
| Gauge | 0 | 12 | 0 | 24 | 8.633 | 35.006 |
| LK-epsilon | 0 | 9 | 3 | 209 | 1569.323 | 1949.529 |
| LK-gauge | 0 | 9 | 3 | 190 | 1528.724 | 1895.454 |

### Matched, decisively solved subset

All five methods decided the same 14 cases: five expected-SAFE and nine
expected-UNSAFE. The SAFE cases are Gridworld d4/d6, Robot Mission d4/d6 and
Network Routing d4. The UNSAFE cases are all Gridworld and Network Routing
dimensions and Robot Mission d4. This easier subset must be reported alongside
the full-suite decision counts.

| Method | Q | W (s) | T (s) |
|---|---:|---:|---:|
| L2 | 76 | 12.825 | 75.953 |
| L-infinity | 83 | 4.497 | 71.121 |
| Gauge | 39 | 8.587 | 45.507 |
| LK-epsilon | 150 | 98.824 | 214.660 |
| LK-gauge | 83 | 4.128 | 71.026 |

Gauge used the fewest oracle calls and least total time on this matched
subset, but L-infinity and LK-gauge used less selection time. Gauge is not
uniformly fastest: on Robot Mission d10 SAFE, L-infinity and Gauge both used
11 oracle calls, with T=17.322 s and 20.762 s respectively. On Network Routing
d10 UNSAFE, both LK variants needed one query (about 1.45 s), whereas Gauge
needed seven (12.224 s).

### UNKNOWN causes and interpretation

Nineteen UNKNOWN outcomes reached the 50-iteration limit. All had Q=50.
The remaining two were:

- Gridworld d10 SAFE / L2: a support-optimality gap of 6.28e-8 exceeded 5e-8,
  after 23 queries. The failed selection at iteration 24 has no selection-time
  record; W=64.245 s covers only the preceding stored records.
- Gridworld d10 SAFE / L-infinity: the degenerate outer-vertex enumeration
  fallback limit was reached at iteration 43, after 42 queries.

In this rerun Gridworld d8 SAFE / L-infinity reached the iteration limit;
its reason is not vertex enumeration failure. Numerical failure locations
from an older run must not be copied into descriptions of this dataset.
Both LK variants failed to decide Robot Mission d6/d8/d10 UNSAFE within the
iteration budget. All other UNKNOWN outcomes concern expected-SAFE cases.

L-infinity's masked-gradient formula and numerical safeguards were unchanged
for this rerun. The oracle now applies and evaluates Storm's native scheduler.
Older runs also differ in input representation and execution order; their
runtime difference is not a controlled measurement of the API change alone.
One repetition cannot establish within-case variability or statistical
significance. Query count, selection cost, geometry failures and the common
decision set must be considered together.
