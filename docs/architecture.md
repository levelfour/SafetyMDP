# Architecture

The implementation separates verification state, query selection, safe-region
geometry, the Storm oracle, and experiment orchestration.

| Component | Responsibility |
|---|---|
| `cli.py`, `specification.py` | CLI settings and terminal/safety JSON loading |
| `formula.py` | Restricted polynomial parsing and shape-assumption checks |
| `verifier.py` | Public API, input/model preparation and work directory |
| `verification/engine.py` | Refinement loop, decisions, retries and measurement |
| `verification/queries.py` | Weighted support and same-scheduler coordinate evaluation |
| `verification/state.py` | Per-run inner/outer approximation and measurement state |
| `verification/results.py` | Result schema, samples and witness assembly |
| `strategies/registry.py` | Shared method factories, names, labels and CLI validation |
| `strategies/norm.py`, `strategies/gauge.py` | Specification-directed strategies |
| `strategies/sandwich.py` | Shared coordinate initialization and inner-facet cache |
| `strategies/sandwich_lk.py` | Absolute/relative maximum-indicator LK strategies |
| `direction_scores.py` | Common normalized support-gap evaluation and audits |
| `safe_regions/` | Membership, support, projection and gauge operations |
| `polytope.py`, `inner_facets.py` | Outer vertices and inner supporting facets |
| `storm.py`, `storm_results.py` | CLI model export, native oracle requests and outward-rounded probability intervals |
| `storm_api.py`, `storm_api_worker.py` | Isolated Stormpy OVI calls, native scheduler application and per-operation deadlines |
| `model.py`, `drn.py` | Weighted-terminal gadget, explicit model and witness scheduler I/O |
| `experiments/` | Manifest parsing, job execution, checkpoints, provenance and aggregation |
| `benchmark_report.py` | Comparative reports, including separate expected-status tables |

All paths in this table are relative to `src/mdp_safety/`.
`regions.py`, `projections.py` and `benchmarking.py` are public re-export
modules, not duplicate implementations.

## Interfaces and invariants

`SafetyVerifier.verify()` creates fresh `VerificationState` for each run.
The oracle protocol returns weighted maximization and same-scheduler coordinate
bounds together, so the native scheduler stays within one Stormpy worker.
The production oracle is Storm; injected oracles in unit tests do not replace
Storm in performance experiments.

## Native scheduler evaluation

The CLI exports the PRISM model once. For each query attempt, Python builds the
weighted-terminal gadget and a fresh worker loads it into Stormpy. The worker
uses `model_checking(..., extract_scheduler=True)` with sound absolute-precision
min-max optimistic value iteration. It completes any undefined choices through
`Scheduler.set_choice`, then calls `SparseMdp.apply_scheduler` on **that same
scheduler object and that same model**. No scheduler is reconstructed from JSON
for evaluation; the JSON action map is only a witness artifact.

The resulting model is a DTMC. The worker retains it in memory and checks
`P=? [F "target"]` for each original target label with the native linear-equation
solver's sound absolute-precision OVI. Both solver families are explicitly
configured before creating the Environment; default API solver settings are
not used. The existing estimate +/- precision plus outward-rounding guard is
retained for coordinate and support intervals.

The induced DTMC still includes the two auxiliary gadget sinks. This does not
alter any original target's reachability: after first entering a target, the
gadget goes only to those absorbing sinks, never to another target. Restricting
the scheduler to original nonterminal states therefore gives exactly the same
target-probability vector in the original model, including nontermination and
the optional first-hit terminalization semantics. Keeping the gadget avoids
transferring a scheduler between differently indexed native models.

One worker handles one optimization plus all `d` coordinate checks. The parent
enforces a new deadline for loading, optimization, scheduler application and
each coordinate check, terminating and reaping the worker on timeout or
interruption. `--timeout none` disables those deadlines. Native diagnostics go
to a separate log; result stdout remains JSON. `--keep-work` retains the
weighted MDP, exported scheduler, induced **DTMC**, coordinate results, worker
request and log. The CLI and linked-library versions can differ and are recorded
separately; `--storm` only selects the CLI used to build the PRISM input.

A query strategy implements `ProjectionStrategy.select()` and its direction
audit. New strategies register through the shared registry.
`supports_inner_refinement` permits a strategy to make progress by expanding
the inner approximation without removing the selected outer point.
`retain_all_support_cuts` also attempts to retain oracle upper cuts in those
updates. Only already-covered, approximately same-direction cuts are skipped;
there is no general redundancy-elimination LP.

The five built-in methods share verification decisions and the Storm oracle.
Their strategies select directions and declare whether inner-hull expansion
and additional support cuts count as progress. General region implementations remain
available as mathematical infrastructure; the published evaluation inputs all
use the formula-input quadratic path.

## Repository data and execution

These paths are relative to the repository root, not `src/mdp_safety/`:

| Path | Responsibility |
|---|---|
| `benchmarks/models/`, `benchmarks/specs/` | Fixed PRISM models and safety specifications only |
| `experiments/paper.json` | Methods, limits and precisions |
| `experiments/manifests/` | Model/specification pairs and expected-status subsets |
| `experiments/metadata/` | Model design and calibration records |
| `scripts/run_study.py` | Study planning, explicit execution and consolidated reporting |
| `scripts/fixed_inputs.py` | Read metadata or copy fixed input bytes without regeneration |

The root-level `experiments/` contains data and instructions; the Python package
`src/mdp_safety/experiments/` implements execution and analysis.
See the [experiment guide](../experiments/README.md) for the paper profile and
measurement definitions.

The study driver saves runtime provenance in `study_provenance.json` and
checks input contents, recursive verifier-source hashes, the runner, executable
identity, environment and profile before resuming. Its input hashes are
calculated from the files actually used; model/calibration metadata is not
loaded into the verification loop.

The lower-level `mdp-safety-bench --resume` checks the manifest, recorded
settings, strategy versions, oracle backend and Stormpy package version, but does not perform the driver's full
source/input/binary provenance comparison. Use the study driver for the paper
workflow. Result schema and checkpoint schema are independent. Do not bypass
the checks to append measurements after changing code, inputs or settings.

## Numerical and regression coverage

See [query selection](query_selection.md) and
[LK implementation](sandwich_lk.md) for score definitions and numerical limits.

Tests include analytic geometry, independent LP comparisons, state-machine
failure paths, result reporting, mock multi-suite orchestration, exact-rational
Storm graph exports for all 12 models, and small real-Storm verification runs.
The synthetic regression fixtures are isolated under `tests/fixtures/`;
they are not listed in any benchmark manifest.
