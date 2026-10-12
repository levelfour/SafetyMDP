# Artifact for "Convex-Analytic View of Safety Verification of Markov Decision Processes"

## Requirements and installation

- Python 3.10 or newer.
- NumPy >= 1.26 and SciPy >= 1.11 (installed by pip).
- Stormpy >= 1.13.2, < 1.14 (installed by pip) for OVI, native scheduler
  application and fixed-scheduler probability evaluation.
- Storm, available as `storm` on PATH. The local regression tests use
  Storm 1.13.0; compatibility with earlier versions is not asserted here.

The local test environment uses Python 3.12.3, Stormpy 1.13.2 and its linked
Storm 1.13.0. The `test` extra installs
pytest >= 8; see [pyproject.toml](pyproject.toml) for package requirements.

From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
storm --version
```

The Storm CLI is a separate dependency; installing this package installs
Stormpy but does not provide the `storm` executable. Use `--storm /path/to/storm`
if it is not on PATH. The CLI builds the PRISM model; quantitative queries
use the Storm library linked to Stormpy, not the executable selected by `--storm`.
Both Storm versions and the Stormpy version are recorded in the result.
Run the commands below from the repository root with this virtual environment
activated. If `mdp-safety` is not found, check that the editable installation
completed in the same environment; `python -m mdp_safety` is an equivalent
entry point.

## Verify one paper instance

```sh
mdp-safety \
  benchmarks/models/gridworld-d4.prism \
  benchmarks/specs/gridworld-d4-safe.json \
  --projection gauge --precision 1e-7 --min-precision 1e-11 \
  --geometry-tolerance 1e-8 --max-iterations 50 --timeout none \
  --output-dir results/gridworld-d4-gauge
```

The model is identical for the SAFE and UNSAFE variants; only the specification
file changes. See [models and specifications](docs/benchmarks.md).
The command prints the result as JSON and writes per-run JSON/CSV artifacts.
Use a fresh output directory: the single-instance CLI replaces existing
artifact files there. Its `--timeout` option limits each Storm CLI/API operation,
not the complete verification run. Native calls run in an isolated worker;
model loading and scheduler application also have separate deadlines.

## Reproduce the paper configuration

```sh
# Read-only: validate inputs and show the 120-job plan.
python scripts/run_study.py --dry-run

# Explicitly start the sequential study in a fresh directory.
python scripts/run_study.py --run --output results/paper

# Resume only when inputs, code, executable and settings still match.
python scripts/run_study.py --run --resume --output results/paper
```

The single shipped profile, [paper.json](experiments/paper.json), uses:

| Setting | Value |
|---|---|
| Domains | Gridworld, Robot Mission, Network Routing |
| Dimensions | 4, 6, 8, 10 |
| Specifications | One quadratic SAFE and one quadratic UNSAFE per MDP |
| Methods | `l2`, `linf`, `gauge`, `sandwich-lk`, `sandwich-lk-gauge` |
| Repetitions | 1 |
| Refinement-loop iteration limit | 50 |
| Whole-run / Storm-call timeouts | Both disabled |
| Initial / minimum OVI precision | `1e-7` / `1e-11` |
| Geometry / Sandwich tolerance | `1e-8` / `1e-6` |

Execution follows Gridworld, Robot Mission, then Network Routing. Within each
family it uses dimensions 4, 6, 8, 10; at each dimension, all five methods run
on SAFE first and then on UNSAFE. Each job prints a flushed `START` line before
execution and a `DONE` line afterward, including its benchmark and method.

Disabling timeouts means there is **no finite wall-clock completion guarantee**.
A numerical failure can produce UNKNOWN before 50 iterations. Initialization
and terminal checks also consume loop iterations; 50 is not a limit of
exactly 50 successful cuts or oracle calls.
The low-level CLIs retain independent general-purpose defaults; use the study
profile or the explicit options above to reproduce the paper configuration.

[Experiment instructions](experiments/README.md) explain outputs,
measurement definitions, separate SAFE/UNSAFE aggregation and provenance.
Use a fresh output directory after any code or input change. Provenance hashes
include comments and docstrings, so documentation-only changes inside Python or
PRISM files also prevent resuming an earlier run.

## Methods and outcomes

| Method | Query selection |
|---|---|
| L2 | Maximum outer-vertex distance to the safety region in the Euclidean norm |
| L-infinity | Maximum outer-vertex L-infinity distance, with a checked masked-gradient direction for smooth formula regions |
| Gauge | Maximum gauge excess relative to the fixed safety region |
| LK-epsilon | Maximum absolute inner/outer approximation gap over inner facets |
| LK-gauge | Maximum relative inner/outer approximation gap over inner facets |

Details and implementation optimizations are documented in
[direction selection](docs/query_selection.md) and
[LK baselines](docs/sandwich_lk.md).

- **SAFE:** every enumerated outer vertex passes the convex safe-region
  membership check under the configured geometry tolerance.
- **UNSAFE:** a scheduler's componentwise probability lower bounds already
  violate the downward-closed safety region beyond that tolerance.
- **UNKNOWN:** neither decision was established before the iteration limit,
  a numerical check failed, refinement made no sufficient progress, or an
  LK approximation tolerance was reached without a safety verdict.
- **ERROR:** an execution/tool failure recorded by the experiment runner.
- **TIMEOUT:** a benchmark subprocess exceeded its whole-run deadline under
  a custom configuration. A per-Storm-call timeout is instead a tool error.
  The paper profile disables both deadlines.

Storm supplies the scalar oracle's sound-mode OVI bounds, but convex geometry
uses floating-point computations and numerical audits. The output is not an
exact-rational certificate of the complete verification procedure.

## Tests

```sh
python -m pytest
```

Storm-dependent tests skip when Storm is absent. With Storm installed, they
check exact rational graph fingerprints for every symbolic model and run
small end-to-end SAFE/UNSAFE cases. These tests are not performance experiments.
Small synthetic inputs under `tests/fixtures/` exist solely for regression
tests and are not evaluation benchmarks.

## Repository layout

```text
src/mdp_safety/             Verifier, geometry, query strategies, Storm and reporting
benchmarks/
  models/                  12 fixed symbolic PRISM inputs
  specs/                   24 fixed quadratic specifications
experiments/
  paper.json               One paper experiment profile
  manifests/               Model/specification pairs and SAFE/UNSAFE subsets
  metadata/                Model design and calibration
scripts/
  run_study.py             Explicit-run study driver (dry-run by default)
  fixed_inputs.py          Read/copy helpers; never generate inputs
tests/                     Unit, numerical-regression and real-Storm tests
docs/                      Algorithms, benchmark definitions and usage guides
results/                   Local generated measurements (ignored by Git)
```

## Documentation

- [Benchmark models](docs/benchmarks.md): state variables, transitions and model sizes.
- [Safety specifications](docs/specifications.md): risk channels and numerical calibration.
- [Query selection](docs/query_selection.md): L2, L-infinity, gauge and optimizations.
- [LK baselines](docs/sandwich_lk.md): absolute/relative inner-hull indicators.
- [Experiment guide](experiments/README.md): execution, exact metric definitions and provenance.
- [Architecture](docs/architecture.md): module responsibilities and extension points.
