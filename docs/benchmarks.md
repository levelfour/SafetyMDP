# Benchmarks

The suite contains three model families at dimensions `d=4,6,8,10`: 12 PRISM
models and 24 model/specification pairs. For each model, SAFE and UNSAFE cases
differ only in their specification. These are purpose-built domain models,
not measurements of deployed systems.

Here `d` is the number of labelled adverse terminal outcomes whose unbounded
reachability probabilities form the objective vector. Each model also has one
benign absorbing outcome, excluded from that vector. Thus the objectives are
subdistributions: `p >= 0` and `sum(p) <= 1`.

## Input locations

All paths below are relative to the repository root.

- `benchmarks/models/`: the 12 fixed symbolic PRISM models.
- `benchmarks/specs/`: the 24 fixed quadratic specifications.
- `experiments/manifests/domains.json`: all 24 cases.
- `experiments/manifests/domains_safe.json` and
  `domains_unsafe.json`: disjoint 12-case subsets.
- `experiments/metadata/model_metadata.json`: model sizes and design parameters.
- `experiments/metadata/specification_metadata.json`: risk channels, scales and calibration.

The runner reads these inputs without generating or recalibrating them.
A missing input is an error. `scripts/fixed_inputs.py` can copy their existing
bytes to another directory; it does not construct models or specifications.

## Gridworld

The four `benchmarks/models/gridworld-d*.prism` files describe an 11 by 11
grid using `row` and `col`, initially `(5,5)`. Rows increase southward.
See the [d=4 model](../benchmarks/models/gridworld-d4.prism).
Obstacles block movement; designated boundary cells are labelled hazards.

At a nonterminal cell, the scheduler chooses `north`, `east`, `south` or
`west`. Each command combines intended motion, lateral slip, a failed-move
self-loop and benign shutdown. Candidate-coordinate formulas implement walls
and boundaries: an invalid move stays in place. Hazard cells and the goal have
only the `absorb` self-loop, so five commands describe the whole grid.
Benign shutdown has probability 1/100 on every nonterminal step. Conditional
on no shutdown, intended motion, left slip, right slip and staying have
probabilities 18/25, 3/25, 3/25 and 1/25; these are not the unconditional
branch probabilities.

The coordinate `(10,10)` is the benign terminal for both goal arrival and
shutdown. The `d` hazard coordinates are distinct absorbing states labelled
`hazard_1` through `hazard_d`. Other boundary cells remain traversable.
All four dimensions have 107 reachable states.

## Robot Mission

The four `benchmarks/models/robot-mission-d*.prism` files use `d` Boolean
completion variables (`done1` through `done4` for `d=4`), observed `weather`
(0=clear, 1=storm), and `outcome`. Initially all completion bits are false, weather is clear and
`outcome=0`. See the [d=4 model](../benchmarks/models/robot-mission-d4.prism).

For each unfinished task, the scheduler chooses efficient or cautious
execution. Success completes the task; recoverable failure leaves the
completion bits unchanged. Weather can change after a nonterminal outcome.
A catastrophic failure enters the terminal associated with that task.
There are `2d+1` commands: two per task and one absorbing command.
After a nonterminal success, clear/storm probabilities are 4/5 and 1/5;
after a retry they are 1/3 and 2/3. The reachable state count is
`2^(d+1)+d-1`, including the absorbing outcomes.

While `outcome=0`, a true bit records successful completion.
`outcome=1..d` denotes task-specific catastrophic failure;
`outcome=d+1` denotes completion of all tasks or benign abort. Absorption
sets every bit to true and weather to 0, giving one state per terminal outcome.
The bits do not describe completion history after absorption.
The last successful task enters the benign terminal directly.
There is no energy state, reward or budget.

## Network Routing

The four `benchmarks/models/network-routing-d*.prism` files use `node`
and observed `congestion`, initially `(0,0)`. With `N=3d+4`,
nodes `0..N-1` are routers, `N` denotes delivery or benign packet drop,
and `N+i` denotes failure class `i`. Failure codes are terminal outcomes,
not physical routers. Congestion is 0 (clear) or 1 (congested);
all absorbing states use congestion 0.
See the [d=4 model](../benchmarks/models/network-routing-d4.prism).

Three commands describe routing decisions: `low_latency` advances by 3,
`balanced` advances by 2, and `secure_detour` goes back by 1,
clamped at node 0. Forward motion reaching `N` delivers the packet.
Retries stay at the current router. Route-dependent primary and secondary
failure classes and probabilities are formulas of node, congestion and mode.

After a successful nonterminal hop, congestion becomes clear/congested with
probabilities 7/8 and 1/8; after retry, the probabilities are 1/4 and 3/4.
A fourth command, `absorb`, loops at terminal outcomes.
The reachable state count is `7d+9`.

## Model sizes

| Domain | d | States | Choices | Transitions | Max actions/state | log10 schedulers |
|---|---:|---:|---:|---:|---:|---:|
| gridworld | 4 | 107 | 413 | 1808 | 4 | 61.41 |
| robot-mission | 4 | 35 | 133 | 741 | 8 | 17.66 |
| network-routing | 4 | 37 | 101 | 653 | 3 | 15.27 |
| gridworld | 6 | 107 | 407 | 1776 | 4 | 60.21 |
| robot-mission | 6 | 133 | 775 | 4567 | 12 | 94.05 |
| network-routing | 6 | 51 | 139 | 907 | 3 | 20.99 |
| gridworld | 8 | 107 | 401 | 1744 | 4 | 59.00 |
| robot-mission | 8 | 519 | 4105 | 24521 | 16 | 445.30 |
| network-routing | 8 | 65 | 177 | 1161 | 3 | 26.72 |
| gridworld | 10 | 107 | 395 | 1712 | 4 | 57.80 |
| robot-mission | 10 | 2057 | 20491 | 122811 | 20 | 1994.67 |
| network-routing | 10 | 79 | 215 | 1415 | 3 | 32.44 |

Counts refer to states reachable from the fixed initial state; symbolic
variable ranges can also encode unreachable valuations. The scheduler column
gives `log10` of the product of enabled-action counts over reachable states:
it counts deterministic memoryless schedulers, not all history-dependent or
randomized schedulers. The implementation does not enumerate these schedulers.

## Specifications and validation

The common safety score combines `2d+3` risk channels into a strictly convex
quadratic inequality. Each MDP's SAFE and UNSAFE cases share a calibrated
raw threshold; the UNSAFE case tightens one channel scale. Calibration is
independent of the query-selection methods. The numerical labels are not
exact-rational proofs; see [specifications](specifications.md) for the formula,
normalizers, guards and witness interpretation.

## Running experiments

See the [experiment guide](../experiments/README.md) for details.
