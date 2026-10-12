# Query selection and implementation optimizations

The three specification-directed methods are `l2`, `linf` and `gauge`.
The two [LK baselines](sandwich_lk.md) use different, inner-hull-based scores.
All five share the verification loop and the Storm OVI scalar oracle.

The MDP frontend uses nonnegative query directions and starts from the
subdistribution simplex `D = {p >= 0: sum(p) <= 1}`. The input safe region
`S` is a closed, convex, downward-closed subset of D. This is the bounded safe
region used in the draft. Extending it downward by the negative orthant would
leave its supports on nonnegative directions unchanged, but the implementation
computes membership, projections and the gauge using the bounded region.
The gauge reference is zero. The frontend does
not search arbitrary cones, negative oracle weights or nonzero references.

The current paper specifications are quadratic polynomials entered as
`type: formula`, handled by `safe_regions/polynomial.py` (`FormulaRegion`).
They do not use the separate matrix-input `QuadraticRegion` backend.

## Common score

For the fixed safety region S and current outer approximation Q, the three
specification-directed methods use

    v_T(w) = (h_Q(w) - h_S(w)) / h_T(w),  w >= 0, w != 0.

| CLI | Denominator | Geometric maximization |
|---|---|---|
| l2 | Euclidean norm of w | Farthest outer vertex under Euclidean projection |
| linf | L1 norm of w | Farthest outer vertex under L-infinity projection |
| gauge | h_S(w) | Maximum gauge excess at an outer vertex |

The L-infinity denominator is the **dual L1 norm**, not the L-infinity norm.
At the start of each loop iteration, the verifier enumerates Q's vertices
and checks containment using `--geometry-tolerance`. If all vertices pass,
it returns SAFE without selecting a new direction. Otherwise, these three
methods score the vertices outside the safe region; they do not sample a
subset or optimize the direction through a local nonconvex search.

For the formula backend, L2 and L-infinity compute or reuse a projection for
each candidate, select the largest distance, and compute the supporting
direction and safe-region support only for the winner. Gauge evaluates all
candidate gauges and similarly performs the normal/support checks only for
the winner. Exact candidate ties retain the first vertex in the deterministic
outer-vertex order. Stored weights are nonnegative and normalized by their
maximum component, not by their sum. The scores are invariant to positive
rescaling, so this does not change the mathematical direction objective.
Absolute oracle precision and numerical separation thresholds are not invariant
under this change of scale: identical finite-precision query sequences are not
guaranteed when comparing different normalization conventions.

The selected direction must pass numerical support and score-agreement
checks. Failure returns UNKNOWN; there is no automatic switch to a different
method or to a safety-facet heuristic.

For an L-infinity projection q of p at distance delta, both

    h_S(w) = w.q
    w.(p-q) = delta * ||w||_1

are needed. Downward closedness permits q=(p-delta)_+.
The direction uses distance-saturating coordinates and adjusts for active
nonnegativity constraints. A boundary gradient alone need not satisfy both
conditions. For smooth formula regions, the normal is the boundary gradient
masked to the distance-saturating coordinates, subject to the numerical
branches and audits described below. It does not accept an arbitrary boundary
normal.

Gauge uses a radial boundary point and checks (h_Q-h_S)/h_S against the
maximum gauge minus one. It additionally requires a numerically positive
absorption radius: `rho*D subset S`, with
`rho > max(1e-12, 100*region.tolerance)`. A merely feasible origin is not
enough. The batch shares the radius calculation within `gauge_scores`;
it is not cached across selections. For formula regions, each candidate
still runs a scalar membership bisection and internally constructs a radial
point; the winning candidate's `gauge` is evaluated again before its normal
is constructed. Thus this optimization is not a closed-form quadratic gauge
nor a claim that all boundary computation occurs only once.

Paths in this section are relative to `src/mdp_safety/`.
`direction_scores.py` defines units and common audits;
`strategies/norm.py` and `strategies/gauge.py` implement the reductions.
Selections record outer/safe support, support gap, normalizer, signed score and
positive part. The three specification-directed strategies record
`global-geometric-reduction-numerically-checked`; this is a numerical
audit, not an exact-arithmetic global-optimality certificate.
LK indicators are documented separately in [sandwich_lk.md](sandwich_lk.md).

## Correspondence to the draft propositions

For the October 4 draft, the three direction constructions correspond to
Propositions 5.4, 5.6 and 5.8. The outer approximation called Q here is U in
the draft, and the error unit T is E there. For
`S = {x in D: f(x) <= theta}`, assume differentiable convex coordinatewise
nondecreasing f and `f(0) < theta`. Let x be a maximizing outer vertex.

| Rule | Boundary point q | Unnormalized direction r |
|---|---|---|
| L2, Proposition 5.4 | Euclidean projection of x onto S, computed numerically | x-q (tiny negative components are clipped in the implementation) |
| L-infinity, Proposition 5.6 | q=(x-delta*1)_+, with delta the minimum feasible uniform decrease | grad f(q) masked to I={i: x_i-q_i=delta} |
| Gauge, Proposition 5.8 | q=x/gamma_S(x), found by radial membership bisection | grad f(q), without masking |

The draft uses `w=r/||r||_1`. The implementation uses
`w=r/max_i(r_i)`; this selects the same ray, not literally the same normalized
vector. L2 projection and all numerical direction certificates are approximate.

For L-infinity with x in D and positive delta, exact arithmetic gives
`f(q)=theta`: otherwise continuity allows a slightly smaller delta while
remaining in S, since the resulting point is still componentwise at most x.
Convexity gives `grad f(q).q >= theta-f(0) > 0`. Since q is zero outside I,
the masked gradient cannot vanish. Thus the draft's otherwise-all-ones branch
is unnecessary in this exact setting; it is not an alternative query heuristic.

The current shared sublevel implementation nevertheless retains a numerical
slack-formula branch that starts with the all-ones simplex normal. With
`a=max(1e-10,100*region.tolerance)`, it takes that branch when
`f(q)<theta-a`; otherwise it clips the gradient to nonnegative components.
For L-infinity it treats residuals within a of the largest residual as
saturated and zeros nonsaturated components at coordinates q_i<=a.
It raises RegionError if the resulting maximum component is at most
region.tolerance; it does not switch a numerically zero masked gradient to
uniform weights. A small positive gradient can trigger this absolute test
after rescaling f and theta, even when the exact safe region is unchanged.
Every constructed candidate still has to pass the support/score audits.
The 2026-10-04 rerun did not change these branches or normalization.

After constructing r, both norm rules and Gauge additionally optimize
`h_S(w)` and check the supporting and score identities. These independent
numerical checks are implementation overhead beyond the closed-form gradient
construction in the propositions. Their optimization/retry time belongs to
query-selection time, and failure can stop the run with UNKNOWN.

## Implementation optimizations

| Component | Optimization |
|---|---|
| Nonlinear L-infinity distance | Monotone scalar bisection of membership of (p-t)_+, instead of a multivariable distance program |
| Gauge | Evaluate all candidate gauge values; construct the supporting normal, solve safe-region support and audit the direction only for the winner |
| Formula-region gauge | Share the absorption-radius calculation within the candidate batch; scalar bisection stops at adjacent floating-point values or its 100-step cap |
| Projection/support | Geometry-aware bounded LRU caches; invalidate on point/direction, coefficients, shape or tolerance changes |
| Outer approximation | Cache vertices until cuts/tolerance change; reuse them for support values and direction audits |
| Polyhedral geometry API | Vectorized gauge ratios and sorted cumulative-sum L-infinity distances |
| Polyhedral L2 API | Bound-based candidate pruning; dual lower bounds and feasible-point upper bounds |

The last two rows apply to `PolytopeRegion`, not to the current paper
specifications. In particular, formula-region L2 selection has no effective
distance-bound pruning: its default candidate bounds are zero and infinity.
Both formula norm methods can reuse cached projections of unchanged vertices.
Projection and safe-support caches hold at most 32,768 and 4,096 entries,
respectively; candidate gauge values themselves are not cached.

Batching/caching preserves the mathematical problem but floating-point ties,
solver accuracy and ordering can affect the exact query sequence.
MDP oracle results are not replaced by a cross-run cache.

## Numerical policy

The tolerance below is `region.tolerance`, set by the CLI to
`--geometry-tolerance / 100` (1e-10 for the paper profile).

- Formula-region L2 projections and supports use SLSQP with analytic gradients.
  Projection candidates undergo feasibility recovery; this alone does not
  certify every candidate's exact projection optimum. The winning direction
  receives the support/score audits described above.
- Formula-region support is accepted only after a feasible-point check and a
  convex tangent-relaxation primal/dual gap check. If the first attempt fails
  this check, there is one retry in axis-radius-scaled coordinates. An unchecked
  solver success flag is insufficient. These are numerical audits, not
  exact-arithmetic or interval certificates for the geometry.
- These SLSQP calls have a 3,000-iteration optimizer limit per call, distinct
  from the verifier's `--max-iterations`. L-infinity and gauge use scalar
  bisections for their distances/radii rather than the same optimizer.
- For a candidate q, gradient g and level theta, the tangent bound is
  g.x <= theta-f(q)+g.q. Nonnegative multipliers lambda,mu with
  lambda*g+mu*1 >= w yield an upper support bound. The check uses the
  probability simplex as well as the tangent inequality.
- Supports are checked after max-absolute-weight normalization, with tolerance
  max(5e-8,10*region.tolerance). A failed check propagates as UNKNOWN.
- The local supporting-direction check uses
  max(5e-7,1000*region.tolerance)*max(1,abs(excess),abs(score)).
  The subsequent Q-support audit uses the same leading factor times
  max(1,abs(excess)). Gauge checks divide by the support denominator, so small
  denominators can amplify numerical errors.
- Outer vertices use SciPy/Qhull with a floating-point combinatorial fallback.
  The fallback is guarded at dimension 8 and 40 inequalities; exceeding either
  limit after the Qhull path is unavailable produces UNKNOWN. This is a limit
  on the fallback's problem size, not a cap of 40 vertices. All five methods,
  including LK, use this enumeration for the shared safety check.
- Feasibility recovery, outward rounding, vertex deduplication and numerical
  tolerances are implementation choices, not exact geometric transformations.
  Very thin or degenerate regions can produce UNKNOWN. Other unhandled geometry
  or tool failures are reported as execution errors.
- The formula parser accepts a restricted polynomial AST. Convexity assumptions
  are part of the input contract; arbitrary functions are not supported.

## Verification and measurement

SAFE follows the vertex containment check. UNSAFE follows a sampled
scheduler's componentwise lower probability vector violating the safe region
by more than `--geometry-tolerance`. The outer vertex used for query selection
is only a candidate counterexample; it is not an UNSAFE witness.
Neither a numerical failure nor exhausting the budget implies UNSAFE.

Each completed oracle attempt comprises one weighted Storm optimization and
d coordinate evaluations of the same native Storm scheduler. Stormpy applies
the scheduler and reuses the induced DTMC; the exported action map is only an
artifact. See [native scheduler evaluation](architecture.md#native-scheduler-evaluation).
If that interval
cannot establish a witness or sufficient refinement progress, the engine
retries the same weight at tighter OVI precision, without repeating selection.
The paper precision schedule is 1e-7, 1e-8, ..., 1e-11.

The paper profile permits 50 loop iterations, with both whole-run and
Storm-call timeouts disabled. Initialization, terminal containment checks and
failed selections can consume iterations; the limit is not 50 successful cuts
or necessarily 50 oracle calls. There is no separate query-selection time
allowance. Numerical limits inside geometry solvers remain in force.

Recorded selection wall/CPU time covers candidate scoring/projection, direction
construction, safe-region support optimization (including its numerical retry)
and direction audits. It excludes the subsequent Storm optimization,
same-scheduler evaluation and OVI precision retries. Verifier-level outer-vertex
enumeration is measured separately; cached vertex access during an audit is
inside selection time.

A successful SAFE containment check adds no selection record. The LK terminal
all-facet tolerance check does add a record, without issuing an oracle query.
A general selection/audit `RegionError` adds no record for that failed
attempt: its time is absent from the selection sum, but included in overall
runtime. Thus the last stored L2/L-infinity record need not be the final loop
attempt, and a short record does not establish that later selections get
cheaper. See the [measurement definitions](../experiments/README.md).

In `outer_halfspaces`, rows such as `probability-lower-1` have normals
`-e_i` because they encode `x_i >= 0`. They are initial domain constraints,
not queries or negative oracle weights, and do not contribute to query counts.

Tests compare scores and distances against analytic examples and independent
LPs, exercise scale invariance and zero coordinates, and check that failed
audits issue neither unverified queries nor false UNSAFE decisions.
Real-Storm regression tests are included; their runtimes are not benchmark
performance results.
