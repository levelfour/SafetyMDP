# Lammel-Kuefer-style sandwich baselines

The two implemented LK baselines are `sandwich-lk` (absolute epsilon) and
`sandwich-lk-gauge` (relative gauge). They use the shared Storm OVI scalar
oracle and verification stopping rules. They are Python implementations, not
calls to Storm's native multi-objective optimization routine.

Reference: I. Lammel and K.-H. Kuefer, *The convergence rate of the Sandwiching
algorithm for convex bounded multiobjective optimization*, Algorithm 1,
Definitions 2.1/2.2 and Algorithm 3
([author-hosted preprint](https://optimization-online.org/wp-content/uploads/2023/12/convergence_rate_sandwiching_preprint.pdf)).
The definitions below specify the implemented probability-maximization variant.

## Inner hull and direction indicators

Let `v` range over componentwise reachability lower bounds for sampled
schedulers. These are lower-bound vectors, not exact Pareto contact points.
After initialization, the downward inner hull is `I = conv(v) - R_+^d`;
`Q` is the current outer approximation. The inner hull's upper facets are
`a_j*x <= b_j`, with `a_j >= 0` and `b_j = max_v a_j*v`.

| CLI method | Indicator maximized over inner facets | Tolerance units |
|---|---|---|
| `sandwich-lk` | `max(0, (h_Q(a_j)-b_j)/L1(a_j))` | Absolute probability difference |
| `sandwich-lk-gauge` | `max(0, (h_Q(a_j)-b_j)/b_j)` | Dimensionless relative excess |

Here `L1(a_j)` is the L1 norm. The absolute indicator is the least nonnegative epsilon such that
`x-epsilon*1 in I` for every `x in Q`.
The relative indicator is the largest `gamma_I(x)-1`, clipped below at zero.
The implementation computes one outer-support LP per inner facet and chooses
the largest indicator. Tests compare these scores with independent
convex-combination LPs over all outer vertices.

After coordinate initialization, a successful selection evaluates every
enumerated inner facet; it does not stop at the first positive or eligible
gap. Exact ties use deterministic lexicographic facet order. Facet normals
are stored and passed to the scalar oracle with L2 normalization; each
component is therefore in [0,1]. Positive rescaling does not change either
indicator. L2 normalization here is not the L2 projection strategy.

The fixed safety region is used for the shared SAFE/UNSAFE decisions, not for
these direction scores. In particular, `gauge` uses the fixed safety region
whereas `sandwich-lk-gauge` uses the evolving inner hull.
The draft's Propositions 5.4, 5.6 and 5.8 concern the specification-directed
rules; they are not the direction-selection rules of these two baselines.

## Initialization and assumptions

1. Start with the subdistribution simplex `Q_0` and no sampled scheduler.
   Unless a verdict or error ends the run first, query `e_1,...,e_d`, once
   each. Initialization advances by the number of completed oracle samples,
   not the number of distinct vectors, so duplicate schedulers do not block it.
   These queries count as ordinary loop iterations and oracle calls.
2. Add each scheduler's lower vector to the inner hull. Except on an UNSAFE
   witness, attempt to retain the Storm upper cut even if it does not exclude
   the selected outer point. A cut is skipped when an existing approximately
   same-direction cut is at least as strong. This is not a general
   redundancy LP: cuts implied by other directions can remain, and a stronger
   same-direction cut does not delete the older weaker one.
   A newly retained cut or inner-hull expansion can provide refinement
   progress; initialization itself also permits the next coordinate query.
3. The relative variant fixes the reference at `c=0`, a lower reference for
   nonnegative probability objectives; it does not estimate a nadir point.
4. After initialization, relative indicators require positive inner support
   in every nonzero nonnegative direction. A zero-support facet makes the
   origin-based gauge unavailable and produces UNKNOWN. There is no artificial
   positive floor or implicit fallback to the absolute method. Nonfinite ratios
   also fail the numerical check.
5. Storm supplies finite-precision upper bounds and scheduler lower vectors,
   rather than exact scalar optima and contact points. Exact-oracle convergence
   rates are not established by this implementation.

The common loop checks outer containment before each selection and checks
every sampled scheduler for UNSAFE immediately after its coordinate evaluation.
Consequently, either verdict can stop the run before all d initialization
queries finish. If precision retries are needed after initialization, they
reuse the selected weight; they are extra oracle calls, not new selections.

## Numerical implementation and termination

Paths here are relative to `src/mdp_safety/`.
`strategies/sandwich.py` handles initialization and caches facets when the
deduplicated sample set is unchanged. `inner_facets.py` scales the sample
coordinates, translates by the all-ones vector and enumerates a bounded polar
polytope using SciPy/Qhull. Supports are recomputed on the original, unrounded
lower samples; no random perturbation is used. This floating-point procedure
is not an exact proof of facet completeness.

`strategies/sandwich_lk.py` recomputes the outer-support LP for every facet
on every post-initialization selection, even when the inner facets are cached.
All-facet LP work, facet construction and indicator evaluation belong to
query-selection time. These are SciPy/HiGHS geometry LPs, not Storm oracle
queries.

The LK selector itself does not enumerate outer vertices. However, the shared
verification loop still enumerates them before selection for the containment
check. LK therefore remains exposed to outer-vertex enumeration costs and
failures. Those verifier-level costs are recorded separately from selection.

Reaching `--sandwich-precision` only establishes the checked approximation
tolerance. If neither outer containment nor an unsafe scheduler has established
a verdict, the result is UNKNOWN, not SAFE. The terminal all-facet check is
stored as a selection record with no weight and no oracle query.

## Reading results

Results use the CLI method name as `projection` and metric prefixes
`sandwich-lk-epsilon-*` or `sandwich-lk-gauge-*`. The suffix identifies the
stage; scores from different stages must not be conflated.

| Metric suffix | Recorded score and behavior |
|---|---|
| `coordinate-initialization` | Absolute coordinate gap `max(0,h_Q(e_i)-h_I(e_i))` for **both** methods; inner support is zero when no scheduler has yet been sampled, and the relative tolerance is not applied at this stage |
| `max-inner-facet-gap` | The selected absolute or relative indicator from the table above; a direction is sent to the oracle |
| `tolerance-check` | The nonnegative maximum checked indicator; empty weight and no oracle query, returning UNKNOWN |

For a nonterminal selection, `counterexample` is an outer-support LP
optimizer. `projected` is an inner supporting sample (zero before any
sample), not a nearest-point projection. In the shared query-record schema,
`safe_support` actually stores the **inner-hull support**, not the support of
the safety specification. `direction_score` is null: the LK indicator is not
the fixed-safe-region score used by L2, L-infinity and Gauge.
The terminal record's `no-eligible-facet-no-oracle` tag means that the
checked maximum is within tolerance, not that no facets were enumerated.

Selection timing excludes the later weighted Storm query, same-scheduler
coordinate evaluations and OVI retries. A general selection `RegionError`
returns UNKNOWN without storing the failed attempt's selection time; the
terminal tolerance check above is the exception that is explicitly timed.
Unhandled LP/tool failures are recorded by the experiment runner as ERROR.
Checkpoints verify method names, implementation versions and the
strategy-catalog revision before resuming.

See the [experiment guide](../experiments/README.md) for metric definitions
and the distinction between loop iterations, selection records and oracle calls.

The [2026-10-04 native-Storm rerun](../experiments/README.md#measured-native-storm-rerun-2026-10-04)
decided 14/24 cases with LK-epsilon and 18/24 with LK-gauge. Their 10 and
6 UNKNOWN outcomes respectively all reached the 50-iteration limit, not
the LK approximation-tolerance stopping rule or an execution error.
Both variants remained undecided on Robot Mission d6/d8/d10 UNSAFE, while
Network Routing d10 UNSAFE was decided by the first initialization query.
These results illustrate that approximating the entire attainable region
need not prioritize the direction revealing a specification violation.

## Running the two baselines

For the complete five-method profile, use the [experiment guide](../experiments/README.md).
To run only the two LK methods on the same 24 cases, use:

```sh
mdp-safety-bench experiments/manifests/domains.json \
  --projections sandwich-lk sandwich-lk-gauge \
  --output-dir results/lk-only \
  --timeout none --run-timeout none --max-iterations 50 \
  --precision 1e-7 --min-precision 1e-11 --geometry-tolerance 1e-8 \
  --sandwich-precision 1e-6 --repetitions 1
```

Run this from the repository root after the installation steps in the
[root README](../README.md). Use a fresh output directory. This schedules
48 jobs, with one repetition and neither whole-run nor Storm-call timeouts.
The fixed order is Gridworld, Robot Mission, Network Routing; within each
family d=4,6,8,10; at each dimension SAFE then UNSAFE; and for each case
`sandwich-lk` then `sandwich-lk-gauge`. The runner flushes a `START` line
before each job and a `DONE` line with its verdict and elapsed time afterward.

The 50-iteration limit includes initialization and terminal checks, not just
successful cuts. There is no separate selection-time cap and no wall-clock
completion guarantee. Equal numerical tolerances in the absolute and relative
variants do not represent equal geometric accuracy. Aggregate expected-SAFE and expected-UNSAFE
cases separately, and report decisions as well as approximation indicators.
