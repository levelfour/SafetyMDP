# Quadratic specifications for the paper benchmarks

The suite contains 24 fixed quadratic specifications on 12 fixed symbolic MDPs:
Gridworld, Robot Mission and Network Routing, each at dimensions 4, 6, 8 and 10.

## Files and reproduction

- `benchmarks/models/`: 12 fixed symbolic PRISM files.
- `benchmarks/specs/`: 24 formula-input degree-two specifications.
- `experiments/metadata/model_metadata.json`: model sizes and design.
- `experiments/metadata/specification_metadata.json`: risk coefficients, scales,
  thresholds, numerical calibration and UNSAFE witness vectors.
- `experiments/manifests/domains.json`: the complete evaluation suite.
- `experiments/paper.json`: the single paper experiment profile.

The following calibration procedure describes how the saved specifications
were chosen, not work performed during each experiment. Verification consumes
the fixed JSON formulas and synthesizes its own schedulers; it does not read
the calibration policies or witness vectors to choose queries or verdicts.

The order in `terminal_labels` defines the probability coordinates:
`x1` is the reachability probability of the first label, and so on.
The benign `safe_completion` outcome is excluded from this list.

## Curved, convex, monotone specifications

Let `a_j >= 0` be the `m=2d+3` risk channels and `b_j > 0` their
normalizing budgets. Define

    z_j(x) = a_j^T x / b_j
    f_b(x) = (1/(2m)) sum_j [z_j(x)^2 + z_j(x)]
    S_b = {x >= 0 : sum_i x_i <= 1, f_b(x) <= theta}.

The JSON stores the expanded, threshold-one quadratic as `type: formula`:

    Q = C^T C / (2m theta), c = C^T 1 / (2m theta), C_j = a_j^T/b_j
    x^T Q x + c^T x <= 1.

All 24 specifications use `FormulaRegion`, including its feasibility recovery,
checked support optimization and supporting-direction audits. The separate
matrix-input geometry API is not used by this evaluation suite. Numerical
policies and implementation optimizations are described in
[query selection](query_selection.md).

Individual channels include all coordinate directions, so `Q` is positive
definite. All coefficients are nonnegative; thus the region is convex and
downward closed, contains a neighborhood of zero in the nonnegative orthant,
and satisfies the assumptions of all three specification-directed methods.
Tests also check that this constraint cuts the probability simplex, so it is
not a redundant curved inequality. This is a genuinely non-polyhedral region,
not `(a^T x)^2 <= b^2`. The linear term additionally makes the score
nonhomogeneous. Both linear and quadratic parts contribute equally when a
channel is at its nominal normalized budget.

The channel construction is shared across domains: d individual risks
`p_i`, d cyclic index pairs `0.6*p_i+p_(i+1)` with `p_(d+1)=p_1`,
total risk, severity coefficients `1/3+2*(i-1)/(3*(d-1))` for `i=1,...,d`,
and alternating coefficients `1,0.4,1,0.4,...`.
These coefficients are synthetic design choices, not measured loss estimates.
Index adjacency need not coincide with an edge in the underlying MDP.
Squared channel terms
penalize simultaneous large risks. The `b_j` are **scales in one aggregate
curved constraint**, not simultaneous hard per-channel caps.

In `specification_metadata.json`, `rows` contains the channel coefficients
and `risk_rows` their names. Top-level `budgets` gives the scales for that
particular SAFE or UNSAFE case; `calibration.safe_budgets` always records
the corresponding SAFE scales. `challenge_index` is a zero-based index of
the single tightened UNSAFE channel (null for SAFE).
`calibration.raw_threshold` is theta before the JSON normalization to one.

## Independent SAFE calibration

Let `alpha>0` lower-bound the one-step probability of reaching any terminal
outcome from every nonterminal state/action, including both adverse and benign
outcomes. It is not just the benign-abort probability.
An independent sparse Bellman LP supplies support estimates `s_j` and
sup Bellman residuals `r_j`. We use guarded upper bounds

    u_j = s_j + r_j/alpha + 1e-7 max(1, max_i a_ji)

with outward rounding. The saved scales use
`b_j = max(s_j + 0.004001, 1.02*u_j)`, combining an absolute margin and
2% relative slack. For every attainable vector `p`, `0 <= a_j^T p <= u_j`,
so `(a_j^T p)^2 <= u_j a_j^T p`. Consequently

    f_b(p) <= w^T p,
    w = (1/(2m)) sum_j a_j [u_j/b_j^2 + 1/b_j].

A further independent Bellman LP bounds `h_K(w)`, with the same scaled
residual guard, and `theta=1.02 * guarded_upper(h_K(w))`. This gives a SAFE
upper enclosure without maximizing the convex quadratic over schedulers,
running any refinement strategy, or selecting cases by measured performance.
It may be conservative; the 2% slack is relative to this upper enclosure, not
a claim of 2% distance from the true nonlinear optimum.

## Independent UNSAFE calibration

Use total risk for Gridworld, severity-weighted risk for Robot Mission,
and alternating risk for Network Routing.
Its greedy Bellman policy is fixed and evaluated using `(I-P)X=R`.
Each coordinate has error allowance `(residual+1e-12)/alpha+1e-10`.
The componentwise lower vector `ell` is clipped at zero.

Keep the SAFE threshold and all other channel budgets fixed. Tighten only
the challenged budget until `f_b(ell) >= 1.01 theta`, with a small additional
rounding margin. Monotonicity then gives an explicit UNSAFE witness under the
numerical bounds. Complete policy choices, probability estimates, lower
vectors, error allowances, and the changed budget are saved in metadata.
The SAFE and UNSAFE cases share the same MDP and raw theta. Tightening one
budget changes the expanded polynomial coefficients, although both JSON
formulas have the literal right-hand side one.

These are double-precision, guarded numerical certificates, not exact-rational
proof objects. No verifier performance or query count is used in calibration.
