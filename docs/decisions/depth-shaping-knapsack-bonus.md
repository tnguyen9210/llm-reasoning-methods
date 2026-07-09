# `mcts_bl_cnt_v03`: reintroducing the depth-decay bonus as its own knapsack variant

*2026-07-09*

Records why the `f_a(z) = 1 - z**alpha` depth-decay bonus — the
selection term that was in the *original*, pre-rewrite
`mcts_bl_cnt_search_v02_00_00.py` and was removed from v02 earlier
today for not matching Fractional KUBE — was brought back as its
own explicit variant, `mcts_bl_cnt_search_v03_00_00.py`, rather than
folded back into v02 or discarded outright. Also records a sign error
caught before implementation.

## Where the idea came from

Proposed objective:

```
max   sum_i m_i * ( mu_hat_i + beta * f_a((d_max - d_i)/d_max) )
s.t.  sum_i m_i * (d_max - d_i) <= B_t
f_a(z) = 1 - z^alpha
```

with the stated intent: encourage exploration of shallow nodes via
the max term, restrict it via the cost constraint. This is a
fractional-knapsack objective (same shape as Fractional KUBE's Eq. 9)
with the UCB confidence bonus replaced by a hand-designed function of
tree position.

## The sign check

As written, `z = (d_max - d_i)/d_max` is the **cost fraction**: `z=1`
at the root (`d_i=0`, maximal remaining cost), `z=0` at `d_i=d_max`.
So `f_a(z) = 1-z^alpha` gives:

- root (`d_i=0`): `z=1` → `f_a = 1 - 1 = 0`
- max depth (`d_i=d_max`): `z=0` → `f_a = 1 - 0 = 1`

That is the **opposite** of the stated intent — the term as written
is *smallest* at shallow nodes and *largest* at deep nodes, i.e. it
rewards depth, not shallowness. Combined with the cost constraint
(which already makes shallow nodes expensive), the original
formula would have doubly penalized shallow exploration: nothing in
the objective was pulling toward it.

**Fix:** index `f_a` on the depth fraction instead of the cost
fraction: `z = d_i/d_max` (0 at root, 1 at max depth), same
`f_a(z) = 1 - z^alpha`. Now `f_a(0)=1` at the root (max bonus),
`f_a(1)=0` at max depth (no bonus) — monotonically decaying from root
to leaf, matching the stated goal. (Equivalently: keep the original
cost-fraction `z` and drop the `1 -`, i.e. `f_a(z) = z^alpha` — same
function, since `1 - (d_i/d_max) = (d_max-d_i)/d_max`. The
depth-fraction form was chosen for the implementation since it reads
directly as "how deep is this node," without an extra sign flip to
verify at the call site.)

## Why a new variant, not a v02 change or a discard

This is the same `f_a(z)=1-z^alpha` shape that was in the *original*
`mcts_bl_cnt_search_v02_00_00.py`, before this morning's rewrite to
match Fractional KUBE — removed then because it has no UCB/visit-
count term at all, so it doesn't implement the algorithm v02 is
supposed to be (`docs/decisions-log.md`, 2026-07-09 KUBE-rewrite
entry). Two options were on the table for reviving it:

- **Fold `f_a` into v02** as an additional term alongside the UCB
  bonus (`q + kube_bonus + depth_beta*f_a`, all over cost) — keeps
  the confidence-bound property while adding an explicit depth
  preference on top.
- **A separate v03** — same knapsack skeleton and cost mapping as
  v02, but `f_a` replaces the UCB bonus entirely; no visit-count
  term of any kind.

Chosen: **separate v03**. `f_a` is a deterministic, evidence-blind
function of tree position — it never shrinks as a node accumulates
visits, so it carries no confidence-bound/regret guarantee of any
kind. Mixing it into v02 would muddy what v02's ablation is testing
(cost normalization vs. PUCT, per
[kube-bonus-schedule.md](kube-bonus-schedule.md)) with an unrelated,
unbounded heuristic term. Keeping it as v03 preserves the clean
three-way comparison: v01 (PUCT), v02 (Fractional KUBE, evidence-
based bonus), v03 (fixed depth-shaping bonus, no exploration
guarantee) — same cost mapping and (optional) affordability
restriction across all three, differing only in what replaces the
"bonus" term.

## What v03 shares with v02, and what it doesn't

Shares: cost mapping (`max_depth - depth`), the knapsack-relaxation
reduction to `argmax over affordable arms of value/cost`, and the
`kube_affordable` feasibility-restriction step
([kube-affordability-restriction.md](kube-affordability-restriction.md))
— the constraint `sum m_i*cost_i <= B_t` is identical, only the
per-arm value term differs, so the same restrict-before-argmax logic
applies unchanged.

Does not share: any visit-count, parent-visit, or global-clock
quantity. `depth_density` takes no `t` and no schedule argument — the
bonus is a pure function of `(depth, max_depth, depth_beta,
depth_alpha)`, computed identically whether a node has been visited
once or the entire search is finished. There is consequently no
theoretical guarantee that this variant converges toward the true
best arm as budget grows; `depth_beta`/`depth_alpha` are tuning knobs
for a heuristic depth preference, not exploration coefficients in the
bandit sense.

## Naming

Knobs are `depth_beta`/`depth_alpha`, not `kube_beta`/`kube_alpha`
(the original pre-rewrite names) — deliberately avoiding the `kube_`
prefix since this variant has no UCB/confidence-bound structure, and
calling it "kube" would repeat the exact mislabeling this morning's
audit corrected for v02 (`docs/decisions/
kube-affordability-restriction.md`).

## Connections

- `core/mcts_bl_cnt_search_v02_00_00.py` — the sibling this shares
  its knapsack skeleton, cost mapping, and affordability logic with.
- [kube-bonus-schedule.md](kube-bonus-schedule.md),
  [kube-affordability-restriction.md](kube-affordability-restriction.md)
  — the v02 decisions this variant's shared machinery was audited
  against.
- `docs/decisions-log.md`, 2026-07-09 KUBE-rewrite entry — records
  why the original depth-decay formula was removed from v02.
- `core/mcts_bl_cnt_search_v03_00_00.py::MCTSNode.depth_density` —
  the implementation.
