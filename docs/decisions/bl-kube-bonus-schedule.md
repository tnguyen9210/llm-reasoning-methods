# BL-KUBE-MCTS's KUBE bonus schedule: global clock vs. per-parent (UCT-style) clock

*2026-07-09 (the variant discussed here is now `mcts_bl_kube_v01`,
then named `mcts_bl_cnt_v02`; this file's own narrative below is
left in its original 2026-07-09 terms, since that's the name the
variant had at the time this decision was made)*

Records the discussion and decision behind
`BLMCTSKubeV01Config.kube_schedule` — which clock drives the
exploration bonus in `MCTSNode.kube_density`'s fractional-KUBE index.
This was a real back-and-forth, not a single call: the first
implementation shipped with a global clock, chosen by analogy to
`mcts_bl_sem_v01`; that analogy was then re-derived from scratch,
found not to transfer, and replaced with a UCT-style local clock,
with both kept as a configurable ablation. A short note on the
zero-visit floor (a related but minor fix) closes the file.

## Starting point: the formula as first written

`kube_density` computed

```
density(x) = (q_value(x) + kube_c*sqrt(log(1+t)/visit_count(x))) / cost(x)
cost(x) = max_depth - depth(x)
```

with `t` = number of frontier selections so far, a single counter
shared by every node — the same schedule already in use for
`mcts_bl_sem_v01`'s diversity bonus
([global-vs-local-exploration-schedule.md](global-vs-local-exploration-schedule.md)).
At the time this felt like a natural transplant: both algorithms
select globally over an explicit `leaf_nodes` frontier rather than
walking root-to-leaf, so "use the schedule that already works for the
other frontier-based algorithm" seemed like the safe default.

## The question that started the re-derivation

*"Does the UCB bonus actually matter, given that visit counts tend to
stay flat for most nodes? And why global `t` for v02 but not for
v01's PUCT?"*

Working through the node lifecycle in
`mcts_bl_kube_search_v01_00_00.py` (then named
`mcts_bl_cnt_search_v02_00_00.py`) answered the first half directly.
`MCTS.create_child` gives every new node exactly one `update()` call
at birth (`new_node.update(candidate_score)`), so `visit_count() == 1`
the instant it enters `leaf_nodes`. The outer loop in `mcts_search`
then does, unconditionally, every iteration:

```python
selected = agent.select_child_from_list(leaf_nodes, t)
leaf_nodes.remove(selected)
```

Every node leaves the frontier the first time it's selected — whether
it expands (children replace it) or is terminal (it backprops and is
gone). No node is ever compared against others on the frontier more
than once. So **own `visit_count()` is 1 for essentially every node
at comparison time**, and it never varies across the frontier at any
given iteration.

Plugging that in: with `visits == 1` universally, `kube_c*sqrt(log(1+t)/visits)`
reduces to `kube_c*sqrt(log(1+t))` — a value shared by every node on
the frontier at that iteration. It still perturbs the ranking (it's
added to the numerator before dividing by each node's different
`cost`, so it proportionally favors low-cost / high-depth nodes), but
it does **not** do a UCB bonus's actual job of boosting under-explored
nodes relative to over-explored ones — there is no visit-count
variation left for it to discriminate on. The bonus was real, but
inert in the one respect it was added for.

## Why v01's local clock was never a candidate for "fix," and why that made v02's case suspicious by contrast

The second half of the original question — why local for v01, global
for v02 — has its own answer, already settled in
[global-vs-local-exploration-schedule.md](global-vs-local-exploration-schedule.md):
v01's PUCT bonus, `cpuct*sqrt(log(parent_visits)/visits)`, is a valid
confidence bound specifically *because* `parent_visits`/`visits` are
exact local counts of how many times this node and its parent were
tried. A global elapsed-time counter would compare mismatched
quantities there. That reasoning is airtight for v01 and was never in
question.

What made the *v02* choice worth re-examining is that it wasn't
reached the same way — it was reached by analogy to `bl_sem`, and the
analogy's load-bearing claim was never checked against KUBE's actual
mechanism. `bl_sem`'s bonus is `sqrt(x^T V^-1 x)`, and `V` is
genuinely global: it accumulates evidence from every selection
anywhere in the tree, so indexing its schedule by global elapsed time
`t` correctly tracks "how much has `V` learned so far," independent of
which node is being scored. KUBE's bonus has no analogous global
accumulator — the only quantity in it besides `t` is each node's own
`visit_count()`, which is local, and (per the lifecycle above) inert.
Carrying over `bl_sem`'s conclusion carried over the *label* ("global
is the principled choice for a flat, globally-shared frontier")
without carrying over the *mechanism* that made the label true there.
It wasn't present in KUBE, so the conclusion didn't transfer.

## Reframing as a tree-search question: UCT is UCB1 made local, not a different bonus

The clarifying question was: since we're applying a multi-armed
bandit idea (KUBE) to tree search, should we think in tree-based
UCB/UCT terms instead of flat UCB1?

Tran-Thanh et al.'s Fractional KUBE (arXiv:1204.1909 sec. 3.3;
reference implementation in the sibling `budget-mab` repo,
`src/algorithms.py::FractionalKUBE`) uses the standard flat-bandit
UCB1 bonus, `sqrt(2 log(t)/n_i)` — one bandit over all arms, one
shared clock `t`. Kocsis & Szepesvári's UCT is not a competing bonus
formula; it's UCB1 re-instantiated once per internal node. In that
per-node bandit, "total pulls so far" is exactly the parent's visit
count, so

```
sqrt(2 log(t)/n_i)               (UCB1, flat bandit)
sqrt(log(N_parent)/n_child)       (UCT, per-node bandit)
```

are the same confidence bound at different scopes — `N_parent` is the
local bandit's clock. Adapting KUBE's UCB1 bonus into a tree the
standard way therefore means indexing the clock by `parent_visits`,
not by a frontier-wide `t`. That is precisely `mcts_bl_cnt_v01`'s PUCT
bonus (`cpuct*sqrt(log(parent_visits)/visits)`,
`mcts_bl_cnt_search_v01_00_00.py::MCTSNode.puct`) — so "think in UCT
terms" and "reuse v01's bonus" turned out to be the same answer
reached from two different directions.

This does depart from the letter of Tran-Thanh's formula (no single
global `t`), but the letter without the mechanism it depends on
(repeated pulls of the *same* arm, `n_i` genuinely growing over time)
buys nothing: a frontier-wide constant isn't "more faithful KUBE," it
is KUBE's bonus term degraded to a no-op by a node lifecycle the
original derivation never had to account for.

### A naming aside: PUCT vs. UCT

Worth flagging honestly: `mcts_bl_cnt_v01`'s `puct()` is UCT-shaped
(`log(parent)/n`, no prior term), not PUCT proper. PUCT (Rosin 2011;
popularized by AlphaZero) adds a prior weighting,
`Q + c*P(s,a)*sqrt(N_parent)/(1+n)`, justified by prior-dependent
regret bounds rather than a Chernoff/UCB1-style bound. We likely
don't need to add one: each node's `q_value` is already initialized
from its PRM score at birth, so a prior-like signal is already folded
into the value estimate rather than sitting in a separate term. A
true PUCT variant (prior = softmax over sibling PRM scores) is a
possible future variant, but it would change the algorithm's
semantics, not just its schedule — out of scope here.

## The conditional that sharpened the case for `"parent"`: short trees, frequent terminals

A further hypothesis was raised: if the tree is short and hits
terminal states often, do node visit counts vary more — and if so,
should v02 use the same local, parent-visit-based bonus as v01?

Checking this against the lifecycle above: no, a short/terminal-heavy
tree does **not** change the conclusion that a node's *own*
`visit_count()` stays at 1 — `leaf_nodes.remove(selected)` still fires
unconditionally on every selection, terminal or not, so no individual
frontier node is ever revisited in any regime. What frequent terminal
hits *do* change is `parent_visits`: every terminal selection
backprops via `update_recursive` up through every ancestor, so short
trees with frequent terminal hits mean this fires often, and
ancestor visit counts grow — and diverge across branches — quickly.

That is exactly the quantity `"parent"` conditions on, and exactly
the quantity `"global"` ignores. `parent_visits` tracks how much
evidence has flowed through a specific branch, varies across the
frontier, and moves fast precisely in the short-tree regime — the
situation where "boost the neglected branch" bonuses are supposed to
earn their keep. In a deep tree with rare terminals, `parent_visits`
stays near 1, `log(1) = 0`, and the index gracefully falls back to
plain `q_value/cost` — a sensible default when there's no real
evidence asymmetry yet. The global clock, by contrast, drifts away
from `q/cost` purely as a function of elapsed time, identically for
every node regardless of tree shape or where terminal hits are
concentrated — the one thing it cannot do is react to *where* in the
tree the evidence is accumulating, which is the thing the short-tree
hypothesis specifically cares about.

## The ablation argument (the decisive practical reason)

With `"parent"`, v02's index becomes:

```
density(x) = puct(x) / cost(x)
```

— `mcts_bl_cnt_v01`'s exact PUCT/UCT score, divided by remaining
depth budget. v01 and v02 then differ in exactly one factor, cost
normalization, which is the entire point of introducing the KUBE
variant in the first place. Under `"global"`, the schedule and the
normalization change simultaneously, confounding any v01-vs-v02
comparison — a worse experimental design regardless of which schedule
turns out to perform better.

## Decision

`BLMCTSKubeV01Config.kube_schedule: str = "parent"` (`"parent"` |
`"global"`, same vocabulary as `BLMCTSSemConfig.ds_alpha_schedule`),
in `MCTSNode.kube_density`:

```
"parent" (default):  kube_c * sqrt(log(parent_visits(x)) / visits(x))
"global":             kube_c * sqrt(log(1 + t) / visits(x))
```

Both are implemented and selectable, not just the default —
`"global"` is kept as an explicit ablation arm (the literal
flat-bandit/KUBE clock, the reading closest to a Coquelin &
Munos-style flat UCB over a growing leaf set) rather than deleted, so
the schedule question stays empirically answerable instead of being
silently foreclosed by whichever default we pick.

**Caveat carried forward honestly, not papered over:** frontier
selection compares nodes with *different* parents against each
other — mixing several local bandit instances into one global
ranking. Neither UCT (path-local bandits, no cross-branch comparison)
nor flat UCB1 (one bandit, a stable arm set) exactly covers a growing,
cross-parent arm set where every arm is pulled at most once before
leaving for good. Whichever schedule is chosen, the frontier-wide
comparison itself remains a heuristic layered on top of either
theory, not a literal instantiation of either. `"parent"` is the
better-justified heuristic of the two, not a fully principled one.

**Why:** the global-clock choice was inherited by analogy without
re-deriving whether the mechanism it depended on (a genuinely global
accumulator) existed in KUBE. It didn't — KUBE's only per-node
quantity is `visit_count()`, which the frontier lifecycle pins at 1.
Reframing as "UCT is UCB1 made local" showed the fix was to index the
clock by `parent_visits`, which both restores real per-node
discrimination (especially under short-tree/frequent-terminal
dynamics) and turns the v01-vs-v02 comparison into a clean,
single-factor ablation.

## Minor follow-up: zero-visit / zero-clock floor

While implementing the above, a small related inconsistency surfaced:
the original code set `bonus = inf` whenever `visit_count() == 0`.
That forces the search to exhaust every newly-created node before any
q_value-informed comparison can happen at all (an `inf` bonus wins
against any already-visited node unconditionally) — affordable only
when the budget can visit everything once for free, which
contradicts the entire premise of a *budget-limited* search.
`mcts_bl_cnt_v01`'s `puct()` already handles this the other way
(`parent_visits == 0 or visits == 0: u = 0.0`, so an unvisited node
scores on `q_value` alone). `kube_density` was changed to match:
`visits == 0` or `clock == 0` now yields `bonus = 0.0`. In the current
lifecycle this is a dead-code-path correction (every node gets its
one `update()` before it can ever be compared), but it keeps the two
sibling formulas consistent by design and is load-bearing if the
node-creation order ever changes.

## Connections

- [global-vs-local-exploration-schedule.md](global-vs-local-exploration-schedule.md)
  — the `bl_sem` schedule decision this one was originally modeled on
  by analogy, and the source of the PUCT-locality argument for v01
  reused above.
- `docs/decisions-log.md`, 2026-07-09 entries — short-form pointers to
  both the original KUBE rewrite and this follow-up.
- `core/mcts_bl_cnt_search_v01_00_00.py::MCTSNode.puct` — the sibling
  formula `kube_density`'s `"parent"` schedule now matches exactly
  (modulo the `/cost(x)` division).
- `core/mcts_bl_kube_search_v01_00_00.py::MCTSNode.kube_density` — the
  implementation.
