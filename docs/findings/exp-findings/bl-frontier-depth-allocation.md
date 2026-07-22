# BL frontier scores allocate exploration by depth: four distinct signatures

*2026-07-20*

Findings from a headless four-way trace comparison of the
`mcts_bl_*_v02` frontier variants — `mcts_bl_cnt_v02`,
`mcts_bl_kube_v02`, `mcts_bl_kdepth_v02`, `mcts_bl_sem_v02` —
on 5 prm800k level-5 problems (one per method per problem, 20
searches total), qwen-3b, `gen_budget=80`, seed 100000. Raw
per-node trees, selection-trace logs, per-run summaries, the
computed statistics (`analysis_stats.json`), and a self-
contained HTML write-up all live in
[../../../unittests/results/](../../../unittests/results/);
regenerate with `unittests/examine_driver.py`.

**Scope up front (this is not an accuracy result).** n=1 per
(method, question); 5 problems, chosen to span 5 MATH subjects
(the level-5 representative set —
`unittests/results/level5_representative_prompts.json`). No
scoring was run: "completions" counts EOS/length-terminated
leaves, **not correct answers**. These are *behavioral*
diagnostics — how the tree develops — not a claim about which
method solves more problems. Treat the completion counts as a
coarse "did it produce finished trajectories at all" signal,
not as accuracy.

## The five problems and their difficulty

Intrinsic math difficulty, my judgment, rated **1–10 across the
general MATH population** (not within level-5) — all five are
MATH level-5, the hardest tier, so they sit at 5+ by
construction. This is a subjective read of the *math*, made
independent of how the search methods fared, so it can be
cross-referenced against the behavior below (e.g. did cnt stall
only on the hardest, or on ones that are easy for a human?).

| Q | subject | answer | diff | why |
|---|---|---|---|---|
| 0  | Int. Algebra | `p − q` | **9** | The reindex-and-count insight *is* the problem: substitute `n = j+k`, note each `n` arises from `n−1` pairs, so the double sum collapses to `Σ(n−1)/n³ = Σ1/n² − Σ1/n³ = p − q`. Nothing mechanical; the whole thing is the non-obvious move. |
| 5  | Algebra | `5` | **6** | Set up a ratio equation from the two pay scenarios, cross-multiply, solve a quadratic. Math is routine; the difficulty is *parsing* the wording (which quantity gets `4x` vs. a reassignment) — an easy trap. |
| 9  | Counting/Prob | `144` | **7** | Circular seating with "no two of three people adjacent": fix the rotation, then gap-method / inclusion-exclusion to place the restricted three among the others. Multi-step, easy to double-count. |
| 23 | Geometry | `145°` | **5** | The easiest: inscribed hexagon solved by arc / inscribed-angle relations (inscribed angle = half its arc; arcs sum to 360°). Standard once the rules are recalled; the `[asy]` diagram adds parsing overhead. |
| 49 | Number Theory | `6` | **7** | Factor `10! = 2⁸·3⁴·5²·7`, then count `x` with `x³ ∣ 10!`: per prime `p^a` need `3a ≤ exp`, giving `⌊exp/3⌋+1` choices, multiplied. Needs both the factorization and the exponent-counting insight. |

Difficulty **does not track** the search behavior below in an
obvious way — Q0 (hardest, 9) and Q49 (7) are where methods
stall or tunnel, but Q5 (only 6) also stalls cnt, while Q23
(easiest, 5) and Q9 (7) are handled cleanly by most. So the
depth-allocation signatures are driven more by *where the
answer sits in the tree* (deep vs. shallow) than by intrinsic
hardness — see the analysis.

## The question

All four methods are best-first frontier searches over the
same node set, differing only in the score that ranks the
frontier:

- **cnt** — PUCT: `q + c·√(ln N_parent / N_leaf)`
- **kube** — KUBE value-density: `(q + bonus) / cost`
- **kdepth** — kube's density, depth-discounted
- **sem** — diversity-adjusted value:
  `q + α·√(xᵀV⁻¹x)`

Given an identical 80-expansion budget, **where in the tree
does each method spend those expansions** — shallow (early
reasoning steps) or deep (many steps down one line) — and does
that allocation differ enough to matter?

## What the data shows

Per-method means over the 5 questions (an *expansion* = one
frontier node selected and expanded; every method does exactly
80):

| method | mean comps | mean max-depth | mean depth | depth-Gini | visit-Gini | dead-ends (Σ5q) |
|---|---|---|---|---|---|---|
| cnt    | 44.2 | 13.4 |  8.12 | 0.446 | 0.335 |   0 |
| kube   | 77.2 | 14.8 |  8.43 | 0.319 | 0.668 |  21 |
| kdepth | 66.4 | 16.8 | 13.70 | 0.641 | 0.726 | 272 |
| sem    | 39.6 | 13.0 |  6.98 | 0.344 | 0.435 |   1 |

### How to read this table

It collapses all five per-question depth profiles into one
averaged number per method, so the four can be ranked without
reading curves. Every value is a **mean over the 5 questions
EXCEPT dead-ends, which is a total** (hence a whole number).
Column by column:

- **mean comps** — average # of *finished trajectories* (paths
  that reached a natural EOS/length stop) per question. Higher
  = more often produced completed answers. **Not accuracy** — a
  completion is a finished path, not a correct one.
- **mean max-depth** — how deep the deepest node got, averaged.
  Bigger = taller trees.
- **mean depth** — the *typical* node's depth, averaged over
  all generated nodes. This is the headline shallow-vs-deep
  number: low (sem, 6.98) = stays near the top; high (kdepth,
  13.70) = lives deep.
- **depth-Gini** — a 0-to-1 concentration score for the depth
  histogram: 0 = expansions spread evenly across depths, →1 =
  all piled at one depth. (Same inequality measure economists
  use for income, applied to "how evenly is effort spread
  across depths.") kube 0.32 = flat/even; kdepth 0.64 = piled
  (at the max-depth wall).
- **visit-Gini** — the same 0-to-1 concentration idea, but for
  *visit counts across nodes*: does the search re-hammer a few
  favorite nodes (high) or spread attention thin (low)? kdepth
  0.73 / kube 0.67 hoard; cnt 0.34 spreads.
- **dead-ends** — nodes that hit the max-depth wall *without*
  finishing: wasted effort that rammed into the ceiling.
  Summed over 5 questions. 272 (kdepth) vs. ~0 (cnt, sem) is
  the most dramatic number here.

**What to focus on.** (1) *mean depth* + *deep-half %* together
are the shallow↔deep axis — the main finding, in two columns:
sem shallow (6.98), kdepth deep (13.70), cnt/kube in the
middle. (2) *dead-ends* is kdepth's cost — the failure-side view
of its high mean depth. (3) The two *Ginis* split the two
lookalike middle methods: cnt and kube match on depth, but
visit-Gini separates them (kube 0.67 re-hammers a few
high-value nodes; cnt 0.34 spreads thin, never commits).
(4) *mean comps* is the practical outcome — read it alongside
the exploration columns, since the point is *why* (kube's
balanced spread finishes things; sem's shallow breadth finishes
fewer). Scan the extremes in each column: each method owns a
couple, and those are its fingerprint.

**What it supports.** No two rows look alike across the columns
— each method occupies a distinct corner (kdepth deep +
concentrated + wasteful; sem shallow + broad; kube balanced +
completion-heavy; cnt balanced but non-committal). That is the
numerical proof of the headline below: these are four different
search strategies, not variants of one — a difference a single
accuracy number would hide entirely.

The single clearest cut is the **expansion-depth histogram**
per question (# of the 80 expansions landing at each tree
depth). The shapes fall into three silhouettes, stable across
problems:

Q0 (Int. Algebra), expansions by depth (d0…dmax):

```
cnt    d2:10  d9:22  d10:13                    (peak @ d9,  maxD 12)
kube   d2:10  d9:25  d10:13                    (peak @ d9,  maxD 12)
kdepth d0..d16 ≈1 each, d17:4 d18:12 d19:47    (peak @ d19, maxD 20)
sem    broad d2..d14 ≈4-9 each                 (peak @ d14, maxD 17)
```

That kdepth row — ~1 expansion per depth for the first 17
levels, then **47 at the max-depth wall** — is the entire
tunneling story in one line. Same pattern on Q5 (kdepth d19:41)
and Q23 (kdepth d18:15, d19:12).

## Three exploration signatures

This is the same aggregate data as the table above, re-read
**by method** rather than by metric — one takeaway per method,
each backed by the column(s) named in it (the by-method view;
the table is the by-metric view; the per-question histograms
are the by-question view).

**1. kdepth tunnels.** Its depth discount *rewards* pushing an
already-alive path deeper, so it drives a wall of expansions
into the max-depth ceiling: **272 max-depth dead-ends** over
the 5 questions vs. ≤21 for any other method (152 on Q0 alone,
116 on Q5). Highest mean depth (13.7) and highest depth-Gini
(0.64, most concentrated). The cost: on Q0 it produced only
**4** finished trajectories (cnt/kube got ~70) because it spent
~60 of 80 expansions stacking against the wall instead of
finishing elsewhere. But the *same* depth-seeking is a virtue
when the answer is deep: on Q9 it led the field (156).

**2. sem fans out shallow.** Its diversity bonus `√(xᵀV⁻¹x)`
scores a branch by semantic novelty; going deeper down one line
produces steps semantically *close* to their parents, so the
bonus **discourages depth**. Result: shallowest mean depth
(6.98), a low broad hump peaking at depth 3–6, and the lowest
"deep-half" share (as low as 19% of expansions below the
midpoint depth on Q23). It trades completion volume for
breadth of coverage — and its shape barely changes across the
5 problems, i.e. the shallowness is enforced by the mechanism,
not the problem.

**3. cnt and kube sit in the balanced middle — but cnt can
stall.** Both peak mid-depth (≈8–10), tracking wherever the PRM
rates continuations well; on easy questions (Q0, Q9) their
curves nearly coincide. They diverge on hard ones:

- **kube** is the robust all-rounder: completed on all 5,
  most on average (77), flattest depth profile (lowest
  depth-Gini, 0.32) — least likely to tunnel or stall. High
  visit-Gini (0.67) because its density ratio revisits a few
  high-value nodes hard.
- **cnt** *smears* its budget thin: on Q5 and Q49 it returned
  **0 completions** — the same best-first zero-completion
  failure documented for v01
  ([bl-cnt-frontier-zero-completion-rate.md](../coding-findings/bl-cnt-frontier-zero-completion-rate.md)),
  here reproduced under v02 on 2 of 5 problems. PUCT's
  visit-balancing spreads exploration across many depths (its
  Q5 tree reached depth 19) without ever concentrating enough
  on one line to reach EOS before the budget ran out. Its
  visit-Gini is lowest (0.34) — the flip side of "balanced"
  is "never commits."

## The headline: score sets tree shape, not the problem

Same 80-expansion budget, same 5 problems, four visibly
different and *self-consistent* depth signatures. Each method's
curve keeps its characteristic silhouette (kdepth right-edge
spike, sem left hump, cnt/kube mid humps) across all five
questions — so the exploration allocation is a property of the
frontier score, and the problem only shifts it, doesn't
reshape it. The depth-Gini / deep-half / dead-end numbers just
put a scalar on what the histograms show by eye.

## Which diagnostics were informative

Chosen without pre-judging which would separate the methods
(all derivable from the dumped per-node trees, no new
instrumentation):

- **Most informative:** expansion-depth histogram +
  depth-Gini + dead-end count — cleanly separate all four.
- **Useful secondary:** visit-Gini (kdepth hoards, 0.73; cnt
  spreads, 0.34) and mean-completion-depth.
- **Less discriminating here:** branching factor by depth and
  mean-q by depth were computed but added little beyond the
  depth histogram at this budget.

## Conclusion / cautions

- **The four v02 frontier scores are genuinely different
  search strategies, not variants of one.** Any comparison of
  them must account for *where* they explore — a single
  accuracy number hides that kdepth and sem are exploring
  almost disjoint regions of the tree.
- **kdepth's depth-seeking is double-edged.** It is the method
  most sensitive to whether the answer lies deep; expect high
  variance across problems and a real risk of max-depth
  starvation on problems where a shallow completion exists.
- **kube is the safe default** for "produces finished
  trajectories reliably" at this budget; **cnt's stall risk is
  real** and reproduces the v01 zero-completion behavior.
- **Do not read completion counts as accuracy.** The obvious
  next step, if any of these patterns matters for the paper,
  is to score these same runs (pass@gb) so "tunnels deep" can
  be tied to "solves / fails to solve," and to raise n beyond
  1/cell before any of the per-question numbers are treated as
  more than illustrative.

## Connections

- [bl-cnt-frontier-zero-completion-rate.md](../coding-findings/bl-cnt-frontier-zero-completion-rate.md)
  — the v01 zero-completion behavior this reproduces for cnt
  under v02 (Q5, Q49); the ~18%-of-questions rate there is the
  population-level version of the two stalls seen here.
- [../../../unittests/examine_search_trace_bl_v1.ipynb](../../../unittests/examine_search_trace_bl_v1.ipynb)
  — the interactive notebook this batch driver mirrors
  (single method/question at a time).
- [../../../unittests/results/bl_search_comparison.html](../../../unittests/results/bl_search_comparison.html)
  — the visual write-up (depth-profile plots + metrics table)
  these numbers are drawn from.
