# Fractional-KUBE alignment audit: `(q+bonus)/cost` confirmed; affordability restriction added; constant folded into `kube_c`

*2026-07-09*

Records a line-by-line verification of
`mcts_bl_cnt_search_v02_00_00.py::MCTSNode.kube_density` and
`MCTS.select_child_from_list` against both the paper (Tran-Thanh et
al., arXiv:1204.1909, Section 3.3, Eq. 9 — read from the local
`budget-mab/paper.pdf`, not from memory) and the reference
implementation (`budget-mab/src/algorithms.py::FractionalKUBE`), plus
the one real gap the audit found and how it was fixed.

## What the audit confirmed: `(q + bonus) / cost` is the right form

The question was whether the selection score should be
`(q + bonus) / cost` — i.e. the bonus normalized by cost along with
the mean — or something else (e.g. `q/cost + bonus`).

**Paper, Section 3.3:** fractional KUBE "pulls the arm that maximises
`μ̂_i,n_i,t / c_i + sqrt(2 ln t / n_i,t) / c_i`" — both the mean and
the confidence bonus divided by cost, separately, which is
algebraically identical to `(μ̂ + sqrt(2 ln t / n)) / c`. The bonus is
inside the cost normalization.

**Reference code** (`FractionalKUBE.run`):

```python
ucb = ucb_values(stats, t, bonus_mode, arm_var)  # = mean + bonus
ucb_density = ucb / env.costs                    # (mean+bonus)/cost
affordable = np.where(env.costs <= residual)[0]
return int(affordable[np.argmax(ucb_density[affordable])])
```

Same structure. So `kube_density`'s `return (q + bonus) / cost` is
aligned with both sources — no change needed there.

## Gap 1 (documented, not fixed): the constant `2` is folded into `kube_c`

Paper and reference both use `sqrt(2·ln(t)/n)`. Our code computes
`kube_c·sqrt(log(clock)/visits)` — no literal `2` under the root.
Since `sqrt(2·x) = sqrt(2)·sqrt(x)`, dropping it just rescales the
bonus by a constant, which the free coefficient `kube_c` absorbs.
This is the exact convention `mcts_bl_cnt_v01`'s `puct()` already
uses for UCT's constant (`cpuct·sqrt(log(N)/n)`, `cpuct=2.0` by
default), so v02 matching it keeps the two siblings comparable —
which matters more here than matching the paper's literal constant,
given the single-factor-ablation design
([kube-bonus-schedule.md](kube-bonus-schedule.md)). The cost is that
`kube_c=2.0` should be read as a tuning starting point, not as "the
paper's √2" — now stated in the module docstring rather than left
implicit.

## Gap 2 (fixed): the affordability restriction was missing

The paper's fractional KUBE never argmaxes over all arms — the
knapsack relaxation at each step is solved subject to the residual
budget, and the reference implementation restricts to
`costs <= residual` **before** the argmax. The reference repo's own
`docs/issues.md#fractional-kube-fallback-discards-ranking` documents
why the ordering matters: argmax over all arms with a fallback to
"cheapest affordable" silently discards the ranking among affordable
arms near budget exhaustion.

Our `select_child_from_list` had no counterpart: it ranked the whole
frontier by density regardless of whether `cost(x) = max_depth -
depth(x)` fit within the remaining generation budget, so late in a
run the search could keep opening shallow branches that could not
possibly reach a completion before the budget ran out. That failure
mode is not hypothetical for this family —
`bl_cnt_v01` has a known ~18% zero-completion rate
(docs/findings, 2026-07-08), and "budget exhausted with no path
finished" is exactly what an affordability filter pushes against: as
`residual` shrinks, the filter excludes shallow (high-cost) nodes and
forces the endgame into deep, nearly-complete paths.

### Implementation choice

Three placements were considered:

- **Set-restriction in `select_child_from_list`** (chosen) — filter
  the candidate list before the ranking loop, mirroring the
  reference's restrict-before-argmax. Keeps feasibility a set
  operation and `kube_density` a pure index; the outer loop only
  supplies `residual = gen_budget - gen_cnt`.
- **`-inf` sentinel in `kube_density`** — smallest diff (reuses the
  `cost <= 0` guard pattern) but conflates constraint with index,
  and degenerates badly when nothing is affordable: the uniform
  tie-break would pick uniformly at random among all-`-inf` nodes, a
  silent behavior change of exactly the class the reference's issue
  doc warns about.
- **Permanent pruning of `leaf_nodes` in the outer loop** —
  unaffordability is monotone (`residual` only falls, costs are
  fixed) so pruning once would be valid and efficient, but it
  mutates frontier state outside the selection function, couples the
  generic loop to KUBE semantics, obscures the trace logs, and
  forecloses the empty-set fallback below.

### Three sub-decisions

1. **Terminal nodes are always affordable.** Selecting a terminal
   node consumes zero generations (it only backprops), so the filter
   reads `is_terminal or cost <= residual`. Excluding terminals
   would strand completed paths and break backprop.
2. **Empty affordable set → relax to the full frontier**, rather
   than the paper's stop. Our `cost` is a *worst-case* completion
   bound — a shallow node can EOS in two steps even when
   `max_depth - depth` says fifteen — so stopping would waste budget
   that can still produce completions, the worse failure mode given
   the zero-completion problem. Unlike the reference's documented
   bug, this fallback relaxes the *set* and re-ranks properly, so
   the ranking is preserved.
3. **Default `kube_affordable: true`.** The canonical v02 config is
   the faithful KUBE package. This does mean the headline v01-vs-v02
   comparison tests cost normalization *and* feasibility filtering
   jointly; `kube_affordable=false` is kept as the middle arm of a
   three-way decomposition (v01 → v02 no-filter → v02 full) for when
   the factors need separating.

### Hash note

Adding `kube_affordable` changes the v02 config hash (third time
today, after `kube_schedule`). Still safe: no real (non-smoketest)
v02 runs exist yet. All three schema changes land before the first
tracked run.

## Connections

- [kube-bonus-schedule.md](kube-bonus-schedule.md) — the bonus-clock
  decision this audit followed; together they cover every term of
  the v02 selection rule.
- `budget-mab/src/algorithms.py::FractionalKUBE` and
  `budget-mab/docs/issues.md` — reference implementation and the
  restrict-before-argmax lesson.
- `budget-mab/paper.pdf`, Section 3.3 / Eq. 9 — the primary source
  the formula was checked against.
- `core/mcts_bl_cnt_search_v02_00_00.py::MCTS.select_child_from_list`
  — the implementation.
