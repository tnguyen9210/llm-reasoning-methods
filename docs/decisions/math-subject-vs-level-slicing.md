# MATH `subject` is an analysis axis, not a run axis

## What

Experiments keep slicing MATH by **`level`** (`data.level=4`,
`data.level=5`), never by `subject`. Subject-wise results are
produced by **re-slicing already-scored runs**, never by
launching subject-filtered runs.

Asked 2026-07-28: "would it be reasonable to run experiments on
the MATH subjects instead of levels?" Answer: yes as an
*analysis* axis, no as a *run* axis.

## Why

**1. The two axes are not symmetric.** `level` is a filter
applied *at launch* — a level slice costs its own run. `subject`
is never filtered: every level run already spans all 7 subjects.
Adding a subject filter would shrink `n` per run at the same
per-problem cost, so it is strictly worse than partitioning
after the fact.

**2. The data is already on disk.** Every scored trial record
carries `subject`, `level`, and `unique_id` alongside the
per-question predictions:

```
['agg_scores', 'answer', 'comp_depth', 'comp_gen', 'comp_phase',
 'completions', 'level', 'phase_depths', 'pred_maj@gb',
 'pred_naive@gb', 'pred_weighted@gb', 'problem', 'q_last_phase',
 'q_nodes_max_depth', 'q_total_gens', 'scores', 'solution',
 'subject', 'unique_id']
```

A subject breakdown of every run ever scored is a groupby, not a
GPU job.

**3. Power.** Subject cells inside a level are 10-36 problems
(table below), giving a per-question SE of roughly .08-.15 for a
proportion near .5. Measured effects: model-family gaps at level
5 are .1-.3 (resolvable), but `w_eff` sweep effects are ~.05
(not resolvable). Note that repeated trials do **not** shrink
this: all trials reuse the same fixed question set, so averaging
reduces generation noise but not question-sampling error.

**4. Theory linkage.** The FBMCTS analysis is stated in terms of
instance hardness (gaps, `H`). `level` is a difficulty proxy
that maps onto that axis, so "does the method help more as
instances get harder" is a claim the theory predicts. `subject`
has no counterpart in the analysis: a subject table can support
a *robustness* claim, never a *mechanism* claim.

## Evidence: subject x level in `prm800k/math_splits/test.jsonl`

500 problems, 7 subjects, 5 levels.

| subject | L1 | L2 | L3 | L4 | L5 | total | L4+L5 |
|---|---|---|---|---|---|---|---|
| Algebra | 17 | 21 | 26 | 30 | 30 | 124 | 48% |
| Intermediate Algebra | 7 | 12 | 19 | 23 | 36 | 97 | 61% |
| Prealgebra | 7 | 19 | 17 | 20 | 19 | 82 | 48% |
| Number Theory | 5 | 10 | 16 | 19 | 12 | 62 | 50% |
| Precalculus | 3 | 13 | 15 | 13 | 12 | 56 | 45% |
| Geometry | 2 | 8 | 8 | 10 | 13 | 41 | 56% |
| Counting & Probability | 2 | 7 | 4 | 13 | 12 | 38 | 66% |

Two readings matter:

- **Not confounded.** Every subject spans all five levels and the
  L4+L5 share only ranges 45% (Precalculus) to 66% (Counting &
  Probability). Subject is not a difficulty proxy in disguise, so
  a subject slice is interpretable — *provided* it conditions on
  level.
- **Small cells.** Level 5 (n=134) splits into 36/30/19/13/12/12/12;
  level 4 (n=128) into 30/23/20/19/13/13/10. That is the power
  ceiling for any subject-wise claim.

## What subject IS worth using for

1. **Failure-mode diagnosis (highest value).** `completions` and
   `comp_gen` are recorded per question, so completion length by
   subject is measurable. This bears directly on
   [context-length-overflow-guard.md](context-length-overflow-guard.md):
   the open question is whether `mml=8000` clears the sem-bl
   overflow, and the suspicion is that long symbolic chains
   (Intermediate Algebra, Precalculus) dominate the failures.
   Measuring beats picking a window and hoping.
2. **One robustness table**, model-family only, at level 5, 7
   subject rows — `n` is adequate for the .1-.3 family effects
   and it answers "are the gains content-specific?".
3. **Grading-reliability checks** — sympy timeout flakes
   (`exp-check` §3) may cluster in subjects with unusual answer
   formats (Geometry, Precalculus).

Explicitly NOT worth it: subject breakdowns of the `lam`/
`ds_alpha` sweeps. No power.

## Considered: run the full 500 (no level filter), slice both ways

The one design that makes subject analysis properly powered:
drop `data.level` entirely, run all 500 problems, and derive
*both* the level tables and the subject tables from the same
run. It is strictly more informative than the current pair of
level runs, because filtering the full set by level reproduces
exactly the same question sets (the filter is deterministic), so
the existing level-4 and level-5 tables stay directly comparable
— no re-baselining.

**What it buys.** Subject cells go from 10-36 to 38-124:

| subject | n | SE at p=.5 |
|---|---|---|
| Algebra | 124 | .045 |
| Intermediate Algebra | 97 | .051 |
| Prealgebra | 82 | .055 |
| Number Theory | 62 | .064 |
| Precalculus | 56 | .067 |
| Geometry | 41 | .078 |
| Counting & Probability | 38 | .081 |

That resolves model-family effects (.1-.3) in every subject and
mid-size effects (~.1) in the four big ones. The ~.05 sweep
effects stay out of reach even here — no design short of a
larger benchmark fixes that.

**What it costs.** Less than the 3.7x that "500 vs 134"
suggests, because two level runs are already being paid for.
Measured rates from cnt-mcts b=320 qwen-7b gptq-int4: 309
s/question at level 4 (10.99 hr/trial, n=128) and 390
s/question at level 5 (14.50 hr/trial, n=134). Assuming levels
1-3 cost 0.8-1.0x the level-4 rate (they are easier, so this is
an upper bound), a full-500 trial estimates to **42-46 hr**
against **25 hr** for the L4+L5 pair — a **1.6-1.8x** increase,
in exchange for levels 1-3 and every subject slice. These are
extrapolations from two measured points, not measurements.

**Why it is not adopted wholesale.** At b=320 a single trial of
42-46 h fits inside a 3-day allocation but 4 trials (~170 h) do
not, so every trial needs its own multi-day slot and the run
leans on resume across allocations. At b=80 the same arithmetic
gives roughly 6-12 hr/trial, which is unremarkable.

**Recommended shape if pursued:** pilot it at **b=80 on one
model-family comparison**, where the cost is ordinary, and keep
b=320 on level slices. Report the headline number per level, not
pooled over all 500 — the pooled figure is dominated by the 238
easy problems (levels 1-3 are 48% of the set) and compresses
exactly the hard-instance differences the method is about.

## Consequences / follow-ups

- No ledger entries, no tables, and no GPU time are allocated to
  subject-filtered runs.
- A re-slice script (join scored trial jsonls on `unique_id`,
  group by `subject` x `level`) is the deliverable if
  subject-wise numbers are wanted; it covers every level-4 and
  level-5 run already scored. Not yet written as of 2026-07-28.
- If a subject table is ever added to an exp-comp doc, it must
  state the per-cell `n` in the table, since the reader cannot
  infer it from the level header.

## Revisit if

- The **full-500 option above** is piloted and the 1.6-1.8x cost
  proves acceptable. Subject then becomes the natural secondary
  axis, and this decision narrows to "don't *filter* by subject"
  rather than "don't report by subject".
- A result appears that is plausibly content-specific (e.g. a
  method that helps only on Geometry), in which case the
  robustness table stops being optional.
