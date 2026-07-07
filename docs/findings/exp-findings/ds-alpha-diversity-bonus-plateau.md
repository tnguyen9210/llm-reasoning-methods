# ds_alpha: the diversity bonus is a switch, not a dial

*2026-06-24*

Findings from the sem-mcts (v02, PRM embeds) `ds_alpha`
sweeps across LLM × PRM, 2 trials/cell — see the two
`ds_alpha sweep (v02)` tuning tables (rlhflow and qwen PRM) in
[exp-comparison.md](../../exp-comparison.md) for the raw
numbers.

## The question

`search.ds_alpha` weights the diversity bonus in sem-mcts
child selection (`q_val = ds_beta*score + ds_alpha*diversity`,
`ds_beta=1.0` fixed). Across different LLM and PRM
combinations, how do `ds_alpha ∈ {0, 10, 100, 1000}` affect
pass@gb and naive@gb?

## What the data shows

pass@gb (mean ± SEM, 2 trials), by LLM × PRM × ds_alpha:

| LLM | PRM | da=0 | da=10 | da=100 | da=1000 |
|---|---|---|---|---|---|
| llama-1b | rlhflow | .434±.031 | .613±.031 | .590±.031 | .594±.031 |
| llama-1b | qwen | — | .621±.030 | .613±.031 | .629±.030 |
| qwen-math-1.5b | rlhflow | .781±.026 | .895±.019 | .879±.020 | .887±.020 |
| qwen-math-1.5b | qwen | — | .879±.020 | .875±.021 | .875±.021 |
| llama-3b | rlhflow | — | .742±.027 | .738±.028 | .734±.028 |
| llama-3b | qwen | — | .770±.026 | .766±.027 | *running* |

naive@gb, same layout:

| LLM | PRM | da=0 | da=10 | da=100 | da=1000 |
|---|---|---|---|---|---|
| llama-1b | rlhflow | .402±.031 | .445±.031 | .434±.031 | .426±.031 |
| llama-1b | qwen | — | .535±.031 | .496±.031 | .508±.031 |
| qwen-math-1.5b | rlhflow | .727±.028 | .762±.027 | .746±.027 | .766±.027 |
| qwen-math-1.5b | qwen | — | .797±.025 | .797±.025 | .801±.025 |
| llama-3b | rlhflow | — | .543±.031 | .547±.031 | .559±.031 |
| llama-3b | qwen | — | .680±.029 | .656±.030 | *running* |

(— = not run. ds_alpha=0 exists only for the two rlhflow
cells; the qwen-PRM 0-rows are deferred. llama-3b ×
qwen × da=1000 still in flight.)

## Two regimes: the 0→on jump vs. the 10→1000 plateau

The data splits cleanly into two regimes with opposite
behavior.

**1. Turning the bonus on (0 → 10) is a large effect.** Where
we have a `ds_alpha=0` baseline (the rlhflow rows):

- llama-1b: pass@gb **.434 → .613** (+17.9 pts), a ~41%
  relative gain — far beyond the ~.03 SEM.
- qwen-math-1.5b: pass@gb **.781 → .895** (+11.4 pts).

naive@gb moves much less at the same step (llama-1b
.402→.445, +4.3; qwen-m .727→.762, +3.5). So the bonus's
benefit lands disproportionately on the **gen-budget**
(`@gb`, best-of-the-tree-at-budget) metric rather than the
naive single-path read — consistent with diversity helping the
*search* surface a better candidate, not making any single
trajectory better.

**2. The magnitude past ~10 does nothing (10 → 100 → 1000).**
Every cell, both metrics, both PRMs, is flat within SEM across
10/100/1000:

- pass@gb spreads per row: llama-1b ~.02 (both PRMs),
  qwen-m ~.02 / ~.00, llama-3b ~.01. None exceeds ~0.7 SEM.
- naive@gb is similarly flat (largest swing is llama-1b/qwen
  .535→.496, Δ=.039 ≈ 1.2 SEM — still within noise, and
  non-monotone, so not a trend).

There is no monotone trend in any row: the bonus saturates by
ds_alpha=10. The selection rule evidently only needs the
diversity term to be *present and non-trivial* relative to the
score term; once it dominates ties, scaling it 100× more
changes which child wins only at the margins.

## Robustness across LLM and PRM

- **Plateau is robust to PRM.** The 10→1000 flatness holds
  identically under rlhflow and qwen scoring — it is a
  property of the selection algorithm, not an artifact of one
  PRM's score scale.
- **Plateau is robust to LLM.** Same flat shape for llama-1b,
  qwen-math-1.5b, and llama-3b.
- **PRM choice shifts levels, not shape.** qwen-PRM lifts
  llama-3b (pass .77 vs .74; naive .66–.68 vs .54–.56) and
  llama-1b naive (~.50 vs ~.44); qwen-math-1.5b is ~tied
  across PRMs. In all cases the across-ds_alpha shape is
  preserved.
- **Model dominates everything.** qwen-math-1.5b (~.88) >
  llama-3b (~.75) > llama-1b (~.61) on pass@gb, regardless of
  ds_alpha or PRM.

## Is the 10→1000 flatness "real," or just underpowered?

It's "no detectable trend at n=2," which for *this* question
is the answer that matters. Unlike a hoped-for improvement,
here the useful finding is a **null** (magnitude doesn't
matter), and the mechanism supports it: scaling an already-
dominant additive bonus term mostly preserves the argmax over
children. To *rule out* a real ≤2-pt drift across the decade
would need many trials (same `n ≈ 16×(σ/Δ)²` logic as the
prm_batch_size finding: with σ≈.084 and Δ≈.02, ~280
trials/cell) — not worth it to characterize a saturated knob.
The 0→on jump, by contrast, is far outside noise and needs no
such defense.

## Conclusion / recommendation

- **Treat `ds_alpha` as a switch, not a dial.** Set it nonzero
  — **10 is sufficient** — and do not tune the magnitude.
  100 (the current default) and 1000 buy nothing measurable
  over 10.
- **The bonus itself is worth keeping**: the 0→10 jump is the
  largest single-knob effect in the sem-mcts sweeps
  (+12–18 pts pass@gb where measured), and it lands on the
  gen-budget metric the project cares about.
- **Spend tuning compute on model and PRM, not ds_alpha** —
  those move the levels; ds_alpha (once on) does not.
- **Two gaps to close before this is airtight:** (a) a
  `ds_alpha=0` row for qwen-PRM per model, to confirm the
  0→on jump is PRM-independent (currently rests on rlhflow
  alone); (b) the in-flight llama-3b × qwen × da=1000 cell.
  Both are tracked in `experiments.yaml`
  (feeds `sem-mcts/ds_alpha-sweep`, `sem-mcts/ds_alpha-sweep-qwen`).

## Connections

- [exp-comparison.md](../../exp-comparison.md) — the two
  `ds_alpha sweep (v02)` tuning tables (rlhflow + qwen PRM)
  these numbers are recorded in.
- [prm-batch-size-throughput-memory.md](prm-batch-size-throughput-memory.md)
  — sibling finding; same "flat within noise, don't chase a
  null" reasoning and the `n ≈ 16(σ/Δ)²` trial-count rule.
- [tuning-semantic-score-weights-and-lambda.md](../../decisions/tuning-semantic-score-weights-and-lambda.md)
  — the mechanism behind this finding: why `ds_alpha` needs to
  be ~100x `ds_beta` (the diversity term's scale at init is
  `1/sqrt(lam)` vs. the `[0,1]`-bounded score term), why
  `ds_beta=1` fixed + `ds_alpha`-only sweeping is lossless, and
  why the sweep range found here is scoped to `lam=0.01`
  specifically (`lam` and `ds_alpha` are coupled, not
  independent knobs).
