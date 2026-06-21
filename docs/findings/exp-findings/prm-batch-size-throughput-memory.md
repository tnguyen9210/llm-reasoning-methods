# prm_batch_size: throughput, memory, and significance

*2026-06-21*

Findings from the cnt-mcts Llama-3.2-1B `prm_batch_size`
sweep (rlhflow/qwen × prm_bs∈{1,4}, 2 trials each, scored
2026-06-21) — see the tuning table in
[exp-comparison.md](../../exp-comparison.md) for the
raw numbers.

## The question

Does `search.prm_batch_size` (the in-loop PRM scoring
micro-batch size) trade off accuracy, speed, or memory?

## What the data shows

| prm | prm_bs | pass@gb | time/trial (hr) | peak GPU mem (GB) |
|---|---|---|---|---|
| rlhflow | 1 | .617±.030 | 2.51 | 30.23 |
| rlhflow | 4 | .641±.030 | 2.38 | 31.68 |
| qwen | 1 | .633±.030 | 2.35 | 27.49 |
| qwen | 4 | .676±.029 | 2.31 | 28.68 |

- **Time/trial: flat.** ~2.3–2.5 hr regardless of `prm_bs`,
  for both PRMs. No throughput win from batching at this
  model (Llama-3.2-1B) / budget (b=80) scale.
- **Peak GPU mem: NOT flat.** `prm_bs=4` costs ~1.2–1.5 GB
  more than `prm_bs=1` for both PRMs (rlhflow: 30.23→31.68
  GB; qwen: 27.49→28.68 GB). Pulled from W&B's auto-logged
  `system.gpu.0.memoryAllocatedBytes` (max over each run's
  history) — not something the code explicitly instruments;
  not in `timing_state.json` or the `wandb.log()` calls in
  generate_mcts_cnt.py, which only log
  `time_per_question_s`/`time_per_trial_hr`.
- **pass@gb: looks higher at prm_bs=4, but this is not a
  real effect** — see next section.

## Is the pass@gb gap real?

No good reason to expect one: `prm_batch_size` only changes
how many (state, completion) pairs are batched into one PRM
forward pass during scoring. It doesn't change which
completions are generated, which nodes get expanded, or any
UCT/selection logic — pass@gb is computed from the same
search trajectories regardless of how the scoring loop is
chunked. The one theoretically real channel (batch-shape-
dependent floating-point non-determinism in the PRM forward,
shifting tie-breaks) would move scores by ULPs, not a
systematic multi-point swing.

**The gaps observed are well within noise:**
- rlhflow: .617→.641 (Δ=.024)
- qwen: .633→.676 (Δ=.043)
- Per-row SEM ≈ .03 (bootstrapped *within* a trial, over
  ~100+ questions — this is a lower bound on the true
  between-trial uncertainty, not an estimate of it).

Both gaps are ≲1.5 within-trial SEM, i.e., not
distinguishable from re-running the identical config with a
different seed.

## How many trials would it take to resolve this for real?

Used the rule of thumb `n_per_group ≈ 16 × (σ/Δ)²` (80%
power, α=.05, two-sided), anchoring σ (trial-to-trial SD of
pass@gb for one config) off the Llama-1B custom-tmpl cell
(.648±.042 at n=4 trials ⇒ σ ≈ .042×√4 ≈ .084 — the only
cell in the project with n>2 to estimate spread from):

- rlhflow gap (Δ=.024): n ≈ 16×(.084/.024)² ≈ **196
  trials/group**.
- qwen gap (Δ=.043): n ≈ 16×(.084/.043)² ≈ **61
  trials/group**.

Both are far beyond what's practical to run just to
characterize a batching knob.

## Conclusion / recommendation

- **Don't chase significance on this comparison.** There's
  no mechanism by which `prm_batch_size` should affect
  pass@gb — treat "flat within noise, as theoretically
  expected" as sufficient, rather than running 60–200 trials
  to confirm a null.
- **`prm_batch_size=1` is the better default**, not just a
  "safe" one: same accuracy (within noise), same speed
  (flat), and ~1.2–1.5 GB less peak memory — relevant given
  the V100S's tight headroom (32GB, sm_70, no bf16).
- If a cheap sanity check is wanted instead of full
  significance, 4 trials/cell (matching this project's usual
  convention) would at least show whether the gap stays
  ~3-4 pts (reassuring — noise) or grows (would flag a real
  bug, e.g. accidental seed reuse).
- Reserve real trial-count investment for comparisons that
  *do* have a plausible mechanism and matter to the
  project's claims (cnt-vs-sem-UCT, custom-vs-native
  template, cnt-vs-cnt-bl) — not a batching-only knob.

## Connections

- [exp-comparison.md](../../exp-comparison.md) —
  tuning table this sweep lives in
