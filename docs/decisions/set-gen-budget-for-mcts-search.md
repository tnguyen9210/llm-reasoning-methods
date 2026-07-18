# `gen_budget` is the primary search-budget knob, set directly

*Decided 2026-06-11 — [decisions-log.md](../decisions-log.md#2026-06-11--configs-gen_budget-is-set-directly-num_batches-dropped)*

## What

Every current search config (`MCTSCntConfig`, `MCTSSemV01Config`,
`MCTSSemV02Config`, `MCTSBLCntConfig` — `utils/configs.py`) exposes
`gen_budget: int = 80` directly, and every `conf/search/*.yaml` sets it
explicitly (`mcts_cnt.yaml`, `mcts_sem_v01.yaml`, `mcts_sem_v02.yaml`,
`mcts_bl_cnt_v01.yaml`, `mcts_bl_kube_v01_prm800k.yaml` (renamed
2026-07-16 from `mcts_bl_cnt_v02_prm800k.yaml` — see
[bl-cnt-to-bl-kube-rename.md](bl-cnt-to-bl-kube-rename.md)) — all
`= 80`).
Launchers pass it through unchanged; it also feeds `config_name`'s
`--b-{budget:03d}` tag (`utils/configs.py:545`), so it's part of a
run's identity.

For MCTS methods (CNT, Semantic, BL), `gen_budget` is the total count
of *generation calls* (each producing `batch_size` candidates) allowed
across the whole search; the algorithm terminates once it's exhausted,
regardless of how deep the tree got. This is charged per *expansion*,
not per depth level — a search that expands few nodes very deeply and
one that expands many nodes shallowly both spend the same budget per
expansion event.

For BoB, the convention (documented in `docs/algorithms.md`
Terminology) is to distribute the total evenly across depths —
`gen_budget / max_depth` generations per depth level — so that BoB's
per-depth generation count is comparable to MCTS's total-budget framing
rather than an independent hyperparameter. (The current
`generate_bob_prm800k_v0101.py` launcher hardcodes its depth/budget
values directly rather than reading a `gen_budget` field from a
config — the distribution rule is the documented target shape for
this launcher, not literally implemented as a shared field today.)

## Why this shape, not `num_batches * max_depths`

The prior config exposed `num_batches`, and launchers derived
`gen_budget = num_batches * max_depths`. That's backwards: `gen_budget`
is the quantity that's actually meaningful for comparing two runs (or
two algorithms) at the same generation cost — `num_batches` was just an
arbitrary factor that happened to produce it. Setting the total
directly means:

- **Sweeps vary the quantity that matters.** A budget sweep varies
  `gen_budget` itself, not a factor that has to be re-multiplied by
  `max_depth` to know what was actually spent.
- **Cross-algorithm comparisons are explicit.** MCTS and BoB can be
  compared "at matched `gen_budget`" directly, without translating
  each one's own factorization back to a common unit first.
- **MCTS's charge-per-expansion semantics don't fit a depth-factored
  budget anyway.** MCTS doesn't spend a fixed amount per depth level —
  it spends per expansion event, wherever the tree happens to expand.
  Trying to keep `num_batches * max_depths` meaningful for MCTS was
  already an artifact of the BoB framing being generalized where it
  didn't apply.

## Revisit if

Never expected to change — this is a straightforward "expose the
meaningful quantity, not a derived factor" simplification, not tied to
any experiment result that could shift.
