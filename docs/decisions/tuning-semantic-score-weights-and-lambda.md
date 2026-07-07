# Tuning semantic score weights and lambda

*2026-07-07*

Mechanism behind the numbers in
[ds-alpha-diversity-bonus-plateau.md](../findings/exp-findings/ds-alpha-diversity-bonus-plateau.md)
(the empirical "switch, not a dial" finding) — this note derives
*why* the default `ds_alpha=100, ds_beta=1, lam=0.01` isn't expressing
"diversity matters 100x more than accuracy," why a `ds_alpha`-only
sweep with `ds_beta` fixed at 1 loses nothing, and why `lam` is not a
free third knob but the other lever on the same scale-matching problem
`ds_alpha` addresses. It closes with a design (not yet executed) for
tuning `lam` and `ds_alpha` jointly, built around a single derived
invariant (`w_eff`, below) rather than a blind 2D grid.

## The question

`core/mcts_sem_search_v02_00_00.py::_diverse_select` scores each
candidate arm as

```python
q_vals = ds_beta * q_scores + ds_alpha * q_diversity
```

`q_scores` is a PRM-derived running mean (`ch.q_value()`); `q_diversity
= sqrt(x^T V^-1 x)`, where `V`'s starting point is set by the ridge
constant `lam` (`V_0 = lam * I`, `MCTSSemV01Config.lam`, inherited by
v02). Are `q_scores` and `q_diversity` on comparable scales? If not,
what does that imply for choosing `ds_alpha`/`ds_beta` — and since
`lam` directly sets `q_diversity`'s scale, should it be tuned
alongside them or treated as fixed?

## q_scores is genuinely bounded in [0,1]

Both PRM implementations emit a softmax probability at each scored
step: `core/reward_models.py:267` (`RLHFlowPRM`, `P(correct)`) and
`core/reward_models.py:455` (`QwenPRM`, `P(+)`) — both `logits.softmax
(...)`, hence in `[0,1]`. `sal.utils.score.aggregate_scores`'s three
strategies all preserve that range: `min` and `last` trivially (the
min/last of values in `[0,1]` is in `[0,1]`); `prod` too (a product of
factors in `[0,1]` stays in `[0,1]`, only shrinking). Depth-truncated
nodes get `negative_reward=0` (`conf/search/mcts_sem_v02.yaml`), also
in range. So every score ever backprop'd is in `[0,1]`, and `q_value()`
(a running mean of values in `[0,1]`) stays in `[0,1]` too.

## q_diversity's starting scale is set by lam, not fixed at ~10

`q_diversity(x) = sqrt(x^T V^-1 x)`, with `x` a unit-norm pooled
embedding (`embeds_normalize: true`, `conf/search/mcts_sem_v02.yaml`).
At initialization, before any embedding has been folded into the
covariance (`MCTS.__init__`, `mcts_sem_search_v02_00_00.py:497-504`),
`V_0 = lam * I`, so in closed form:

```
V_inv = (1/lam) * I
q_diversity(x) = sqrt(x^T (1/lam * I) x) = ||x|| / sqrt(lam) = 1/sqrt(lam)
                                                    (since ||x||=1)
```

At the project's default `lam=0.01`: `q_diversity(x) = 1/sqrt(0.01) =
10`. **This value is a direct, deterministic function of `lam`** —
it is not an incidental fact about the embeddings or the model, it is
exactly `1/sqrt(lam)` by construction. So at the default, `q_diversity
≈ 10` at the very first selection at a node — two orders of magnitude
above a `q_scores` value in `[0,1]`. As more vectors accumulate
(`V_inv` updated via `select_child`'s exact-inverse or Sherman-
Morrison path), `q_diversity` shrinks along directions already
represented in `V`, but it starts at `1/sqrt(lam)` and only converges
toward comparability with the score term after substantial
accumulation.

This is exactly why the project's default is `ds_alpha=100` (not
~1-10) at `lam=0.01`: it's compensating for `q_diversity`'s much
larger natural scale at the point selection actually happens, not
encoding a belief that diversity should dominate 100x. Change `lam`
and this compensating value changes with it — see the next section.

## lam and ds_alpha are coupled, not independent knobs

Because `q_diversity`'s initial scale is exactly `1/sqrt(lam)`, `lam`
and `ds_alpha` are not two independent things to tune — **`lam`
determines what "matched scale" even means for `ds_alpha`.** Lowering
`lam` raises `q_diversity`'s starting scale (smaller ridge term = less
regularized, more confident `V_inv`), which means a *larger*
`ds_alpha` is needed to reach the same effective compensation; raising
`lam` does the opposite. Concretely, if `lam` were changed from `0.01`
to `0.0001`, `q_diversity`'s initial scale would jump from `10` to
`100` — an already-tuned `ds_alpha=100` would then only achieve
1/10th of its previous compensating weight relative to the diversity
term, silently changing the effective selection behavior without
`ds_alpha` itself having moved.

**Practical implication:** a `ds_alpha` sweep run at one `lam` value
does not transfer to a different `lam` — the "sufficient range"
finding (`ds_alpha ∈ {0, 10, 100}`, see below) is scoped to `lam=0.01`
specifically. Sweeping `lam` and `ds_alpha` independently, on separate
axes, would each partially answer "how does the effective bonus scale
change" — the informative single quantity is really the *ratio*
`ds_alpha * sqrt(lam)` (the effective bonus weight once `lam`'s
contribution is normalized out), not either raw value alone. No sweep
over `lam` currently exists in the repo (`lam=0.01` fixed throughout
[exp-comparison.md](../exp-comparison.md)'s sem-mcts tables) — this is
an open gap, not a settled finding.

## ds_beta=1, tune only ds_alpha — lossless

`q_vals = ds_beta*q_scores + ds_alpha*q_diversity` — scaling both
terms by the same constant doesn't change the argmax over children, so
only the *ratio* `ds_alpha/ds_beta` matters for which arm is selected.
Fixing `ds_beta=1` and sweeping `ds_alpha` covers the full
one-parameter family; nothing is lost by not also varying `ds_beta`.
This is what the project's `ds_alpha sweep (v02)` tables in
[exp-comparison.md](../exp-comparison.md) already do (`ds_beta=1.0`
fixed throughout).

## Recommended sweep range (at lam=0.01, the only lam tested so far)

Given the scale above, `ds_alpha=0` (pure q-value) vs. any nonzero
value tests qualitatively different regimes (bonus off vs. on); once
on, the effective weight relative to `q_diversity`'s ≈10-at-init scale
(at the current `lam=0.01`) is what should be swept. The empirical
finding
([ds-alpha-diversity-bonus-plateau.md](../findings/exp-findings/ds-alpha-diversity-bonus-plateau.md))
confirms this reasoning: `ds_alpha ∈ {0, 10, 100}` is sufficient —
`0` establishes the on/off jump (the large, real effect), and `10`
already saturates the plateau that `100`/`1000` sit on. Testing `1000`
is redundant compute; if more confidence is wanted, spend trials
re-running `10` or `100` at higher n rather than sweeping a fourth
magnitude. **This range is scoped to `lam=0.01`** — per the coupling
above, a different `lam` would shift which raw `ds_alpha` values are
"on/off" vs. "plateaued," even though the underlying effective-weight
story would be the same.

This section predates the `lam`-sweep design below and should not be
read as "the" answer for other `lam` values — it is the one point on
the `w_eff` curve (see below) that has actually been measured.

## w_eff: the single quantity a joint lam/ds_alpha sweep should target

2026-07-07 follow-on discussion, prompted by asking what a *joint*
`lam`/`ds_alpha` tuning strategy should look like, without assuming
the current defaults (`lam=0.01`, `ds_alpha=100`) are well chosen.

Define the **effective weight**

```
w_eff = ds_alpha / sqrt(lam)      (ds_beta=1 fixed)
```

— the actual multiplier on `q_diversity` relative to the `[0,1]`-
scaled score term at the moment of selection (since `q_diversity`'s
initial scale is exactly `1/sqrt(lam)`, per the derivation above). Two
consequences:

1. **`lam` and `ds_alpha` are redundant along the `w_eff` axis.** Any
   `(lam, ds_alpha)` pair with the same `w_eff` produces the same
   selection behavior at init — so a full 2D grid mostly measures the
   same thing twice, unless `lam`'s second role (below) also matters.
2. **`lam` is not *purely* redundant with `ds_alpha`, though.** Beyond
   setting `q_diversity`'s init scale, `lam` is the ridge term in
   `V_inv`, so it also controls how fast `q_diversity` decays as real
   embeddings accumulate: a large `lam` means the prior dominates
   longer (slow adaptation); a tiny `lam` makes `V_inv` sensitive to
   the very first embedding it sees (fast adaptation, but closer to
   numerical instability in the Sherman-Morrison path once near-
   duplicate embeddings arrive). A `w_eff`-only sweep at one fixed
   `lam` cannot see this second effect.

Re-reading the existing plateau finding through `w_eff`: at
`lam=0.01`, `sqrt(1/lam)=10`, so the tested range
`ds_alpha ∈ {0,10,100,1000}` is actually `w_eff ∈ {0,100,1000,10000}`.
**The known plateau starts at `w_eff≈100`, not `w_eff≈1`** — i.e., the
existing sweep only ever explored `w_eff ≥ 100`, all inside the
plateau. The region `w_eff ∈ (0, 100)` — where the diversity term
goes from "matched scale" to "starting to saturate" — has never been
tested. That gap, not a re-run of the known plateau, is what a
`lam`-inclusive sweep should target.

### Proposed lam range

Read `lam` as a pseudo-count of prior observations, in the same units
as a real (unit-norm) embedding's rank-1 contribution to `V`:

| lam | q_diversity at init | interpretation |
|---|---|---|
| 1.0 | 1 | prior ~1 real observation; diversity term needs no compensation to match score scale |
| 0.1 | ~3.2 | |
| 0.01 (current default) | 10 | prior very weak; matches the only value tested so far |
| below 0.01 | >10, rising | approaches numerical risk (near-singular `V_inv` after few near-duplicate embeddings) in the `sm` update path — treat as a practical floor, not worth exploring for tuning insight alone |

Recommended: **`lam ∈ {1.0, 0.1, 0.01}`** — three points, one order of
magnitude apart, spanning "no compensation needed" down to the
already-tested "10x compensation" regime.

### Proposed ds_alpha range per lam

Target the same `w_eff` checkpoints at each `lam`, via
`ds_alpha = w_eff * sqrt(lam)`, filling in the previously-untested
region below the known plateau (`w_eff=100`):

| lam | w_eff=0 | w_eff=1 | w_eff=10 | w_eff=100 (known: saturated) |
|---|---|---|---|---|
| 1.0 | 0 | 1 | 10 | 100 |
| 0.1 | 0 | 0.316 | 3.16 | 31.6 |
| 0.01 | 0 | 0.1 | 1 | 10 |

Note this does **not** reuse the repo's existing raw `ds_alpha` grid
at `lam=0.01` (`{0,10,100,1000}`) — those all sit at `w_eff≥100`,
inside the confirmed plateau. This table is deliberately the
unexplored region beneath it.

### Tuning procedure

1. **Confirm `w_eff` is the right invariant (cheap, one model/PRM
   combo).** Run one matched-`w_eff` pair at two different `lam`
   values (e.g. `lam=1.0,ds_alpha=10` vs. `lam=0.01,ds_alpha=0.1`,
   both `w_eff=10`). If pass@gb/naive@gb agree within SEM, `lam`'s
   second (decay-speed) role doesn't matter in practice and the
   problem collapses to a 1D sweep over `w_eff` at one convenient
   fixed `lam` — skip the full grid below. If they disagree, `lam`'s
   independent effect is real and the full grid is needed.
2. **Run the grid (or its 1D reduction) on the cheapest, clearest
   signal.** Use llama-1b/rlhflow — the combo that showed the
   largest 0→on effect in the existing plateau finding — not
   qwen-math-1.5b (near-ceiling, ~.88-.90, too little headroom to
   see a tuning effect) or llama-3b first. 2 trials/cell, as in the
   existing sweeps, is enough for a first pass at shape, not a
   precise optimum.
3. **Read out where sensitivity lives.** Per `lam` row, find the
   first `w_eff` where naive@gb/pass@gb moves outside 2xSEM going up
   from the previous `w_eff`, then the point where it flattens again.
   That interval is the informative middle — likely below the
   current default's `w_eff=1000` (`ds_alpha=100, lam=0.01`), which
   the existing finding already shows is past the plateau's onset.
4. **Only if step 1 found `lam` matters independently:** re-run the
   best `w_eff` from step 3 at 2-3 different `lam` values, holding
   `w_eff` fixed, to isolate the decay-speed effect from the scale-
   setting effect.
5. **Validate on a second model/PRM combo** before generalizing the
   result or changing the sweep tables' default.

This is a 1-2 session plan: step 1 is a single pair of runs that
decides how much step 2 costs; steps 2-3 are the bulk of the compute;
steps 4-5 fire only conditionally. Not yet started — queued alongside
the `ds_alpha` sweep-mechanics todo (`00_inbox/exp_todos.md`,
2026-07-07).

## Connections

- [ds-alpha-diversity-bonus-plateau.md](../findings/exp-findings/ds-alpha-diversity-bonus-plateau.md)
  — the empirical result this note explains the mechanism behind.
- `core/mcts_sem_search_v02_00_00.py::_diverse_select`,
  `MCTS.__init__`, `MCTS.select_child` — the code this note traces.
- No `lam` sweep has been *run* yet — the design above
  (`w_eff`, the `lam`/`ds_alpha` tables, the 5-step procedure) is
  queued alongside the `ds_alpha` sweep-mechanics todo in the vault
  (`00_inbox/exp_todos.md`, 2026-07-07), not yet executed.
