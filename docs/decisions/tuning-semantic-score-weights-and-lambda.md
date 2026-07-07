# Tuning semantic score weights and lambda

*2026-07-07*

Mechanism behind the numbers in
[ds-alpha-diversity-bonus-plateau.md](../findings/exp-findings/ds-alpha-diversity-bonus-plateau.md)
(the empirical "switch, not a dial" finding) — this note derives
*why* the default `ds_alpha=100, ds_beta=1, lam=0.01` isn't expressing
"diversity matters 100x more than accuracy," why a `ds_alpha`-only
sweep with `ds_beta` fixed at 1 loses nothing, and why `lam` is not a
free third knob but the other lever on the same scale-matching problem
`ds_alpha` addresses.

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

## Recommended sweep range (at lam=0.01)

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

## Connections

- [ds-alpha-diversity-bonus-plateau.md](../findings/exp-findings/ds-alpha-diversity-bonus-plateau.md)
  — the empirical result this note explains the mechanism behind.
- `core/mcts_sem_search_v02_00_00.py::_diverse_select`,
  `MCTS.__init__`, `MCTS.select_child` — the code this note traces.
- No `lam` sweep exists yet in the repo — if one is designed, it
  belongs alongside the `ds_alpha` sweep-mechanics todo already queued
  in the vault (`00_inbox/exp_todos.md`, 2026-07-07), not as a
  separate, uncoordinated axis.
