# sem-mcts selection: why ds_alpha needs to be ~100x ds_beta

*2026-07-07*

Mechanism behind the numbers in
[ds-alpha-diversity-bonus-plateau.md](../exp-findings/ds-alpha-diversity-bonus-plateau.md)
(the empirical "switch, not a dial" finding) — this note derives
*why* the default `ds_alpha=100, ds_beta=1` isn't expressing
"diversity matters 100x more than accuracy," and why a `ds_alpha`-only
sweep with `ds_beta` fixed at 1 loses nothing.

## The question

`core/mcts_sem_search_v02_00_00.py::_diverse_select` scores each
candidate arm as

```python
q_vals = ds_beta * q_scores + ds_alpha * q_diversity
```

`q_scores` is a PRM-derived running mean (`ch.q_value()`); `q_diversity
= sqrt(x^T V^-1 x)`. Are these two terms on comparable scales? If not,
what does that imply for choosing `ds_alpha`/`ds_beta`?

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

## q_diversity starts around 10, not around 1

`q_diversity(x) = sqrt(x^T V^-1 x)`, with `x` a unit-norm pooled
embedding (`embeds_normalize: true`, `conf/search/mcts_sem_v02.yaml`).
At initialization, before any embedding has been folded into the
covariance (`MCTS.__init__`, `mcts_sem_search_v02_00_00.py:497-504`):

```
V_inv = (1/lam) * I,  lam = 0.01  =>  V_inv = 100 * I
q_diversity(x) = sqrt(x^T (100*I) x) = 10 * ||x|| = 10   (since ||x||=1)
```

So **at the very first selection at a node, `q_diversity ≈ 10`** — two
orders of magnitude above a `q_scores` value in `[0,1]`. As more
vectors accumulate (`V_inv` updated via `select_child`'s exact-inverse
or Sherman-Morrison path), `q_diversity` shrinks along directions
already represented in `V`, but it starts an order of magnitude above
the score term and only converges toward comparability after
substantial accumulation.

This is exactly why the project's default is `ds_alpha=100` (not
~1-10): it's compensating for `q_diversity`'s much larger natural
scale at the point selection actually happens, not encoding a belief
that diversity should dominate 100x.

## ds_beta=1, tune only ds_alpha — lossless

`q_vals = ds_beta*q_scores + ds_alpha*q_diversity` — scaling both
terms by the same constant doesn't change the argmax over children, so
only the *ratio* `ds_alpha/ds_beta` matters for which arm is selected.
Fixing `ds_beta=1` and sweeping `ds_alpha` covers the full
one-parameter family; nothing is lost by not also varying `ds_beta`.
This is what the project's `ds_alpha sweep (v02)` tables in
[exp-comparison.md](../../exp-comparison.md) already do (`ds_beta=1.0`
fixed throughout).

## Recommended sweep range

Given the scale above, `ds_alpha=0` (pure q-value) vs. any nonzero
value tests qualitatively different regimes (bonus off vs. on); once
on, the effective weight relative to `q_diversity`'s ≈10-at-init scale
is what should be swept. The empirical finding
([ds-alpha-diversity-bonus-plateau.md](../exp-findings/ds-alpha-diversity-bonus-plateau.md))
confirms this reasoning: `ds_alpha ∈ {0, 10, 100}` is sufficient —
`0` establishes the on/off jump (the large, real effect), and `10`
already saturates the plateau that `100`/`1000` sit on. Testing `1000`
is redundant compute; if more confidence is wanted, spend trials
re-running `10` or `100` at higher n rather than sweeping a fourth
magnitude.

## Connections

- [ds-alpha-diversity-bonus-plateau.md](../exp-findings/ds-alpha-diversity-bonus-plateau.md)
  — the empirical result this note explains the mechanism behind.
- `core/mcts_sem_search_v02_00_00.py::_diverse_select`,
  `MCTS.__init__`, `MCTS.select_child` — the code this note traces.
