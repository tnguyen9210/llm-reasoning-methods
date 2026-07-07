# Child selection: two scenarios, dispatched by visit count

No prior log entry covers this — written directly from code
(`core/mcts_sem_search_v02_00_00.py`), verified 2026-07-07, following a
session discussion of how sem-mcts picks a child at each tree
descent. See also
[decisions/sherman-morrison-covariance-update.md](sherman-morrison-covariance-update.md)
for the covariance-update mechanics shared across both scenarios.

## The two scenarios

`MCTS.select_child(node)` dispatches on `node.visit_count()`:

**Scenario 1 — first visit (`node.visit_count() == 1`) →
`_select_by_q_value`.** Pure argmax on q-value, no diversity term at
all:

```python
qs = [ch.q_value() for ch in node.children]
best_q = max(qs)
best_childs = [ch for ch, q in zip(node.children, qs) if abs(best_q - q) <= tol]
return random.choice(best_childs)
```

Ties within `tol=1e-4` are broken by uniform random sampling.

**Scenario 2 — subsequent visits (`node.visit_count() > 1`) →
`_select_by_diversity`.** Combines q-value and diversity via
`_diverse_select`:

```python
log_nvisit_parent = np.sqrt(np.log(1 + node.visit_count()))
best_idx = _diverse_select(
    self.V_inv, embeds, q_values,
    ds_alpha * log_nvisit_parent,   # effective alpha grows with visits
    ds_beta,
)
```

`_diverse_select` internally computes `q_vals = ds_beta*q_scores +
ds_alpha_effective*q_diversity` and argmaxes that, with the same
`tol`-based random tie-break as scenario 1.

## Why the split exists

Right after a node is expanded, every child has `visit_count() == 1`
and its q-value is just its raw PRM candidate score — nothing has been
backpropagated through it yet. At that exact instant, the diversity
term `sqrt(x^T V^-1 x)` would be uninformative: `V` hasn't accumulated
*any* of these specific children's embeddings yet, so every child's
diversity bonus reflects only what earlier, unrelated selections
contributed to `V` — not anything about how these children relate to
each other. Mixing that in would add noise to what is otherwise a
clean q-value comparison, not real exploration signal.

Once a node has been revisited, the diversity bonus becomes meaningful
— `V` has now accumulated at least one of this node's children's
embeddings, so `sqrt(x^T V^-1 x)` genuinely reflects which directions
among *these* children are under-explored. The `sqrt(log(1 +
parent_visits))` factor on the effective `ds_alpha` scales exploration
pressure up as visits accumulate — a UCB-style schedule where diversity
matters progressively more the longer a node has been sunk into.

## The one thing shared across both scenarios

Regardless of which selection path fires, `select_child` **unconditionally**
folds the selected child's embedding into the covariance (`V`/`V_inv`)
afterward — including on the first-visit path, which never *reads*
`V_inv` at all. This matters because that path still commits to a
child: if its direction didn't also get folded in, `V_inv` would go
stale relative to what was actually selected (no longer equal to
`inv(V)`), silently corrupting every later diversity bonus computed
from it. See
[decisions/sherman-morrison-covariance-update.md](sherman-morrison-covariance-update.md)
for how that fold-in is computed under each `cov_update` mode.

## Summary

| | trigger | uses diversity? | tie-break |
|---|---|---|---|
| `_select_by_q_value` | `node.visit_count()==1` (first visit after expansion) | no — pure q-value argmax | uniform random within `tol` |
| `_select_by_diversity` | `node.visit_count()>1` | yes — `ds_beta*q + ds_alpha*sqrt(log(1+visits))*diversity` | uniform random within `tol` (inside `_diverse_select`) |

Both paths write into `V`/`V_inv` unconditionally after selecting.

## Revisit if

The first-visit special case is ever found to matter for a v01
comparison too — v01's selection is structured differently (a
within-call greedy-K batch selector, not a persistent-state dispatcher;
see
[decisions/sherman-morrison-covariance-update.md](sherman-morrison-covariance-update.md)),
so this exact two-scenario split is v02-specific and hasn't been
verified to apply to v01's selection shape.
