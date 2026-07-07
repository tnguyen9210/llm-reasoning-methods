# Sherman-Morrison covariance update: what it is, and a real v01-vs-v02 divergence

*Originating rename:
[decisions-log.md #2026-06-20](../decisions-log.md#2026-06-20--configs-cov_update-value-renamed-sherman_morrison---sm) —
that entry only covers the value spelling
(`"sherman_morrison"` → `"sm"`); the algorithm itself and its
implementation are not covered by any log entry, so this doc is
written directly from the current code (verified 2026-07-07).*

## What `cov_update` controls

Sem-mcts's diversity bonus is `sqrt(x^T V^-1 x)`, where `V = λI + Σ uuᵀ`
is a covariance accumulated over selected embeddings. Computing the
bonus needs `V^-1`; `cov_update` picks how it's kept up to date as new
`u` vectors are folded in:

- **`"exact"`** — keep `V` itself; recompute `V^-1` from scratch
  (`np.linalg.solve(V, I)`) every selection. **O(d³)** per selection.
- **`"sm"`** (Sherman-Morrison) — keep `V^-1` directly and rank-1
  update it in closed form each selection, without ever forming or
  inverting `V`:

  ```
  (V + uuᵀ)^-1 = V^-1 - (V^-1 u)(V^-1 u)ᵀ / (1 + uᵀ V^-1 u)
  ```

  **O(d²)** per selection — the whole point of the mode.

Both start from the same closed-form initial state: `V_0 = λI`, so
`V_0^-1 = (1/λ)I` with no inverse call needed at all, and the initial
diversity bonus is uniform across arms regardless of mode.

**Current default:** the dataclass default (`MCTSSemV01Config`,
`utils/configs.py`) is `"exact"`. `conf/search/mcts_sem_v01.yaml` keeps
that default. **`conf/search/mcts_sem_v02.yaml` overrides it to
`"sm"`** — so `sm` is v02's live default, not a schema-wide default;
v01 runs `exact` unless explicitly overridden.

## A real divergence between v01 and v02's `sm` implementations

The two files' Sherman-Morrison code is **not** the same
implementation reused — they differ in shape and in one specific
numerical-stability detail.

**v02** (`core/mcts_sem_search_v02_00_00.py::MCTS.select_child`) — a
single persistent `V_inv`, living on the `MCTS` instance, updated
**once per selection**, across the whole search:

```python
Vu = self.V_inv @ u
denom = 1.0 + float(u.T @ Vu)
self.V_inv = self.V_inv - (Vu @ Vu.T) / denom
self.V_inv = 0.5 * (self.V_inv + self.V_inv.T)   # <- symmetrize
```

The unconditional covariance-fold-in happens on *every* selection,
including the first-visit q-value-only path (`_select_by_q_value`)
that never reads `V_inv` at all — that path still commits to a child,
so its direction must enter the covariance or `V_inv` would go stale
relative to what was actually selected, corrupting every later
diversity bonus computed from it.

**v01** (`core/mcts_sem_search_v01_00_00.py::_diverse_select`) — a
**local, within-call** `_V_inv`, deep-copied from the caller's `V` at
the start of each call, greedily picking `K` arms in a loop inside one
function invocation:

```python
_V = copy.deepcopy(V)
_V_inv = np.linalg.inv(_V) if cov_update == "sm" else None
for _ in range(K):
    ...
    Vu = _V_inv @ u
    denom = 1.0 + float(u.T @ Vu)
    _V_inv = _V_inv - (Vu @ Vu.T) / denom
    # no symmetrize step here
```

**The difference that matters:** v02 re-symmetrizes `V_inv` after
every single update (`0.5 * (V_inv + V_inv.T)`), explicitly to stop
floating-point asymmetry from compounding over a long-running search.
v01's within-call loop has no equivalent step — `_V_inv` accumulates
without correction across its `K` within-call iterations. This isn't
necessarily a bug (v01's `K` is bounded per call, so the drift has a
bounded window to compound in, unlike v02's `V_inv` which persists and
updates across the *entire* search for a question), but it means the
two "sm" paths carry different numerical-stability guarantees, not
just different call shapes. This matches what the v02 YAML comment
already hints at (*"the dataclass default stays exact for v01, whose
sm path isn't refactored"*) — v01's `sm` mode exists and is selectable,
but hasn't received the same refactor/hardening v02's has.

## Why "sm" is trusted as the default for v02 specifically

Per the v02 YAML comment: `sm` was **"validated machine-precision-
identical to exact (0 selection mismatches)"** before being made the
default there — i.e., an empirical equivalence check (same selections,
same run) was performed for v02's implementation specifically. No
equivalent validation claim exists in the repo for v01's `sm` path;
v01 keeping `"exact"` as its default is consistent with that gap
(not validated to the same standard, so not defaulted to).

## Revisit if

v01's `sm` path is ever made the default there too — at minimum, add
the same re-symmetrization step v02 has, and run the same
machine-precision-identical validation before trusting it as a
default rather than an opt-in. Also revisit if `sm`'s accumulated
asymmetry (in either file) is ever observed to cause a real numerical
problem (e.g. `V_inv` losing positive-definiteness) — currently
theoretical, not observed.
