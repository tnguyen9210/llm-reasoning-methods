# Covariance precision: `cov_dtype` ("fp32" | "fp64")

*Built 2026-07-14 —
[decisions-log.md #2026-07-14](../decisions-log.md#2026-07-14--search-configs-cov_dtype-fp32--fp64-explicit-covariance-precision).*

## What was actually true before this flag existed

`V`/`V_inv` (`core/mcts_sem_search_v02_00_00.py`, `MCTS.__init__` /
`select_child`) were seeded via `np.eye(embeds_dim)` and
`np.linalg.solve(V, np.eye(...))` with **no `dtype=` argument** —
both default to `np.float64`. The pooled embeddings fed into them
(`_extract_embeds`: `pooled.detach().cpu().float().numpy()`) are
**float32**. Every op that combines them (`V_inv @ u`, `u @ u.T`, the
`einsum` in `_diverse_select`) was silently upcast to float64 by
NumPy's mixed-dtype promotion rules. So the covariance math was
**already fp64**, just implicitly, with no way to compare against a
uniform-fp32 alternative and no explicit record of the choice.

## What the flag does

`cov_dtype: "fp32" | "fp64"` (default `"fp64"`, on `MCTSSemV01Config`,
inherited by v02) makes this explicit and controllable:
- `"fp64"` — unchanged behavior, now stated rather than implied.
- `"fp32"` — `V`/`V_inv` seeded as float32 (`np.eye(...,
  dtype=np.float32)`), and every array multiplied against them (`u`
  in `select_child`, `q_embeds` in `_diverse_select`) explicitly cast
  to float32 first, so the whole covariance path — not just the seed
  matrices — runs at a single controlled precision instead of
  drifting to float64 via promotion.

Threaded through: `MCTS.__init__` (seeds `self.cov_dtype` via a
module-level `_COV_DTYPES = {"fp32": np.float32, "fp64": np.float64}`
lookup, raising on an unrecognized value), `select_child` (casts `u`
and the `np.eye` identity in the exact-update branch), and
`_diverse_select` (new `cov_dtype` kwarg, casts `q_embeds`).

**2026-07-15 fix — pydantic field declaration.** The first version
(2026-07-14) only assigned `self.cov_dtype = ...` inside `__init__`,
never declaring it as a class-level field. `MCTS` is a **pydantic
`BaseModel`**, which raises `ValueError: "MCTS" object has no field
"cov_dtype"` on any `self.attr = value` for an undeclared attribute —
this crashed every live orchestrator launch instantly. Fixed by
adding `cov_dtype: Any = np.float64` as a declared field on `MCTS`,
alongside the existing `V`/`V_inv`/`completed_nodes` declarations.
See [decisions-log.md #2026-07-15](../decisions-log.md#2026-07-15--search-configs-cov_dtype--embeds_center_mode--reimplemented-after-a-pydantic-field-declaration-bug).

## Why this precision axis matters here

Three reasons this isn't just defensive noise, elaborated in the
2026-07-14 discussion that motivated this flag:
1. **Error accumulates multiplicatively.** Each Sherman-Morrison
   update reuses the previous `V_inv`, which already carries rounding
   error from every prior update — an iterative recurrence, not an
   isolated computation.
2. **Ill-conditioning amplifies rounding error.** Relative error in
   `V^-1` scales with `condition_number * machine_epsilon`; at
   `embeds_dim` up to 512 with correlated hidden-state directions,
   `V`'s condition number can grow enough that fp32's `~1.2e-7`
   epsilon becomes visible, while fp64's `~2.2e-16` still has margin.
3. **Near-cancellation** in `x^T V_inv x` and the SM denominator
   `1 + u^T V_inv u` is the classic setting where small relative
   errors become large ones (catastrophic cancellation) — could flip
   which candidate looks "more diverse."

`rep_exp` (the closest published relative of this method — see
[rep-exp-elliptical-bonus-review.md](rep-exp-elliptical-bonus-review.md))
independently upgrades its own projected features to float64 right
before its covariance math, while running everything else (the LLM
forward pass) in bf16 — an independent second data point for the same
"cheap subroutine gets full precision" split.

## Cost

At `embeds_dim` = 32–512, the `d x d` covariance ops (`O(d^2)` per
selection for `cov_update=sm`, `O(d^3)` for `exact`) are microseconds
even at fp64 on a V100S — immaterial next to an LLM forward pass, and
untouched by the [[project_env_volta_constraint]] no-bf16 constraint
(this flag only ever touches the small covariance tensor, never
anything that runs through the policy/PRM forward pass).

## Hash handling

Same reusable mechanism as `embeds_center_mode`
([embeds-centering-design.md](embeds-centering-design.md)):
`_HASH_EXCLUDE_IF_DEFAULT["search"]["cov_dtype"] = "fp64"` drops the
field from the config identity iff it equals the pinned neutral value
`"fp64"`, so every pre-existing hash is unaffected.

**Verified (2026-07-14):**
- Baseline (no override): `cfg-c371341f` — unchanged from before this
  flag existed.
- `+cov_dtype=fp32`: `cfg-573c095f` — distinct, as expected.
- Explicit `cov_dtype=fp64` override: `cfg-c371341f` — identical to
  baseline, confirming exclude-if-default works.
- `python status.py --group sem-mcts`: full ledger unaffected, 0
  orphans/mismatches.

## Status: implemented, hash-verified, and smoke-tested

**Verified (2026-07-15, live smoke test):** `1q/1trial`,
`results_subdir=smoketest`, `cov_dtype=fp32` explicit,
`WANDB_MODE=offline` — ran end-to-end (model load → full
`num_phases=1000` search → scoring → scored dataset written) with no
crash, confirming the pydantic field-declaration fix above actually
works at runtime, not just at the hash-check level.

Still open: no live A/B run comparing `fp32` vs. `fp64` selected-child
*sequences* (i.e. do the two precisions actually pick different
children at any point) has been executed. That comparison is the
natural next step if/when this is prioritized, and is the empirical
check for the claim that fp64 matters here at all, rather than
resting on the argument above alone.

## Scope

v01 and v02 both get the field (defined once on `MCTSSemV01Config`,
inherited — same as `embeds_center_mode`'s actual landing spot,
correcting an earlier assumption that it was duplicated per-class).
The `select_child`/`_diverse_select` wiring lives in the v02 core
file; v01's own core file was not touched in this pass — confirm
before assuming v01 respects the flag at runtime.
