# LLM Reasoning — MCTS Experiment Comparison — PRM800K Level 5

> **Provenance:** structure mirrored from [exp-comparison.md](exp-comparison.md) (the level-4 doc) on 2026-07-10; every table reset to `planned` — no level-5 runs exist yet. Launch commands are the level-4 counterparts' plus `data.level=5` (config hashes and `--level-5--` run names follow automatically). Intro/`Fixed` prose is inherited from the level-4 doc: table definitions remain valid, but any inherited claim about completeness or findings describes level-4 state — trust the (all-planned) tables here over such prose until level-5 results land. The level-5 grid also **drops two models** relative to level 4 — llama-3b gptq and qwen-3b gptq-int4 — so inherited “7-model” grid prose reads as 5 models here (llama-1b, llama-3b fp16, qwen-3b fp16, qwen-7b gptq-int4, qwen-math-1.5b).

Central tracker for every MCTS search experiment (cnt / sem /
cnt-bl / sem-bl) on PRM800K — per-algorithm tuning tables grouped
by gen_budget, plus a cross-algorithm best-config summary.

## Purpose
The four algorithm tracks (`llm-reasoning-mcts-exp`,
`llm-reasoning-mcts-bl-exp`, + the `sem` variants) own
*implementation* milestones. This doc owns *experiment tracking
and comparison*: per-algorithm tuning grids and the cross-algorithm
verdict — the views none of the per-algorithm tracks give.

**This is a living log, not a milestone doc.** No `progress: N/M`;
"done" isn't a state here.

## Structure (why it's shaped this way)
Two activities, two shapes:
- **Tuning *within* an algorithm, at a given budget and
  model** → small, algorithm-specific grid of its own knobs
  (template, cpuct, embeds-strategy…). One
  `## Tuning tables [gen_budget=N]` section per budget →
  nested `### algorithm` → `##### model` (or an `llm` column
  when several model tables share one comparison), so each
  table is just the config rows for one (budget, algorithm,
  model) cell. Different algorithms show different columns —
  no forced shared schema.
- **Comparing *across* algorithms at a fixed budget** →
  sparse, one row per algorithm × model. The **Summary**
  table (your fixed columns), one row per algorithm × model
  × budget.

`gen_budget` is a top-level tuning-tables section; algorithm
and model are *subsection* levels within it, not columns —
model gets its own level (or column, for grouped comparisons)
because its GPU constraints (cf. the trial-loop OOM) and
scaling behavior are what you tune around. The
within-algorithm scaling curve (80→160→320) is read by
scanning one algorithm/model across the `gen_budget=N` tuning
sections; the cross-algorithm-per-budget cut lives in the
Summary. (Larger budgets need less tuning, so those sections
will be sparser — the nesting keeps that asymmetry clean
instead of one wide sparse grid.)

## How to use
- **Plan/run** → add a config row under the matching
  `## Tuning tables [gen_budget=N]` → `### algorithm` →
  `##### model`; log hypothesis + follow-up in the Run log.
- **A config wins at a budget** → promote it to the Summary
  as that (algorithm, model, budget)'s best config, linking
  back to the tuning row.
- **Metrics**: tuning tables carry pass@gb only (terse);
  the Summary carries the full set (pass/naive/weighted/
  maj@gb). Everything else (depth, ncomps, timing) stays in
  W&B / the result dir, linked.
- **Best-config rule:** best-scoring row *at that budget*,
  same LLM/level/trials, picked across **all tuning knobs
  jointly** (template, cpuct, …) — not "best template" with
  other knobs held fixed. Don't promote a config from a
  different budget or LLM into a Summary row — that breaks
  the comparison.

---

## Cross-algorithm summary (QwenPRM)
> One table per model, one row per algorithm — pulled directly from
> each algorithm's own "model family, size, quantization comparison
> (QwenPRM)" table above/below (`cnt-mcts`, `sem-mcts`,
> `cnt-mcts-bl-v01`, `cnt-mcts-bl-v02`, `cnt-mcts-bl-v03`,
> `sem-mcts-bl-v01`). All rows fixed at b=80, bs-4, d-20,
> agg_strategy=`last`, tmpl=model-family default (native for Qwen,
> custom for Llama), prm=qwen. `cnt-mcts` row is method=`mcts_cnt_v01`
> (the only cnt-mcts entry point at this level — see the
> `### cnt-mcts` section above). `sem-mcts` row is `mcts_sem_v02` (PRM embeds),
> `ds_alpha=100` (w_eff not applicable — that knob is bl_sem-specific).
> `sem-mcts-bl-v01` row uses the `w_eff=100` table (more complete than
> `w_eff=10` at time of writing: 5/7 vs. 4/7 cells scored); see that
> algorithm's own section for the `w_eff=10` comparison point.
> `cnt-mcts-bl-v02` (Fractional KUBE) and `cnt-mcts-bl-v03`
> (depth-shaping) are each filled at 5 of 7 models as of 2026-07-09
> (see `docs/decisions/kube-bonus-schedule.md` /
> `kube-affordability-restriction.md` and
> `docs/decisions/depth-shaping-knapsack-bonus.md` for the
> algorithms).

**llama-1b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | — | planned | — | — | — | — | — |
| sem-mcts | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v01 | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v02 | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v03 | — | planned | — | — | — | — | — |
| sem-mcts-bl-v01 | — | planned | — | — | — | — | — |

**llama-3b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | — | planned | — | — | — | — | — |
| sem-mcts | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v01 | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v02 | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v03 | — | planned | — | — | — | — | — |
| sem-mcts-bl-v01 | — | planned | — | — | — | — | — |

**qwen-3b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | — | planned | — | — | — | — | — |
| sem-mcts | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v01 | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v02 | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v03 | — | planned | — | — | — | — | — |
| sem-mcts-bl-v01 | — | planned | — | — | — | — | — |

**qwen-7b gptq-int4**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | — | planned | — | — | — | — | — |
| sem-mcts | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v01 | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v02 | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v03 | — | planned | — | — | — | — | — |
| sem-mcts-bl-v01 | — | planned | — | — | — | — | — |

**qwen-math-1.5b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | — | planned | — | — | — | — | — |
| sem-mcts | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v01 | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v02 | — | planned | — | — | — | — | — |
| cnt-mcts-bl-v03 | — | planned | — | — | — | — | — |
| sem-mcts-bl-v01 | — | planned | — | — | — | — | — |

> **Analysis.** No level-5 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-4 counterpart's command plus `data.level=5`.

---

## Tuning tables [gen_budget=80]
> Hierarchy: `### <algorithm>` → `##### <model family + size>`
> → a table whose rows are configs (template × cpuct …).
> Algorithm and model are *subsection* levels, not columns
> (model sometimes becomes an `llm` column instead, for
> grouped comparisons across models). Higher budgets get their
> own `## Tuning tables [gen_budget=N]` section as runs land
> (expected sparser — less tuning at high budget).

### cnt-mcts
> method=`mcts_cnt_v01`, post-`PRM._split_steps` fix (2026-07-06 —
> see
> [findings/coding-findings/prm-step-split-trailing-separator.md](findings/coding-findings/prm-step-split-trailing-separator.md)
> and `docs/decisions-log.md`), which affected `agg_strategy="last"`
> scoring for non-terminal candidates in every table below. (The
> level-4 doc also carries the older pre-fix `mcts_cnt` section for
> comparison against already-scored data; level 5 has no scored
> pre-fix runs, so that section is dropped here — `mcts_cnt_v01` is
> the only cnt-mcts entry point for this level.)


#### model family, size, quantization comparison (RLHFlowPRM)
> **Fixed:** method=`mcts_cnt_v01`, prm=rlhflow, agg_strategy=
> `last`, cpuct=2.0, bs-4, d-20, b=80, prm_batch_size=1 (default —
> every row uses the same default, so fp16/GPTQ runtimes are
> directly comparable), tmpl=model-family
> default (native for Qwen, custom for Llama).
>
> **W&B:** none yet (no level-5 runs).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Fixed a real `compute_stats.py` hang while filling this row.**
> `qwen-3b gptq-int4` (rlhflow) reproducibly hung — bisected to one
> record (`test/precalculus/920.json`, a matrix-power question)
> whose model completion boxed a whole equation instead of a value;
> comparing it via `sympy` hung so hard that `signal.alarm`
> (`utils/metrics.py::run_with_timeout`/`_grade_pred`) couldn't
> interrupt it — signals only fire between Python bytecode
> instructions, and the stuck call was in `sympy`'s C-level code.
> Fixed by passing `timeout=True` to `grader2.math_equal` at both
> call sites, routing symbolic comparison through `grader2.py`'s
> already-existing (but previously unused by `metrics.py`) hard-kill
> subprocess path (`call_with_timeout`/`symbolic_equal_process`,
> `multiprocessing.Process.terminate()`) instead of in-process
> comparison. Verified: the poison record now resolves in ~1-10s
> instead of hanging forever; both trial files replayed clean
> end-to-end; the real `compute_stats.py` invocation for this cell
> now completes in ~1 minute. See
> [findings/coding-findings/compute-stats-sympy-hang.md](findings/coding-findings/compute-stats-sympy-hang.md)
> for the full write-up.

#### model family, size, quantization comparison (QwenPRM)
> **Fixed:** method=`mcts_cnt_v01`, prm=qwen, agg_strategy=
> `last`, cpuct=2.0, bs-4, d-20, b=80, prm_batch_size=1 (default,
> matched across every row — same rationale as the rlhflow
> table above), tmpl=model-family default (native for Qwen,
> custom for Llama). Companion to the rlhflow-PRM table above;
> same 7 model/quant configs, different scoring PRM.
>
> **W&B:** llama-1b `05lky8bc`, llama-3b `grfdicia`, qwen-3b
> `wns54ql3`, qwen-math-1.5b `43zjzxmj`.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .3619<br>±.0294 | .2724<br>±.0272 | .2127<br>±.0250 | .1903<br>±.0240 | 2.98 |
| llama-3b fp16 | 2 | scored | .5522<br>±.0304 | .4291<br>±.0303 | .4104<br>±.0301 | .3619<br>±.0294 | 5.13 |
| qwen-3b fp16 | 2 | scored | .6978<br>±.0281 | .5896<br>±.0301 | .5896<br>±.0301 | .5410<br>±.0305 | 4.63 |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | 2 | scored | .7575<br>±.0262 | .6418<br>±.0293 | .6455<br>±.0293 | .6269<br>±.0296 | 3.37 |

#### agg_strategy comparison (qwen-3b, qwen-math-1.5b)
> **Compares:** `gen.agg_strategy` (`"min"` | `"prod"` | `"last"` —
> `core/scoring.py::aggregate_scores`) — how a candidate's
> per-step PRM scores collapse to one scalar. `"last"` is every
> other table's fixed default; `"min"` and `"prod"` are
> implemented but not yet reported anywhere in this doc. Prompted
> by the `_split_steps` fix (`agg="last"`-specific bug, see
> `### cnt-mcts` header above) — `"min"` in particular
> is a useful cross-check since it's structurally less exposed to
> that bug (a holistic bogus score rarely wins a min() over a
> trajectory with a genuinely bad step).
>
> **Fixed:** method=`mcts_cnt_v01`, cpuct=2.0, bs-4, d-20, b=80,
> tmpl=model-family default (native for both models here).

| llm | prm | agg_strategy | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | qwen | min | — | planned | — | — | — | — | — |
| qwen-3b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-3b | qwen | last | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | min | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | last | — | planned | — | — | — | — | — |

### sem-mcts (v02)
> **Runnable as of 2026-06-18** (rename + migration landed).
> Two methods = two embedding sources: `mcts_sem_v01` (policy
> embeds, 2nd vLLM engine) and `mcts_sem_v02` (PRM embeds, no
> 2nd engine). knobs beyond template: ds_alpha, ds_beta,
> lam, embeds_strategy (last/avg), embeds_normalize, and for
> v02 embeds_proj (none/sparse, dim 512) + cov_update
> (exact/sherman_morrison). Defaults in conf/search/mcts_sem_v0*.
> Run v01 and v02 at matched model/level/trials vs. cnt-mcts —
> the comparison the project exists for. sem-mcts has no
> `cpuct` knob (selection is q-value-only on first visit, then
> a ds_alpha/ds_beta-weighted diversity bonus on later visits;
> see `core/mcts_sem_search_v02_00_00.py:select_child`).


#### embeds_strategy × scope sweep (QwenPRM)
> **Compares:** how the PRM hidden state is pooled into the
> covariance bonus — `embeds_strategy` (`last` = final-token
> hidden state vs. `avg` = mean over tokens) crossed with
> `embeds_scope` (`full` = the whole prompt+response sequence
> vs. `response` = only the assistant response tokens). The
> question is whether averaging or response-only scoping
> changes the diversity signal enough to move pass@gb.
>
> **Fixed:** method=`mcts_sem_v02`, llm=llama-3b, prm=qwen,
> tmpl=custom (llama default), bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, ds_beta=1.0.
>
> ⚠️ `embeds_scope=response` is **not supported on v02** (PRM
> source): `_extract_embeds` computes `response_start_idx`
> with the generator tokenizer, which doesn't apply to PRM
> hidden states, so the core raises for that combination (see
> `core/mcts_sem_search_v02_00_00.py:227`). The two `response`
> rows are therefore **blocked** — shown for completeness but
> not runnable until that fix lands; they are NOT queued in
> `experiments.yaml`. `last`×`full` is the v02 default config
> and is already done (it's the qwen-PRM llama-3b default run,
> W&B `kbwjqw96`-family — same cfg as the ds_alpha=100 cell).
>
> **W&B:** none yet (no level-5 runs).
>
> **`lam=0.1` addendum (2026-07-08):** two extra `w_eff` checkpoints
> per strategy, at `lam=0.1` (`w_eff = ds_alpha/sqrt(lam)`, so
> `w_eff=10 → ds_alpha=3.16`, `w_eff=100 → ds_alpha=31.6`). The
> `last` rows at both checkpoints reuse already-scored cells from
> the `lam / ds_alpha joint sweep (v02, llama-3b)` table (same
> `lam=0.1, ds_alpha` pairs, same fixed config) — no new run. The
> `avg` rows at both checkpoints are new; both now run and scored
> (2026-07-08/09).
>
> **`lam=0.01` addendum (2026-07-08):** same two `w_eff` checkpoints
> at the table's default `lam=0.01` (`w_eff=10 → ds_alpha=1.0`,
> `w_eff=100 → ds_alpha=10`). The `last` rows reuse already-scored
> cells from the `lam / ds_alpha joint sweep (v02, llama-3b)` table
> (`cfg-23f6c64a`, `cfg-baa5b18e`) — no new run. The `avg` rows are
> new; both now run and scored (2026-07-08/09).

| strategy | scope | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| last | full | 0.01 | 100 | 1000 | — | planned | — | — | — | — | — |
| last | full | 0.01 | 1.0 | 10 | — | planned | — | — | — | — | — |
| last | full | 0.01 | 10 | 100 | — | planned | — | — | — | — | — |
| last | full | 0.1 | 3.16 | 10 | — | planned | — | — | — | — | — |
| last | full | 0.1 | 31.6 | 100 | — | planned | — | — | — | — | — |
| avg | full | 0.01 | 100 | 1000 | — | planned | — | — | — | — | — |
| avg | full | 0.01 | 1.0 | 10 | — | planned | — | — | — | — | — |
| avg | full | 0.01 | 10 | 100 | — | planned | — | — | — | — | — |
| avg | full | 0.1 | 3.16 | 10 | — | planned | — | — | — | — | — |
| avg | full | 0.1 | 31.6 | 100 | — | planned | — | — | — | — | — |
| last | response | — | — | — | — | planned | — | — | — | — | — |
| avg | response | — | — | — | — | planned | — | — | — | — | — |

> **Limitations / follow-up:** the two `response` rows are blocked
> on PRM-source `response_start_idx` support; queue them once the
> v02 core handles `embeds_scope=response` for `embeds_source=prm`.
> A v01 (policy-embeds) version of this table would unblock the
> `response` axis, since v01 supports it.

#### lam / ds_alpha joint sweep (llama-1b)
> **Compares:** whether `lam` (the ridge constant setting `V`'s
> initial scale, `V_0 = lam*I`) and `ds_alpha` are truly redundant
> along the derived invariant `w_eff = ds_alpha/sqrt(lam)`, or
> whether `lam`'s second role (controlling how fast `V_inv` adapts
> as embeddings accumulate) has an independent effect on pass@gb.
> The two `ds_alpha sweep (v02)` tables above only ever tested
> `lam=0.01`; this table holds `w_eff` fixed across rows and varies
> `lam`/`ds_alpha` jointly, filling in `w_eff` values below the
> confirmed plateau (`w_eff≥100`, i.e. `ds_alpha≥10` at `lam=0.01`)
> that have never been tested. `w_eff ∈ {0.1, 0.3, 3.0}` extend the
> grid below `w_eff=1` — the existing plateau finding only shows
> *that* a switch happens somewhere in `(0, 100)`, not the shape of
> the on-ramp; these log-spaced points (matching the existing
> `{0.316, 3.16, 31.6}` spacing at `lam=0.1`) probe whether the
> switch is sharp near `w_eff≈1` or a gradual ramp from `w_eff=0`.
>
> **Fixed:** tmpl=model-family default, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=qwen, ds_beta=1.0,
> prm_batch_size=1, llm=llama-1b. Queued *after* the llama-3b table
> below, which is the one actually running first.
>
> See
> [tuning-semantic-score-weights-and-lambda.md](decisions/tuning-semantic-score-weights-and-lambda.md)
> for the `w_eff` derivation and the full 5-step tuning procedure
> this table is step 1/2 of. **Step 1** is the two bolded cells
> below (`w_eff=10`, matched across `lam=1.0` and `lam=0.01`) — if
> their pass@gb/naive@gb agree within SEM, `lam`'s independent role
> is negligible and the remaining cells collapse to a 1D sweep over
> `w_eff` at one fixed `lam` (skip the rest of the grid); if they
> disagree outside SEM, run the full grid.
>
> ✅ Step 1 done (2026-07-08); every `w_eff∈{1,3,10,100}` row across
> all three `lam` values scored via `compute_stats.py` as of
> 2026-07-08 — no `experiments.yaml` entries were added for these runs
> (launched and generated directly; see below). Under `prm=qwen`,
> llama-1b still has **no existing `ds_alpha=0` (w_eff=0) baseline**
> (the qwen-PRM `ds_alpha sweep (v02, qwen PRM)` table above only has
> 10/100/1000 for llama-1b; only the *rlhflow* table has a llama-1b
> `ds_alpha=0` row) — that row is still not started, and neither are
> the `w_eff∈{0.1,0.3}` on-ramp cells. The `w_eff=100, lam=0.01` row
> reuses the already-scored llama-1b `ds_alpha=10` cell from the
> qwen-PRM sweep table.

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-1b | qwen | 0.01 | 0 | 0 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 1.0 | 0.1 | 0.1 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.1 | 0.0316 | 0.1 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.01 | 0.01 | 0.1 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 1.0 | 0.3 | 0.3 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.1 | 0.0949 | 0.3 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.01 | 0.03 | 0.3 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 1.0 | 1 | 1 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.1 | 0.316 | 1 | 2 | scored | .2910<br>±.0278 | .2537<br>±.0266 | .2239<br>±.0255 | .2052<br>±.0247 | 4.82 |
| llama-1b | qwen | 0.01 | 0.1 | 1 | 2 | scored | .2985<br>±.0280 | .2425<br>±.0262 | .2313<br>±.0258 | .2127<br>±.0250 | 4.98 |
| llama-1b | qwen | 1.0 | 3.0 | 3.0 | 2 | scored | .3507<br>±.0292 | .2799<br>±.0275 | .2500<br>±.0265 | .2239<br>±.0255 | 4.97 |
| llama-1b | qwen | 0.1 | 0.949 | 3.0 | 2 | scored | .3433<br>±.0291 | .2687<br>±.0271 | .2537<br>±.0266 | .2164<br>±.0252 | 4.91 |
| llama-1b | qwen | 0.01 | 0.3 | 3.0 | 2 | scored | .3321<br>±.0288 | .2575<br>±.0268 | .2575<br>±.0268 | .2276<br>±.0257 | 4.93 |
| llama-1b | qwen | **1.0** | **10** | **10** | 2 | scored | .3582<br>±.0293 | .2649<br>±.0270 | .2463<br>±.0264 | .2090<br>±.0249 | 4.96 |
| llama-1b | qwen | 0.1 | 3.16 | 10 | 2 | scored | .3433<br>±.0291 | .2425<br>±.0262 | .2500<br>±.0265 | .2090<br>±.0249 | 5.03 |
| llama-1b | qwen | **0.01** | **1.0** | **10** | 2 | scored | .3209<br>±.0286 | .2425<br>±.0262 | .2313<br>±.0258 | .2015<br>±.0245 | 4.88 |
| llama-1b | qwen | 1.0 | 100 | 100 | 2 | scored | .3284<br>±.0287 | .2351<br>±.0260 | .1828<br>±.0237 | .1642<br>±.0227 | 4.82 |
| llama-1b | qwen | 0.1 | 31.6 | 100 | 2 | scored | .2910<br>±.0278 | .2201<br>±.0254 | .1903<br>±.0240 | .1567<br>±.0222 | 4.82 |
| llama-1b | qwen | 0.01 | 10 | 100 | 2 | scored | .3433<br>±.0291 | .2537<br>±.0266 | .1978<br>±.0244 | .1679<br>±.0229 | 4.85 |

> **Analysis.** 11/18 cells scored (2 trials each); `lam=1.0,
> ds_alpha=0.1` (w_eff=0.1) and the `ds_alpha=0` (w_eff=0)
> gap-closer remain — the `w_eff=0.1` failure is a launch attempt
> that died before `wandb.init` on 2026-07-11 and was re-queued.
> Step 1 pair (`w_eff=10`, `lam=1.0` vs `lam=0.01`): pass@gb .3582
> vs .3209 — within SEM (±.029/±.029), consistent with `lam`
> having no strong independent effect at this level, matching the
> level-4 finding.
> **Limitations / follow-up:** n=2 trials is preliminary (wide
> SEMs); `w_eff=0` and `w_eff=0.1, lam=1.0` still pending.

#### lam / ds_alpha joint sweep (llama-3b)
> **Compares:** the same `lam`/`ds_alpha` joint-tuning question as
> the llama-1b table above, on llama-3b. **This is the table queued
> to run first** — llama-3b has no existing `ds_alpha=0` baseline
> under either PRM (closing that gap and running step 1 happen in
> the same pass), and its result determines how much of the llama-1b
> table (and any later cross-check per procedure step 5) is worth
> running. `w_eff ∈ {0.1, 0.3, 3.0}` fill in the on-ramp below
> `w_eff=1`, same rationale as the llama-1b table above.
>
> **Fixed:** tmpl=model-family default, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=qwen, ds_beta=1.0,
> prm_batch_size=1, llm=llama-3b.
>
> See
> [tuning-semantic-score-weights-and-lambda.md](decisions/tuning-semantic-score-weights-and-lambda.md)
> for the `w_eff` derivation and 5-step procedure. **Step 1** is the
> two bolded cells below (`w_eff=10`); the `w_eff=0` cell closes the
> pre-existing llama-3b `ds_alpha=0` gap (neither PRM had this row
> before) and is independent of the step-1 outcome, so it's fine to
> launch alongside step 1 rather than wait.
>
> ✅ Step 1 done (2026-07-08) — both step-1 cells generated 2026-07-07
> and scored via `compute_stats.py` 2026-07-08 (no `experiments.yaml`
> entries added; launched and generated directly). The `w_eff=0` gap-
> closer (`lam=0.01, ds_alpha=0`) is still not generated. Launch
> command used: `generate_mcts_sem.py --config-name
> mcts_sem_v02_prm800k llm=llama_3b prm=qwen_prm
> search.lam=<lam> search.ds_alpha=<ds_alpha> search.ds_beta=1.0`.

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-3b | qwen | 0.01 | **0** | **0** | — | planned | — | — | — | — | — |
| llama-3b | qwen | 1.0 | 0.1 | 0.1 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.1 | 0.0316 | 0.1 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.01 | 0.01 | 0.1 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 1.0 | 0.3 | 0.3 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.1 | 0.0949 | 0.3 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 0.01 | 0.03 | 0.3 | — | planned | — | — | — | — | — |
| llama-3b | qwen | 1.0 | 1 | 1 | 2 | scored | .5336<br>±.0305 | .4179<br>±.0302 | .4142<br>±.0301 | .3843<br>±.0298 | 6.12 |
| llama-3b | qwen | 0.1 | 0.316 | 1 | 2 | scored | .5299<br>±.0305 | .4179<br>±.0302 | .4030<br>±.0300 | .3918<br>±.0299 | 6.31 |
| llama-3b | qwen | 0.01 | 0.1 | 1 | 2 | scored | .5261<br>±.0306 | .4216<br>±.0302 | .4254<br>±.0303 | .3918<br>±.0299 | 6.34 |
| llama-3b | qwen | 1.0 | 3.0 | 3.0 | 2 | scored | .5336<br>±.0305 | .4291<br>±.0303 | .4104<br>±.0301 | .3843<br>±.0298 | 6.50 |
| llama-3b | qwen | 0.1 | 0.949 | 3.0 | 2 | scored | .5336<br>±.0305 | .4440<br>±.0304 | .4254<br>±.0303 | .4142<br>±.0301 | 6.60 |
| llama-3b | qwen | 0.01 | 0.3 | 3.0 | 2 | scored | .5261<br>±.0306 | .4403<br>±.0304 | .4216<br>±.0302 | .4030<br>±.0300 | 6.52 |
| llama-3b | qwen | **1.0** | **10** | **10** | 2 | scored | .5373<br>±.0305 | .4142<br>±.0301 | .3731<br>±.0296 | .3545<br>±.0293 | 6.98 |
| llama-3b | qwen | 0.1 | 3.16 | 10 | 2 | scored | .5597<br>±.0304 | .4366<br>±.0304 | .4067<br>±.0301 | .3769<br>±.0297 | 6.95 |
| llama-3b | qwen | **0.01** | **1.0** | **10** | 2 | scored | .5784<br>±.0302 | .4403<br>±.0304 | .4291<br>±.0303 | .3881<br>±.0298 | 6.93 |
| llama-3b | qwen | 1.0 | 100 | 100 | 2 | scored | .5560<br>±.0304 | .4067<br>±.0301 | .3433<br>±.0291 | .3209<br>±.0286 | 6.98 |
| llama-3b | qwen | 0.1 | 31.6 | 100 | 2 | scored | .5634<br>±.0304 | .4291<br>±.0303 | .3694<br>±.0295 | .3358<br>±.0289 | 7.06 |
| llama-3b | qwen | 0.01 | 10 | 100 | 2 | scored | .5485<br>±.0305 | .4328<br>±.0303 | .3619<br>±.0294 | .3321<br>±.0288 | 6.99 |
| llama-3b | qwen | 1.0 | 1000 | 1000 | 2 | scored | .5485<br>±.0305 | .4216<br>±.0302 | .3507<br>±.0292 | .3060<br>±.0282 | 6.92 |
| llama-3b | qwen | 0.1 | 316.2 | 1000 | 2 | scored | .5410<br>±.0305 | .3843<br>±.0298 | .3321<br>±.0288 | .2985<br>±.0280 | 7.26 |
| llama-3b | qwen | 0.01 | 100 | 1000 | 2 | scored | .5896<br>±.0301 | .4104<br>±.0301 | .3694<br>±.0295 | .3358<br>±.0289 | 7.00 |

> **Analysis.** 15/21 cells scored (2 trials each). All three
> `w_eff=1` and `w_eff=3` rows land tightly clustered (pass@gb
> .526–.534) — no `lam`-dependence signal yet, consistent with the
> llama-1b table. The `w_eff=10` step-1 pair is now fully
> resolved: `lam=1.0` .5373 vs `lam=0.01` .5784 — a real gap this
> time, favoring low `lam`. That direction holds through
> `w_eff=100` (`lam=1.0` .5560 vs `lam=0.01` .5485, roughly flat)
> and strengthens at `w_eff=1000` (`lam=1.0` .5485 vs `lam=0.01`
> .5896, the widest lam-driven spread yet in this table) — tentative
> read: as `w_eff` grows, lower `lam` increasingly outperforms.
> **Limitations / follow-up:** 7/21 cells still running or
> queued — the `w_eff=0.1/0.3` on-ramp rows (6) and the `w_eff=0`
> gap-closer (1) are the remaining tail; only 2 trials/cell so the
> lam-driven spreads above are suggestive, not conclusive.

#### lam / ds_alpha joint sweep (qwen-math-1.5b)
> **Compares:** the same `lam`/`ds_alpha` joint-tuning question as
> the llama-1b/llama-3b tables above, on qwen-math-1.5b. `w_eff ∈
> {0.1, 0.3, 3.0}` fill in the on-ramp below `w_eff=1`, same
> rationale as those tables; `lam=0.01, ds_alpha=0` closes the
> pre-existing qwen-math-1.5b `ds_alpha=0` gap under `prm=qwen`
> (the `ds_alpha sweep (v02, qwen PRM)` table above only has
> 10/100/1000 for this model).
>
> **Fixed:** tmpl=model-family default (native), bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=qwen, ds_beta=1.0,
> prm_batch_size=1, llm=qwen-math-1.5b.
>
> See
> [tuning-semantic-score-weights-and-lambda.md](decisions/tuning-semantic-score-weights-and-lambda.md)
> for the `w_eff` derivation and 5-step procedure. **Step 1** is the
> two bolded cells below (`w_eff=10`).

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-math-1.5b | qwen | 0.01 | **0** | **0** | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 1.0 | 0.1 | 0.1 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.1 | 0.0316 | 0.1 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.01 | 0.01 | 0.1 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 1.0 | 0.3 | 0.3 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.1 | 0.0949 | 0.3 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.01 | 0.03 | 0.3 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 1.0 | 1 | 1 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.1 | 0.316 | 1 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.01 | 0.1 | 1 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 1.0 | 3.0 | 3.0 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.1 | 0.949 | 3.0 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.01 | 0.3 | 3.0 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | **1.0** | **10** | **10** | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.1 | 3.16 | 10 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | **0.01** | **1.0** | **10** | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 1.0 | 100 | 100 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.1 | 31.6 | 100 | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | 0.01 | 10 | 100 | — | planned | — | — | — | — | — |

> **Analysis.** No level-5 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-4 counterpart's command plus `data.level=5`.

#### lam / ds_alpha joint sweep (qwen-7b gptq-int4)
> **Compares:** the same `lam`/`ds_alpha` joint-tuning question as
> the llama-1b/llama-3b/qwen-math-1.5b tables above, on qwen-7b
> gptq-int4. `w_eff ∈ {0.1, 0.3, 3.0}` fill in the on-ramp below
> `w_eff=1`, same rationale as those tables; `lam=0.01, ds_alpha=0`
> closes the pre-existing qwen-7b gptq-int4 `ds_alpha=0` gap under
> `prm=qwen` (the `ds_alpha sweep (v02, qwen PRM)` table above has
> no qwen-7b row at all).
>
> **Fixed:** tmpl=model-family default, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=qwen, ds_beta=1.0,
> prm_batch_size=1, llm=qwen-7b gptq-int4.
>
> See
> [tuning-semantic-score-weights-and-lambda.md](decisions/tuning-semantic-score-weights-and-lambda.md)
> for the `w_eff` derivation and 5-step procedure. **Step 1** is the
> two bolded cells below (`w_eff=10`).

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-7b gptq-int4 | qwen | 0.01 | **0** | **0** | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 1.0 | 0.1 | 0.1 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.1 | 0.0316 | 0.1 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.01 | 0.1 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 1.0 | 0.3 | 0.3 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.1 | 0.0949 | 0.3 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.03 | 0.3 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 1.0 | 1 | 1 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.1 | 0.316 | 1 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.1 | 1 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 1.0 | 3.0 | 3.0 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.1 | 0.949 | 3.0 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.3 | 3.0 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | **1.0** | **10** | **10** | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.1 | 3.16 | 10 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | **0.01** | **1.0** | **10** | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 1.0 | 100 | 100 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.1 | 31.6 | 100 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.01 | 10 | 100 | — | planned | — | — | — | — | — |

> **Analysis.** No level-5 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-4 counterpart's command plus `data.level=5`.

#### model family, size, quantization comparison (RLHFlowPRM)
> **Compares:** model family, size, and quantization jointly —
> same shape as cnt-mcts's table above, for cross-method
> comparability.
>
> **Fixed:** bs-4, d-20, b=80, tmpl=model-family default,
> method=`mcts_sem_v02` (PRM embeds), `embeds_proj=sparse512`,
> `cov_update=sherman_morrison` (sm) — the project's default path
> (path-identical to exact, proven, see decisions-log.md).
>
> ⚠️ `prm_batch_size` differs by row (1 for llama-3b/gptq/qwen-3b/
> qwen-3b-gptq-int4/qwen-7b-gptq-int4; 2 for llama-1b/
> qwen-math-1.5b — no prmbs-1+rlhflow run exists yet for those
> two), so hr/trial isn't perfectly apples-to-apples across every
> row.
>
> **W&B:** none yet (no level-5 runs).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No level-5 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-4 counterpart's command plus `data.level=5`.

#### model family, size, quantization comparison (QwenPRM)
> **Compares:** the same 7-model family/size/quantization sweep
> as the sem table above, but scored with `prm=qwen`
> (Qwen-Math-7B-PRM) instead of the default `prm=rlhflow`
> (Llama-8B-PRM). Read against that table, it isolates whether
> the model-family ranking (and the GPTQ accuracy/speed
> tradeoff) is robust to the PRM, or specific to rlhflow
> scoring.
>
> **Fixed:** method=`mcts_sem_v02` (PRM embeds), prm=qwen,
> bs-4, d-20, b=80, tmpl=model-family default (native for Qwen,
> custom for Llama), `embeds_proj=sparse512`,
> `cov_update=sherman_morrison` (sm), ds_alpha=100, ds_beta=1.0,
> prm_batch_size=1.
>
> ⚠️ All 7 cells scored as of 2026-07-07. **qwen-3b fp16**
> (`cfg-77cae091`) and **qwen-math-1.5b fp16** (`cfg-7a4be169`)
> are read from the pre-fix `--prefix-backup` copies (numbers
> below), not the in-progress precautionary regen described in
> `docs/decisions-log.md` (2026-07-07 entry) — that regen is expected
> to reproduce these exact numbers (verified no-op at existing
> hashes); re-check this row once it lands and is diffed.
>
> **W&B:** none yet (no level-5 runs).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No level-5 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-4 counterpart's command plus `data.level=5`.


#### agg_strategy comparison (qwen-3b, qwen-math-1.5b)
> **Compares:** `gen.agg_strategy` (`"min"` | `"prod"` | `"last"` —
> `core/scoring.py::aggregate_scores`) — how a candidate's
> per-step PRM scores collapse to one scalar. Scoring-side
> counterpart to the cnt-mcts table of the same name. Note:
> sem-mcts's `_generate_candidates` already strips the trailing
> `"\n\n"` before calling `prm.score` (verified — the embed and
> score paths share the same cleaned `candidate_texts`), so unlike
> cnt-mcts this table isn't tied to the `_split_steps` fix; it's a
> fresh sweep, not a rerun.
>
> **Fixed:** method=`mcts_sem_v02`, bs-4, d-20, b=80,
> tmpl=model-family default (native for both models here),
> proj=sparse512, cov=sm, ds_alpha=100.0, ds_beta=1.0.

| llm | prm | agg_strategy | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | qwen | min | — | planned | — | — | — | — | — |
| qwen-3b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-3b | qwen | last | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | min | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | last | — | planned | — | — | — | — | — |

#### agg_strategy comparison (qwen-3b, qwen-math-1.5b, lam=0.1, w_eff=10)
> **Compares:** same `gen.agg_strategy` knob as the table above, at
> `lam=0.1` instead of the default `lam=0.01` — matched `w_eff` (via
> `w_eff = ds_alpha/sqrt(lam)`) rather than matched `ds_alpha`, so
> this is a cross-check on the `agg_strategy` finding under a
> different `lam` operating point, not a new axis.
>
> **Fixed:** method=`mcts_sem_v02`, bs-4, d-20, b=80,
> tmpl=model-family default (native for both models here),
> proj=sparse512, cov=sm, lam=0.1, ds_alpha=3.16 (w_eff=10),
> ds_beta=1.0.

| llm | prm | agg_strategy | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | qwen | min | — | planned | — | — | — | — | — |
| qwen-3b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-3b | qwen | last | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | min | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | last | — | planned | — | — | — | — | — |

> **Analysis.** No level-5 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-4 counterpart's command plus `data.level=5`.

#### agg_strategy comparison (qwen-3b, qwen-math-1.5b, lam=0.1, w_eff=100)
> **Compares:** same as the `w_eff=10` table above, at the next
> `w_eff` checkpoint.
>
> **Fixed:** method=`mcts_sem_v02`, bs-4, d-20, b=80,
> tmpl=model-family default (native for both models here),
> proj=sparse512, cov=sm, lam=0.1, ds_alpha=31.6 (w_eff=100),
> ds_beta=1.0.

| llm | prm | agg_strategy | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | qwen | min | — | planned | — | — | — | — | — |
| qwen-3b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-3b | qwen | last | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | min | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | prod | — | planned | — | — | — | — | — |
| qwen-math-1.5b | qwen | last | — | planned | — | — | — | — | — |

> **Analysis.** No level-5 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-4 counterpart's command plus `data.level=5`.


### cnt-mcts-bl-v01
> knobs: template, cpuct (bs-4, d-20 fixed). method=`mcts_bl_cnt_v01`.
> No cpuct sweep yet — every row is the default 2.0. Same
> selection rule as cnt-mcts: the Summary above promotes
> whichever row scores highest on **pass@gb** across all
> knobs jointly. (`num_phases` cap exists but isn't a tuned
> knob yet — open backlog question in
> `llm-reasoning-mcts-bl-exp-todo` on whether to keep it,
> replace it, or remove it.)

#### model family, size, quantization comparison (QwenPRM)
> **Compares:** model family, size, and quantization jointly —
> same 7-model/quant grid as cnt-mcts's equivalent
> table above, so a direct bl_cnt-vs-cnt read is possible once
> both are filled. All 7 cells are new for bl_cnt_v01 with the
> qwen PRM.
>
> **Fixed:** method=`mcts_bl_cnt_v01`, prm=qwen, agg_strategy=
> `last`, cpuct=2.0, bs-4, d-20, b=80, prm_batch_size=1 (the
> new default — see `generate_mcts_bl_cnt.py`/
> `BLMCTSCntConfig` alignment fix), tmpl=model-family default
> (native for Qwen, custom for Llama).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No level-5 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-4 counterpart's command plus `data.level=5`.

### cnt-mcts-bl-v02
> knobs: template, kube_c, kube_schedule, kube_affordable (bs-4,
> d-20 fixed). method=`mcts_bl_cnt_v02`. No kube_c sweep yet —
> every row is the default (kube_c=2.0, kube_schedule=parent,
> kube_affordable=true). Same best-first frontier as
> cnt-mcts-bl-v01, but selects by fractional-KUBE density (a UCB
> confidence bonus divided by remaining cost) instead of PUCT,
> following Tran-Thanh et al. arXiv:1204.1909 sec. 3.3. See
> `docs/algorithms.md` ("BL-MCTS") and
> `docs/decisions/kube-bonus-schedule.md` /
> `docs/decisions/kube-affordability-restriction.md` for the
> algorithm and its schedule/feasibility design.

#### model family, size, quantization comparison (QwenPRM)
> **Compares:** model family, size, and quantization jointly —
> same 7-model/quant grid as cnt-mcts-bl-v01's equivalent table
> above, so a direct v01-vs-v02 (PUCT-vs-KUBE) read is possible
> once filled. All 7 cells are new for bl_cnt_v02.
>
> **Fixed:** method=`mcts_bl_cnt_v02`, prm=qwen, agg_strategy=
> `last`, kube_c=2.0, kube_schedule=parent (default — UCT-style
> local clock, matches v01's PUCT bonus so the v01-vs-v02
> comparison isolates cost normalization; see
> `docs/decisions/kube-bonus-schedule.md`), kube_affordable=true
> (default), bs-4, d-20, b=80, prm_batch_size=1, tmpl=model-family
> default (native for Qwen, custom for Llama).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No level-5 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-4 counterpart's command plus `data.level=5`.

### cnt-mcts-bl-v03
> knobs: template, depth_beta, depth_alpha (bs-4, d-20 fixed).
> method=`mcts_bl_cnt_v03`. No depth_beta/depth_alpha sweep yet —
> every row is the default (depth_beta=2.0, depth_alpha=1.0).
> Sibling of cnt-mcts-bl-v01 (PUCT) and cnt-mcts-bl-v02
> (Fractional KUBE): same best-first frontier / knapsack-style
> selection and cost mapping as both, but the leaf-selection
> bonus is a fixed depth-preference function instead of a
> confidence bound — no visit-count/exploration term, no
> bandit/regret guarantee. See `docs/algorithms.md` ("BL-MCTS")
> and `docs/decisions/depth-shaping-knapsack-bonus.md` for the
> algorithm and the sign-correction note (`f_a` is indexed on
> depth fraction, not cost fraction, so it favors shallow nodes
> as intended).

#### model family, size, quantization comparison (QwenPRM)
> **Compares:** model family, size, and quantization jointly —
> same 7-model/quant grid as cnt-mcts-bl-v01's equivalent table
> above, so a direct bl_cnt-v01-vs-v03 (and, once v02 has runs,
> a three-way PUCT/KUBE/depth-shaping) read is possible once
> filled. All 7 cells are new for bl_cnt_v03.
>
> **Fixed:** method=`mcts_bl_cnt_v03`, prm=qwen, agg_strategy=
> `last`, depth_beta=2.0, depth_alpha=1.0, kube_affordable=true
> (default), bs-4, d-20, b=80, prm_batch_size=1, tmpl=model-family
> default (native for Qwen, custom for Llama).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No level-5 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-4 counterpart's command plus `data.level=5`.

### sem-mcts-bl
> knobs: model family/size/quantization (this table); lam,
> ds_alpha, ds_alpha_schedule not yet swept — every row below
> is the same fixed point. method=`mcts_bl_sem_v01`
> (`core/mcts_bl_sem_search_v01_00_00.py`), best-first frontier
> selection with the sem family's diversity-adjusted value
> (frontier counterpart of sem-mcts v02, as cnt-mcts-bl-v01 is to
> cnt-mcts). Run from `generate_mcts_sem.py`,
> `algo=mcts_bl_sem_v01`. See `docs/algorithms.md`
> ("BL-Sem-MCTS") and `docs/decisions-log.md` (2026-07-08) for
> the algorithm and its `ds_alpha_schedule` design.

#### model family, size, quantization comparison (QwenPRM, w_eff=100)
> **Compares:** model family, size, and quantization jointly —
> same 7-model/quant grid as cnt-mcts-bl-v01's equivalent table
> above, so a direct bl_sem-vs-bl_cnt read is possible once both
> are filled. All 7 cells are new.
>
> **Fixed:** method=`mcts_bl_sem_v01`, prm=qwen (both scoring
> AND diversity embeds — `embeds_source=prm` is the schema
> default, no second pooling engine), agg_strategy=`last`, bs-4,
> d-20, b=80, prm_batch_size=1, `ds_alpha_schedule=global`
> (default — see decisions-log), `cov_update=sm`,
> `embeds_dim=512`/`embeds_proj=sparse` (defaults), tmpl=
> model-family default (native for Qwen, custom for Llama).
> **lam=0.1, ds_alpha=31.6** (`w_eff = ds_alpha/sqrt(lam) = 100`
> — see
> [decisions/tuning-semantic-score-weights-and-lambda.md](decisions/tuning-semantic-score-weights-and-lambda.md)'s
> `lam=0.1` row; `ds_beta=1.0` fixed throughout, so only the
> ratio matters).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No level-5 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-4 counterpart's command plus `data.level=5`.

#### model family, size, quantization comparison (QwenPRM, w_eff=10)
> **Compares:** same 7-model/quant grid as the `w_eff=100` table
> above, at one order of magnitude lower effective diversity
> weight — the two tables together give a first (coarse) read on
> whether the model-family ranking is sensitive to `w_eff` for
> this algorithm, ahead of a proper `w_eff` sweep.
>
> **Fixed:** identical to the `w_eff=100` table above (method=
> `mcts_bl_sem_v01`, prm=qwen, agg_strategy=`last`, bs-4, d-20,
> b=80, prm_batch_size=1, `ds_alpha_schedule=global`,
> `cov_update=sm`, `embeds_dim=512`/`embeds_proj=sparse`, tmpl=
> model-family default) except the diversity weight.
> **lam=0.1, ds_alpha=3.16** (`w_eff = ds_alpha/sqrt(lam) = 10`
> — see
> [decisions/tuning-semantic-score-weights-and-lambda.md](decisions/tuning-semantic-score-weights-and-lambda.md)'s
> `lam=0.1` row).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No level-5 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-4 counterpart's command plus `data.level=5`.

## Tuning tables [gen_budget=160, 320, …] *(future)*
> Add a new `## Tuning tables [gen_budget=N]` section, then
> `###` per algorithm and `#####` per model as above, when
> those runs start. Expected sparser (less tuning at high
> budget). The within-algorithm scaling curve (80→160→320) is
> read across the `gen_budget=N` tuning sections; the Summary
> above carries the cross-algorithm cut per budget.

### cnt-mcts

#### model family comparison (b=320, QwenPRM)
> **Compares:** the same 7-model family/size/quantization sweep
> as the `[gen_budget=80]` table above, but at
> `search.gen_budget=320` (4× the b=80 budget) with
> `prm=qwen_prm` instead of the b=80 table's default
> `llama_prm`. Two axes change at once — budget and PRM — so
> this table isn't a clean isolation of either; it answers
> "does the b=80 ranking across model family/size/quantization
> hold at a much larger search budget under qwen scoring," not
> "what does budget alone do." A matched-PRM (llama) b=320 row
> per model would be needed to separate the two effects.
>
> **Fixed:** cpuct=2.0, bs-4, d-20, b=320, prm=qwen,
> tmpl=model-family default (native for Qwen, custom for Llama).
>
> ⚠️ Entirely `planned` — no runs yet. Budget=320 is a 4×
> generation-count increase over the b=80 table; expect roughly
> 4× the per-trial wall-clock of the corresponding b=80 row
> (e.g. qwen-7b gptq-int4 was 3.21 hr/trial at b=80).
>
> **W&B:** none yet (no level-5 runs).

| llm | prm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | — | planned | — | — | — | — | — |
| llama-3b fp16 | qwen | — | planned | — | — | — | — | — |
| qwen-3b fp16 | qwen | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | qwen | — | planned | — | — | — | — | — |

> **Analysis.** No level-5 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-4 counterpart's command plus `data.level=5`.

### sem-mcts

#### model family comparison (b=320, QwenPRM, lam=0.1, w_eff=10)
> **Compares:** the same 7-model family/size/quantization sweep
> as the `[gen_budget=80]` sem-mcts (QwenPRM) table above, but
> at `search.gen_budget=320` (4× the b=80 budget) and at
> `lam=0.1, ds_alpha=3.16` (`w_eff = ds_alpha/sqrt(lam) = 10`)
> instead of that table's default point (`lam=0.01,
> ds_alpha=100`, i.e. `w_eff=1000`). Three axes move at once
> relative to that b=80 table — budget, lam, and ds_alpha — so
> this isn't a clean isolation of any one of them; paired with
> the `w_eff=100` table below (same budget, same lam, 10×
> ds_alpha) it does isolate `w_eff` at b=320.
>
> **Fixed:** method=`mcts_sem_v02` (PRM embeds), prm=qwen, bs-4,
> d-20, b=320, prm_batch_size=1, `ds_alpha_schedule=global`
> (default), `cov_update=sm`, `embeds_dim=512`/
> `embeds_proj=sparse` (defaults), tmpl=model-family default
> (native for Qwen, custom for Llama). **lam=0.1, ds_alpha=3.16**
> (`w_eff=10` — see
> [decisions/tuning-semantic-score-weights-and-lambda.md](decisions/tuning-semantic-score-weights-and-lambda.md)'s
> `lam=0.1` row, same point used by the `sem-mcts-bl` w_eff=10
> table).
>
> ⚠️ Entirely `planned` — no runs yet. Budget=320 is a 4×
> generation-count increase over the b=80 table; expect roughly
> 4× the per-trial wall-clock of the corresponding b=80/w_eff=10
> row (see the `sem-mcts-bl` w_eff=10 table's hr/trial column for
> a rough b=80 reference point at this lam/ds_alpha, though that's
> the bl_sem frontier variant, not phase-based sem-mcts).
>
> **W&B:** none yet (no level-5 runs).

| llm | prm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | — | planned | — | — | — | — | — |
| llama-3b fp16 | qwen | — | planned | — | — | — | — | — |
| qwen-3b fp16 | qwen | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | qwen | — | planned | — | — | — | — | — |

> **Analysis.** No level-5 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-4 counterpart's command plus `data.level=5`.

#### model family comparison (b=320, QwenPRM, lam=0.1, w_eff=100)
> **Compares:** identical setup to the `w_eff=10` table above,
> at `ds_alpha=31.6` instead of `3.16` (10× the diversity
> weight, same `lam=0.1`) — the b=320 counterpart of the
> `sem-mcts-bl` w_eff=100 table, and the paired point needed to
> isolate `w_eff` alone at this budget.
>
> **Fixed:** identical to the `w_eff=10` table above (method=
> `mcts_sem_v02`, prm=qwen, bs-4, d-20, b=320, prm_batch_size=1,
> `ds_alpha_schedule=global`, `cov_update=sm`,
> `embeds_dim=512`/`embeds_proj=sparse`, tmpl=model-family
> default) except the diversity weight. **lam=0.1,
> ds_alpha=31.6** (`w_eff=100`).
>
> ⚠️ Entirely `planned` — no runs yet. Same 4× wall-clock
> expectation vs. b=80 as the `w_eff=10` table above.
>
> **W&B:** none yet (no level-5 runs).

| llm | prm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | — | planned | — | — | — | — | — |
| llama-3b fp16 | qwen | — | planned | — | — | — | — | — |
| qwen-3b fp16 | qwen | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | qwen | — | planned | — | — | — | — | — |

> **Analysis.** No level-5 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-4 counterpart's command plus `data.level=5`.

---

## Run log (newest first)
> One dated block per run/comparison: hypothesis → result →
> follow-up. Append-only; newest at top.

*(empty — no level-5 runs yet)*

## Standing comparison questions
- Does sem-UCT beat cnt-UCT at matched budget? (needs
  sem-mcts runnable)
- Does the BL frontier protocol beat phase-based walks?
  (cnt-mcts-bl-v01 vs cnt-mcts @80, QwenPRM — no runs yet, see
  `#### model family, size, quantization comparison (QwenPRM)`
  under `### cnt-mcts-bl-v01`)
- Custom vs native template: consistent across algorithms?
- cpuct sensitivity: same optimum across algorithms? (no
  sweep for any algorithm yet)
- Does the cnt/sem (or BL) gap hold as gen_budget grows
  80→160→320? (the cross-budget question the Summary exists
  to answer)

## Links & connections
- Findings: [findings/exp-findings/prm-batch-size-throughput-memory.md](../findings/exp-findings/prm-batch-size-throughput-memory.md) —
  prm_bs sweep throughput/memory result + why the pass@gb
  gap isn't statistically real
- Findings index: [findings/README.md](../findings/README.md)
