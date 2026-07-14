# LLM Reasoning — MCTS Experiment Comparison — PRM800K Level 5

> **Provenance:** structure mirrored from [exp-comp-prm800k-level4.md](exp-comp-prm800k-level4.md) (the level-4 doc) on 2026-07-10; every table reset to `planned` — no level-5 runs exist yet. Launch commands are the level-4 counterparts' plus `data.level=5` (config hashes and `--level-5--` run names follow automatically). Intro/`Fixed` prose is inherited from the level-4 doc: table definitions remain valid, but any inherited claim about completeness or findings describes level-4 state — trust the (all-planned) tables here over such prose until level-5 results land. The level-5 grid also **drops two models** relative to level 4 — llama-3b gptq and qwen-3b gptq-int4 — so inherited “7-model” grid prose reads as 5 models here (llama-1b, llama-3b fp16, qwen-3b fp16, qwen-7b gptq-int4, qwen-math-1.5b).

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

## Structure and use
Two activities, two shapes:
- **Tuning *within* an algorithm, at a given budget and
  model** → a `## Tuning tables [gen_budget=N]` section per
  budget → nested `### algorithm` → `##### model` (or an `llm`
  column when several model tables share one comparison), so
  each table is just the config rows for one (budget,
  algorithm, model) cell. Different algorithms show different
  columns — no forced shared schema. `gen_budget`, algorithm,
  and model are subsection levels, not columns, since model
  size drives the GPU constraints and scaling behavior you're
  tuning around; larger budgets need less tuning, so those
  sections are sparser.
  → **Plan/run:** add a config row here; log hypothesis +
  follow-up in the Run log.
- **Comparing *across* algorithms at a fixed budget** →
  the sparse **Summary** table, one row per algorithm × model
  × budget, carrying the full metric set (pass/naive/weighted/
  maj@gb) — tuning tables carry pass@gb only. Everything else
  (depth, ncomps, timing) stays in W&B / the result dir,
  linked. The within-algorithm scaling curve (80→160→320) is
  read by scanning the `gen_budget=N` sections; the
  cross-algorithm-per-budget cut lives here.
  → **A config wins at a budget:** promote it to the Summary
  as that (algorithm, model, budget)'s best config — picked
  across **all tuning knobs jointly** (template, cpuct, …), not
  "best template" with other knobs held fixed — linking back to
  the tuning row. Don't promote a config from a different
  budget or LLM into a Summary row; that breaks the comparison.

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
> `sem-mcts-bl-v01` row uses the `w_eff=100` table; see that
> algorithm's own section for the `w_eff=10` comparison point.
> `cnt-mcts-bl-v02` (Fractional KUBE) and `cnt-mcts-bl-v03`
> (depth-shaping) — see `docs/decisions/kube-bonus-schedule.md` /
> `kube-affordability-restriction.md` and
> `docs/decisions/depth-shaping-knapsack-bonus.md` for the
> algorithms.

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

#### model family, size, quantization comparison (RLHFlowPRM)
> **Fixed:** method=`mcts_cnt_v01`, prm=rlhflow, agg_strategy=
> `last`, cpuct=2.0, bs-4, d-20, b=80, prm_batch_size=1,
> tmpl=model-family default (native for Qwen, custom for Llama).
>
> **W&B:** none yet (no level-5 runs).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

#### model family, size, quantization comparison (QwenPRM)
> **Fixed:** method=`mcts_cnt_v01`, prm=qwen, agg_strategy=
> `last`, cpuct=2.0, bs-4, d-20, b=80, prm_batch_size=1,
> tmpl=model-family default (native for Qwen, custom for Llama).
> Companion to the rlhflow-PRM table above; same 7 model/quant
> configs, different scoring PRM.
>
> **W&B:** llama-1b `05lky8bc`, llama-3b `grfdicia`, qwen-3b
> `wns54ql3`, qwen-7b gptq-int4 `hrfcyqx4`, qwen-math-1.5b
> `43zjzxmj`.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .3619<br>±.0294 | .2724<br>±.0272 | .2127<br>±.0250 | .1903<br>±.0240 | 2.98 |
| llama-3b fp16 | 2 | scored | .5522<br>±.0304 | .4291<br>±.0303 | .4104<br>±.0301 | .3619<br>±.0294 | 5.13 |
| qwen-3b fp16 | 2 | scored | .6978<br>±.0281 | .5896<br>±.0301 | .5896<br>±.0301 | .5410<br>±.0305 | 4.63 |
| qwen-7b gptq-int4 | 2 | scored | .7537<br>±.0264 | .6157<br>±.0298 | .5784<br>±.0302 | .5634<br>±.0304 | 4.19 |
| qwen-math-1.5b fp16 | 2 | scored | .7575<br>±.0262 | .6418<br>±.0293 | .6455<br>±.0293 | .6269<br>±.0296 | 3.37 |

#### agg_strategy comparison (qwen-3b, qwen-math-1.5b)
> **Compares:** `gen.agg_strategy` (`"min"` | `"prod"` | `"last"` —
> `core/scoring.py::aggregate_scores`) — how a candidate's
> per-step PRM scores collapse to one scalar. `"last"` is every
> other table's fixed default; `"min"` and `"prod"` aren't yet
> reported anywhere in this doc.
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
> source) — the two `response` rows are **blocked**, 
> shown for completeness. 
> See [embeds-scope-design.md](decisions/embeds-scope-design.md) 
> for the full explanation.
>
> **W&B:** none yet (no level-5 runs).

| llm | prm | strategy | scope | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-3b | qwen | last | full | 0.01 | 1.0 | 10 | — | planned | — | — | — | — | — |
| llama-3b | qwen | last | full | 0.01 | 10 | 100 | — | planned | — | — | — | — | — |
| llama-3b | qwen | last | full | 0.1 | 3.16 | 10 | — | planned | — | — | — | — | — |
| llama-3b | qwen | last | full | 0.1 | 31.6 | 100 | — | planned | — | — | — | — | — |
| llama-3b | qwen | avg | full | 0.01 | 1.0 | 10 | — | planned | — | — | — | — | — |
| llama-3b | qwen | avg | full | 0.01 | 10 | 100 | — | planned | — | — | — | — | — |
| llama-3b | qwen | avg | full | 0.1 | 3.16 | 10 | — | planned | — | — | — | — | — |
| llama-3b | qwen | avg | full | 0.1 | 31.6 | 100 | — | planned | — | — | — | — | — |
| llama-3b | qwen | last | response | — | — | — | — | planned | — | — | — | — | — |
| llama-3b | qwen | avg | response | — | — | — | — | planned | — | — | — | — | — |

> **Limitations / follow-up:** the two `response` rows are blocked
> on PRM-source `response_start_idx` support; queue them once the
> v02 core handles `embeds_scope=response` for `embeds_source=prm`.
> A v01 (policy-embeds) version of this table would unblock the
> `response` axis, since v01 supports it.

#### lam / ds_alpha joint sweep (llama-1b)
> **Compares:** whether `lam` and `ds_alpha` affect selection
> primarily through the effective diversity weight
> `w_eff = ds_alpha / sqrt(lam)`, or whether `lam` also has
> an independent effect on pass@gb. `lam` sets the initial
> ridge scale, `V_0 = lam * I`, and determines how quickly
> `V_inv` changes as embeddings accumulate.
>
> For each `w_eff` checkpoint, the table varies `lam` and
> `ds_alpha` jointly while keeping `w_eff` fixed. Stable
> results within a checkpoint would support `w_eff` as the
> main tuning variable. Differences would indicate a separate
> adaptation-rate effect from `lam`.
>
> The `w_eff = 0` row provides the no-diversity baseline.
> The `0.1` and `0.3` checkpoints probe the on-ramp below
> `w_eff = 1`, while `3` fills the interval between `1` and
> `10`. Together with the higher checkpoints, these values
> test whether pass@gb rises gradually from zero, changes
> sharply near `w_eff ≈ 1`, and eventually reaches the known
> plateau near `w_eff = 100`.
>
> **Fixed:** tmpl=model-family default, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=qwen, ds_beta=1.0,
> prm_batch_size=1, llm=llama-1b.

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-1b | qwen | 0.01 | 0 | 0 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 1.0 | 0.1 | 0.1 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.1 | 0.0316 | 0.1 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.01 | 0.01 | 0.1 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 1.0 | 0.3 | 0.3 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.1 | 0.0949 | 0.3 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.01 | 0.03 | 0.3 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 1.0 | 1 | 1 | 2 | scored | .3358<br>±.0289 | .2649<br>±.0270 | .2575<br>±.0268 | .2313<br>±.0258 | 4.94 |
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
| llama-1b | qwen | 1.0 | 1000 | 1000 | 2 | scored | .3396<br>±.0290 | .2388<br>±.0261 | .1828<br>±.0237 | .1381<br>±.0211 | 5.02 |
| llama-1b | qwen | 0.1 | 316.2 | 1000 | 2 | scored | .3470<br>±.0291 | .2313<br>±.0258 | .1679<br>±.0229 | .1455<br>±.0216 | 5.00 |
| llama-1b | qwen | 0.01 | 100 | 1000 | 2 | scored | .3769<br>±.0297 | .2649<br>±.0270 | .2276<br>±.0257 | .1716<br>±.0231 | 4.88 |

> **Analysis.** 15/22 cells scored (2 trials each); `lam=1.0,
> ds_alpha=0.1` (w_eff=0.1) and the `ds_alpha=0` (w_eff=0)
> gap-closer remain — the `w_eff=0.1` failure is a launch attempt
> that died before `wandb.init` on 2026-07-11 and was re-queued.
> Step 1 pair (`w_eff=10`, `lam=1.0` vs `lam=0.01`): pass@gb .3582
> vs .3209 — within SEM (±.029/±.029), consistent with `lam`
> having no strong independent effect at this level, matching the
> level-4 finding. `w_eff=1` (`lam=1.0` .3358) sits above both
> `lam=0.1`/`lam=0.01` rows at the same checkpoint (.2910/.2985),
> a wider spread than the step-1 pair shows — worth another look
> once the `w_eff=0.1` cell lands and fills out the low end. The
> new `w_eff=1000` step is now fully scored: pass@gb .3396/.3470/
> .3769 (`lam=1.0/0.1/0.01`) — `lam=0.01` again trends highest, a
> milder version of the pattern seen on llama-3b at this budget.
> **Limitations / follow-up:** n=2 trials is preliminary (wide
> SEMs); `w_eff=0` and `w_eff=0.1, lam=1.0` still pending.

#### lam / ds_alpha joint sweep (llama-3b)
> **Compares:** the same `lam`/`ds_alpha` joint-tuning question as
> the llama-1b table above, on llama-3b.
>
> **Fixed:** tmpl=model-family default, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=qwen, ds_beta=1.0,
> prm_batch_size=1, llm=llama-3b.

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
> the llama-1b/llama-3b tables above, on qwen-math-1.5b.
>
> **Fixed:** tmpl=model-family default (native), bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=qwen, ds_beta=1.0,
> prm_batch_size=1, llm=qwen-math-1.5b.

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
| qwen-math-1.5b | qwen | **1.0** | **10** | **10** | 2 | scored | .7425<br>±.0268 | .6381<br>±.0294 | .6157<br>±.0298 | .6082<br>±.0299 | 4.85 |
| qwen-math-1.5b | qwen | 0.1 | 3.16 | 10 | 2 | scored | .7015<br>±.0280 | .6119<br>±.0298 | .5970<br>±.0300 | .5821<br>±.0302 | 4.81 |
| qwen-math-1.5b | qwen | **0.01** | **1.0** | **10** | 2 | scored | .7500<br>±.0265 | .6343<br>±.0295 | .6157<br>±.0298 | .6007<br>±.0300 | 4.79 |
| qwen-math-1.5b | qwen | 1.0 | 100 | 100 | 2 | scored | .6866<br>±.0284 | .5970<br>±.0300 | .5709<br>±.0303 | .5410<br>±.0305 | 4.75 |
| qwen-math-1.5b | qwen | 0.1 | 31.6 | 100 | 2 | scored | .7164<br>±.0276 | .6082<br>±.0299 | .6007<br>±.0300 | .5933<br>±.0301 | 4.76 |
| qwen-math-1.5b | qwen | 0.01 | 10 | 100 | 2 | scored | .7164<br>±.0276 | .5896<br>±.0301 | .5746<br>±.0303 | .5597<br>±.0304 | 4.83 |
| qwen-math-1.5b | qwen | 1.0 | 1000 | 1000 | 2 | scored | .6754<br>±.0287 | .6007<br>±.0300 | .5709<br>±.0303 | .5597<br>±.0304 | 4.83 |
| qwen-math-1.5b | qwen | 0.1 | 316.2 | 1000 | 2 | scored | .7463<br>±.0266 | .6306<br>±.0295 | .5933<br>±.0301 | .5522<br>±.0304 | 4.81 |
| qwen-math-1.5b | qwen | 0.01 | 100 | 1000 | 2 | scored | .7201<br>±.0275 | .6119<br>±.0298 | .5896<br>±.0301 | .5560<br>±.0304 | 4.73 |

> **Analysis.** 9/22 cells scored (2 trials each). Step-1 pair
> (`w_eff=10`, `lam=1.0` vs `lam=0.01`): pass@gb .7425 vs .7500 —
> within SEM (±.027/±.027), no strong `lam` effect at this
> checkpoint. `lam=0.1` sits between them at .7015, noticeably
> lower than both — the widest spread in this table so far,
> though still within ~1 SEM of either endpoint. `w_eff=100`
> shows a similar pattern (`lam=1.0` .6866 vs `lam=0.01`/`lam=0.1`
> both ~.716) — `lam=1.0` trending lowest at both checkpoints. The
> new `w_eff=1000` step continues this: `lam=1.0` .6754 is the
> lowest of the three again, vs. `lam=0.1` .7463 (the highest) and
> `lam=0.01` .7201 — `lam=1.0` now trending lowest at all three
> checkpoints tested (`w_eff=10/100/1000`).
> **Limitations / follow-up:** 13/22 cells still unrun — the
> `w_eff=1/3` blocks, the `w_eff=0.1/0.3` on-ramp, and the
> `w_eff=0` gap-closer; only 2 trials/cell so the `lam=1.0`
> low-trend above is suggestive, not conclusive.

#### lam / ds_alpha joint sweep (qwen-7b gptq-int4)
> **Compares:** the same `lam`/`ds_alpha` joint-tuning question as
> the llama-1b/llama-3b/qwen-math-1.5b tables above, on qwen-7b
> gptq-int4.
>
> **Fixed:** tmpl=model-family default, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, prm=qwen, ds_beta=1.0,
> prm_batch_size=1, llm=qwen-7b gptq-int4.

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
| qwen-7b gptq-int4 | qwen | **1.0** | **10** | **10** | 2 | scored | .7575<br>±.0262 | .6007<br>±.0300 | .5858<br>±.0301 | .5672<br>±.0303 | 5.10 |
| qwen-7b gptq-int4 | qwen | 0.1 | 3.16 | 10 | 2 | scored | .7761<br>±.0255 | .6231<br>±.0297 | .5933<br>±.0301 | .5858<br>±.0301 | 5.48 |
| qwen-7b gptq-int4 | qwen | **0.01** | **1.0** | **10** | 2 | scored | .7687<br>±.0258 | .6231<br>±.0297 | .6194<br>±.0297 | .6119<br>±.0298 | 5.42 |
| qwen-7b gptq-int4 | qwen | 1.0 | 100 | 100 | 2 | scored | .7761<br>±.0255 | .6157<br>±.0298 | .5299<br>±.0305 | .5299<br>±.0305 | 5.41 |
| qwen-7b gptq-int4 | qwen | 0.1 | 31.6 | 100 | 2 | scored | .7799<br>±.0254 | .6119<br>±.0298 | .5560<br>±.0304 | .5336<br>±.0305 | 5.43 |
| qwen-7b gptq-int4 | qwen | 0.01 | 10 | 100 | 2 | scored | .7873<br>±.0250 | .6045<br>±.0299 | .5634<br>±.0304 | .5634<br>±.0304 | 5.54 |
| qwen-7b gptq-int4 | qwen | 1.0 | 1000 | 1000 | 2 | scored | .7500<br>±.0265 | .5672<br>±.0303 | .5448<br>±.0305 | .5373<br>±.0305 | 5.43 |
| qwen-7b gptq-int4 | qwen | 0.1 | 316.2 | 1000 | 2 | scored | .7799<br>±.0254 | .6045<br>±.0299 | .5522<br>±.0304 | .5336<br>±.0305 | 5.46 |
| qwen-7b gptq-int4 | qwen | 0.01 | 100 | 1000 | 2 | scored | .7649<br>±.0260 | .6082<br>±.0299 | .5634<br>±.0304 | .5224<br>±.0306 | 5.40 |

> **Analysis.** 9/22 cells scored (2 trials each). Step-1 pair
> (`w_eff=10`, `lam=1.0` vs `lam=0.01`): pass@gb .7575 vs .7687 —
> within SEM (±.026/±.026), no strong `lam` effect. All 6
> `w_eff=10/100` cells cluster tightly on pass@gb (.7575–.7873)
> regardless of `lam` or `w_eff`, the flattest spread of any
> model-family table in this sweep so far. `w_eff=1000` breaks
> that flatness slightly: `lam=1.0` .7500 is the lowest of the
> three (still within ~1 SEM of the others), `lam=0.1` .7799 the
> highest — a small, `lam=1.0`-trending-lowest pattern echoing
> qwen-math-1.5b's table, though far milder here.
> **Limitations / follow-up:** 13/22 cells still unrun — the
> `w_eff=1/3` blocks, the `w_eff=0.1/0.3` on-ramp, and the
> `w_eff=0` gap-closer; only 2 trials/cell.

#### model family, size, quantization comparison (RLHFlowPRM)
> **Compares:** model family, size, and quantization jointly —
> same shape as cnt-mcts's table above, for cross-method
> comparability.
>
> **Fixed:** bs-4, d-20, b=80, tmpl=model-family default,
> method=`mcts_sem_v02` (PRM embeds), `embeds_proj=sparse512`,
> `cov_update=sherman_morrison` (sm).
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
> **Compares:** the same 5-model family/size/quantization sweep
> as the RLHFlowPRM table above, but scored with `prm=qwen`
> (Qwen-Math-7B-PRM) instead of the default `prm=rlhflow`
> (Llama-8B-PRM).
>
> **Fixed:** method=`mcts_sem_v02` (PRM embeds), prm=qwen,
> bs-4, d-20, b=80, tmpl=model-family default (native for Qwen,
> custom for Llama), `embeds_proj=sparse512`,
> `cov_update=sherman_morrison` (sm), ds_alpha=100, ds_beta=1.0,
> prm_batch_size=1.
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
> counterpart to the cnt-mcts table of the same name.
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

#### model family, size, quantization comparison (QwenPRM)
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as cnt-mcts's equivalent
> table above, so a direct bl_cnt-vs-cnt read is possible once
> both are filled.
>
> **Fixed:** method=`mcts_bl_cnt_v01`, prm=qwen, agg_strategy=
> `last`, cpuct=2.0, bs-4, d-20, b=80, prm_batch_size=1,
> tmpl=model-family default (native for Qwen, custom for Llama).

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

#### model family, size, quantization comparison (QwenPRM)
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as cnt-mcts-bl-v01's equivalent table
> above, so a direct v01-vs-v02 (PUCT-vs-KUBE) read is possible
> once filled.
>
> **Fixed:** method=`mcts_bl_cnt_v02`, prm=qwen, agg_strategy=
> `last`, kube_c=2.0, kube_schedule=parent, kube_affordable=true,
> bs-4, d-20, b=80, prm_batch_size=1, tmpl=model-family default
> (native for Qwen, custom for Llama). See
> `docs/decisions/kube-bonus-schedule.md` for the schedule choice.

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

#### model family, size, quantization comparison (QwenPRM)
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as cnt-mcts-bl-v01's equivalent table
> above, so a direct bl_cnt-v01-vs-v03 (and, once v02 has runs,
> a three-way PUCT/KUBE/depth-shaping) read is possible once
> filled.
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

#### model family, size, quantization comparison (QwenPRM, w_eff=100)
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as cnt-mcts-bl-v01's equivalent table
> above, so a direct bl_sem-vs-bl_cnt read is possible once both
> are filled.
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
> **Compares:** same 5-model/quant grid as the `w_eff=100` table
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
> **Compares:** the same 5-model family/size/quantization sweep
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
> **Compares:** the same 5-model family/size/quantization sweep
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
