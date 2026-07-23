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
> `cnt-mcts-bl-v01`, `kube-mcts-bl-v01`, `kdepth-mcts-bl-v01`,
> `sem-mcts-bl-v01`). All rows fixed at b=80, bs-4, d-20,
> agg_strategy=`last`, tmpl=model-family default (native for Qwen,
> custom for Llama), prm=qwen. `cnt-mcts` row is method=`mcts_cnt_v01`
> (the only cnt-mcts entry point at this level — see the
> `### cnt-mcts` section above). `sem-mcts` row is `mcts_sem_v02` (PRM embeds),
> `ds_alpha=100` (w_eff not applicable — that knob is bl_sem-specific).
> `sem-mcts-bl-v01` row uses the `w_eff=100` table; see that
> algorithm's own section for the `w_eff=10` comparison point.
> `kube-mcts-bl-v01` (Fractional KUBE) and `kdepth-mcts-bl-v01`
> (depth-shaping) — see `docs/decisions/bl-kube-bonus-schedule.md` /
> `kube-affordability-restriction.md` and
> `docs/decisions/bl-kdepth-knapsack-bonus.md` for the
> algorithms.

> Each cell is the **best available** result for that (algorithm,
> model) pair by pass@gb (tie → wei@gb). Non-bl rows (`cnt-mcts`,
> `sem-mcts`) pull from their own model-family QwenPRM tables below;
> `sem-mcts` takes the better of its ds_alpha=1/10 arms. The four
> `*-bl-v01` rows pull the best config from that family's tuning
> tables (kube: parent_blend arms; kdepth: the depth_alpha sweep;
> sem-bl: ds_alpha=1/10). bl-v01 SEMs are from direct
> compute_stats scoring (2 trials); hr/trial not captured for those.

**llama-1b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | 2 | scored | .3619<br>±.0294 | .2724 | .2127 | .1903 | 2.98 |
| sem-mcts | 2 | scored | .3433<br>±.0291 | .2537 | .1978 | .1679 | 4.85 |
| cnt-mcts-bl-v01 | 2 | scored | .2313<br>±.0258 | .2090 | .1940 | .1940 | 2.74 |
| kube-mcts-bl-v01 | 2 | scored | .3060<br>±.0282 | .2612 | .2463 | .2276 | 3.11 |
| kdepth-mcts-bl-v01 | — | running | — | — | — | — | — |
| sem-mcts-bl-v01 | — | running | — | — | — | — | — |

**llama-3b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | 2 | scored | .5522<br>±.0304 | .4291 | .4104 | .3619 | 5.13 |
| sem-mcts | 2 | scored | .5784<br>±.0302 | .4403 | .4291 | .3881 | 6.93 |
| cnt-mcts-bl-v01 | 2 | scored | .3731<br>±.0296 | .3209 | .3321 | .3209 | 4.76 |
| kube-mcts-bl-v01 | 2 | scored | .4851<br>±.0306 | .3918 | .3769 | .3731 | — |
| kdepth-mcts-bl-v01 | 2 | scored | .5000<br>±.0306 | .4104 | .4030 | .3955 | — |
| sem-mcts-bl-v01 | — | running | — | — | — | — | — |

**qwen-3b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | 2 | scored | .6978<br>±.0281 | .5896 | .5896 | .5410 | 4.63 |
| sem-mcts | 2 | scored | .6903<br>±.0283 | .5784 | .5597 | .5373 | 6.20 |
| cnt-mcts-bl-v01 | — | running | — | — | — | — | — |
| kube-mcts-bl-v01 | 2 | scored | .6157<br>±.0298 | .5410 | .5224 | .5075 | 4.10 |
| kdepth-mcts-bl-v01 | — | running | — | — | — | — | — |
| sem-mcts-bl-v01 | — | running | — | — | — | — | — |

**qwen-7b gptq-int4**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | 2 | scored | .7537<br>±.0264 | .6157 | .5784 | .5634 | 4.19 |
| sem-mcts | 2 | scored | .7873<br>±.0250 | .6045 | .5634 | .5634 | 5.54 |
| cnt-mcts-bl-v01 | 2 | scored | .6343<br>±.0295 | .5709 | .5672 | .5522 | 3.97 |
| kube-mcts-bl-v01 | 2 | scored | .7164<br>±.0276 | .6157 | .5858 | .5746 | — |
| kdepth-mcts-bl-v01 | — | running | — | — | — | — | — |
| sem-mcts-bl-v01 | 2 | scored | .7537<br>±.0264 | .5597 | .5037 | .4478 | — |

**qwen-math-1.5b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| cnt-mcts | 2 | scored | .7575<br>±.0262 | .6418 | .6455 | .6269 | 3.37 |
| sem-mcts | 2 | scored | .7500<br>±.0265 | .6343 | .6157 | .6007 | 4.79 |
| cnt-mcts-bl-v01 | 2 | scored | .4366<br>±.0304 | .4142 | .4104 | .3955 | 3.31 |
| kube-mcts-bl-v01 | 2 | scored | .6493<br>±.0292 | .5784 | .5672 | .5522 | — |
| kdepth-mcts-bl-v01 | 2 | scored | .6455<br>±.0293 | .5522 | .5485 | .5336 | — |
| sem-mcts-bl-v01 | 2 | scored | .6567<br>±.0291 | .5410 | .4627 | .4552 | — |

> **Analysis.** Best-available snapshot, 24/30 cells scored
> (all 5 model blocks visible as of 2026-07-22; the llama-1b and
> qwen-3b blocks were restored after their kube-bl cells landed).
> Ordering is model-driven: qwen-math-1.5b and qwen-7b-gptq top
> every algorithm (pass@gb .65–.79), qwen-3b sits mid (.62–.70),
> llama-1b trails (.23–.36). Non-bl cnt/sem-mcts lead their bl
> counterparts on every model with data (qwen-7b: sem-mcts .7873
> vs sem-bl-v01 .7537, cnt-mcts .7537 vs cnt-bl-v01 .6343;
> qwen-3b: cnt-mcts .6978 vs kube-bl .6157). Within the bl
> families no single variant dominates: sem-bl-v01 tops qwen-7b
> (.7537) and qwen-math (.6567), kdepth tops llama-3b (.5000),
> kube tops the two models where the others are unfilled
> (llama-1b .3060, qwen-3b .6157) — and the kube/kdepth/sem-bl
> spreads sit within ~1 SEM of each other per model, so treat
> the bl-internal ranking as provisional. The robust reads:
> cnt-bl-v01 is uniformly weakest (its qwen-math collapse .4366
> is the standout anomaly), and every bl variant trails plain
> cnt/sem-mcts on every model.
> **Limitations / follow-up:** 6 cells still running — llama-1b
> kdepth/sem-bl, qwen-3b cnt-bl/kdepth/sem-bl, qwen-7b kdepth
> (runs done on disk, awaiting compute_stats + verification).
> hr/trial missing for most bl cells (compute_stats path doesn't
> capture it; backfill from timing_state.json pending).

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

#### embeds_center_mode comparison (lam=0.01/ds_alpha=1)
> **Compares:** `embeds_center_mode="local"` (rep_exp-style
> sibling-group centering) against `embeds_center=false` (no
> centering — today's default). `"fixed"` mode isn't in this table
> yet — no precomputed held-out mean exists at this level. See
> [rep-exp-elliptical-bonus-review.md](decisions/rep-exp-elliptical-bonus-review.md)
> follow-up #3 and
> [embeds-centering-design.md](decisions/embeds-centering-design.md)
> for the full discussion.
>
> **Fixed:** method=`mcts_sem_v02` (PRM embeds), prm=qwen, bs-4,
> d-20, b=80, proj=sparse512, cov_update=sm, cov_dtype=fp64 (default),
> ds_beta=1.0, prm_batch_size=1, tmpl=model-family default (native for
> Qwen, custom for Llama), **lam=0.01, ds_alpha=1.0** (`w_eff =
> ds_alpha/sqrt(lam) = 10`) — the same checkpoint recommended as the
> cross-model default in the `lam`/`ds_alpha` joint-sweep tables
> above.
>
> **W&B:** baselines cited from each model's own `lam`/`ds_alpha`
> joint-sweep table above — llama-1b `tdyxh9sr`, llama-3b `tc6d70jy`,
> qwen-7b gptq-int4 `3l1vzy8m`, qwen-math-1.5b `bb6rpjps`. `local`
> rows: llama-1b `dcgo3trx`, llama-3b `x0oosb8l`, qwen-3b
> `lrd7oa38`, qwen-7b gptq-int4 `oks5m0gi`, qwen-math-1.5b
> `x8eokhhd`. qwen-3b `none`: `8ssy5kpj`.

| llm | prm | center | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| llama-1b | qwen | none | 2 | scored | .3209<br>±.0286 | .2425<br>±.0262 | .2313<br>±.0258 | .2015<br>±.0245 | 4.88 |
| llama-1b | qwen | local | 2 | scored | .3806<br>±.0297 | .2724<br>±.0272 | .2575<br>±.0268 | .2201<br>±.0254 | 4.86 |
| llama-3b | qwen | none | 2 | scored | .5784<br>±.0302 | .4403<br>±.0304 | .4291<br>±.0303 | .3881<br>±.0298 | 6.93 |
| llama-3b | qwen | local | 2 | scored | .5746<br>±.0303 | .4104<br>±.0301 | .3993<br>±.0300 | .3694<br>±.0295 | 7.02 |
| qwen-3b | qwen | none | 2 | scored | .6903<br>±.0283 | .5784<br>±.0302 | .5597<br>±.0304 | .5373<br>±.0305 | 6.20 |
| qwen-3b | qwen | local | 2 | scored | .6978<br>±.0281 | .5634<br>±.0304 | .5336<br>±.0305 | .5112<br>±.0306 | 6.33 |
| qwen-7b gptq-int4 | qwen | none | 2 | scored | .7687<br>±.0258 | .6231<br>±.0297 | .6194<br>±.0297 | .6119<br>±.0298 | 5.42 |
| qwen-7b gptq-int4 | qwen | local | 2 | scored | .7724<br>±.0257 | .6231<br>±.0297 | .5821<br>±.0302 | .5709<br>±.0303 | 5.57 |
| qwen-math-1.5b | qwen | none | 2 | scored | .7500<br>±.0265 | .6343<br>±.0295 | .6157<br>±.0298 | .6007<br>±.0300 | 4.79 |
| qwen-math-1.5b | qwen | local | 2 | scored | .7500<br>±.0265 | .6418<br>±.0293 | .6082<br>±.0299 | .6045<br>±.0299 | 4.85 |

> **Analysis.** 10/10 cells scored (2 trials each). `local` vs
> `none` splits both directions: llama-1b (.3209→.3806) and
> qwen-3b (.6903→.6978) trend higher under `local`; llama-3b
> (.5784→.5746) and qwen-7b gptq-int4 (.7687→.7724, essentially
> flat) show little change; qwen-math-1.5b is identical on
> pass@gb (.7500 both). Every gap is within ~1 SEM — no
> consistent centering-mode effect at this trial count.
> **Limitations / follow-up:** n=2 trials/cell is preliminary. A
> `"fixed"`-mode column is a natural follow-up once a held-out
> mean is computed for at least one model (see
> [embeds-centering-design.md](decisions/embeds-centering-design.md)
> for how the fixed-mean file is built and loaded).

#### embeds_center_mode comparison (lam=0.01/ds_alpha=10)
> **Compares:** same as the `ds_alpha=1` table above, at the next
> `w_eff` checkpoint (`w_eff = ds_alpha/sqrt(lam) = 100`).
>
> **Fixed:** identical to the `ds_alpha=1` table above (method=
> `mcts_sem_v02`, prm=qwen, bs-4, d-20, b=80, proj=sparse512,
> cov_update=sm, cov_dtype=fp64, ds_beta=1.0, prm_batch_size=1,
> tmpl=model-family default) except **ds_alpha=10** (`w_eff=100`).
>
> **W&B:** baselines cited from each model's own `lam`/`ds_alpha`
> joint-sweep table above — llama-1b `2sd0cen5`, llama-3b `q7yxcuq7`,
> qwen-7b gptq-int4 `es99bc0h`, qwen-math-1.5b `scmsaxeq`. `local`
> rows: llama-1b `1mtu94qz`, llama-3b `mi4pzhba`, qwen-3b
> `22z34pwz`, qwen-7b gptq-int4 `250t5r5m`, qwen-math-1.5b
> `cu0ntvth`. qwen-3b `none`: `c7a9bmxt`.
>
> ⚠️ The `none` baselines cited for llama-1b (`2sd0cen5`),
> qwen-7b gptq-int4 (`es99bc0h`), and qwen-math-1.5b (`scmsaxeq`)
> were independently recomputed this pass via the runs cited in
> the `lam`/`ds_alpha` joint-sweep tables above (`f2r2nsv7`,
> `qhyxutx6`, `ryzf69dm` — same config, same trial count) and
> disagreed with the doc's existing numbers beyond rounding; see
> the mismatch note under those tables. The `none` values shown
> here are left as the doc's pre-existing (possibly stale)
> numbers — do not treat this row's `none` baseline as verified
> until that mismatch is resolved.

| llm | prm | center | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| llama-1b | qwen | none | 2 | scored | .3433<br>±.0291 | .2537<br>±.0266 | .1978<br>±.0244 | .1679<br>±.0229 | 4.85 |
| llama-1b | qwen | local | 2 | scored | .3246<br>±.0287 | .2276<br>±.0257 | .2313<br>±.0258 | .1866<br>±.0238 | 4.90 |
| llama-3b | qwen | none | 2 | scored | .5485<br>±.0305 | .4328<br>±.0303 | .3619<br>±.0294 | .3321<br>±.0288 | 6.99 |
| llama-3b | qwen | local | 2 | scored | .5896<br>±.0301 | .4366<br>±.0304 | .4179<br>±.0302 | .3955<br>±.0299 | 6.85 |
| qwen-3b | qwen | none | 2 | scored | .6642<br>±.0289 | .5634<br>±.0304 | .5261<br>±.0306 | .5000<br>±.0306 | 6.07 |
| qwen-3b | qwen | local | 2 | scored | .6903<br>±.0283 | .5560<br>±.0304 | .5261<br>±.0306 | .4963<br>±.0306 | 6.20 |
| qwen-7b gptq-int4 | qwen | none | 2 | scored | .7873<br>±.0250 | .6045<br>±.0299 | .5634<br>±.0304 | .5634<br>±.0304 | 5.54 |
| qwen-7b gptq-int4 | qwen | local | 2 | scored | .7836<br>±.0252 | .5933<br>±.0301 | .5784<br>±.0302 | .5560<br>±.0304 | 5.63 |
| qwen-math-1.5b | qwen | none | 2 | scored | .7164<br>±.0276 | .5896<br>±.0301 | .5746<br>±.0303 | .5597<br>±.0304 | 4.83 |
| qwen-math-1.5b | qwen | local | 2 | scored | .7463<br>±.0266 | .6194<br>±.0297 | .6045<br>±.0299 | .5746<br>±.0303 | 4.81 |

> **Analysis.** 10/10 cells scored (2 trials each), but three of
> the five `none` baselines shown are flagged stale (see the
> ⚠️ above) — treat any `local` vs `none` comparison here as
> provisional until that's resolved. Of the two unaffected
> baselines (llama-3b, qwen-3b), both trend higher under `local`
> (.5485→.5896, .6642→.6903), within ~1 SEM.
> **Limitations / follow-up:** resolve the `none`-baseline
> mismatch (dead/stale W&B citation vs. hand-entered value vs.
> genuine re-run) before drawing conclusions from this table;
> n=2 trials/cell throughout.

#### agg_strategy comparison (qwen-3b, qwen-math-1.5b, lam=0.01/ds_alpha=1)
> **Compares:** `gen.agg_strategy` (`"min"` | `"prod"` | `"last"` —
> `core/scoring.py::aggregate_scores`) — how a candidate's per-step
> PRM scores collapse to one scalar — at `lam=0.01, ds_alpha=1.0`
> (`w_eff = ds_alpha/sqrt(lam) = 10`), the same checkpoint used in
> the `embeds_center_mode` tables above.
>
> **Fixed:** method=`mcts_sem_v02`, bs-4, d-20, b=80,
> tmpl=model-family default (native for both models here),
> proj=sparse512, cov=sm, lam=0.01, ds_alpha=1.0 (w_eff=10),
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

#### agg_strategy comparison (qwen-3b, qwen-math-1.5b, lam=0.01/ds_alpha=10)
> **Compares:** same as the `ds_alpha=1.0` table above, at the next
> `w_eff` checkpoint (`w_eff = ds_alpha/sqrt(lam) = 100`).
>
> **Fixed:** method=`mcts_sem_v02`, bs-4, d-20, b=80,
> tmpl=model-family default (native for both models here),
> proj=sparse512, cov=sm, lam=0.01, ds_alpha=10 (w_eff=100),
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

#### model family, size, quantization comparison (QwenPRM, lam=0.01/ds_alpha=1)
> **Compares:** model family, size, and quantization jointly,
> scored with `prm=qwen` (Qwen-Math-7B-PRM), at `lam=0.01,
> ds_alpha=1.0` (`w_eff = ds_alpha/sqrt(lam) = 10`) — the same
> checkpoint used in the `embeds_center_mode` and `agg_strategy`
> tables above.
>
> **Fixed:** method=`mcts_sem_v02` (PRM embeds), prm=qwen,
> bs-4, d-20, b=80, tmpl=model-family default (native for Qwen,
> custom for Llama), `embeds_proj=sparse512`,
> `cov_update=sherman_morrison` (sm), lam=0.01, ds_alpha=1.0
> (w_eff=10), ds_beta=1.0, prm_batch_size=1.
>
> **W&B:** baselines cited from each model's own `lam`/`ds_alpha`
> joint-sweep table above — llama-1b `tdyxh9sr`, llama-3b `tc6d70jy`,
> qwen-7b gptq-int4 `3l1vzy8m`, qwen-math-1.5b `bb6rpjps`. qwen-3b:
> `8ssy5kpj`.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .3209<br>±.0286 | .2425<br>±.0262 | .2313<br>±.0258 | .2015<br>±.0245 | 4.88 |
| llama-3b fp16 | 2 | scored | .5784<br>±.0302 | .4403<br>±.0304 | .4291<br>±.0303 | .3881<br>±.0298 | 6.93 |
| qwen-3b fp16 | 2 | scored | .6903<br>±.0283 | .5784<br>±.0302 | .5597<br>±.0304 | .5373<br>±.0305 | 6.20 |
| qwen-7b gptq-int4 | 2 | scored | .7687<br>±.0258 | .6231<br>±.0297 | .6194<br>±.0297 | .6119<br>±.0298 | 5.42 |
| qwen-math-1.5b fp16 | 2 | scored | .7500<br>±.0265 | .6343<br>±.0295 | .6157<br>±.0298 | .6007<br>±.0300 | 4.79 |

> **Analysis.** 5/5 cells filled (qwen-3b newly computed this
> pass; the other 4 cited from existing scored data — no new
> compute). Ranking by pass@gb: qwen-7b gptq-int4 (.7687) ≈
> qwen-math-1.5b (.7500) > qwen-3b (.6903) > llama-3b (.5784) >
> llama-1b (.3209).
> **Limitations / follow-up:** none — table complete at this
> checkpoint.

#### model family, size, quantization comparison (QwenPRM, lam=0.01/ds_alpha=10)
> **Compares:** same as the `ds_alpha=1` table above, at the next
> `w_eff` checkpoint (`w_eff = ds_alpha/sqrt(lam) = 100`).
>
> **Fixed:** identical to the `ds_alpha=1` table above except
> **ds_alpha=10** (w_eff=100).
>
> **W&B:** baselines cited from each model's own `lam`/`ds_alpha`
> joint-sweep table above — llama-1b `2sd0cen5`, llama-3b `q7yxcuq7`,
> qwen-7b gptq-int4 `es99bc0h`, qwen-math-1.5b `scmsaxeq`. qwen-3b:
> `c7a9bmxt`.
>
> ⚠️ Same mismatch flag as the `embeds_center_mode (ds_alpha=10)`
> table above: independently recomputing the cited llama-1b,
> qwen-7b gptq-int4, and qwen-math-1.5b baselines this pass
> produced different numbers than shown below (same config, same
> trial count). Not overwritten — see that table's note.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .3433<br>±.0291 | .2537<br>±.0266 | .1978<br>±.0244 | .1679<br>±.0229 | 4.85 |
| llama-3b fp16 | 2 | scored | .5485<br>±.0305 | .4328<br>±.0303 | .3619<br>±.0294 | .3321<br>±.0288 | 6.99 |
| qwen-3b fp16 | 2 | scored | .6642<br>±.0289 | .5634<br>±.0304 | .5261<br>±.0306 | .5000<br>±.0306 | 6.07 |
| qwen-7b gptq-int4 | 2 | scored | .7873<br>±.0250 | .6045<br>±.0299 | .5634<br>±.0304 | .5634<br>±.0304 | 5.54 |
| qwen-math-1.5b fp16 | 2 | scored | .7164<br>±.0276 | .5896<br>±.0301 | .5746<br>±.0303 | .5597<br>±.0304 | 4.83 |

> **Analysis.** 5/5 cells filled (qwen-3b newly computed this
> pass; the other 4 cited from existing scored data — 3 of which
> are flagged stale, see ⚠️ above).
> **Limitations / follow-up:** resolve the flagged mismatch
> before treating llama-1b/qwen-7b gptq-int4/qwen-math-1.5b rows
> as verified.

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
| llama-1b fp16 | 2 | scored | .2313<br>±.0258 | .2090<br>±.0249 | .1940<br>±.0242 | .1940<br>±.0242 | 2.74 |
| llama-3b fp16 | 2 | scored | .3731<br>±.0296 | .3209<br>±.0286 | .3321<br>±.0288 | .3209<br>±.0286 | 4.76 |
| qwen-3b fp16 | 2 | scored | .3657<br>±.0295 | .3545<br>±.0293 | .3582<br>±.0293 | .3582<br>±.0293 | 4.26 |
| qwen-7b gptq-int4 | 2 | scored | .6343<br>±.0295 | .5709<br>±.0303 | .5672<br>±.0303 | .5522<br>±.0304 | 3.97 |
| qwen-math-1.5b fp16 | 2 | scored | .4366<br>±.0304 | .4142<br>±.0301 | .4104<br>±.0301 | .3955<br>±.0299 | 3.31 |

> **Analysis.** 4/5 cells scored (2026-07-21); qwen-3b mid-run.
> bl-v01 trails cnt-mcts on every model — qwen-7b gptq .6343 vs
> .7873 pass@gb (−.15); qwen-math-1.5b .4366 vs .7164 (−.28, the
> largest gap). The math specialist's advantage largely vanishes
> under the uniform baseline: qwen-math-1.5b (.4366) barely
> clears llama-3b (.3731), while it is a top model under
> cnt-mcts. Remaining ordering is size/family-consistent:
> llama-1b < llama-3b < qwen-math-1.5b < qwen-7b gptq.
> **Limitations / follow-up:** qwen-3b pending (1/2 trials,
> ~4.26 hr/trial). Queue-only block — no experiments.yaml
> entries to flip (per 2026-07-20 note in queue.yaml).

### cnt-mcts-bl-v02

#### score_mode sweep: parent_blend (alpha) vs. path_decay (gamma × cpuct) (qwen-3b, QwenPRM)
> **Compares:** the two selectable v02 frontier scores head-to-head
> on one model. parent_blend arms sweep `alpha` (one-hop blend of a
> leaf's q with its parent's) at the file-default cpuct=2.0;
> path_decay arms sweep the full `gamma × cpuct` cross. Why cpuct
> is crossed with gamma rather than held fixed: path_decay's
> exploration term uses the AlphaZero shape
> `cpuct·sqrt(N_parent)/(1+N_leaf)` — no log damping, so at the
> default cpuct=2.0 it can swamp the `q_path` value term (range
> ~[0,1]) after a few backprops regardless of gamma. A flat gamma
> effect at cpuct=2.0 alone would be uninterpretable (gamma
> useless, or drowned out?). The cross separates the two stories:
> gamma mattering at cpuct=0.5 but not 2.0 confirms
> scale-domination; gamma inert at both scales is real evidence
> against full-path value reading. gamma semantics: 1.0 = plain
> path average, 0.8 = moderate decay, 0.5 = steep/near-local.
>
> **Fixed:** method=`mcts_bl_cnt_v02`, llm=qwen-3b fp16 (native
> tmpl), prm=qwen, agg_strategy=`last`, bs-4, d-20, b=80,
> prm_batch_size=1, level=5.
>
> ⚠️ Entirely planned, no runs yet. cpuct is NOT comparable across
> the two modes (different exploration-term shapes) — compare
> cpuct values within a mode only. The alpha=1.0 row is the exact
> v01 control arm (recovers v01's puct identically); path_decay
> has no v01-equivalent arm.
>
> **W&B:** pb-a1.0 `uu7p59lq`, pd-g1.0-c0.5 `gx1u385h`; others
> none yet.

| score_mode | alpha | gamma | cpuct | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|
| parent_blend | 1.0 | — | 2.0 | 2 | scored | .4403<br>±.0304 | .4142<br>±.0301 | .4067<br>±.0301 | .3955<br>±.0299 | 4.28 |
| parent_blend | 0.8 | — | 2.0 | 2 | scored | .4403<br>±.0304 | .3993<br>±.0300 | .3993<br>±.0300 | .3843<br>±.0298 | 4.41 |
| parent_blend | 0.6 | — | 2.0 | 2 | scored | .4701<br>±.0305 | .4478<br>±.0304 | .4366<br>±.0304 | .4291<br>±.0303 | 4.39 |
| path_decay | — | 1.0 | 2.0 | 2 | scored | .6493<br>±.0292 | .5522<br>±.0304 | .5522<br>±.0304 | .5410<br>±.0305 | 4.41 |
| path_decay | — | 0.8 | 2.0 | 2 | scored | .6418<br>±.0293 | .5746<br>±.0303 | .5634<br>±.0304 | .5485<br>±.0305 | 4.39 |
| path_decay | — | 0.5 | 2.0 | — | running | — | — | — | — | — |
| path_decay | — | 1.0 | 0.5 | 2 | scored | .6194<br>±.0297 | .5821<br>±.0302 | .5746<br>±.0303 | .5560<br>±.0304 | 4.31 |
| path_decay | — | 0.8 | 0.5 | 2 | scored | .6082<br>±.0299 | .5821<br>±.0302 | .5709<br>±.0303 | .5560<br>±.0304 | 4.52 |
| path_decay | — | 0.5 | 0.5 | 2 | scored | .5746<br>±.0303 | .5373<br>±.0305 | .5261<br>±.0306 | .5261<br>±.0306 | 4.54 |

> **Analysis.** 8/9 arms scored (2026-07-22). The three reads:
> (1) one-hop blending does NOT help — pb arms sit at
> .4403/.4403/.4701 (a0.6 nominally best, within 1 SEM);
> (2) gamma is NOT scale-dominated: path_decay wins at BOTH
> cpuct values, and cpuct=2.0 is nominally better (g1.0: .6493
> vs .6194) — so the pd-vs-pb gap is attributable to the PATH
> VALUE itself, not the exploration scale; (3) best pd arm
> (g1.0-c2.0 .6493) beats best pb arm (a0.6 .4701) by +.18
> (~6 SEM) at equal cost — **path_decay is the survivor** for
> cnt-v02; gamma ordering is monotone (1.0 > 0.8 > 0.5 at both
> scales), i.e. the plain full-path average is best and decay
> only hurts. pd searches deeper (depth ~11 vs ~9) and keeps
> ~1.7x more completions on the same 80-gen budget.
> **Limitations / follow-up:** pd-g0.5-c2.0 still running
> (3rd launch, gpu_standard). Ledger:
> experiments/prm800k-level5.yaml, feeds
> `level5-cnt-bl-v02-score-mode-qwen3b`. Single model (qwen-3b);
> extend path_decay g1.0 to the 5-model grid if the v03
> decision confirms.

### kube-mcts-bl-v01

#### model family, size, quantization comparison (QwenPRM)
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as cnt-mcts-bl-v01's equivalent table
> above, so a direct v01-vs-v02 (PUCT-vs-KUBE) read is possible
> once filled.
>
> **Fixed:** method=`mcts_bl_kube_v01` (renamed 2026-07-16 from
> `mcts_bl_cnt_v02`), prm=qwen, agg_strategy=
> `last`, kube_c=2.0, kube_schedule=parent, kube_affordable=true,
> bs-4, d-20, b=80, prm_batch_size=1, tmpl=model-family default
> (native for Qwen, custom for Llama). See
> `docs/decisions/bl-kube-bonus-schedule.md` for the schedule choice.
>
> **W&B:** llama-1b `gu5l0k7p`, llama-3b `rywqssh0`, qwen-3b
> `79kyy2a7`, qwen-7b gptq `tmtery3w`, qwen-math-1.5b `ktgdyyfx`.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .3060<br>±.0282 | .2612<br>±.0269 | .2463<br>±.0264 | .2276<br>±.0257 | 3.11 |
| llama-3b fp16 | 2 | scored | .4851<br>±.0306 | .3918<br>±.0299 | .3769<br>±.0297 | .3731<br>±.0296 | 4.65 |
| qwen-3b fp16 | 2 | scored | .6157<br>±.0298 | .5410<br>±.0305 | .5224<br>±.0306 | .5075<br>±.0306 | 4.10 |
| qwen-7b gptq-int4 | 2 | scored | .7164<br>±.0276 | .6157<br>±.0298 | .5858<br>±.0301 | .5746<br>±.0303 | 3.43 |
| qwen-math-1.5b fp16 | 2 | scored | .6493<br>±.0292 | .5784<br>±.0302 | .5672<br>±.0303 | .5522<br>±.0304 | 3.25 |

> **Analysis.** All 5 cells scored (2026-07-22). Ordering is
> size/family-consistent and matches the other bl-family grids:
> llama-1b < llama-3b < qwen-3b < qwen-math-1.5b < qwen-7b gptq
> (best .7164). llama-3b, qwen-7b, and qwen-math values match
> the cross-algorithm summary tables exactly (consistency check
> passed); llama-1b and qwen-3b are net-new. Vs. cnt-mcts-bl-v01
> cell-for-cell (PUCT-vs-KUBE at identical budget): KUBE wins
> every model — llama-1b .3060 vs .2313, llama-3b .4851 vs
> .3731, qwen-7b .7164 vs .6343, qwen-math .6493 vs .4366 (the
> largest gap, +.21).
> **Limitations / follow-up:** queue-only block
> (`kube-bl-v01-l5-*` entries, all done — deletable after this
> recording); no experiments.yaml entries.

### kube-mcts-bl-v02

#### score_mode sweep: parent_blend (alpha) vs. path_decay (gamma × kube_c) (qwen-3b, QwenPRM)
> **Compares:** the two selectable v02 frontier densities
> head-to-head, mirroring the cnt-mcts-bl-v02 score_mode sweep
> above cell-for-cell (same model, PRM, level, budget, arm grid) —
> so the two families' sweeps are directly comparable: under
> kube_schedule=parent (fixed here), kube's path_decay density is
> exactly bl_cnt v02's path_decay score divided by remaining cost,
> and kube's parent_blend bonus is exactly bl_cnt's PUCT bonus
> over cost. Any ranking difference between the families is
> attributable to the /cost division alone. The gamma × kube_c
> cross rationale is the same as the cnt table's (see its blurb):
> the AZ-shaped path_decay bonus has no log damping, so
> kube_c=2.0 may swamp the value term regardless of gamma — the
> cross separates "gamma useless" from "gamma drowned out".
>
> **Fixed:** method=`mcts_bl_kube_v02`, kube_schedule=`parent`,
> kube_affordable=true, llm=qwen-3b fp16 (native tmpl), prm=qwen,
> agg_strategy=`last`, bs-4, d-20, b=80, prm_batch_size=1,
> level=5.
>
> ⚠️ 9/9 cells scored (2 trials each). kube_c is NOT comparable
> across the two modes (different bonus shapes: log form vs. AZ
> form) — compare kube_c values within a mode only. The alpha=1.0
> row is the exact v01 control arm; path_decay has no
> v01-equivalent arm.
>
> **W&B:** parent_blend alpha=1.0 `26ty6v7n`, alpha=0.8 `z15wgie9`,
> alpha=0.6 `fxcn54og`; path_decay gamma=0.8/kube_c=0.5 `nvcah979`,
> gamma=0.5/kube_c=2.0 `69mrzt65`, gamma=0.8/kube_c=2.0 `mi5gfiwv`,
> gamma=1.0/kube_c=2.0 `ugkxb2de`, gamma=1.0/kube_c=0.5 `cs0rfgii`,
> gamma=0.5/kube_c=0.5 `gthxems1`.

| score_mode | alpha | gamma | kube_c | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|
| parent_blend | 1.0 | — | — | 2 | scored | .6381<br>±.0294 | .5485<br>±.0305 | .5112<br>±.0306 | .5224<br>±.0306 | — |
| parent_blend | 0.8 | — | — | 2 | scored | .6194<br>±.0297 | .5299<br>±.0305 | .5224<br>±.0306 | .5037<br>±.0306 | — |
| parent_blend | 0.6 | — | — | 2 | scored | .6082<br>±.0299 | .5149<br>±.0306 | .5075<br>±.0306 | .4925<br>±.0306 |  4.20 |
| path_decay | — | 1.0 | 2.0 | 2 | scored | .6082<br>±.0299 | .5149<br>±.0306 | .4776<br>±.0306 | .4739<br>±.0306 |  3.97 |
| path_decay | — | 0.8 | 2.0 | 2 | scored | .6157<br>±.0298 | .5261<br>±.0306 | .4925<br>±.0306 | .4701<br>±.0305 |  3.87 |
| path_decay | — | 0.5 | 2.0 | 2 | scored | .6269<br>±.0296 | .5112<br>±.0306 | .5000<br>±.0306 | .4851<br>±.0306 |  4.14 |
| path_decay | — | 1.0 | 0.5 | 2 | scored | .6269<br>±.0296 | .5187<br>±.0306 | .4664<br>±.0305 | .4440<br>±.0304 |  3.91 |
| path_decay | — | 0.8 | 0.5 | 2 | scored | .6269<br>±.0296 | .5261<br>±.0306 | .4813<br>±.0306 | .4813<br>±.0306 | — |
| path_decay | — | 0.5 | 0.5 | 2 | scored | .6231<br>±.0297 | .5187<br>±.0306 | .4925<br>±.0306 | .4888<br>±.0306 |  3.97 |

> **Analysis.** All 9 cells scored. `parent_blend, alpha=1.0`
> (pass@gb .6381 — the v01 control, no blend) is the best arm
> overall; parent_blend is monotone in alpha (1.0 > 0.8 > 0.6),
> i.e. blending toward the parent's q consistently hurts here.
> path_decay tops out at .6269 (three tied cells: gamma=0.5/
> kube_c=2.0, gamma=1.0/kube_c=0.5, gamma=0.8/kube_c=0.5) with no
> clean gamma trend, and trails the best parent_blend cell. Reads
> mirror the cnt table's three (blend vs. v01 control; gamma at
> kube_c=0.5 vs. 2.0; best arm per mode), plus the cross-family
> read this table uniquely enables: same-arm kube-vs-cnt cells
> isolate the effect of cost normalization on each score_mode. The
> winning mode is the survivor; the loser is slated for deletion
> (docs/decisions-log.md 2026-07-19).
> **Limitations / follow-up:** all 9 cells scored — see
> experiments.yaml group `kube-mcts-bl-v02`, feeds
> `level5-kube-bl-v02-score-mode-qwen3b`. kube_schedule=global not
> swept here (would double the grid); a separate global-schedule
> table is a later decision.

#### model family, size, quantization comparison (QwenPRM, parent_blend/alpha=0.8)
> **Compares:** model family, size, and quantization jointly at
> the winning-candidate frontier score `score_mode=parent_blend`
> with `alpha=0.8` — same 5-model/quant grid as cnt-mcts-bl-v01's
> equivalent table above, so a direct kube_v02-vs-cnt-v01 (and,
> across the bl families' model-family tables) read is possible
> once filled. The qwen-3b cell is the **exact same run** as the
> `parent_blend/alpha=0.8` arm of the score_mode sweep above
> (cfg-a027f260, W&B `z15wgie9`) — reused, not re-run.
>
> **Fixed:** method=`mcts_bl_kube_v02`, **score_mode=parent_blend,
> alpha=0.8**, kube_schedule=`parent`, kube_c=2.0 (default),
> kube_affordable=true (default), prm=qwen, agg_strategy=`last`,
> bs-4, d-20, b=80, prm_batch_size=1, level=5, tmpl=model-family
> default (native for Qwen, custom for Llama).
>
> ⚠️ 1/5 cells scored (qwen-3b, reused from the score_mode sweep);
> the other 4 inqueue (orchestration/queue.yaml
> `kube-bl-v02-l5-mf-a0.8-*`, priority 2, queued 2026-07-22 —
> deliberately behind the alpha×kube_c sweep, which decides
> whether alpha=0.8 is the arm worth propagating).
>
> **W&B:** qwen-3b `z15wgie9`; others none yet (no runs exist).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | inqueue | — | — | — | — | — |
| llama-3b fp16 | — | inqueue | — | — | — | — | — |
| qwen-3b fp16 | 2 | scored | .6194<br>±.0297 | .5299<br>±.0305 | .5224<br>±.0306 | .5037<br>±.0306 | — |
| qwen-7b gptq-int4 | — | inqueue | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | inqueue | — | — | — | — | — |

> **Analysis.** Only the qwen-3b cell is filled (pass@gb .6194).
> Once the other four land, the key read is how parent_blend's
> one-hop q-blend generalizes across model families and sizes at
> the fixed alpha=0.8, and — cell-for-cell against the analogous
> cnt-mcts-bl and kdepth-mcts-bl model-family tables — whether the
> /cost normalization helps or hurts per model.
> **Limitations / follow-up:** 4 cells planned — see
> experiments.yaml group `kube-mcts-bl-v02`, feeds
> `level5-kube-bl-v02-model-family-parent-blend-qwen`. qwen-3b
> feeds both this table and the score_mode-sweep table.

#### model family, size, quantization comparison (QwenPRM, parent_blend/alpha=1.0)
> **Compares:** the same 5-model/quant grid as the
> `parent_blend/alpha=0.8` table above, but at **alpha=1.0** — the
> exact-v01 control arm (no parent blend: `blended_q = q(leaf)`,
> recovering `BLMCTSKubeV01Config`'s kube_density exactly). Read
> against the alpha=0.8 table, this isolates whether the one-hop
> q-blend helps or hurts per model family. On qwen-3b the control
> currently wins (.6381 vs .6194), so this table tests whether
> that holds across models. qwen-3b reuses the score_mode sweep's
> alpha=1.0 arm (cfg-4db0f6ff, W&B `26ty6v7n`).
>
> **Fixed:** method=`mcts_bl_kube_v02`, **score_mode=parent_blend,
> alpha=1.0**, kube_schedule=`parent`, kube_c=2.0 (default),
> kube_affordable=true (default), prm=qwen, agg_strategy=`last`,
> bs-4, d-20, b=80, prm_batch_size=1, level=5, tmpl=model-family
> default (native for Qwen, custom for Llama).
>
> All 5 cells scored (2026-07-22).
>
> **W&B:** llama-1b `oypd0uyv`, llama-3b `apcu4aqr`, qwen-3b
> `26ty6v7n`, qwen-7b gptq `68tp8ltv`, qwen-math-1.5b `g9qs2c4z`.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .3321<br>±.0288 | .2761<br>±.0274 | .2463<br>±.0264 | .2425<br>±.0262 | 2.99 |
| llama-3b fp16 | 2 | scored | .4851<br>±.0306 | .3881<br>±.0298 | .3881<br>±.0298 | .3731<br>±.0296 | 4.67 |
| qwen-3b fp16 | 2 | scored | .6381<br>±.0294 | .5485<br>±.0305 | .5112<br>±.0306 | .5224<br>±.0306 | — |
| qwen-7b gptq-int4 | 2 | scored | .6978<br>±.0281 | .6045<br>±.0299 | .5261<br>±.0306 | .5187<br>±.0306 | 3.42 |
| qwen-math-1.5b fp16 | 2 | scored | .6679<br>±.0288 | .5709<br>±.0303 | .5410<br>±.0305 | .5261<br>±.0306 | 3.42 |

> **Analysis.** All 5 scored. Vs. kube-v01 cell-for-cell (the
> eager-terminal-backprop isolation — alpha=1.0 recovers v01's
> kube_density exactly, so the ONLY difference is v02's fold):
> pass@gb deltas are small and mixed — llama-1b +.026, llama-3b
> .000, qwen-3b +.022, qwen-7b −.019, qwen-math +.019, all
> within ~1 SEM. Eager backprop is roughly **neutral on
> pass@gb**. The real signal is wei@gb: v02 hurts weighted
> aggregation on all three Qwen models, sharply on qwen-7b
> (.5261 vs v01's .5858, −.06 ≈ 2 SEM) — eager terminal folding
> appears to distort the score distribution the weighted vote
> relies on. If wei@gb matters, v01's delayed backprop is the
> safer default. Cost identical (hr/trial ±.02); v02 runs far
> fewer phases (~77 vs 95–140) for the same budget. qwen-7b's
> trees are notably shallower (depth ~9 vs 12–14 elsewhere).
> **Limitations / follow-up:** 4 cells planned — see
> experiments.yaml group `kube-mcts-bl-v02`, feeds
> `level5-kube-bl-v02-model-family-parent-blend-a1.0-qwen`. qwen-3b
> feeds both this table and the score_mode-sweep table.

#### model family, size, quantization comparison (QwenPRM, parent_blend/alpha=0.0)
> **Compares:** the same 5-model/quant grid at the **value-blindness
> extreme alpha=0.0** — the leaf's own q is ignored entirely and
> every child is scored by its parent's q alone
> (`blended_q = q(parent)`). Within a sibling set the value term is
> then identical for all children, so selection among siblings is
> driven purely by the exploration/visit term and the kube bonus.
> This is an ablation, not a tuning arm: it asks whether the leaf's
> own PRM signal matters at all under KUBE, bounding the alpha
> sweep from below. Read against the alpha=1.0 (pure own-q) and
> alpha=0.8 tables: a small 0.0→1.0 gap would mean per-leaf PRM
> discrimination contributes little beyond tree structure; a large
> gap confirms the leaf q is load-bearing. Unlike the 0.8/1.0
> tables, the qwen-3b cell is NOT reusable from the score_mode
> sweep (no alpha=0.0 arm exists there) — all 5 cells are net-new.
>
> **Fixed:** method=`mcts_bl_kube_v02`, **score_mode=parent_blend,
> alpha=0.0**, kube_schedule=`parent`, kube_c=2.0 (default),
> kube_affordable=true (default), prm=qwen, agg_strategy=`last`,
> bs-4, d-20, b=80, prm_batch_size=1, level=5, tmpl=model-family
> default (native for Qwen, custom for Llama).
>
> ⚠️ Entirely planned, no runs yet.
>
> **W&B:** none yet (no runs exist).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | scored | — | — | — | — | — |
| llama-3b fp16 | — | running | — | — | — | — | — |
| qwen-3b fp16 | — | running | — | — | — | — | — |
| qwen-7b gptq-int4 | — | running | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | running | — | — | — | — | — |

> **Analysis.** No data yet — nothing to take away. Once filled,
> the key read per model is the alpha=0.0-vs-1.0 gap (how much the
> leaf's own PRM q buys over parent-only scoring), with alpha=0.8
> as the intermediate point; a rough second read is whether the
> root-adjacent depth-1 phase (where the parent is the root with no
> meaningful q) degrades uniformly across models. All 5 launched
> 2026-07-22 (llama-1b/3b + qwen-3b 09:30; qwen-7b +
> qwen-math-1.5b 11:22).
> **Limitations / follow-up:** ledger
> experiments/prm800k-level5.yaml, feeds
> `level5-kube-bl-v02-model-family-parent-blend-a0.0-qwen`.
> Single-alpha ablation; only worth extending if the 0.0-vs-1.0
> gap is surprisingly small.

#### alpha × kube_c joint sweep (llama-3b, QwenPRM, parent_blend)
> **Compares:** the parent_blend value-composition knob `alpha`
> jointly with the exploration scale `kube_c`, as a 3×3 factorial
> — NOT two sequential 1-D sweeps. The two parameters are
> partially aliased by construction: among siblings the parent-q
> term is a shared constant, so value discrimination is exactly
> `alpha·Δq` and behavior depends only on the ratio
> `kube_c/alpha`; alpha's genuinely new effect (importing the
> parent's q into cross-branch comparisons) is only identifiable
> against the factorial's interaction pattern. Reads: (1) kube_c
> main effect down the alpha=1.0 column; (2) alpha spread at
> kube_c=0.5/0.1 vs. at 2.0 — separation only at low kube_c
> confirms scale-domination (tune alpha in the low-c regime);
> flat everywhere is real evidence against one-hop blending;
> (3) the (a=1.0,c=2.0) vs (a=0.5,c=1.0-equivalent) diagonal is
> approximated by comparing constant-ratio pairs across the grid.
> kube_c grid {0.1, 0.5, 2.0} straddles the default from below —
> q ∈ [0,1] bounds sibling Δq to a few tenths, so the
> value↔bonus crossover plausibly sits below the default, not
> above (see the score_mode sweep's scale-domination rationale).
>
> **Fixed:** method=`mcts_bl_kube_v02`, llm=llama-3b fp16 (custom
> tmpl), **score_mode=parent_blend**, kube_schedule=`parent`,
> kube_affordable=true (default), prm=qwen, agg_strategy=`last`,
> bs-4, d-20, b=80, prm_batch_size=1, level=5.
>
> ⚠️ 1/9 cells scored — the (alpha=1.0, kube_c=2.0) cell is the
> **exact same run** as the alpha=1.0 model-family table's
> llama-3b cell (cfg-63051bb1), reused, not re-run. The
> (alpha=0.8, kube_c=2.0) cell will likewise reuse the alpha=0.8
> model-family table's llama-3b cell once that queue entry runs
> (`kube-bl-v02-l5-mf-a0.8-llama3b`). The other 7 are net-new.
> kube_c is NOT numerically comparable to cnt-v02's cpuct
> (different bonus shapes).
>
> **W&B:** (1.0, 2.0) `apcu4aqr`; others none yet.

| alpha | kube_c | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 2.0 | 2 | scored | .4851<br>±.0306 | .3881<br>±.0298 | .3881<br>±.0298 | .3731<br>±.0296 | 4.67 |
| 1.0 | 0.5 | — | running | — | — | — | — | — |
| 1.0 | 0.1 | — | running | — | — | — | — | — |
| 0.8 | 2.0 | — | inqueue | — | — | — | — | — |
| 0.8 | 0.5 | — | running | — | — | — | — | — |
| 0.8 | 0.1 | — | running | — | — | — | — | — |
| 0.5 | 2.0 | — | inqueue | — | — | — | — | — |
| 0.5 | 0.5 | — | running | — | — | — | — | — |
| 0.5 | 0.1 | — | running | — | — | — | — | — |

> **Analysis.** No scored cells yet — nothing to take away. Once
> filled, the decision rule: if alpha separates only at low
> kube_c, fix the low-c regime and tune alpha there; if the
> alpha=1.0 column dominates everywhere, one-hop blending is dead
> for kube and the model-family grids stay at alpha=1.0; the
> winning (alpha, kube_c) pair + the alpha=1.0 control then
> propagate to the other models' grids — the full 3×3 is NOT
> repeated per model.
> **Limitations / follow-up:** 7 cells net-new, ~8h/trial each —
> queue-only block (orchestration/queue.yaml
> `kube-bl-v02-l5-ac-sweep-llama3b-*`), no experiments.yaml
> entries yet. 2 trials/cell → SEM ~±.03; effects under ~.06
> pass@gb are not resolvable. Single model (llama-3b); the
> propagation step, not this table, covers generalization.

### kdepth-mcts-bl-v01

#### model family, size, quantization comparison (QwenPRM)
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as cnt-mcts-bl-v01's equivalent table
> above, so a direct bl_cnt-v01-vs-v03 (and, once v02 has runs,
> a three-way PUCT/KUBE/depth-shaping) read is possible once
> filled.
>
> **Fixed:** method=`mcts_bl_kdepth_v01` (renamed 2026-07-17 from
> `mcts_bl_cnt_v03`), prm=qwen, agg_strategy=
> `last`, depth_beta=2.0, depth_alpha=1.0, kube_affordable=true
> (default), bs-4, d-20, b=80, prm_batch_size=1, tmpl=model-family
> default (native for Qwen, custom for Llama).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .2910<br>±.0278 | .2612<br>±.0269 | .2351<br>±.0260 | .2164<br>±.0252 | — |
| llama-3b fp16 | 2 | scored | .4739<br>±.0306 | .4179<br>±.0302 | .3918<br>±.0299 | .3806<br>±.0297 | — |
| qwen-3b fp16 | 2 | scored | .6343<br>±.0295 | .5709<br>±.0303 | .5634<br>±.0304 | .5560<br>±.0304 | 3.97 |
| qwen-7b gptq-int4 | 2 | scored | .7388<br>±.0269 | .6418<br>±.0293 | .6045<br>±.0299 | .6007<br>±.0300 | — |
| qwen-math-1.5b fp16 | 2 | scored | .6642<br>±.0289 | .5634<br>±.0304 | .5560<br>±.0304 | .5448<br>±.0305 | — |

> **Analysis.** No level-5 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned; launch is
> the level-4 counterpart's command plus `data.level=5`.

#### model family, size, quantization comparison (QwenPRM, depth_alpha=0.5)
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as the `depth_alpha=1.0` table above,
> but with a **concave** depth-bonus curve (`f(z)=1-z^0.5`, bonus
> stays high deeper into the tree). Read against the α=1.0 and
> α=2.0 tables, this isolates how the depth-bonus curvature
> interacts with model family/size at fixed `depth_beta=2.0`.
>
> **Fixed:** method=`mcts_bl_kdepth_v01`, prm=qwen, agg_strategy=
> `last`, depth_beta=2.0, **depth_alpha=0.5**, kube_affordable=true
> (default), bs-4, d-20, b=80, prm_batch_size=1, tmpl=model-family
> default (native for Qwen, custom for Llama).
>
> ⚠️ Entirely planned, no runs yet.
>
> **W&B:** none yet (no runs exist).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .2910<br>±.0278 | .2724<br>±.0272 | .2575<br>±.0268 | .2425<br>±.0262 | 2.88 |
| llama-3b fp16 | 2 | scored | .4963<br>±.0306 | .4291<br>±.0303 | .4291<br>±.0303 | .4216<br>±.0302 | 4.50 |
| qwen-3b fp16 | 2 | scored | .5970<br>±.0300 | .5597<br>±.0304 | .5560<br>±.0304 | .5485<br>±.0305 | 3.97 |
| qwen-7b gptq-int4 | 2 | scored | .7313<br>±.0271 | .6119<br>±.0298 | .6157<br>±.0298 | .6119<br>±.0298 | 3.52 |
| qwen-math-1.5b fp16 | 2 | scored | .6418<br>±.0293 | .5970<br>±.0300 | .6045<br>±.0299 | .6007<br>±.0300 | 3.11 |

> **Analysis.** No data yet — nothing to take away. Once filled,
> the key read is whether a concave (depth-tolerant) bonus helps
> or hurts relative to the linear α=1.0 baseline, and whether
> that effect is consistent across model families.
> **Limitations / follow-up:** all cells planned — see
> experiments.yaml group `kdepth-mcts-bl`, feeds
> `level5-kdepth-bl-v01-model-family-qwen-da0.5`.

#### model family, size, quantization comparison (QwenPRM, depth_alpha=2.0)
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as the `depth_alpha=1.0` table above,
> but with a **convex** depth-bonus curve (`f(z)=1-z^2`, bonus
> collapses fast → strongly favors shallow nodes). The α=2.0
> counterpart to the α=0.5 table, so the three tables
> (0.5/1.0/2.0) span the curvature axis at fixed
> `depth_beta=2.0`.
>
> **Fixed:** method=`mcts_bl_kdepth_v01`, prm=qwen, agg_strategy=
> `last`, depth_beta=2.0, **depth_alpha=2.0**, kube_affordable=true
> (default), bs-4, d-20, b=80, prm_batch_size=1, tmpl=model-family
> default (native for Qwen, custom for Llama).
>
> ⚠️ 1/5 cells scored (qwen-math-1.5b); the other 4 planned.
>
> **W&B:** qwen-math-1.5b on disk as cfg-ad001285 (scored via
> prepare_scored_dataset + compute_stats_basics, 2 trials); no
> W&B run id captured. Others: none yet.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .3022<br>±.0281 | .2500<br>±.0265 | .2276<br>±.0257 | .2127<br>±.0250 | 3.22 |
| llama-3b fp16 | 2 | scored | .5000<br>±.0306 | .4104<br>±.0301 | .4030<br>±.0300 | .3955<br>±.0299 | 4.78 |
| qwen-3b fp16 | 2 | scored | .6082<br>±.0299 | .5224<br>±.0306 | .5224<br>±.0306 | .5075<br>±.0306 | 3.92 |
| qwen-7b gptq-int4 | — | failed | — | — | — | — | — |
| qwen-math-1.5b fp16 | 2 | scored | .6455<br>±.0293 | .5522<br>±.0304 | .5485<br>±.0305 | .5336<br>±.0305 |  3.26 |

> **Analysis.** One data point so far (qwen-math-1.5b): pass@gb
> .6455 ± .0293 at α=2.0. For the same model/quant the α=0.5
> and α=1.0 tables give the cross-curvature comparison; the
> full read (whether the convex, shallow-favoring bonus shifts
> realized depth and accuracy relative to α=1.0, and whether
> `1/cost` deep-node inflation cancels it) waits on the other 4
> cells (see docs/decisions/bl-kdepth-knapsack-bonus.md).
> **Limitations / follow-up:** all cells planned — see
> experiments.yaml group `kdepth-mcts-bl`, feeds
> `level5-kdepth-bl-v01-model-family-qwen-da2.0`.

### kdepth-mcts-bl-v02

#### score_mode sweep: parent_blend (alpha) vs. path_decay (gamma) (qwen-3b, QwenPRM)
> **Compares:** the two selectable v02 frontier densities
> head-to-head, mirroring the kube-mcts-bl-v02 score_mode sweep
> above — but on kdepth's density, whose exploration term is the
> fixed DEPTH bonus (`depth_beta*(1-depth_frac**depth_alpha)`),
> not a visit/clock bonus. So the score_mode blend here touches
> ONLY the value (q) term; the depth bonus and `/cost` are shared
> across both modes (see docs/decisions-log.md 2026-07-21). There
> is no `kube_c` axis (kdepth has no bonus coefficient to sweep —
> `depth_beta`/`depth_alpha` are held fixed), so path_decay varies
> gamma alone. The read: does path-aware value (one-hop parent
> blend, or gamma-decayed full path) beat plain own-q under a
> depth-shaped frontier?
>
> **Fixed:** method=`mcts_bl_kdepth_v02`, depth_beta=2.0,
> depth_alpha=1.0, kube_affordable=true, llm=qwen-3b fp16 (native
> tmpl), prm=qwen, agg_strategy=`last`, bs-4, d-20, b=80,
> prm_batch_size=1, level=5.
>
> ⚠️ Entirely planned, no runs yet. The `parent_blend, alpha=1.0`
> row is the exact v01 control arm (own-q only, recovers
> `BLMCTSKdepthV01Config`'s depth_density exactly);
> `path_decay, gamma=0.0` would also reduce to own-q but is NOT
> included as a distinct control (v01 == parent_blend alpha=1.0).
>
> **W&B:** none yet (no runs exist).

| score_mode | alpha | gamma | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| parent_blend | 1.0 | — | — | running | — | — | — | — | — |
| parent_blend | 0.8 | — | — | running | — | — | — | — | — |
| parent_blend | 0.6 | — | — | planned | — | — | — | — | — |
| path_decay | — | 1.0 | — | planned | — | — | — | — | — |
| path_decay | — | 0.8 | — | planned | — | — | — | — | — |
| path_decay | — | 0.5 | — | planned | — | — | — | — | — |

> **Analysis.** No data yet — nothing to take away. Once filled,
> the key reads are: (1) whether any path-aware value beats the
> alpha=1.0 own-q control under a depth-shaped frontier (the
> kube-v02 sweep found the control won there — does that hold
> when the exploration term is depth, not visits?); (2)
> parent_blend's alpha trend; (3) path_decay's gamma trend; and
> (4) the cross-family read against the kube-v02 and cnt-v02
> score_mode sweeps — same value-blend, different exploration
> term (depth vs. AZ-visit vs. PUCT-visit).
> **Limitations / follow-up:** all 6 cells planned — see
> experiments.yaml group `kdepth-mcts-bl`, feeds
> `level5-kdepth-bl-v02-score-mode-qwen3b`. depth_beta/depth_alpha
> not swept here (fixed at the v01 defaults); a curvature ×
> score_mode grid is a later decision.

#### model family, size, quantization comparison (QwenPRM, parent_blend/alpha=0.8)
> **Compares:** model family, size, and quantization jointly at
> `score_mode=parent_blend` with `alpha=0.8` — same 5-model/quant
> grid as kdepth-mcts-bl-v01's and kube-mcts-bl-v02's equivalent
> tables, so a direct v01-vs-v02 (blend vs. no channel) and
> cross-family (kdepth vs. kube) read is possible once filled. The
> qwen-3b cell is the **exact same run** as the
> `parent_blend/alpha=0.8` arm of the score_mode sweep above
> (cfg-414a9f81) — reused, not re-run.
>
> **Fixed:** method=`mcts_bl_kdepth_v02`, **score_mode=parent_blend,
> alpha=0.8**, depth_beta=2.0, depth_alpha=1.0, kube_affordable=true
> (default), prm=qwen, agg_strategy=`last`, bs-4, d-20, b=80,
> prm_batch_size=1, level=5, tmpl=model-family default (native for
> Qwen, custom for Llama).
>
> ⚠️ Entirely planned, no runs yet (qwen-3b reuses the score_mode
> sweep's alpha=0.8 arm once that runs).
>
> **W&B:** none yet (no runs exist).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | failed | — | — | — | — | — |
| llama-3b fp16 | 2 | scored | .4776<br>±.0306 | .4104<br>±.0301 | .3993<br>±.0300 | .3993<br>±.0300 | — |
| qwen-3b fp16 | 2 | scored | .6231<br>±.0297 | .5709<br>±.0303 | .5522<br>±.0304 | .5410<br>±.0305 | — |
| qwen-7b gptq-int4 | 2 | scored | .7239<br>±.0274 | .6194<br>±.0297 | .6231<br>±.0297 | .6157<br>±.0298 | — |
| qwen-math-1.5b fp16 | 2 | scored | .6679<br>±.0288 | .5933<br>±.0301 | .5896<br>±.0301 | .5746<br>±.0303 | — |

> **Analysis.** No data yet. Once filled, the key read is how
> parent_blend's one-hop q-blend generalizes across model families
> at fixed alpha=0.8 under a depth-shaped frontier, and —
> cell-for-cell against the kube-v02 and cnt-v02 model-family
> tables — whether the depth bonus (vs. visit bonus) helps or
> hurts per model.
> **Limitations / follow-up:** all 5 cells planned — see
> experiments.yaml group `kdepth-mcts-bl`, feeds
> `level5-kdepth-bl-v02-model-family-parent-blend-qwen`. qwen-3b
> feeds both this table and the score_mode-sweep table.

#### model family, size, quantization comparison (QwenPRM, parent_blend/alpha=1.0)
> **Compares:** the same 5-model/quant grid as the
> `parent_blend/alpha=0.8` table above, but at **alpha=1.0** — the
> exact-v01 control arm (no parent blend: `blended_q = q(leaf)`,
> recovering `BLMCTSKdepthV01Config`'s depth_density exactly). Read
> against the alpha=0.8 table, this isolates whether the one-hop
> q-blend helps or hurts per model family under a depth-shaped
> frontier. qwen-3b reuses the score_mode sweep's alpha=1.0 arm
> (cfg-0483dfe8).
>
> **Fixed:** method=`mcts_bl_kdepth_v02`, **score_mode=parent_blend,
> alpha=1.0**, depth_beta=2.0, depth_alpha=1.0, kube_affordable=true
> (default), prm=qwen, agg_strategy=`last`, bs-4, d-20, b=80,
> prm_batch_size=1, level=5, tmpl=model-family default (native for
> Qwen, custom for Llama).
>
> ⚠️ Entirely planned, no runs yet (qwen-3b reuses the score_mode
> sweep's alpha=1.0 arm once that runs).
>
> **W&B:** none yet (no runs exist).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .2948<br>±.0279 | .2649<br>±.0270 | .2425<br>±.0262 | .2313<br>±.0258 | — |
| llama-3b fp16 | 2 | scored | .5000<br>±.0306 | .4291<br>±.0303 | .4254<br>±.0303 | .4216<br>±.0302 | — |
| qwen-3b fp16 | 2 | scored | .6455<br>±.0293 | .5896<br>±.0301 | .5933<br>±.0301 | .5933<br>±.0301 | — |
| qwen-7b gptq-int4 | 2 | scored | .7351<br>±.0270 | .6007<br>±.0300 | .5784<br>±.0302 | .5821<br>±.0302 | — |
| qwen-math-1.5b fp16 | 2 | scored | .6604<br>±.0290 | .5970<br>±.0300 | .5821<br>±.0302 | .5672<br>±.0303 | — |

> **Analysis.** No data yet. Once filled, the key read is the
> per-model alpha=1.0-vs-0.8 delta — whether "no blend beats
> blend" (as the kube-v02 sweep found on qwen-3b) holds across the
> family under a depth-shaped frontier.
> **Limitations / follow-up:** all 5 cells planned — see
> experiments.yaml group `kdepth-mcts-bl`, feeds
> `level5-kdepth-bl-v02-model-family-parent-blend-a1.0-qwen`.
> qwen-3b feeds both this table and the score_mode-sweep table.

### sem-mcts-bl-v01
could you help me add these sem-mcts-bl-v01 experiments to queue, all priority=1 except priority for llama-1b and qweb-3b=1.7
#### model family, size, quantization comparison (QwenPRM, lam=0.01/ds_alpha=10)
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as cnt-mcts-bl-v01's equivalent table
> above, so a direct bl_sem-vs-bl_cnt read is possible once both
> are filled. Anchored to the **same (lam, ds_alpha) checkpoint
> as sem-mcts (v02)'s `lam=0.01/ds_alpha=10` table** above (not
> level-4 bl_sem_v01's `lam=0.1` convention), so bl_sem-vs-sem_v02
> is apples-to-apples at this `w_eff`.
>
> **Fixed:** method=`mcts_bl_sem_v01` (PRM embeds — prm=qwen for
> both scoring AND diversity via `embeds_source=prm`, the schema
> default, no second pooling engine), prm=qwen, agg_strategy=
> `last`, bs-4, d-20, b=80, prm_batch_size=1,
> `ds_alpha_schedule=global` (default — see decisions-log),
> `cov_update=sm` (sherman_morrison), `embeds_proj=sparse512`
> (`embeds_dim=512`, defaults), ds_beta=1.0, tmpl=model-family
> default (native for Qwen, custom for Llama).
> **lam=0.01, ds_alpha=10** (`w_eff = ds_alpha/sqrt(lam) = 100`).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | running | — | — | — | — | — |
| llama-3b fp16 | — | running | — | — | — | — | — |
| qwen-3b fp16 | 2 | scored | .5970<br>±.0300 | .4403<br>±.0304 | .3619<br>±.0294 | .3172<br>±.0285 | 8.1 |
| qwen-7b gptq-int4 | 2 | scored | .7537<br>±.0264 | .5597<br>±.0304 | .5037<br>±.0306 | .4478<br>±.0304 | 6.6 |
| qwen-math-1.5b fp16 | 2 | scored | .6567<br>±.0291 | .5410<br>±.0305 | .4627<br>±.0305 | .4552<br>±.0305 | 9.2 |

> **Analysis.** No level-5 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned. Launch per
> cell: `generate_mcts_sem.py --config-name
> mcts_bl_sem_v01_prm800k llm=<model> prm=qwen_prm data.level=5
> search.lam=0.01 search.ds_alpha=10` (NOTE: `lam=0.01`, not the
> level-4 bl_sem_v01 tables' `lam=0.1` — this table is anchored
> to the sem_v02 `lam=0.01` checkpoint, see Compares).

#### model family, size, quantization comparison (QwenPRM, lam=0.01/ds_alpha=1)
> **Compares:** same 5-model/quant grid as the `lam=0.01/
> ds_alpha=10` table above, at one order of magnitude lower
> effective diversity weight — the two tables together give a
> first (coarse) read on whether the model-family ranking is
> sensitive to `w_eff` for this algorithm, ahead of a proper
> `w_eff` sweep. Anchored to sem-mcts (v02)'s `lam=0.01/
> ds_alpha=1` table above (`w_eff=10`).
>
> **Fixed:** identical to the `lam=0.01/ds_alpha=10` table above
> (method=`mcts_bl_sem_v01`, prm=qwen, agg_strategy=`last`, bs-4,
> d-20, b=80, prm_batch_size=1, `ds_alpha_schedule=global`,
> `cov_update=sm`, `embeds_proj=sparse512`, ds_beta=1.0, tmpl=
> model-family default) except the diversity weight.
> **lam=0.01, ds_alpha=1.0** (`w_eff = ds_alpha/sqrt(lam) = 10`).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | planned | — | — | — | — | — |
| llama-3b fp16 | — | planned | — | — | — | — | — |
| qwen-3b fp16 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | — | planned | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | planned | — | — | — | — | — |

> **Analysis.** No level-5 data yet — nothing to take away.
> **Limitations / follow-up:** entire table planned. Launch per
> cell: `generate_mcts_sem.py --config-name
> mcts_bl_sem_v01_prm800k llm=<model> prm=qwen_prm data.level=5
> search.lam=0.01 search.ds_alpha=1` (NOTE: `lam=0.01`, not the
> level-4 bl_sem_v01 tables' `lam=0.1`; anchored to the sem_v02
> `lam=0.01` checkpoint, see Compares).

### sem-mcts-bl-v02

#### model family, size, quantization comparison (QwenPRM, parent_blend/alpha=1.0, lam=0.01/ds_alpha=10)
> **Compares:** model family, size, and quantization jointly for
> sem-mcts-bl-v02 at its selectable frontier value term fixed to
> `score_mode=parent_blend, alpha=1.0` (the no-blend control:
> `q_term = q(leaf)`, byte-identical to `score_mode=own`), with the
> diversity knobs pinned to the same `lam=0.01/ds_alpha=10`
> (`w_eff=100`) checkpoint as the sem-mcts-bl-v01 table above and
> sem-mcts (v02)'s tables — so bl_sem_v02-vs-bl_sem_v01 and, cell-
> for-cell, bl_sem_v02-vs-kube_v02 (both at parent_blend/alpha=1.0)
> reads are possible once filled.
>
> **Fixed:** method=`mcts_bl_sem_v02`, **score_mode=parent_blend,
> alpha=1.0**, embeds_source=prm, **lam=0.01, ds_alpha=10**
> (`w_eff=100`), ds_beta=1.0 (default), ds_alpha_schedule=global
> (default), prm=qwen, agg_strategy=`last`, bs-4, d-20, b=80,
> prm_batch_size=1, level=5, tmpl=model-family default (native for
> Qwen, custom for Llama).
>
> ⚠️ Entirely planned, no runs yet.
>
> **W&B:** none yet (no runs exist).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | running | — | — | — | — | — |
| llama-3b fp16 | — | running | — | — | — | — | — |
| qwen-3b fp16 | — | running | — | — | — | — | — |
| qwen-7b gptq-int4 | — | running | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | running | — | — | — | — | — |

> **Analysis.** No data yet — nothing to take away. Once filled,
> the key reads are (1) bl_sem_v02 vs. bl_sem_v01 at the same
> lam/ds_alpha — what the v02 eager-terminal + score_mode machinery
> buys on the diversity family, and (2) the cell-for-cell
> parent_blend/alpha=1.0 comparison against the kube-mcts-bl-v02
> model-family table, isolating the diversity value term vs. the
> KUBE density at a shared no-blend control.
> **Limitations / follow-up:** entire table planned — see
> experiments.yaml group `sem-mcts-bl` (feeds
> `level5-sem-bl-v02-model-family-parent-blend-a1.0-qwen`). Launch
> per cell: `generate_mcts_sem.py --config-name
> mcts_bl_sem_v02_prm800k llm=<model> prm=qwen_prm data.level=5
> search.score_mode=parent_blend search.alpha=1.0 search.lam=0.01
> search.ds_alpha=10`.

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
