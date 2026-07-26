# LLM Reasoning — MCTS Experiment Comparison — PRM800K Level 5

> **Provenance:** structure mirrored from [exp-comp-prm800k-level4.md](exp-comp-prm800k-level4.md) (the level-4 doc) on 2026-07-10; every table reset to `planned` — no level-5 runs exist yet. Launch commands are the level-4 counterparts' plus `data.level=5` (config hashes and `--level-5--` run names follow automatically). Intro/`Fixed` prose is inherited from the level-4 doc: table definitions remain valid, but any inherited claim about completeness or findings describes level-4 state — trust the (all-planned) tables here over such prose until level-5 results land. The level-5 grid also **drops two models** relative to level 4 — llama-3b gptq and qwen-3b gptq-int4 — so inherited “7-model” grid prose reads as 5 models here (llama-1b, llama-3b fp16, qwen-3b fp16, qwen-7b gptq-int4, qwen-math-1.5b).

Central tracker for every MCTS search experiment (cnt / sem /
cnt-bl / sem-bl) on PRM800K — per-algorithm tuning tables grouped
by gen_budget, plus a cross-algorithm best-config summary.


<!-- toc:begin -- generated, do not hand-edit -->
## Contents

- [**Purpose**](#purpose)
- [**Structure and use**](#structure-and-use)
- [**Cross-algorithm summary (QwenPRM)**](#cross-algorithm-summary-qwenprm)
- [**Tuning tables \[gen_budget=80\]**](#tuning-tables-gen_budget80)
  - [cnt-mcts](#cnt-mcts)
    - [model family, size, quantization comparison (RLHFlowPRM)](#model-family-size-quantization-comparison-rlhflowprm) · `tbl-d6065d`
    - [model family, size, quantization comparison (QwenPRM)](#model-family-size-quantization-comparison-qwenprm) · `tbl-afdda0`
    - [agg_strategy comparison (qwen-3b, qwen-math-1.5b)](#agg_strategy-comparison-qwen-3b-qwen-math-15b) · `tbl-a45ce2`
  - [sem-mcts-v02](#sem-mcts-v02)
    - [embeds_strategy × scope sweep (QwenPRM)](#embeds_strategy-scope-sweep-qwenprm) · `tbl-666cb6`
    - [lam / ds_alpha joint sweep (llama-1b)](#lam-ds_alpha-joint-sweep-llama-1b) · `tbl-a554c7`
    - [lam / ds_alpha joint sweep (llama-3b)](#lam-ds_alpha-joint-sweep-llama-3b) · `tbl-591232`
    - [lam / ds_alpha joint sweep (qwen-math-1.5b)](#lam-ds_alpha-joint-sweep-qwen-math-15b) · `tbl-a12d4f`
    - [lam / ds_alpha joint sweep (qwen-7b gptq-int4)](#lam-ds_alpha-joint-sweep-qwen-7b-gptq-int4) · `tbl-21bde4`
    - [embeds_center_mode comparison (lam=0.01/ds_alpha=1)](#embeds_center_mode-comparison-lam001ds_alpha1) · `tbl-e58353`
    - [embeds_center_mode comparison (lam=0.01/ds_alpha=10)](#embeds_center_mode-comparison-lam001ds_alpha10) · `tbl-2e75f2`
    - [agg_strategy comparison (qwen-3b, qwen-math-1.5b, lam=0.01/ds_alpha=1)](#agg_strategy-comparison-qwen-3b-qwen-math-15b-lam001ds_alpha1) · `tbl-ae7863`
    - [agg_strategy comparison (qwen-3b, qwen-math-1.5b, lam=0.01/ds_alpha=10)](#agg_strategy-comparison-qwen-3b-qwen-math-15b-lam001ds_alpha10) · `tbl-4cc5b9`
    - [model family, size, quantization comparison (QwenPRM, lam=0.01/ds_alpha=1)](#model-family-size-quantization-comparison-qwenprm-lam001ds_alpha1) · `tbl-73533c`
    - [model family, size, quantization comparison (QwenPRM, lam=0.01/ds_alpha=10)](#model-family-size-quantization-comparison-qwenprm-lam001ds_alpha10) · `tbl-cf8fea`
  - [cnt-mcts-bl-v01](#cnt-mcts-bl-v01)
    - [model family, size, quantization comparison (QwenPRM)](#model-family-size-quantization-comparison-qwenprm-1) · `tbl-6557b7`
  - [cnt-mcts-bl-v02](#cnt-mcts-bl-v02)
    - [score_mode sweep: parent_blend (alpha) vs. path_decay (gamma × cpuct) (qwen-3b, QwenPRM)](#score_mode-sweep-parent_blend-alpha-vs-path_decay-gamma-cpuct-qwen-3b-qwenprm) · `tbl-249fa2`
  - [kube-mcts-bl-v01](#kube-mcts-bl-v01)
    - [model family, size, quantization comparison (QwenPRM)](#model-family-size-quantization-comparison-qwenprm-2) · `tbl-622bce`
    - [kube_c sweep × model family (QwenPRM)](#kube_c-sweep-model-family-qwenprm) · `tbl-61a2b9`
  - [kube-mcts-bl-v02](#kube-mcts-bl-v02)
    - [score_mode sweep: parent_blend (alpha) vs. path_decay (gamma × kube_c) (qwen-3b, QwenPRM)](#score_mode-sweep-parent_blend-alpha-vs-path_decay-gamma-kube_c-qwen-3b-qwenprm) · `tbl-dac772`
    - [model family, size, quantization comparison (QwenPRM, parent_blend/alpha=0.8)](#model-family-size-quantization-comparison-qwenprm-parent_blendalpha08) · `tbl-c85c90`
    - [model family, size, quantization comparison (QwenPRM, parent_blend/alpha=1.0)](#model-family-size-quantization-comparison-qwenprm-parent_blendalpha10) · `tbl-3fb9a1`
    - [model family, size, quantization comparison (QwenPRM, parent_blend/alpha=0.0)](#model-family-size-quantization-comparison-qwenprm-parent_blendalpha00) · `tbl-a55139`
    - [alpha × kube_c joint sweep (llama-3b, QwenPRM, parent_blend)](#alpha-kube_c-joint-sweep-llama-3b-qwenprm-parent_blend) · `tbl-a9e420`
    - [gamma × kube_c joint sweep (qwen-3b, QwenPRM, path_decay)](#gamma-kube_c-joint-sweep-qwen-3b-qwenprm-path_decay) · `tbl-46d9c7`
  - [kdepth-mcts-bl-v01](#kdepth-mcts-bl-v01)
    - [model family, size, quantization comparison (QwenPRM)](#model-family-size-quantization-comparison-qwenprm-3) · `tbl-d1a3ce`
    - [model family, size, quantization comparison (QwenPRM, depth_alpha=0.5)](#model-family-size-quantization-comparison-qwenprm-depth_alpha05) · `tbl-43590e`
    - [model family, size, quantization comparison (QwenPRM, depth_alpha=2.0)](#model-family-size-quantization-comparison-qwenprm-depth_alpha20) · `tbl-9d088e`
  - [kdepth-mcts-bl-v02](#kdepth-mcts-bl-v02)
    - [score_mode sweep: parent_blend (alpha) vs. path_decay (gamma) (qwen-3b, QwenPRM)](#score_mode-sweep-parent_blend-alpha-vs-path_decay-gamma-qwen-3b-qwenprm) · `tbl-1b443b`
    - [model family, size, quantization comparison (QwenPRM, parent_blend/alpha=0.8)](#model-family-size-quantization-comparison-qwenprm-parent_blendalpha08-1) · `tbl-2fe92e`
    - [model family, size, quantization comparison (QwenPRM, parent_blend/alpha=1.0)](#model-family-size-quantization-comparison-qwenprm-parent_blendalpha10-1) · `tbl-76f66a`
  - [sem-mcts-bl-v01](#sem-mcts-bl-v01)
    - [model family, size, quantization comparison (QwenPRM, lam=0.01/ds_alpha=10)](#model-family-size-quantization-comparison-qwenprm-lam001ds_alpha10-1) · `tbl-c43f9b`
    - [model family comparison (QwenPRM, lam=0.01/ds_alpha=10, max_model_len=6000)](#model-family-comparison-qwenprm-lam001ds_alpha10-max_model_len6000) · `tbl-9f7cda`
    - [model family, size, quantization comparison (QwenPRM, lam=0.01/ds_alpha=1)](#model-family-size-quantization-comparison-qwenprm-lam001ds_alpha1-1) · `tbl-369e81`
  - [sem-mcts-bl-v02](#sem-mcts-bl-v02)
    - [model family, size, quantization comparison (QwenPRM, parent_blend/alpha=1.0, lam=0.01/ds_alpha=10)](#model-family-size-quantization-comparison-qwenprm-parent_blendalpha10-lam001ds_alpha10) · `tbl-e9dbbb`
- [**Tuning tables \[gen_budget=160, 320, …\] *(future)***](#tuning-tables-gen_budget160-320-future)
  - [cnt-mcts](#cnt-mcts-1)
    - [model family comparison (b=320, QwenPRM)](#model-family-comparison-b320-qwenprm) · `tbl-867868`
  - [sem-mcts-v02](#sem-mcts-v02-1)
    - [model family comparison (b=320, QwenPRM, lam=0.01/ds_alpha=1)](#model-family-comparison-b320-qwenprm-lam001ds_alpha1) · `tbl-900e87`
    - [model family comparison (b=320, QwenPRM, lam=0.01/ds_alpha=10)](#model-family-comparison-b320-qwenprm-lam001ds_alpha10) · `tbl-01c466`

*37 tables. Regenerate with `python scripts/gen_toc.py`.*
<!-- toc:end -->

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
<!-- table-id: tbl-d6065d -->
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
<!-- table-id: tbl-afdda0 -->
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
<!-- table-id: tbl-a45ce2 -->
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

### sem-mcts-v02

#### embeds_strategy × scope sweep (QwenPRM)
<!-- table-id: tbl-666cb6 -->
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
<!-- table-id: tbl-a554c7 -->
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
<!-- table-id: tbl-591232 -->
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
<!-- table-id: tbl-a12d4f -->
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
<!-- table-id: tbl-21bde4 -->
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
<!-- table-id: tbl-e58353 -->
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
<!-- table-id: tbl-2e75f2 -->
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
<!-- table-id: tbl-ae7863 -->
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
<!-- table-id: tbl-4cc5b9 -->
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
<!-- table-id: tbl-73533c -->
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
<!-- table-id: tbl-cf8fea -->
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
<!-- table-id: tbl-6557b7 -->
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
<!-- table-id: tbl-249fa2 -->
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
> **W&B:** pb-a1.0 `uu7p59lq`, pd-g1.0-c0.5 `gx1u385h`,
> pd-g0.5-c2.0 `9k378zhj`; remaining arms' ids not recorded
> here (see result dirs).

| llm | score_mode | alpha | gamma | cpuct | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | parent_blend | 1.0 | — | 2.0 | 2 | scored | .4403<br>±.0304 | .4142<br>±.0301 | .4067<br>±.0301 | .3955<br>±.0299 | 4.28 |
| qwen-3b | parent_blend | 0.8 | — | 2.0 | 2 | scored | .4403<br>±.0304 | .3993<br>±.0300 | .3993<br>±.0300 | .3843<br>±.0298 | 4.41 |
| qwen-3b | parent_blend | 0.6 | — | 2.0 | 2 | scored | .4701<br>±.0305 | .4478<br>±.0304 | .4366<br>±.0304 | .4291<br>±.0303 | 4.39 |
| qwen-3b | path_decay | — | 1.0 | 2.0 | 2 | scored | .6493<br>±.0292 | .5522<br>±.0304 | .5522<br>±.0304 | .5410<br>±.0305 | 4.41 |
| qwen-3b | path_decay | — | 0.8 | 2.0 | 2 | scored | .6418<br>±.0293 | .5746<br>±.0303 | .5634<br>±.0304 | .5485<br>±.0305 | 4.39 |
| qwen-3b | path_decay | — | 0.5 | 2.0 | 2 | scored | .5485<br>±.0305 | .5224<br>±.0306 | .5336<br>±.0305 | .5149<br>±.0306 | 4.35 |
| qwen-3b | path_decay | — | 1.0 | 0.5 | 2 | scored | .6194<br>±.0297 | .5821<br>±.0302 | .5746<br>±.0303 | .5560<br>±.0304 | 4.31 |
| qwen-3b | path_decay | — | 0.8 | 0.5 | 2 | scored | .6082<br>±.0299 | .5821<br>±.0302 | .5709<br>±.0303 | .5560<br>±.0304 | 4.52 |
| qwen-3b | path_decay | — | 0.5 | 0.5 | 2 | scored | .5746<br>±.0303 | .5373<br>±.0305 | .5261<br>±.0306 | .5261<br>±.0306 | 4.54 |

> **Analysis.** 9/9 arms scored (final cell 2026-07-23). The
> three reads: (1) one-hop blending does NOT help — pb arms sit
> at .4403/.4403/.4701 (a0.6 nominally best, within 1 SEM);
> (2) gamma is NOT scale-dominated: path_decay wins at BOTH
> cpuct values, and cpuct=2.0 is nominally better (g1.0: .6493
> vs .6194) — so the pd-vs-pb gap is attributable to the PATH
> VALUE itself, not the exploration scale; (3) best pd arm
> (g1.0-c2.0 .6493) beats best pb arm (a0.6 .4701) by +.18
> (~6 SEM) at equal cost — **path_decay is the survivor** for
> cnt-v02; gamma ordering is monotone (1.0 > 0.8 > 0.5 at both
> scales), i.e. the plain full-path average is best and decay
> only hurts. The last cell sharpens this: the g0.5 penalty
> compounds with exploration scale (−.10 vs g1.0 at cpuct=2.0,
> −.04 at 0.5) — aggressive decay is worst exactly where the
> search branches most. pd searches deeper (depth ~11 vs ~9)
> and keeps ~1.7x more completions on the same 80-gen budget.
> **Limitations / follow-up:** ledger
> orchestration/ledgers/prm800k-level5.yaml, feeds
> `level5-cnt-bl-v02-score-mode-qwen3b`. Single model (qwen-3b);
> extend path_decay g1.0 to the 5-model grid if the v03
> decision confirms.

### kube-mcts-bl-v01

#### model family, size, quantization comparison (QwenPRM)
<!-- table-id: tbl-622bce -->
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

#### kube_c sweep × model family (QwenPRM)
<!-- table-id: tbl-61a2b9 -->
> **Compares:** sensitivity of kube-bl-v01 to the KUBE
> exploration coefficient `kube_c`, swept {0.1, 0.5, 2.0, 8.0}
> on the full 5-model/quant grid. 2.0 is the default (those 5
> cells are the exact runs of the model-family table above —
> reused, not re-run); 0.1/0.5 probe below it, 8.0 brackets
> above. Unlike the v02 alpha × kube_c joint sweep (llama-3b,
> parent_blend), this is a clean 1-D sweep — v01 has no
> score_mode/alpha knob, so kube_c is the only exploration
> scale and up/down bracketing is unaliased. Per-model read:
> whether the KUBE advantage over cnt-bl-v01 (seen at 2.0 on
> every model) is robust to the bonus scale, and whether the
> optimum shifts with model strength.
>
> **Fixed:** method=`mcts_bl_kube_v01`, prm=qwen,
> agg_strategy=`last`, kube_schedule=parent,
> kube_affordable=true, bs-4, d-20, b=80, prm_batch_size=1,
> level=5, tmpl=model-family default (native for Qwen, custom
> for Llama).
>
> ⚠️ kube_c values are NOT comparable to the v02 tables'
> (different bonus shapes across modes).
>
> **W&B:** kube_c=2.0 reused — llama-1b `gu5l0k7p`, llama-3b
> `rywqssh0`, qwen-3b `79kyy2a7`, qwen-7b gptq `tmtery3w`,
> qwen-math-1.5b `ktgdyyfx`; llama-1b sweep — c0.1 `ciaa0mnv`,
> c0.5 `bgrjt17l`, c8.0 `2nuo4pt9`; llama-3b sweep — c0.1
> `z9vhyakl`, c0.5 `25t829u4`, c8.0 `lku8g5in`; qwen-3b sweep —
> c0.1 `r9hdz8uz`, c0.5 `cjnfk9uc`, c8.0 `p50sut32`;
> qwen-math-1.5b sweep — c0.1 `bs81j93w`, c0.5 `gyhlf3hu`,
> c8.0 `tzslh57a`.

| llm | kube_c | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 0.1 | 2 | scored | .3134<br>±.0284 | .2649<br>±.0270 | .2313<br>±.0258 | .2239<br>±.0255 | 3.10 |
| llama-1b fp16 | 0.5 | 2 | scored | .3172<br>±.0285 | .2612<br>±.0269 | .2500<br>±.0265 | .2388<br>±.0261 | 3.11 |
| llama-1b fp16 | 2.0 | 2 | scored | .3060<br>±.0282 | .2612<br>±.0269 | .2463<br>±.0264 | .2276<br>±.0257 | 3.11 |
| llama-1b fp16 | 8.0 | 2 | scored | .3097<br>±.0283 | .2612<br>±.0269 | .2537<br>±.0266 | .2463<br>±.0264 | 3.14 |
| llama-3b fp16 | 0.1 | 2 | scored | .5149<br>±.0306 | .4142<br>±.0301 | .3993<br>±.0300 | .3955<br>±.0299 | 4.67 |
| llama-3b fp16 | 0.5 | 2 | scored | .4813<br>±.0306 | .4142<br>±.0301 | .3843<br>±.0298 | .3657<br>±.0295 | 4.64 |
| llama-3b fp16 | 2.0 | 2 | scored | .4851<br>±.0306 | .3918<br>±.0299 | .3769<br>±.0297 | .3731<br>±.0296 | 4.65 |
| llama-3b fp16 | 8.0 | 2 | scored | .4925<br>±.0306 | .4179<br>±.0302 | .4030<br>±.0300 | .3881<br>±.0298 | 4.75 |
| qwen-3b fp16 | 0.1 | 2 | scored | .6007<br>±.0300 | .5448<br>±.0305 | .5261<br>±.0306 | .5149<br>±.0306 | 3.95 |
| qwen-3b fp16 | 0.5 | 2 | scored | .6231<br>±.0297 | .5746<br>±.0303 | .5485<br>±.0305 | .5410<br>±.0305 | 3.95 |
| qwen-3b fp16 | 2.0 | 2 | scored | .6157<br>±.0298 | .5410<br>±.0305 | .5224<br>±.0306 | .5075<br>±.0306 | 4.10 |
| qwen-3b fp16 | 8.0 | 2 | scored | .6530<br>±.0291 | .5672<br>±.0303 | .5784<br>±.0302 | .5560<br>±.0304 | 4.08 |
| qwen-7b gptq-int4 | 0.1 | 2 | scored | .7127<br>±.0277 | .6157<br>±.0298 | .6082<br>±.0299 | .5933<br>±.0301 | 3.41 |
| qwen-7b gptq-int4 | 0.5 | 2 | scored | .7313<br>±.0271 | .6007<br>±.0300 | .5597<br>±.0304 | .5560<br>±.0304 | 3.50 |
| qwen-7b gptq-int4 | 2.0 | 2 | scored | .7164<br>±.0276 | .6157<br>±.0298 | .5858<br>±.0301 | .5746<br>±.0303 | 3.43 |
| qwen-7b gptq-int4 | 8.0 | 2 | scored | .7276<br>±.0272 | .5933<br>±.0301 | .5821<br>±.0302 | .5933<br>±.0301 | 3.43 |
| qwen-math-1.5b fp16 | 0.1 | 2 | scored | .6157<br>±.0298 | .5448<br>±.0305 | .5373<br>±.0305 | .5224<br>±.0306 | 3.24 |
| qwen-math-1.5b fp16 | 0.5 | 2 | scored | .6343<br>±.0295 | .5672<br>±.0303 | .5485<br>±.0305 | .5373<br>±.0305 | 3.24 |
| qwen-math-1.5b fp16 | 2.0 | 2 | scored | .6493<br>±.0292 | .5784<br>±.0302 | .5672<br>±.0303 | .5522<br>±.0304 | 3.25 |
| qwen-math-1.5b fp16 | 8.0 | 2 | scored | .6306<br>±.0295 | .5299<br>±.0305 | .5261<br>±.0306 | .5149<br>±.0306 | 3.28 |

> **Analysis.** 20/20 scored (2026-07-23) — the sweep is
> complete. **Every row is flat.** Across an 80× `kube_c` range
> (0.1 → 8.0), no model shows a monotone trend and no row's
> spread reaches 2 SEM:
>
> | model | pass@gb range | spread | in SEMs | argmax c |
> |---|---|---|---|---|
> | llama-1b | .3060–.3172 | .011 | 0.40 | 0.5 |
> | llama-3b | .4813–.5149 | .034 | 1.10 | 0.1 |
> | qwen-3b | .6007–.6530 | .052 | 1.77 | 8.0 |
> | qwen-7b gptq | .7127–.7313 | .019 | 0.68 | 0.5 |
> | qwen-math-1.5b | .6157–.6493 | .034 | 1.14 | 2.0 |
>
> Two features mark this as noise rather than weak structure:
> **no row is monotone** in `kube_c`, and the `argmax` lands on
> a *different* c in four of the five rows (0.5 / 0.1 / 8.0 /
> 0.5 / 2.0). A real exploration optimum would sit in a
> consistent place, or at least drift systematically with model
> strength; instead it scatters, which is what independent
> per-cell noise at n=134 × 2 trials produces.
>
> This settles the optimum-shift question the table was built
> for, and settles it more strongly than the two-endpoint
> bracket could: **the flatness is not a weak-model artifact,
> nor a mid-range one.** Insensitivity holds uniformly across a
> .31 → .73 pass@gb span — the weakest and strongest models in
> the grid, and the three in between, respond identically (i.e.
> not at all) to the exploration scale. At b=80/level-5 the KUBE
> bonus coefficient is second-order regardless of model
> strength; the value/PRM signal, not exploration tuning, is
> what binds. `c=8.0` never over-explores in any row (≤ .03 from
> the row mean everywhere), so the upper bracket is not merely
> tolerable but indistinguishable. Same verdict as the v02
> alpha × kube_c sweep below (llama-3b), now generalized across
> the family.
>
> The largest single deviation is **qwen-3b c=8.0** (.6530,
> +.030 over its row mean, ~1.25 SEM) — the only cell that
> gestures at top-end benefit. It is the expected maximum of 20
> draws under the flat hypothesis, its row is non-monotone
> around it (.6007/.6231/.6157/.6530), and the other four rows
> put c=8.0 within .006 of their means. Treated as noise; worth
> a second look only if a future qwen-3b sweep reproduces it.
>
> **Limitations / follow-up:** 2 trials/cell — SEM ~.03, so
> effects below ~.06 are invisible here; the claim is "no large
> effect," not "no effect." All cells relaunched after the
> 2026-07-22 results-disk-full incident (disk moved to /groups
> 2026-07-23). Ledger orchestration/ledgers/prm800k-level5.yaml,
> feeds `level5-kube-bl-v01-kubec-sweep-qwen`. Remaining
> follow-up: the bl_cnt-v01-vs-KUBE delta per model.

### kube-mcts-bl-v02

#### score_mode sweep: parent_blend (alpha) vs. path_decay (gamma × kube_c) (qwen-3b, QwenPRM)
<!-- table-id: tbl-dac772 -->
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

| llm | score_mode | alpha | gamma | kube_c | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | parent_blend | 1.0 | — | — | 2 | scored | .6381<br>±.0294 | .5485<br>±.0305 | .5112<br>±.0306 | .5224<br>±.0306 | — |
| qwen-3b | parent_blend | 0.8 | — | — | 2 | scored | .6194<br>±.0297 | .5299<br>±.0305 | .5224<br>±.0306 | .5037<br>±.0306 | — |
| qwen-3b | parent_blend | 0.6 | — | — | 2 | scored | .6082<br>±.0299 | .5149<br>±.0306 | .5075<br>±.0306 | .4925<br>±.0306 |  4.20 |
| qwen-3b | path_decay | — | 1.0 | 2.0 | 2 | scored | .6082<br>±.0299 | .5149<br>±.0306 | .4776<br>±.0306 | .4739<br>±.0306 |  3.97 |
| qwen-3b | path_decay | — | 0.8 | 2.0 | 2 | scored | .6157<br>±.0298 | .5261<br>±.0306 | .4925<br>±.0306 | .4701<br>±.0305 |  3.87 |
| qwen-3b | path_decay | — | 0.5 | 2.0 | 2 | scored | .6269<br>±.0296 | .5112<br>±.0306 | .5000<br>±.0306 | .4851<br>±.0306 |  4.14 |
| qwen-3b | path_decay | — | 1.0 | 0.5 | 2 | scored | .6269<br>±.0296 | .5187<br>±.0306 | .4664<br>±.0305 | .4440<br>±.0304 |  3.91 |
| qwen-3b | path_decay | — | 0.8 | 0.5 | 2 | scored | .6269<br>±.0296 | .5261<br>±.0306 | .4813<br>±.0306 | .4813<br>±.0306 | — |
| qwen-3b | path_decay | — | 0.5 | 0.5 | 2 | scored | .6231<br>±.0297 | .5187<br>±.0306 | .4925<br>±.0306 | .4888<br>±.0306 |  3.97 |

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
<!-- table-id: tbl-c85c90 -->
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
> 5/5 cells scored (2026-07-26). qwen-3b is reused from the
> score_mode sweep.
>
> **W&B:** qwen-3b `z15wgie9`; llama-1b/llama-3b/qwen-7b from the
> 2026-07-23 batch.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .3284<br>±.0287 | .2463<br>±.0264 | .2276<br>±.0257 | .2239<br>±.0255 | 3.0 |
| llama-3b fp16 | 2 | scored | .5224<br>±.0306 | .4254<br>±.0303 | .4142<br>±.0301 | .4067<br>±.0301 | 4.7 |
| qwen-3b fp16 | 2 | scored | .6194<br>±.0297 | .5299<br>±.0305 | .5224<br>±.0306 | .5037<br>±.0306 | — |
| qwen-7b gptq-int4 | 2 | scored | .7239<br>±.0274 | .6082<br>±.0299 | .5896<br>±.0301 | .5746<br>±.0303 | 3.4 |
| qwen-math-1.5b fp16 | 2 | scored | .6567<br>±.0291 | .5709<br>±.0303 | .5672<br>±.0303 | .5522<br>±.0304 | 3.4 |

> **Analysis.** 5/5 scored (2026-07-26). Ordering at b=80 is
> monotone in capability — qwen-7b .7239 > qwen-math-1.5b .6567 >
> qwen-3b .6194 > llama-3b .5224 > llama-1b .3284 pass@gb — with
> adjacent pairs separated by 1–4 SEM, so unlike the alpha×kube_c
> sweep (flat by construction, one model) this grid genuinely
> resolves. **qwen-math-1.5b beats qwen-3b at half the size and
> 72% of the cost** (3.4 vs qwen-7b's 3.4 hr/trial), repeating
> the math-post-training-beats-parameter-count result the AIME
> tables show — and its maj@gb (.5522) ties qwen-3b's .5037
> upward, making it the best accuracy-per-hour cell in the table.
> **Selection retention (maj@gb / pass@gb) is remarkably stable
> here**: 79% (qwen-7b), 84% (qwen-math), 81% (qwen-3b), 78%
> (llama-3b), 68% (llama-1b) — far tighter than the same read at
> b=320 in the
> cnt table above (38–68%). That contrast is the useful one: at
> b=80 the searcher proposes few enough candidates that the PRM
> picks well across the whole family; the retention collapse is a
> *budget-induced* problem, appearing when b=320 floods
> aggregation with near-miss leaves. Worth testing directly
> before scaling budget further on weak policies.
> **Limitations / follow-up:** 2 trials/cell (SEM ~±.03). qwen-3b
> has no hr/trial recorded (reused run, timing not captured).
> The cell-for-cell read against the cnt-mcts-bl and
> kdepth-mcts-bl model-family tables — whether /cost
> normalization helps per model — is now possible and unwritten.

#### model family, size, quantization comparison (QwenPRM, parent_blend/alpha=1.0)
<!-- table-id: tbl-3fb9a1 -->
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
<!-- table-id: tbl-a55139 -->
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
> All 5 cells scored (2026-07-22).
>
> **W&B:** llama-1b `ehm46m6l`, llama-3b `ez4boxm3`, qwen-3b
> `e0uvhi9j`, qwen-7b gptq `1nzsifou`, qwen-math-1.5b `y4wj7fe9`.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .2687<br>±.0271 | .2052<br>±.0247 | .1828<br>±.0237 | .1679<br>±.0229 | 2.98 |
| llama-3b fp16 | 2 | scored | .5000<br>±.0306 | .3657<br>±.0295 | .3433<br>±.0291 | .3284<br>±.0287 | 4.84 |
| qwen-3b fp16 | 2 | scored | .6045<br>±.0299 | .5000<br>±.0306 | .4888<br>±.0306 | .4701<br>±.0305 | 4.26 |
| qwen-7b gptq-int4 | 2 | scored | .7388<br>±.0269 | .5821<br>±.0302 | .5261<br>±.0306 | .5000<br>±.0306 | 3.54 |
| qwen-math-1.5b fp16 | 2 | scored | .6306<br>±.0295 | .5112<br>±.0306 | .5075<br>±.0306 | .4888<br>±.0306 | 3.16 |

> **Analysis.** The ablation splits by metric, not by model. On
> the answer-selection metrics the leaf q is load-bearing
> everywhere: naive@gb drops at a0.0 vs a1.0 on all 5 models
> (−.022 to −.071), and wei@gb/maj@gb likewise (only qwen-7b's
> wei@gb ties). On pass@gb the gap is model-dependent: llama-1b
> −.063 (~2 SEM, the clearest value-blindness cost), qwen-3b
> −.034 and qwen-math −.037 (~1 SEM each), llama-3b +.015
> (noise) — but qwen-7b **+.041** (.7388 vs .6978), tying the
> best bl pass@gb cell in the doc. Reading: value-blind sibling
> selection pushes the search toward uniform/visit-driven
> exploration, which can widen coverage (pass@gb) for a strong
> generator while filling the candidate pool with worse-ranked
> solutions (all selection metrics fall). So the alpha sweep is
> bounded from below only for selection quality; for raw
> coverage on strong models, a0.0 is surprisingly competitive.
> All 5 launched 2026-07-22 (llama-1b/3b + qwen-3b 09:30;
> qwen-7b + qwen-math-1.5b 11:22).
> **Limitations / follow-up:** ledger
> orchestration/ledgers/prm800k-level5.yaml, feeds
> `level5-kube-bl-v02-model-family-parent-blend-a0.0-qwen`.
> Single-alpha ablation; only worth extending if the 0.0-vs-1.0
> gap is surprisingly small.

#### alpha × kube_c joint sweep (llama-3b, QwenPRM, parent_blend)
<!-- table-id: tbl-a9e420 -->
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
> ⚠️ 8/9 cells scored (2026-07-23) — the (alpha=1.0, kube_c=2.0)
> cell is the **exact same run** as the alpha=1.0 model-family
> table's llama-3b cell (cfg-63051bb1), reused, not re-run. The
> (alpha=0.8, kube_c=2.0) cell will likewise reuse the alpha=0.8
> model-family table's llama-3b cell once that queue entry runs
> (`kube-bl-v02-l5-mf-a0.8-llama3b`). The other 7 are net-new.
> kube_c is NOT numerically comparable to cnt-v02's cpuct
> (different bonus shapes).
>
> **W&B:** (1.0, 2.0) `apcu4aqr`, (1.0, 0.5) `xb9bpt5y`,
> (1.0, 0.1) `na9zpn1u`, (0.8, 0.5) `2qd6obc0`, (0.8, 0.1)
> `q45ropnt`, (0.5, 2.0) `w19d5y18`, (0.5, 0.5) `fpit1vuv`,
> (0.5, 0.1) `eb34obcy`.

| llm | alpha | kube_c | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| llama-3b | 1.0 | 2.0 | 2 | scored | .4851<br>±.0306 | .3881<br>±.0298 | .3881<br>±.0298 | .3731<br>±.0296 | 4.67 |
| llama-3b | 1.0 | 0.5 | 2 | scored | .5037<br>±.0306 | .4328<br>±.0303 | .4104<br>±.0301 | .3881<br>±.0298 | 4.61 |
| llama-3b | 1.0 | 0.1 | 2 | scored | .4888<br>±.0306 | .4030<br>±.0300 | .3918<br>±.0299 | .3694<br>±.0295 | 4.75 |
| llama-3b | 0.8 | 2.0 | 2 | scored | .5224<br>±.0306 | .4254<br>±.0303 | .4142<br>±.0301 | .4067<br>±.0301 | 4.7 |
| llama-3b | 0.8 | 0.5 | 2 | scored | .5037<br>±.0306 | .4478<br>±.0304 | .4142<br>±.0301 | .3955<br>±.0299 | 4.62 |
| llama-3b | 0.8 | 0.1 | 2 | scored | .4776<br>±.0306 | .4179<br>±.0302 | .4179<br>±.0302 | .3993<br>±.0300 | 4.64 |
| llama-3b | 0.5 | 2.0 | 2 | scored | .4776<br>±.0306 | .4067<br>±.0301 | .3806<br>±.0297 | .3731<br>±.0296 | 4.71 |
| llama-3b | 0.5 | 0.5 | 2 | scored | .4888<br>±.0306 | .4142<br>±.0301 | .3843<br>±.0298 | .3806<br>±.0297 | 4.58 |
| llama-3b | 0.5 | 0.1 | 2 | scored | .4963<br>±.0306 | .4030<br>±.0300 | .3993<br>±.0300 | .3955<br>±.0299 | 4.61 |
| llama-3b | 0.0 | 2.0 | 2 | scored | .5000<br>±.0306 | .3657<br>±.0295 | .3433<br>±.0291 | .3284<br>±.0287 | 4.84 |
| llama-3b | 0.0 | 0.5 | 2 | scored | .4851<br>±.0306 | .3470<br>±.0291 | .3396<br>±.0290 | .3284<br>±.0287 | 4.9 |
| llama-3b | 0.0 | 0.1 | — | running | — | — | — | — | — |

> **Analysis.** 11/12 scored (updated 2026-07-26 with the
> alpha=0.8/c=2.0 and alpha=0.0/c=0.5 cells). **On pass@gb the
> grid is still FLAT** — .4776–.5224 across a 20× kube_c swing
> and the full alpha range, ~1 SEM end to end. Against the
> designed reads: (1) no kube_c main effect down the alpha=1.0
> column (.4851/.5037/.4888); (2) no alpha spread at low kube_c
> (c0.1: .4888/.4776/.4963); (3) constant-ratio pairs are equal.
>
> **But the alpha=0.0 anchor added 07-25/26 changes the verdict
> on the aggregated metrics.** With blending switched fully off,
> selection quality drops consistently while pass@gb does not:
>
> | alpha | naive@gb | maj@gb |
> |---|---|---|
> | 1.0 | .3881–.4328 | .3694–.3881 |
> | 0.8 | .4179–.4478 | .3955–.4067 |
> | 0.5 | .4030–.4142 | .3731–.3955 |
> | **0.0** | **.3470–.3657** | **.3284–.3284** |
>
> Both alpha=0.0 cells sit ~2–3 SEM below every alpha>0 cell on
> maj@gb (.3284 vs .37–.41) and below all but one on naive@gb.
> So the earlier "one-hop blending is dead" reading (written when
> the grid stopped at alpha=0.5) was **too strong**: *having*
> parent blending helps the searcher pick winners it already
> found; *how much* of it (0.5 vs 0.8 vs 1.0) is what doesn't
> matter. The pre-registered flat-everywhere rule fired on
> pass@gb alone, which is exactly the metric blind to selection.
> Keeping the model-family grids at alpha=1.0 remains right — but
> for the reason "any alpha>0 is fine", not "alpha is inert".
> The kube_c insensitivity stands, echoing the kube-v01 llama-1b
> row: at b=80/level-5 the llamas look value-noise-dominated, not
> exploration-scale-dominated. hr/trial flat (4.58–4.9).
> **Limitations / follow-up:** 2 trials/cell → SEM ~±.03, and the
> alpha=0.0 claim rests on 2 cells (the third, c=0.1, failed at
> startup 07-24 and is a requeue candidate — it would test
> whether the drop holds at low exploration too). "Flat" on
> pass@gb means no effect ≥ .06 resolvable, not zero effect.
> Single model (llama-3b). The pending (0.8, 2.0) cell cannot
> change the verdict (its row and column are already flat).
> Consequence: the 4 kube-a0.8 model-family requeues
> (`kube-bl-v02-l5-mf-a0.8-*`, inqueue) lose their motivation —
> Tuan to decide run vs. drop. Ledger:
> orchestration/ledgers/prm800k-level5.yaml
> (`kube-bl-v02-l5-ac-sweep-llama3b-*`), feeds
> `level5-kube-bl-v02-alpha-kubec-sweep-llama3b`.

#### gamma × kube_c joint sweep (qwen-3b, QwenPRM, path_decay)
<!-- table-id: tbl-46d9c7 -->
> **Compares:** the `path_decay` score_mode's two knobs jointly
> — the per-hop value decay `gamma` (rows) against the
> exploration-bonus coefficient `kube_c` (columns). The
> path_decay sibling of the alpha × kube_c parent_blend sweep
> above: same grid shape, the *other* v02 scorer. Reads whether
> gamma separates once kube_c is swept, and where the g0.5
> penalty (seen in the score_mode tables) sits across the bonus
> range.
>
> **Fixed:** method=`mcts_bl_kube_v02`, llm=qwen-3b fp16 (native
> tmpl), **score_mode=path_decay**, kube_schedule=`parent`,
> kube_affordable=true (default), prm=qwen, agg_strategy=`last`,
> bs-4, d-20, b=80, prm_batch_size=1, level=5.
>
> ⚠️ 6/12 cells scored (2026-07-23) — the kube_c∈{0.5, 2.0}
> columns at gamma∈{0.5, 0.8, 1.0} are the **exact same runs** as
> the qwen-3b score_mode sweep (`tbl-dac772`), reused, not re-run.
> The kube_c=0.1 column and the **gamma=0 row** (added 2026-07-24)
> are net-new: gamma=0 reads only the leaf's own q (no path
> decay), anchoring the bottom of the gamma axis — no existing run
> to reuse at gamma=0, so all 3 of its cells are fresh. kube_c is
> NOT numerically comparable to parent_blend's kube_c or to
> cnt-v02's cpuct (different bonus shapes: AZ form here). gamma has
> no v01 control arm (unlike parent_blend's alpha=1.0).
>
> **W&B:** (0.5, 2.0) `69mrzt65`, (0.5, 0.5) `gthxems1`,
> (0.8, 2.0) `mi5gfiwv`, (0.8, 0.5) `nvcah979`, (1.0, 2.0)
> `ugkxb2de`, (1.0, 0.5) `cs0rfgii`.

| llm | gamma | kube_c | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | 1.0 | 2.0 | 2 | scored | .6082<br>±.0299 | .5149<br>±.0306 | .4776<br>±.0306 | .4739<br>±.0306 | 3.97 |
| qwen-3b | 1.0 | 0.5 | 2 | scored | .6269<br>±.0296 | .5187<br>±.0306 | .4664<br>±.0305 | .4440<br>±.0304 | 3.91 |
| qwen-3b | 1.0 | 0.1 | — | inqueue | — | — | — | — | — |
| qwen-3b | 0.8 | 2.0 | 2 | scored | .6157<br>±.0298 | .5261<br>±.0306 | .4925<br>±.0306 | .4701<br>±.0305 | 3.87 |
| qwen-3b | 0.8 | 0.5 | 2 | scored | .6269<br>±.0296 | .5261<br>±.0306 | .4813<br>±.0306 | .4813<br>±.0306 | — |
| qwen-3b | 0.8 | 0.1 | — | running | — | — | — | — | — |
| qwen-3b | 0.5 | 2.0 | 2 | scored | .6269<br>±.0296 | .5112<br>±.0306 | .5000<br>±.0306 | .4851<br>±.0306 | 4.14 |
| qwen-3b | 0.5 | 0.5 | 2 | scored | .6231<br>±.0297 | .5187<br>±.0306 | .4925<br>±.0306 | .4888<br>±.0306 | 3.97 |
| qwen-3b | 0.5 | 0.1 | — | running | — | — | — | — | — |
| qwen-3b | 0.0 | 2.0 | — | inqueue | — | — | — | — | — |
| qwen-3b | 0.0 | 0.5 | — | running | — | — | — | — | — |
| qwen-3b | 0.0 | 0.1 | — | running | — | — | — | — | — |

> **Analysis.** 6/9 scored (2026-07-23); the kube_c=0.1 column
> is queued. Early read on the two scored columns: pass@gb is
> tight (.6082–.6269, within ~1 SEM) — gamma shows no clean
> main effect on pass@gb, but the deeper metrics hint that
> **g1.0 (no decay) is weakest on wei/maj** (.4664/.4440 at
> c0.5; .4776/.4739 at c2.0) while g0.5/g0.8 hold higher
> (~.48–.50) — consistent with the g0.5-penalty-inverts note
> from the score_mode tables, here reading as g1.0 doing worse,
> not better. The kube_c=0.1 column will test whether that
> ordering holds when exploration is starved. Contrast with the
> parent_blend sweep above (flat everywhere): path_decay's gamma
> at least *moves* the tail metrics, so it is the less-dead of
> the two v02 modes on qwen-3b.
> **Limitations / follow-up:** 2 trials/cell → SEM ~±.03; the
> gamma read is on wei/maj only (pass@gb is flat). Single model
> (qwen-3b); the parent_blend companion is llama-3b, so the two
> v02-mode tables are not same-model comparable yet. Ledger:
> orchestration/ledgers/prm800k-level5.yaml
> (`kube-bl-v02-l5-qwen3b-pd-*` + the net-new
> `kube-bl-v02-l5-pd-sweep-qwen3b-*`), feeds `tbl-46d9c7`.

### kdepth-mcts-bl-v01

#### model family, size, quantization comparison (QwenPRM)
<!-- table-id: tbl-d1a3ce -->
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
<!-- table-id: tbl-43590e -->
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
<!-- table-id: tbl-9d088e -->
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
> ⚠️ 2/5 cells scored (qwen-math-1.5b, qwen-7b gptq-int4); the
> other 3 planned.
>
> **W&B:** qwen-math-1.5b on disk as cfg-ad001285 (scored via
> prepare_scored_dataset + compute_stats_basics, 2 trials); no
> W&B run id captured. qwen-7b gptq-int4 `e0p7hhug`. Others:
> none yet.

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .3022<br>±.0281 | .2500<br>±.0265 | .2276<br>±.0257 | .2127<br>±.0250 | 3.22 |
| llama-3b fp16 | 2 | scored | .5000<br>±.0306 | .4104<br>±.0301 | .4030<br>±.0300 | .3955<br>±.0299 | 4.78 |
| qwen-3b fp16 | 2 | scored | .6082<br>±.0299 | .5224<br>±.0306 | .5224<br>±.0306 | .5075<br>±.0306 | 3.92 |
| qwen-7b gptq-int4 | 2 | scored | .7052<br>±.0279 | .6007<br>±.0300 | .5522<br>±.0304 | .5448<br>±.0305 | 3.18 |
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
<!-- table-id: tbl-1b443b -->
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

| llm | score_mode | alpha | gamma | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | parent_blend | 1.0 | — | — | running | — | — | — | — | — |
| qwen-3b | parent_blend | 0.8 | — | — | running | — | — | — | — | — |
| qwen-3b | parent_blend | 0.6 | — | — | planned | — | — | — | — | — |
| qwen-3b | path_decay | — | 1.0 | — | planned | — | — | — | — | — |
| qwen-3b | path_decay | — | 0.8 | — | planned | — | — | — | — | — |
| qwen-3b | path_decay | — | 0.5 | — | planned | — | — | — | — | — |

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
<!-- table-id: tbl-2fe92e -->
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
> ⚠️ 5/5 scored (llama-1b completed 2026-07-23). qwen-3b reuses
> the score_mode sweep's alpha=0.8 arm (cfg-414a9f81).
>
> **W&B:** llama-1b `2ucgri6p`; other cells' ids not recorded
> here (see result dirs).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | 2 | scored | .3134<br>±.0284 | .2724<br>±.0272 | .2575<br>±.0268 | .2537<br>±.0266 | 3.05 |
| llama-3b fp16 | 2 | scored | .4776<br>±.0306 | .4104<br>±.0301 | .3993<br>±.0300 | .3993<br>±.0300 | — |
| qwen-3b fp16 | 2 | scored | .6231<br>±.0297 | .5709<br>±.0303 | .5522<br>±.0304 | .5410<br>±.0305 | — |
| qwen-7b gptq-int4 | 2 | scored | .7239<br>±.0274 | .6194<br>±.0297 | .6231<br>±.0297 | .6157<br>±.0298 | — |
| qwen-math-1.5b fp16 | 2 | scored | .6679<br>±.0288 | .5933<br>±.0301 | .5896<br>±.0301 | .5746<br>±.0303 | — |

> **Analysis.** 5/5 scored (2026-07-23). Family ordering is the
> standard one (qwen-7b .7239 > qwen-math .6679 > qwen-3b .6231
> > llama-3b .4776 > llama-1b .3134) — same ranking as every
> other method's model-family table. Cell-for-cell against
> kube-v01's kube_c=2.0 column (visit bonus): kdepth-a0.8 is
> nominally ahead on 4/5 models (+.007 to +.019) but every gap
> is under 1 SEM — **depth bonus ≈ visit bonus** at this
> budget/trial count. Blend-vs-control read (vs the alpha=1.0
> table below) is only available on qwen-3b so far: a0.8 .6231
> vs a1.0 .6381 (−.015, ~0.5 SEM) — no sign the one-hop blend
> helps under a depth-shaped frontier either, consistent with
> the kube ac-sweep's flat-alpha verdict and the cnt score-mode
> sweep's weak pb arms.
> **Limitations / follow-up:** ledger
> orchestration/ledgers/prm800k-level5.yaml, feeds
> `level5-kdepth-bl-v02-model-family-parent-blend-qwen`. qwen-3b
> feeds both this table and the score_mode-sweep table. hr/trial
> missing for 4 cells (scored before the timing convention).
> The full blend-vs-control read needs the alpha=1.0 grid's
> remaining models.

#### model family, size, quantization comparison (QwenPRM, parent_blend/alpha=1.0)
<!-- table-id: tbl-76f66a -->
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

#### model family, size, quantization comparison (QwenPRM, lam=0.01/ds_alpha=10)
<!-- table-id: tbl-c43f9b -->
> **Compares:** model family, size, and quantization jointly —
> same 5-model/quant grid as cnt-mcts-bl-v01's equivalent table
> above, so a direct bl_sem-vs-bl_cnt read is possible once both
> are filled. Anchored to the **same (lam, ds_alpha) checkpoint
> as sem-mcts-v02's `lam=0.01/ds_alpha=10` table** above (not
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

#### model family comparison (QwenPRM, lam=0.01/ds_alpha=10, max_model_len=6000)
<!-- table-id: tbl-9f7cda -->
> **Compares:** the SAME 5-model/quant grid as the
> `max_model_len=5000` table directly above (`tbl-c43f9b`), but
> with the vLLM context window raised to **6000** tokens
> (`llm.max_model_len=6000`). The question is diagnostic, not a
> new tuning axis: does the extra headroom let the sem-bl-v01
> **llama** cells run to completion? At the default 5000 those
> deep-search trajectories can overflow the context and the run
> raises (the same class of failure tracked in
> `docs/decisions/context-length-overflow-guard.md`). A
> row-by-row read against the 5000 table tells whether 6000 is
> enough headroom, and — where both complete — whether the
> larger window shifts pass@gb at all.
>
> **Fixed:** method=`mcts_bl_sem_v01` (PRM embeds — prm=qwen for
> both scoring AND diversity via `embeds_source=prm`, the schema
> default), prm=qwen, agg_strategy=`last`, bs-4, d-20, b=80,
> prm_batch_size=1, `ds_alpha_schedule=global`, `cov_update=sm`,
> `embeds_proj=sparse512` (`embeds_dim=512`), ds_beta=1.0,
> tmpl=model-family default (native for Qwen, custom for Llama).
> **lam=0.01, ds_alpha=10** (`w_eff = ds_alpha/sqrt(lam) = 100`).
> **`max_model_len=6000`** (this table's whole point; hash-
> relevant, so every cell is a distinct config from the 5000
> table — nothing is shared).
>
> ⚠️ Entirely planned — no runs yet. `max_model_len` is a
> hash group field, so these 5 cells have brand-new config
> hashes and result dirs; they do NOT resume the 5000 runs.
> Higher context costs more KV-cache VRAM per step (watch the
> gptq/7b cell at `gpu_memory_utilization=0.3`).
>
> **W&B:** none yet (no runs exist).

| llm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|
| llama-1b fp16 | — | failed | — | — | — | — | — |
| llama-3b fp16 | — | failed | — | — | — | — | — |
| qwen-3b fp16 | 2 | scored | .6045<br>±.0299 | .4216<br>±.0302 | .3507<br>±.0292 | .2985<br>±.0280 | 6.5 |
| qwen-7b gptq-int4 | — | running | — | — | — | — | — |
| qwen-math-1.5b fp16 | — | failed | — | — | — | — | — |

> **Analysis.** 1/5 scored (2026-07-26) — and the diagnostic
> question this table exists to answer is **not** settled by it.
> qwen-3b (which already completed at 5000) lands at .6045
> pass@gb; the matched `tbl-c43f9b` row is the comparison to
> make when reading whether 6000 moves anything for models that
> never hit the cap. The **llama rows both failed again at
> 6000**, and the traceback (W&B `khe3unjb`, checked 2026-07-26)
> says exactly why: `prompt contains at least 6001 input tokens`
> against a 6000 limit — **over by one token**. So context length
> is confirmed as the cause and 6000 is simply not enough
> headroom; nothing exotic is wrong with the llama cells. The
> right retry is a clearly-oversized window (8000+) rather than
> another marginal bump, since 6000 was chosen to just clear the
> observed 5000 failure and the chains evidently grow past
> whatever bound you pick. That makes
> `docs/decisions/context-length-overflow-guard.md` (still
> unimplemented) the real fix: a guard that truncates or aborts
> the offending rollout beats chasing the window upward.
> qwen-math failed for the unrelated architectural reason (4096
> `max_position_embeddings` < 6000 — vLLM rejects it at engine
> construction; permanent for this table).
> **Limitations / follow-up:** 5 cells planned — see
> orchestration/ledgers/prm800k-level5.yaml group `sem-mcts-bl`,
> feeds `tbl-9f7cda`. Launch per cell:
> `generate_mcts_sem.py --config-name mcts_bl_sem_v01_prm800k
> llm=<model> prm=qwen_prm data.level=5 search.lam=0.01
> search.ds_alpha=10 llm.max_model_len=6000`.

#### model family, size, quantization comparison (QwenPRM, lam=0.01/ds_alpha=1)
<!-- table-id: tbl-369e81 -->
> **Compares:** same 5-model/quant grid as the `lam=0.01/
> ds_alpha=10` table above, at one order of magnitude lower
> effective diversity weight — the two tables together give a
> first (coarse) read on whether the model-family ranking is
> sensitive to `w_eff` for this algorithm, ahead of a proper
> `w_eff` sweep. Anchored to sem-mcts-v02's `lam=0.01/
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
<!-- table-id: tbl-e9dbbb -->
> **Compares:** model family, size, and quantization jointly for
> sem-mcts-bl-v02 at its selectable frontier value term fixed to
> `score_mode=parent_blend, alpha=1.0` (the no-blend control:
> `q_term = q(leaf)`, byte-identical to `score_mode=own`), with the
> diversity knobs pinned to the same `lam=0.01/ds_alpha=10`
> (`w_eff=100`) checkpoint as the sem-mcts-bl-v01 table above and
> sem-mcts-v02's tables — so bl_sem_v02-vs-bl_sem_v01 and, cell-
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
| qwen-3b fp16 | 2 | scored | .5933<br>±.0301 | .4179<br>±.0302 | .3470<br>±.0291 | .2910<br>±.0278 | — |
| qwen-7b gptq-int4 | 2 | scored | .7537<br>±.0264 | .5597<br>±.0304 | .4925<br>±.0306 | .4664<br>±.0305 | — |
| qwen-math-1.5b fp16 | 2 | scored | .6828<br>±.0285 | .5373<br>±.0305 | .4888<br>±.0306 | .4478<br>±.0304 | — |

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
<!-- table-id: tbl-867868 -->
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
> 4/5 scored 2026-07-26. Budget=320 is a 4× generation-count
> increase over the b=80 table; measured cost came in at
> 12–21 hr/trial (vs the ~4× projection off b=80's 3.21 hr/trial
> for qwen-7b — the scaling is worse than linear in budget for
> the llamas).
>
> **W&B:** 2026-07-24 batch, `tnguyen10/llm-reasoning`.

| llm | prm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | 2 | scored | .5672<br>±.0303 | .3246<br>±.0287 | .2463<br>±.0264 | .2164<br>±.0252 | 12.1 |
| llama-3b fp16 | qwen | 2 | scored | .7015<br>±.0280 | .4328<br>±.0303 | .4142<br>±.0301 | .3619<br>±.0294 | 20.9 |
| qwen-3b fp16 | qwen | 2 | scored | .8172<br>±.0237 | .6306<br>±.0295 | .5858<br>±.0301 | .5522<br>±.0304 | 18.2 |
| qwen-7b gptq-int4 | qwen | 2 | scored | .8433<br>±.0222 | .6381<br>±.0294 | .5858<br>±.0301 | .5746<br>±.0303 | 14.5 |
| qwen-math-1.5b fp16 | qwen | — | running | — | — | — | — | — |

> **Analysis.** 4/5 scored (2026-07-26). At b=320 the model
> ordering is clean and wide — qwen-7b .8433 ≳ qwen-3b .8172 >
> llama-3b .7015 > llama-1b .5672 pass@gb — with the two qwens
> separated from the llamas by 4–8 SEM. **The interesting gap is
> pass@gb vs maj@gb**, i.e. how much of what the search finds
> survives aggregation: qwen-7b keeps .5746/.8433 = 68% of its
> reachable answers, qwen-3b 68%, llama-3b 52%, llama-1b just
> 38%. So the weaker the policy, the more the b=320 budget buys
> leaves the PRM then fails to pick — extra budget helps the
> llamas find answers but not *commit* to them, which is a
> scoring/aggregation problem, not a search one.
> Cost is non-monotone in model size (llama-3b 20.9 hr/trial is
> the most expensive cell, above qwen-7b gptq's 14.5) — the
> gptq-int4 quantization pays for itself at this budget.
> **Limitations / follow-up:** 2 trials/cell (SEM ~±.03); the
> qwen-math cell is the `mml4096` diagnostic still in trial 2
> (see the sem tables below for the 4096-window question). A
> matched-PRM (llama) b=320 row per model is still missing, so
> budget and PRM remain confounded against the b=80 table — as
> the header warns.

### sem-mcts-v02

#### model family comparison (b=320, QwenPRM, lam=0.01/ds_alpha=1)
<!-- table-id: tbl-900e87 -->
> **Compares:** the same 5-model family/size/quantization sweep
> as the b=80 `model family, size, quantization comparison
> (QwenPRM, lam=0.01/ds_alpha=1)` table above, but at
> `search.gen_budget=320` (4× the b=80 budget). Same diversity
> point as that b=80 table — `lam=0.01, ds_alpha=1.0`
> (`w_eff = ds_alpha/sqrt(lam) = 10`) — so budget is the only
> axis that moves; paired with the `ds_alpha=10` table below
> (same budget, same lam, 10× ds_alpha) it also isolates `w_eff`
> at b=320.
>
> **Fixed:** method=`mcts_sem_v02` (PRM embeds), prm=qwen, bs-4,
> d-20, b=320, prm_batch_size=1, **`llm.max_model_len=6000`**,
> `ds_alpha_schedule=global`
> (default), `cov_update=sm`, `embeds_dim=512`/
> `embeds_proj=sparse` (defaults), tmpl=model-family default
> (native for Qwen, custom for Llama). **lam=0.01, ds_alpha=1.0**
> (`w_eff=10` — the same point used by the b=80
> `lam=0.01/ds_alpha=1` table).
>
> **Why `max_model_len=6000` (not the 5000 default):** at
> **level 5** specifically, b=320 search builds prompts that
> overflow a 5000-token window — the 2026-07-24 b=320 attempts at
> the 5000 default died at startup with `decoder prompt (5000) +
> output tokens > max_model_len 5000` (observed on cnt llama-3b;
> the sibling cells were relaunched at 6000 rather than
> individually reproduced). **Correction (2026-07-26): this is
> NOT level-5-specific.** The earlier claim here — that level-4
> b=320 completes at mml=5000, so level-5's longer chains are the
> driver — was based on level-4 cells that had only finished
> trial 1. `sem-mcts-56ae22f5` (level-4, b=320, llama-3b,
> mml=5000) then died in **trial 2** with the identical
> `decoder prompt (5000) + output > max_model_len 5000` error
> after 20h44m of clean trial-1 running (W&B `7c79wk6z`). So
> **b=320 overflows a 5000-token window at level 4 as well** —
> the failure is budget-driven and merely *probabilistic*: it
> needs one sufficiently long chain, which more trials eventually
> supply. Level-4 b=320 cells that read `scored` at mml=5000 are
> survivors, not evidence of safety. 6000 is therefore fixed
> config for this table, not a variation; `max_model_len` is
> hash-relevant, so these cells address different config hashes
> than any 5000 attempt, and no b=320/mml=5000 companion table is
> possible at level 5.
> Note b=80 tables use the 5000 default, so b=80↔b=320
> comparisons move two settings; mml only binds when prompts
> approach the cap, which b=80 does not, so the confound is mild.
>
> ⚠️ **qwen-math-1.5b is excluded from this table at mml6000:**
> its `max_position_embeddings=4096`, so vLLM rejects
> `max_model_len=6000` at engine construction — that part is
> observed and unambiguous.
>
> Whether qwen-math could instead run level-5 b=320 *at its own
> 4096 window* is **still open**. It was attempted on 2026-07-24
> (`cfg-d87ee48f`, W&B `08m3c7r9`): the model loaded, entered
> trial 0, and ran ~1 h 25 min with no context error before the
> process was killed externally (unrelated kill sweep at 07:09) —
> so that attempt neither confirms nor refutes an overflow. For
> contrast, qwen-math *completed* level-4 b=320 at mml=4096
> (`cfg-799bfbc6`, `cfg-c67e46ee`, 2/2 trials each). The
> level-5-needs->5000 evidence comes from one cell only (cnt
> llama-3b at mml=5000, W&B `qyve9h2t`, `failed` after ~23 min);
> extending it to qwen-math is inference, not measurement. One
> rerun at `llm.max_model_len=4096` would settle it.
>
> 3/5 scored 2026-07-26; llama-3b relaunched after a walltime
> timeout took its second trial.
>
> **W&B:** runs in `tnguyen10/llm-reasoning` (2026-07-24 batch).

| llm | prm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | 2 | scored | .5187<br>±.0306 | .3134<br>±.0284 | .2799<br>±.0275 | .2500<br>±.0265 | 18.5 |
| llama-3b fp16 | qwen | — | running | — | — | — | — | — |
| qwen-3b fp16 | qwen | 2 | scored | .8097<br>±.0240 | .6157<br>±.0298 | .5858<br>±.0301 | .5784<br>±.0302 | 23.4 |
| qwen-7b gptq-int4 | qwen | 2 | scored | .7985<br>±.0245 | .6418<br>±.0293 | .6269<br>±.0296 | .6157<br>±.0298 | 14.6 |
| qwen-math-1.5b fp16 | qwen | — | running (mml4096) | — | — | — | — | — |

> **Analysis.** 3/5 scored (2026-07-26). The two qwens are
> statistically tied on pass@gb (.8097 vs .7985, ≪1 SEM) and both
> ~9 SEM above llama-1b (.5187) — at b=320 with w_eff=10, model
> family dominates and the 3b-vs-7b size/quantization gap
> vanishes. **Against the matched cnt table above (tbl-867868,
> same models, same budget, same PRM), sem's advantage is in
> selection, not reach:** qwen-7b pass@gb .7985 (sem) vs .8433
> (cnt) — sem finds *fewer* correct leaves — yet maj@gb .6157
> (sem) vs .5746 (cnt). Retention climbs 68% → 77%. The
> diversity term is doing what it was designed to do: trading a
> little raw coverage for a candidate set the PRM can rank.
> Same pattern on qwen-3b (.5784 vs .5522 maj@gb) and llama-1b
> (.2500 vs .2164). Cost is the price: 14.6–23.4 hr/trial vs the
> cnt table's 12.1–18.2.
> **Limitations / follow-up:** llama-3b relaunched 2026-07-26
> (job 23419789) after its first attempt lost trial 2 to a 3-day
> allocation timeout — note its trial-1 alone took ~25h, nearly
> the 28h `expected_hr` budgeted for *both*, so this family's
> estimates need recalibrating. The qwen-math row is the open
> 4096-window question: its `mml4096` diagnostic (`cfg-06533f44`)
> has been running cleanly for 18h+ — well past the ~23-min
> context death seen at mml5000 — which increasingly suggests
> b=320 *does* fit in 4096 and the "permanently blocked" framing
> above applies only to mml6000, not to the model.

#### model family comparison (b=320, QwenPRM, lam=0.01/ds_alpha=10)
<!-- table-id: tbl-01c466 -->
> **Compares:** identical setup to the `ds_alpha=1` table above,
> at `ds_alpha=10` instead of `1.0` (10× the diversity weight,
> same `lam=0.01`) — the b=320 counterpart of the b=80
> `lam=0.01/ds_alpha=10` table, and the paired point needed to
> isolate `w_eff` alone at this budget.
>
> **Fixed:** identical to the `ds_alpha=1` table above (method=
> `mcts_sem_v02`, prm=qwen, bs-4, d-20, b=320, prm_batch_size=1,
> **`llm.max_model_len=6000`**,
> `ds_alpha_schedule=global`, `cov_update=sm`,
> `embeds_dim=512`/`embeds_proj=sparse`, tmpl=model-family
> default) except the diversity weight. **lam=0.01,
> ds_alpha=10** (`w_eff=100`).
>
> **Why `max_model_len=6000`:** same reason as the `ds_alpha=1`
> table above — at level 5, b=320 prompts overflow the 5000
> default, so 6000 is fixed config here and no mml=5000 companion
> table is possible. (Level-4 b=320 ran fine at 5000; the limit is
> level-5-specific.)
>
> ⚠️ **qwen-math-1.5b is blocked at level-5 b=320** (4096
> `max_position_embeddings` vs. the >5000 requirement) — same
> architectural conflict as the `ds_alpha=1` table; it does run at
> level-4 b=320 with mml=4096.
>
> 3/5 scored 2026-07-26; llama-3b still running.
>
> **W&B:** runs in `tnguyen10/llm-reasoning` (2026-07-24 batch).

| llm | prm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | 2 | scored | .5373<br>±.0305 | .2649<br>±.0270 | .2127<br>±.0250 | .1828<br>±.0237 | 20.0 |
| llama-3b fp16 | qwen | — | running | — | — | — | — | — |
| qwen-3b fp16 | qwen | 2 | scored | .8396<br>±.0225 | .5896<br>±.0301 | .5560<br>±.0304 | .5299<br>±.0305 | 23.8 |
| qwen-7b gptq-int4 | qwen | 2 | scored | .8582<br>±.0213 | .6007<br>±.0300 | .6007<br>±.0300 | .5821<br>±.0302 | 19.1 |
| qwen-math-1.5b fp16 | qwen | — | running (mml4096) | — | — | — | — | — |

> **Analysis.** 3/5 scored (2026-07-26). **The w_eff pairing this
> table exists for now has an answer on 3 of 5 models, and it is
> a clean split by metric.** Against the `ds_alpha=1` table
> (w_eff=10), raising diversity 10× to w_eff=100:
>
> | model | pass@gb (10 → 100) | maj@gb (10 → 100) |
> |---|---|---|
> | llama-1b | .5187 → .5373 | .2500 → **.1828** |
> | qwen-3b | .8097 → .8396 | .5784 → **.5299** |
> | qwen-7b | .7985 → .8582 | .6157 → **.5821** |
>
> Every model gains on pass@gb (+.019/+.030/+.060) and every
> model *loses* on maj@gb (−.067/−.049/−.034) — consistent in
> sign across all three, and for llama-1b the maj drop is ~2.7
> SEM. So **more diversity widens what the search reaches and
> degrades what aggregation keeps**: w_eff=100 pushes the tree
> toward semantically spread-out branches whose leaves the PRM
> then ranks worse. Note the loss shrinks as the policy gets
> stronger (−.067 llama-1b → −.034 qwen-7b), consistent with the
> retention story in the sibling tables: weak policies need the
> PRM's ranking most and are hurt most when diversity dilutes it.
> Practical read: **w_eff=10 is the better operating point at
> b=320** unless pass@gb is the target metric, and w_eff should
> arguably scale with policy strength rather than being fixed.
> Cost is ~flat-to-worse (19.1–23.8 vs 14.6–23.4 hr/trial).
> **Limitations / follow-up:** 2 trials/cell (SEM ~±.03), so
> individually only llama-1b's maj drop clears 2 SEM — the
> *consistency of sign* across 3 models is what carries the
> claim, not any single cell. llama-3b would be the fourth test
> and is still running. The qwen-math row awaits the mml4096
> question (see the `ds_alpha=1` table's note); if 4096 proves
> sufficient, both b=320 sem tables can be completed at 5/5.

---
