# LLM Reasoning — MCTS Experiment Comparison — PRM800K Level 5

> **Provenance:** structure mirrored from [exp-comp-prm800k-level4.md](exp-comp-prm800k-level4.md) (the level-4 doc) on 2026-07-10; every table reset to `planned` — no level-5 runs exist yet. Launch commands are the level-4 counterparts' plus `data.level=5` (config hashes and `--level-5--` run names follow automatically). Intro/`Fixed` prose is inherited from the level-4 doc: table definitions remain valid, but any inherited claim about completeness or findings describes level-4 state — trust the (all-planned) tables here over such prose until level-5 results land. The level-5 grid also **drops two models** relative to level 4 — llama-3b gptq and qwen-3b gptq-int4 — so inherited “7-model” grid prose reads as 5 models here (llama-1b, llama-3b fp16, qwen-3b fp16, qwen-7b gptq-int4, qwen-math-1.5b).

Central tracker for every MCTS search experiment (cnt / sem /
cnt-bl / sem-bl) on PRM800K — per-algorithm tuning tables grouped
by gen_budget, plus a cross-algorithm best-config summary.

<!-- toc:begin -- generated, do not hand-edit -->
## Contents

- [**Purpose**](#purpose)
- [**Structure and use**](#structure-and-use)
- [**Cross-algorithm summary \[gen_budget=80\] (QwenPRM)**](#cross-algorithm-summary-gen_budget80-qwenprm)
- [**Cross-algorithm summary \[gen_budget=320\] (QwenPRM)**](#cross-algorithm-summary-gen_budget320-qwenprm)
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
  - [sem-mcts-v02 \[cov_scope=local\]](#sem-mcts-v02-cov_scopelocal)
    - [lam / ds_alpha joint sweep (llama-1b)](#lam-ds_alpha-joint-sweep-llama-1b-1) · `tbl-375fa0`
    - [lam / ds_alpha joint sweep (qwen-7b gptq-int4)](#lam-ds_alpha-joint-sweep-qwen-7b-gptq-int4-1) · `tbl-898c25`
    - [lam / ds_alpha joint sweep (qwen-3b)](#lam-ds_alpha-joint-sweep-qwen-3b) · `tbl-fa65d4`
    - [lam / ds_alpha joint sweep (llama-1b, embeds_ref=relative)](#lam-ds_alpha-joint-sweep-llama-1b-embeds_refrelative) · `tbl-ba6b11`
    - [lam / ds_alpha joint sweep (llama-3b, embeds_ref=relative)](#lam-ds_alpha-joint-sweep-llama-3b-embeds_refrelative) · `tbl-cf849a`
    - [lam / ds_alpha joint sweep (qwen-3b, embeds_ref=relative)](#lam-ds_alpha-joint-sweep-qwen-3b-embeds_refrelative) · `tbl-b1cb82`
    - [lam / ds_alpha joint sweep (qwen-7b gptq-int4, embeds_ref=relative)](#lam-ds_alpha-joint-sweep-qwen-7b-gptq-int4-embeds_refrelative) · `tbl-5d64b1`
    - [lam / ds_alpha joint sweep (qwen-math-1.5b, embeds_ref=relative)](#lam-ds_alpha-joint-sweep-qwen-math-15b-embeds_refrelative) · `tbl-3a76ce`
    - [embeds_ref comparison (llama-1b, cov_scope=local)](#embeds_ref-comparison-llama-1b-cov_scopelocal) · `tbl-ea8196`
    - [embeds_ref comparison (llama-3b, cov_scope=local)](#embeds_ref-comparison-llama-3b-cov_scopelocal) · `tbl-7ee727`
    - [embeds_ref comparison (qwen-3b, cov_scope=local)](#embeds_ref-comparison-qwen-3b-cov_scopelocal) · `tbl-6ac460`
    - [embeds_ref comparison (qwen-7b gptq-int4, cov_scope=local)](#embeds_ref-comparison-qwen-7b-gptq-int4-cov_scopelocal) · `tbl-5cf136`
    - [embeds_ref comparison (qwen-math-1.5b, cov_scope=local)](#embeds_ref-comparison-qwen-math-15b-cov_scopelocal) · `tbl-78da65`
    - [model family comparison (QwenPRM, cov_scope=local)](#model-family-comparison-qwenprm-cov_scopelocal) · `tbl-bf15ee`
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
    - [model family comparison (b=320, QwenPRM, lam=0.01/ds_alpha=1, embeds_center_mode=local)](#model-family-comparison-b320-qwenprm-lam001ds_alpha1-embeds_center_modelocal) · `tbl-6a015e`
    - [model family comparison (b=320, QwenPRM, lam=0.01/ds_alpha=10, embeds_center_mode=local)](#model-family-comparison-b320-qwenprm-lam001ds_alpha10-embeds_center_modelocal) · `tbl-560ce2`

*53 tables. Regenerate with `python scripts/gen_toc.py`.*
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

## Cross-algorithm summary [gen_budget=80] (QwenPRM)
> One table per model, one row per algorithm, matching the
> `docs/exp-comp-aime2025.md` summaries. Each row is the best
> config for that (algorithm, model) pair, picked across **all**
> of that variant's tuning knobs jointly by pass@gb, ties broken
> naive → wei → maj. Every cell is **2 trials**, `scored` (the two
> constant columns were dropped; a row of em-dashes means
> `planned`). Fixed: b=80, bs-4, d-20, agg_strategy=`last`,
> tmpl=model-family default (native for Qwen, custom for Llama),
> prm=qwen, data.level=5.
>
> Sources: `cnt-mcts` = method `mcts_cnt_v01` (`tbl-afdda0`, the
> only cnt-mcts entry point at this level). `sem-mcts-v02` = the
> global-scope section, pooling its lam × ds_alpha sweeps with the
> `embeds_center_mode` arms. `sem-mcts-v02 (local)` = the whole
> `### sem-mcts-v02 [cov_scope=local]` section, pooling the
> `embeds_ref` absolute and relative sweeps.
>
> Search-cost columns (mean ± SEM over questions × trials, from
> `compute_stats` / W&B `eval/*` — see `utils/metrics.py`
> `_eval_question`): `ncomps` = completed solutions per question
> (`len(completions)`); `depth` = mean depth of those completions
> (`comp_depth`); `nphases` = the phase index the search ended on
> (`q_last_phase`); `ndepths` = mean per-phase depth
> (`phase_depths`). `ncomps` is the one to read first — it is how
> much of the generation budget actually became a usable solution.
>
> ⚠️ The previous six-algorithm snapshot (adding `cnt-mcts-bl-v01`,
> `kube-mcts-bl-v01`, `kdepth-mcts-bl-v01`, `sem-mcts-bl-v01`) is
> commented out directly below, not deleted. Those families are
> unchanged; they are simply out of scope for this cut. See
> `docs/decisions/bl-kube-bonus-schedule.md`,
> `kube-affordability-restriction.md` and
> `docs/decisions/bl-kdepth-knapsack-bonus.md` for the algorithms.

<!-- TEMPORARILY COMMENTED OUT 2026-08-06: six-algorithm
     best-available snapshot (cnt/sem/cnt-bl/kube-bl/kdepth-bl/
     sem-bl). Superseded below by the three-row cnt-mcts /
     sem-mcts-v02 / sem-mcts-v02 (local) tables, matching the
     AIME2025 doc. To restore: uncomment and replace U+2011
     (non-breaking hyphen) with ASCII hyphens in the separator
     rows; they were swapped only because a doubled ASCII hyphen
     cannot appear inside an HTML comment.
**llama-1b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|‑‑‑|‑‑‑|‑‑‑|‑‑‑|‑‑‑|‑‑‑|‑‑‑|‑‑‑|
| cnt-mcts | 2 | scored | .3619<br>±.0294 | .2724 | .2127 | .1903 | 2.98 |
| sem-mcts | 2 | scored | .3433<br>±.0291 | .2537 | .1978 | .1679 | 4.85 |
| cnt-mcts-bl-v01 | 2 | scored | .2313<br>±.0258 | .2090 | .1940 | .1940 | 2.74 |
| kube-mcts-bl-v01 | 2 | scored | .3060<br>±.0282 | .2612 | .2463 | .2276 | 3.11 |
| kdepth-mcts-bl-v01 | — | running | — | — | — | — | — |
| sem-mcts-bl-v01 | — | running | — | — | — | — | — |

**llama-3b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|‑‑‑|‑‑‑|‑‑‑|‑‑‑|‑‑‑|‑‑‑|‑‑‑|‑‑‑|
| cnt-mcts | 2 | scored | .5522<br>±.0304 | .4291 | .4104 | .3619 | 5.13 |
| sem-mcts | 2 | scored | .5784<br>±.0302 | .4403 | .4291 | .3881 | 6.93 |
| cnt-mcts-bl-v01 | 2 | scored | .3731<br>±.0296 | .3209 | .3321 | .3209 | 4.76 |
| kube-mcts-bl-v01 | 2 | scored | .4851<br>±.0306 | .3918 | .3769 | .3731 | — |
| kdepth-mcts-bl-v01 | 2 | scored | .5000<br>±.0306 | .4104 | .4030 | .3955 | — |
| sem-mcts-bl-v01 | — | running | — | — | — | — | — |

**qwen-3b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|‑‑‑|‑‑‑|‑‑‑|‑‑‑|‑‑‑|‑‑‑|‑‑‑|‑‑‑|
| cnt-mcts | 2 | scored | .6978<br>±.0281 | .5896 | .5896 | .5410 | 4.63 |
| sem-mcts | 2 | scored | .6903<br>±.0283 | .5784 | .5597 | .5373 | 6.20 |
| cnt-mcts-bl-v01 | — | running | — | — | — | — | — |
| kube-mcts-bl-v01 | 2 | scored | .6157<br>±.0298 | .5410 | .5224 | .5075 | 4.10 |
| kdepth-mcts-bl-v01 | — | running | — | — | — | — | — |
| sem-mcts-bl-v01 | — | running | — | — | — | — | — |

**qwen-7b gptq-int4**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|‑‑‑|‑‑‑|‑‑‑|‑‑‑|‑‑‑|‑‑‑|‑‑‑|‑‑‑|
| cnt-mcts | 2 | scored | .7537<br>±.0264 | .6157 | .5784 | .5634 | 4.19 |
| sem-mcts | 2 | scored | .7873<br>±.0250 | .6045 | .5634 | .5634 | 5.54 |
| cnt-mcts-bl-v01 | 2 | scored | .6343<br>±.0295 | .5709 | .5672 | .5522 | 3.97 |
| kube-mcts-bl-v01 | 2 | scored | .7164<br>±.0276 | .6157 | .5858 | .5746 | — |
| kdepth-mcts-bl-v01 | — | running | — | — | — | — | — |
| sem-mcts-bl-v01 | 2 | scored | .7537<br>±.0264 | .5597 | .5037 | .4478 | — |

**qwen-math-1.5b fp16**

| algorithm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|‑‑‑|‑‑‑|‑‑‑|‑‑‑|‑‑‑|‑‑‑|‑‑‑|‑‑‑|
| cnt-mcts | 2 | scored | .7575<br>±.0262 | .6418 | .6455 | .6269 | 3.37 |
| sem-mcts | 2 | scored | .7500<br>±.0265 | .6343 | .6157 | .6007 | 4.79 |
| cnt-mcts-bl-v01 | 2 | scored | .4366<br>±.0304 | .4142 | .4104 | .3955 | 3.31 |
| kube-mcts-bl-v01 | 2 | scored | .6493<br>±.0292 | .5784 | .5672 | .5522 | — |
| kdepth-mcts-bl-v01 | 2 | scored | .6455<br>±.0293 | .5522 | .5485 | .5336 | — |
| sem-mcts-bl-v01 | 2 | scored | .6567<br>±.0291 | .5410 | .4627 | .4552 | — |
-->
**llama-1b fp16**

| algorithm | pass@gb | naive@gb | wei@gb | maj@gb | ncomps | depth | nphases | ndepths | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| cnt-mcts | .3619<br>±.0294 | .2724<br>±.0272 | .2127<br>±.0250 | .1903<br>±.0240 | 10.8<br>±0.5 | 10.5<br>±0.2 | 36.0<br>±9.8 | 11.5<br>±0.2 | 2.98 |
| sem-mcts-v02 | .3806<br>±.0297 | .2724<br>±.0272 | .2575<br>±.0268 | .2201<br>±.0254 | 10.8<br>±0.5 | 11.0<br>±0.2 | 50.5<br>±12.2 | 12.1<br>±0.3 | 4.86 |
| sem-mcts-v02 (local) | .3731<br>±.0296 | .2873<br>±.0277 | .2537<br>±.0266 | .1903<br>±.0240 | 11.2<br>±0.6 | 10.4<br>±0.2 | 54.0<br>±12.0 | 11.6<br>±0.3 | 4.88 |

**llama-3b fp16**

| algorithm | pass@gb | naive@gb | wei@gb | maj@gb | ncomps | depth | nphases | ndepths | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| cnt-mcts | .5522<br>±.0304 | .4291<br>±.0303 | .4104<br>±.0301 | .3619<br>±.0294 | 21.0<br>±1.0 | 9.0<br>±0.2 | 48.0<br>±11.1 | 9.9<br>±0.2 | 5.13 |
| sem-mcts-v02 | .5896<br>±.0301 | .4366<br>±.0304 | .4179<br>±.0302 | .3955<br>±.0299 | 21.8<br>±1.0 | 9.2<br>±0.2 | 80.6<br>±15.3 | 9.8<br>±0.3 | 6.85 |
| sem-mcts-v02 (local) | .5821<br>±.0302 | .4291<br>±.0303 | .3918<br>±.0299 | .3731<br>±.0296 | 21.5<br>±1.0 | 9.2<br>±0.2 | 81.7<br>±15.2 | 9.7<br>±0.3 | 6.85 |

**qwen-3b fp16**

| algorithm | pass@gb | naive@gb | wei@gb | maj@gb | ncomps | depth | nphases | ndepths | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| cnt-mcts | .6978<br>±.0281 | .5896<br>±.0301 | .5896<br>±.0301 | .5410<br>±.0305 | 21.1<br>±1.0 | 10.1<br>±0.2 | 10.1<br>±0.7 | 10.9<br>±0.2 | 4.63 |
| sem-mcts-v02 | .6978<br>±.0281 | .5634<br>±.0304 | .5336<br>±.0305 | .5112<br>±.0306 | 21.8<br>±1.0 | 10.4<br>±0.2 | 13.5<br>±3.7 | 11.1<br>±0.2 | 6.33 |
| sem-mcts-v02 (local) | .7164<br>±.0276 | .5821<br>±.0302 | .5634<br>±.0304 | .5410<br>±.0305 | 22.8<br>±1.0 | 10.3<br>±0.2 | 30.7<br>±7.8 | 10.8<br>±0.2 | 6.21 |

**qwen-7b gptq-int4**

| algorithm | pass@gb | naive@gb | wei@gb | maj@gb | ncomps | depth | nphases | ndepths | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| cnt-mcts | .7537<br>±.0264 | .6157<br>±.0298 | .5784<br>±.0302 | .5634<br>±.0304 | 33.3<br>±1.1 | 7.0<br>±0.2 | 75.5<br>±13.8 | 7.1<br>±0.2 | 4.19 |
| sem-mcts-v02 | .7873<br>±.0250 | .6045<br>±.0299 | .5634<br>±.0304 | .5634<br>±.0304 | 36.2<br>±1.2 | 7.3<br>±0.2 | 101.8<br>±14.8 | 7.1<br>±0.2 | 5.54 |
| sem-mcts-v02 (local) | .7836<br>±.0252 | .6045<br>±.0299 | .5821<br>±.0302 | .5821<br>±.0302 | 34.8<br>±1.1 | 7.0<br>±0.1 | 222.6<br>±23.8 | 6.9<br>±0.2 | 5.54 |

**qwen-math-1.5b fp16**

| algorithm | pass@gb | naive@gb | wei@gb | maj@gb | ncomps | depth | nphases | ndepths | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| cnt-mcts | .7575<br>±.0262 | .6418<br>±.0293 | .6455<br>±.0293 | .6269<br>±.0296 | 16.5<br>±0.8 | 10.7<br>±0.2 | 9.9<br>±1.7 | 11.7<br>±0.2 | 3.37 |
| sem-mcts-v02 | .7500<br>±.0265 | .6418<br>±.0293 | .6082<br>±.0299 | .6045<br>±.0299 | 17.5<br>±1.0 | 10.6<br>±0.2 | 25.9<br>±7.7 | 11.6<br>±0.2 | 4.85 |
| sem-mcts-v02 (local) | .7612<br>±.0261 | .6269<br>±.0296 | .6194<br>±.0297 | .5821<br>±.0302 | 17.2<br>±0.8 | 10.6<br>±0.2 | 8.4<br>±0.4 | 11.5<br>±0.2 | 4.82 |


> **Analysis.** Promoted configs. `sem-mcts-v02`: llama-1b, qwen-3b
> and qwen-math-1.5b `lam=0.01, ds_alpha=1, embeds_center=local`
> (`tbl-e58353`); llama-3b `lam=0.01, ds_alpha=10, center=local`
> (`tbl-2e75f2`); qwen-7b gptq-int4 `lam=0.01, ds_alpha=10`
> (`tbl-21bde4`). `sem-mcts-v02 (local)` (all `lam=0.01`,
> `embeds_ref=relative`): llama-1b and llama-3b `ds_alpha=0.3`
> (`tbl-ba6b11`, `tbl-cf849a`); qwen-3b and qwen-7b gptq-int4
> `ds_alpha=0.1` (`tbl-b1cb82`, `tbl-5d64b1`); qwen-math-1.5b
> `ds_alpha=1.0` (`tbl-3a76ce`).
>
> **A sem-mcts variant beats cnt-mcts on all five models** —
> llama-1b +.0187, llama-3b +.0374, qwen-3b +.0186, qwen-7b
> gptq-int4 +.0336, qwen-math-1.5b +.0037. No single gap clears
> 1.3 SEM (±.026-.031 at n≈267 question-trials), so none is
> individually significant; but five out of five in the same
> direction is a sign test at p≈0.03. Treat that as the finding,
> not any one row. Caveat: the five models share one PRM and one
> question set, so the five signs are not fully independent.
>
> Global takes three models (llama-1b, llama-3b, qwen-7b), local
> two (qwen-3b, qwen-math-1.5b), and every global winner uses
> `embeds_center=local` rather than the plain lam × ds_alpha grid.
> Note the contrast with `docs/exp-comp-aime2025.md`, where local
> beat global on 4/4 models at b=320 but the two traded wins at
> b=80. Level 5 is a b=80 cut and lands mixed, which is what the
> budget-dependence reading of that result predicts.
>
> `ncomps` does **not** show the pathology seen at AIME b=320
> (where global completed ~45 % fewer solutions). Here all three
> methods complete within ~1 SEM of each other on every model
> (e.g. qwen-7b 33.3 / 36.2 / 34.8), so the budget is being
> converted comparably and the accuracy differences are not a
> completion-count artifact. `nphases` still separates them —
> local runs 222.6 on qwen-7b against cnt-mcts 75.5 — but at
> b=80 that extra phase count does not cost completions.
> **Limitations / follow-up:** 2 trials only; every conclusion
> above is one replication deep. hr/trial shows sem costing
> 30-60 % more than cnt-mcts on every model, unchanged from the
> previous snapshot. The four `*-bl-v01` families are commented
> out above, not retired — restore that block if the bl
> comparison is wanted again.

---

## Cross-algorithm summary [gen_budget=320] (QwenPRM)
> Same construction as the b=80 summary above, at
> `gen_budget=320`. Sources: `cnt-mcts` from `tbl-867868`;
> `sem-mcts-v02` pooling the four b=320 model-family tables
> (`tbl-900e87` ds_alpha=1, `tbl-01c466` ds_alpha=10, `tbl-6a015e`
> and `tbl-560ce2` the same two with `embeds_center_mode=local`),
> all at `lam=0.01`. Every cell is **2 trials**, `scored` (the two
> constant columns were dropped; a row of em-dashes means
> `planned`). Cost columns as defined in the b=80 preamble.
>
> ⚠️ `sem-mcts-v02 (local)` is `planned` on every model: there is
> **no `cov_scope=local` run at b=320 for level 5**. The two-way
> cnt-vs-global comparison below is therefore not the same
> comparison as the b=80 tables or the AIME b=320 tables, and in
> particular it cannot speak to the local-beats-global-at-large-
> budget reading. Filling this row needs a b=320 ds_alpha sweep
> with `search.cov_scope=local search.embeds_ref=relative`.
>
> ⚠️ qwen-math-1.5b ran at `max_model_len=4096` where the other
> four models used 6000 — both its rows, so the within-model
> comparison holds, but do not read its absolute numbers against
> the other models.

**llama-1b fp16**

| algorithm | pass@gb | naive@gb | wei@gb | maj@gb | ncomps | depth | nphases | ndepths | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| cnt-mcts | .5672<br>±.0303 | .3246<br>±.0287 | .2463<br>±.0264 | .2164<br>±.0252 | 52.4<br>±2.7 | 10.8<br>±0.2 | 77.0<br>±10.4 | 11.8<br>±0.3 | 12.1 |
| sem-mcts-v02 | .5373<br>±.0305 | .2649<br>±.0270 | .2127<br>±.0250 | .1828<br>±.0237 | 54.2<br>±2.6 | 10.9<br>±0.2 | 107.6<br>±13.4 | 11.6<br>±0.3 | 20.0 |
| sem-mcts-v02 (local) | — | — | — | — | — | — | — | — | — |

**llama-3b fp16**

| algorithm | pass@gb | naive@gb | wei@gb | maj@gb | ncomps | depth | nphases | ndepths | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| cnt-mcts | .7015<br>±.0280 | .4328<br>±.0303 | .4142<br>±.0301 | .3619<br>±.0294 | 107.6<br>±4.9 | 9.4<br>±0.2 | 120.1<br>±13.0 | 9.7<br>±0.3 | 20.9 |
| sem-mcts-v02 | .7201<br>±.0275 | .4590<br>±.0305 | .3955<br>±.0299 | .3545<br>±.0293 | 106.8<br>±5.7 | 9.3<br>±0.2 | 177.2<br>±18.7 | 10.0<br>±0.3 | 27.67 |
| sem-mcts-v02 (local) | — | — | — | — | — | — | — | — | — |

**qwen-3b fp16**

| algorithm | pass@gb | naive@gb | wei@gb | maj@gb | ncomps | depth | nphases | ndepths | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| cnt-mcts | .8172<br>±.0237 | .6306<br>±.0295 | .5858<br>±.0301 | .5522<br>±.0304 | 107.7<br>±4.1 | 9.9<br>±0.2 | 62.6<br>±6.6 | 10.6<br>±0.2 | 18.2 |
| sem-mcts-v02 | .8396<br>±.0225 | .5896<br>±.0301 | .5560<br>±.0304 | .5299<br>±.0305 | 96.7<br>±3.6 | 10.9<br>±0.1 | 77.7<br>±8.8 | 11.5<br>±0.2 | 23.8 |
| sem-mcts-v02 (local) | — | — | — | — | — | — | — | — | — |

**qwen-7b gptq-int4**

| algorithm | pass@gb | naive@gb | wei@gb | maj@gb | ncomps | depth | nphases | ndepths | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| cnt-mcts | .8433<br>±.0222 | .6381<br>±.0294 | .5858<br>±.0301 | .5746<br>±.0303 | 175.1<br>±5.9 | 6.9<br>±0.1 | 286.7<br>±21.3 | 6.7<br>±0.1 | 14.5 |
| sem-mcts-v02 | .8694<br>±.0206 | .6119<br>±.0298 | .5672<br>±.0303 | .5522<br>±.0304 | 168.2<br>±5.9 | 7.1<br>±0.1 | 386.1<br>±24.3 | 6.8<br>±0.2 | 19.00 |
| sem-mcts-v02 (local) | — | — | — | — | — | — | — | — | — |

**qwen-math-1.5b fp16 ⚠ mml=4096**

| algorithm | pass@gb | naive@gb | wei@gb | maj@gb | ncomps | depth | nphases | ndepths | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| cnt-mcts | .8470<br>±.0220 | .6940<br>±.0282 | .6567<br>±.0291 | .6269<br>±.0296 | 86.0<br>±3.6 | 10.5<br>±0.2 | 56.2<br>±6.4 | 11.5<br>±0.2 | 13.48 |
| sem-mcts-v02 | .8507<br>±.0218 | .6455<br>±.0293 | .6381<br>±.0294 | .6269<br>±.0296 | 96.4<br>±3.7 | 10.6<br>±0.2 | 100.3<br>±12.1 | 11.3<br>±0.2 | 19.49 |
| sem-mcts-v02 (local) | — | — | — | — | — | — | — | — | — |

> **Analysis.** Promoted `sem-mcts-v02` configs (all `lam=0.01`):
> llama-1b and qwen-3b `ds_alpha=10` (`tbl-01c466`); llama-3b and
> qwen-7b gptq-int4 `ds_alpha=10, embeds_center=local`
> (`tbl-560ce2`); qwen-math-1.5b `ds_alpha=1, center=local`
> (`tbl-6a015e`).
>
> sem-mcts-v02 beats cnt-mcts on four of five models (llama-3b
> +.0186, qwen-3b +.0224, qwen-7b gptq-int4 +.0261,
> qwen-math-1.5b +.0037) and **loses on llama-1b** (.5373 vs
> .5672, −.0299, the largest single gap in the table). That is
> the reverse of the b=80 cut, where llama-1b was one of the
> models sem won. Four of five in one direction is not
> significant on its own (sign test p≈0.19), and no individual
> gap clears 1.2 SEM.
>
> `ncomps` shows no completion collapse: within ~1 SEM on four
> models, with qwen-3b the one exception (96.7 vs 107.7, ~2 SEM
> fewer completions for a *higher* pass@gb). `nphases` runs
> 1.2-1.5× higher for sem on every model, consistent with the
> b=80 picture and far from the 8-10× blowup seen on AIME b=320.
> Cost: sem is 31-66 % slower per trial, and the gap widens with
> budget (b=80 was 30-60 %).
> **Limitations / follow-up:** 2 trials; the missing
> `cov_scope=local` arm is the important gap — until it is run,
> this table cannot be compared like-for-like with either the
> level-5 b=80 summary or the AIME b=320 summary.

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

### sem-mcts-v02 [cov_scope=local]

> **Same implementation, one flag.** Everything in this section
> runs `core/mcts_sem_search_v02_00_00.py` — the identical file
> the `sem-mcts-v02` section above uses — with
> `search.cov_scope=local`. There is no separate method, no
> separate `config_root`, and no separate launcher; the ledger
> `method` stays `mcts_sem_v02` and the cells differ from their
> global twins only by that override (and therefore by
> `config_hash`). The section exists because **local scope needs
> its own hyperparameter tuning**, not because it is a different
> algorithm.
>
> **What the flag does.** Global scope keeps one covariance `V`
> for the whole tree: every selection anywhere folds into it,
> every bonus reads it. Local scope gives each node its own `V`
> over the children *that node* has selected, so sibling
> subtrees never see each other's folds. The bonus at node `n`
> becomes "which child points somewhere `n` has not committed to
> yet?" instead of "...somewhere the entire search has not
> visited?".
>
> ⚠️ **The global operating point does not transfer — do not
> reuse the `w_eff` grid from the section above.** With
> L2-normalized embeddings and `V = lam*I + sum u u^T`, a
> direction already covered `k` times scores `~1/sqrt(lam + k)`.
> Under **global**, `k` is the run's *total* selections —
> hundreds by mid-run — so the bonus decays far below its
> nominal weight `w_eff = ds_alpha/sqrt(lam)`, which is only the
> value at the ridge init. Under **local**, `k` is the node's
> own fold count — typically 1–5, tens at the root — so the
> bonus stays *near* `w_eff` for the whole run. Concretely at
> `lam=0.01`: `1/sqrt(3.01) = .576` locally against
> `1/sqrt(300) = .058` globally, a **10x stronger diversity
> push for the same `ds_alpha`**.
>
> The prediction that follows: local's optimum should sit near
> **`w_eff ≈ 1`**, roughly an order of magnitude below global's
> measured optimum of `w_eff = 10`. Sweeping local on the global
> grid would put every cell at or above the predicted optimum
> and make local look uniformly over-diversified — a
> measurement artifact, not a result. The sweep below is
> therefore shifted down and denser at the low end.
>
> **`lam` is swept at one value only.** The global joint sweeps
> (`tbl-a554c7` and siblings) found no independent `lam` effect
> at this level once `w_eff` is held fixed — llama-1b `w_eff=10`
> gave .3582 / .3433 / .3209 across `lam` 1.0 / 0.1 / 0.01, all
> inside one SEM. That finding is inherited here, which cuts the
> sweep from 22 cells to 6. If the local results look
> `lam`-sensitive in a way the global ones didn't, that
> assumption is the first thing to revisit.

#### lam / ds_alpha joint sweep (llama-1b)
<!-- table-id: tbl-375fa0 -->
> **Compares:** the effective diversity weight
> `w_eff = ds_alpha/sqrt(lam)` under per-node covariance, to
> locate local scope's operating point. Direct counterpart of
> the global `lam / ds_alpha joint sweep (llama-1b)`
> (`tbl-a554c7`) above, at the same `lam=0.01` column, so every
> row here pairs with a global row at the same nominal `w_eff`.
>
> ⚠️ **Named for its global counterpart, but `lam` is held at
> 0.01** — this is the `lam=0.01` column of a joint sweep, not
> the full 3x grid. See the section preamble for why (the
> global sweeps found no independent `lam` effect at fixed
> `w_eff`, which cuts 22 cells to 6). Adding the `lam=1.0` and
> `lam=0.1` columns would make the title literal at 3x the
> cost; do that only if the local results look `lam`-sensitive.
>
> Grid shifted down per the section note: `w_eff` ∈ {0, 0.1,
> 0.3, 1, 3, 10}. The `w_eff=0` row is the no-diversity
> baseline and is **scope-independent** — with `ds_alpha=0` the
> covariance is multiplied by zero, so local and global must
> agree exactly (asserted in
> `unittests/check_cov_scope_embeds_ref.py`). It is included as
> a correctness anchor: if that row differs from its global
> twin, something is wrong with the plumbing, not the ablation.
>
> **Fixed:** method=`mcts_sem_v02`, **`cov_scope=local`**,
> `embeds_ref=absolute`, prm=qwen, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, cov_dtype=fp64, ds_beta=1.0,
> prm_batch_size=1, llm=llama-1b, **lam=0.01**, data.level=5,
> run.num_trials=2.
>
> ✅ All 6 cells `scored` as of 2026-08-02.
> Hashes resolved: `w_eff=0` `419d1d2d`, `0.1` `34e3e90a`,
> `0.3` `abef6586`, `1` `3dc685e4`, `3` `c3a4212b`, `10`
> `a941cc35`.
>
> **W&B:** mx11pexl (`w_eff=0.3`), 2xfjs5bq (`w_eff=1`),
> qpsupawe (`w_eff=3`), axf4tdpb (`w_eff=10`).

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-1b | qwen | 0.01 | 0 | 0 | 2 | scored | .2724<br>±.0272 | .2425<br>±.0262 | .2201<br>±.0254 | .1754<br>±.0233 | 4.44 |
| llama-1b | qwen | 0.01 | 0.01 | 0.1 | 2 | scored | .3358<br>±.0289 | .2575<br>±.0268 | .2537<br>±.0266 | .2127<br>±.0250 | 4.73 |
| llama-1b | qwen | 0.01 | 0.03 | 0.3 | 2 | scored | .3358<br>±.0289 | .2649<br>±.0270 | .2425<br>±.0262 | .2239<br>±.0255 | 4.83 |
| llama-1b | qwen | 0.01 | 0.1 | 1 | 2 | scored | .3284<br>±.0287 | .2575<br>±.0268 | .2276<br>±.0257 | .2015<br>±.0245 | 4.73 |
| llama-1b | qwen | 0.01 | 0.3 | 3 | 2 | scored | .3582<br>±.0293 | .2724<br>±.0272 | .2612<br>±.0269 | .1940<br>±.0242 | 4.70 |
| llama-1b | qwen | 0.01 | 1.0 | 10 | 2 | scored | .3694<br>±.0295 | .2799<br>±.0275 | .2127<br>±.0250 | .1754<br>±.0233 | 4.90 |

> **Analysis.** Complete (6/6, closed 2026-08-02). pass@gb rises
> across the sweep, .2724 at `w_eff=0` to **.3694 at `w_eff=10`**
> — a +.097 span, the widest of any level-5 local sweep, though
> only ~2.4 SE at these error bars. The interior is not clean:
> .3358 at both 0.1 and 0.3, a dip to .3284 at 1, then .3582 and
> .3694. Treat the 0.1–1 stretch as flat, not as structure.
> Read against `tbl-a554c7`'s `lam=0.01` rows — the same nominal
> `w_eff` under global scope: .2985 (`w_eff=1`), .3321 (`3`),
> .3209 (`10`), .3433 (`100`), .3769 (`1000`) — local scope
> leads at every shared point, by +.0485 at `w_eff=10`, and
> global needs two more decades of `w_eff` to reach a comparable
> .3769. That is the case for local scope on this model.
> **The prediction this table was built to test failed.** The
> `embeds_ref` and model-family tables below were authored at a
> predicted optimum of `w_eff=1`, which is the sweep's *dip*.
> The measured optimum is `w_eff=10`, an order of magnitude
> higher.
> **Limitations / follow-up:** every gap here is 1–2.4 SE at
> n≈267 pooled over 2 trials, and the SEs are unpaired, so the
> ordering is suggestive rather than established — the same
> questions are graded in both arms, so the true paired error is
> smaller than quoted, but by an unmeasured amount. The sweep
> also never turns over: `w_eff=10` is the largest point tested
> and is the maximum, so the optimum may lie beyond it (the
> global arm keeps climbing to `w_eff=1000`). Extending to
> `w_eff` ∈ {100, 1000} under local scope is the natural
> follow-up, and re-pointing the downstream tables off `w_eff=1`
> should wait for it.

#### lam / ds_alpha joint sweep (qwen-7b gptq-int4)
<!-- table-id: tbl-898c25 -->
> **Compares:** the same `w_eff` grid as the llama-1b sweep
> above (`tbl-375fa0`), on the strongest model in the level-5
> family grid. Counterpart of the global `lam / ds_alpha joint
> sweep (qwen-7b gptq-int4)` (`tbl-21bde4`) in the section
> above.
>
> **Why this model second.** llama-1b and qwen-7b gptq-int4
> bracket the policy-strength range of the whole grid — .3209
> against .7687 pass@gb at the global `w_eff=10` point. Running
> the sweep at both ends tests whether **local scope's optimum
> depends on policy strength**, which is not a neutral question:
> the b=320 `w_eff` pairing found that raising diversity costs
> maj@gb, and that the cost *shrinks* as the policy strengthens
> (−.067 llama-1b, −.049 qwen-3b, −.034 qwen-7b). If that
> pattern carries over, qwen-7b should tolerate a higher
> `w_eff` than llama-1b under local scope, and a single
> section-wide operating point would be the wrong abstraction.
>
> ⚠️ **Named for its global counterpart, but `lam` is held at
> 0.01** — same as the llama-1b table; see the section preamble.
>
> ⚠️ **The global twin's low end is unmeasured.** That table has
> `lam=0.01` scored only at `w_eff` 10 / 100 / 1000 (.7687 /
> .7873 / .7649); its `w_eff` 0 / 0.1 / 0.3 / 1 / 3 rows are
> still `planned`. So only the `w_eff=10` row here pairs with a
> measured global value — the rest is new territory for this
> model under either scope. If the local low end looks
> interesting, queueing the five matching global cells is what
> makes it interpretable as a scope effect rather than a
> low-`w_eff` effect.
>
> **Fixed:** method=`mcts_sem_v02`, **`cov_scope=local`**,
> `embeds_ref=absolute`, prm=qwen, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, cov_dtype=fp64, ds_beta=1.0,
> prm_batch_size=1, llm=qwen-7b gptq-int4, **lam=0.01**,
> data.level=5, run.num_trials=2.
>
> All 6 cells **queued at priority 1** on 2026-07-28.
> Hashes: `w_eff=0` `14255c8b`, `0.1` `07d7f95a`, `0.3`
> `6d120627`, `1` **`d5a1327c`**, `3` `0f4168dc`, `10`
> `94840f6b`. The `w_eff=1` cell is **shared with the model
> family table below** (`tbl-bf15ee`) — one ledger entry, two
> `feeds` — so this table's net new cost is 5 cells.
>
> **W&B:** none yet.

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-7b gptq-int4 | qwen | 0.01 | 0 | 0 | 2 | scored | .6306<br>±.0295 | .5933<br>±.0301 | .5709<br>±.0303 | .5299<br>±.0305 | 2.59 |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.01 | 0.1 | 2 | scored | .6679<br>±.0288 | .5896<br>±.0301 | .5746<br>±.0303 | .5560<br>±.0304 | 3.89 |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.03 | 0.3 | 2 | scored | .7276<br>±.0272 | .6157<br>±.0298 | .6045<br>±.0299 | .5933<br>±.0301 | 4.57 |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.1 | 1 | 2 | scored | .7425<br>±.0268 | .6194<br>±.0297 | .5970<br>±.0300 | .5821<br>±.0302 | 4.97 |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.3 | 3 | 2 | scored | .7537<br>±.0264 | .5933<br>±.0301 | .5709<br>±.0303 | .5634<br>±.0304 | 5.35 |
| qwen-7b gptq-int4 | qwen | 0.01 | 1.0 | 10 | 2 | scored | .7537<br>±.0264 | .5709<br>±.0303 | .5597<br>±.0304 | .5448<br>±.0305 | 5.61 |

> **Analysis.** No data yet. The one global anchor is
> `w_eff=10` = .7687 pass@gb (.6231 naive, .6194 wei, .6119
> maj, 5.42 hr/trial).
> **Limitations / follow-up:** read this table *against*
> `tbl-375fa0`, not just against its global twin — the question
> it exists for is whether the two models' local optima land at
> the same `w_eff`. If they do, the section can adopt one
> operating point and the model-family table below is
> well-posed as written. If they don't, the model-family table
> needs a per-model `w_eff` rather than the single provisional
> `w_eff=1` it currently assumes, and that is a bigger
> re-think than a re-point. At ~5.4 hr/trial × 2 trials the six
> cells are ~65 GPU-hours, against ~59 for the llama-1b sweep.

#### lam / ds_alpha joint sweep (qwen-3b)
<!-- table-id: tbl-fa65d4 -->
> **Compares:** the same `w_eff` grid as the two sweeps above,
> on the mid-strength model of the family grid.
>
> **Why this model third.** With llama-1b (.3209) and qwen-7b
> gptq-int4 (.7687) already bracketing the range, qwen-3b
> (.6903) turns the policy-strength question from a two-point
> comparison into a **three-point curve**. That matters for what
> can be concluded: two endpoints can only say "the optimum
> moved" or "it didn't", whereas three points can distinguish a
> monotone drift in the optimum from a non-monotone one, and it
> is the monotone case that would justify making `w_eff` a
> function of policy strength rather than a section-wide
> constant.
>
> ⚠️ **No global counterpart sweep exists for this model.** The
> section above sweeps `lam`/`ds_alpha` for llama-1b, llama-3b,
> qwen-math-1.5b and qwen-7b gptq-int4 — **not** qwen-3b. So
> unlike `tbl-375fa0` and `tbl-898c25`, this table cannot be
> read as a local-vs-global scope effect at any point except
> `w_eff=10`, where qwen-3b's global value is known from
> `tbl-73533c` / `tbl-e58353` (.6903 pass@gb). Everywhere else
> it is a standalone local tuning curve. If a scope comparison
> on qwen-3b is wanted, the global sweep has to be authored
> too — which is why llama-3b or qwen-math-1.5b would have been
> the cheaper third pick for a *scope* question, and qwen-3b is
> the right pick for a *policy-strength* one.
>
> ⚠️ **Named for its global counterpart, but `lam` is held at
> 0.01** — same as the two tables above; see the section
> preamble.
>
> **Fixed:** method=`mcts_sem_v02`, **`cov_scope=local`**,
> `embeds_ref=absolute`, prm=qwen, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, cov_dtype=fp64, ds_beta=1.0,
> prm_batch_size=1, llm=qwen-3b fp16, **lam=0.01**,
> data.level=5, run.num_trials=2.
>
> All 6 cells **queued at priority 1** on 2026-07-28.
> Hashes: `w_eff=0` `c6e6733d`, `0.1` `47459a5b`, `0.3`
> `fa088080`, `1` **`77b736ec`**, `3` `af5cb8a9`, `10`
> `83febb1b`. The `w_eff=1` cell is **shared with the model
> family table below** (`tbl-bf15ee`), so this table's net new
> cost is 5 cells.
>
> **W&B:** none yet.

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | qwen | 0.01 | 0 | 0 | 2 | scored | .5970<br>±.0300 | .5410<br>±.0305 | .5224<br>±.0306 | .5000<br>±.0306 | 4.28 |
| qwen-3b | qwen | 0.01 | 0.01 | 0.1 | 2 | scored | .6679<br>±.0288 | .5746<br>±.0303 | .5746<br>±.0303 | .5560<br>±.0304 | 5.90 |
| qwen-3b | qwen | 0.01 | 0.03 | 0.3 | 2 | scored | .6828<br>±.0285 | .5970<br>±.0300 | .5896<br>±.0301 | .5560<br>±.0304 | 5.93 |
| qwen-3b | qwen | 0.01 | 0.1 | 1 | 2 | scored | .7052<br>±.0279 | .5672<br>±.0303 | .5672<br>±.0303 | .5373<br>±.0305 | 6.12 |
| qwen-3b | qwen | 0.01 | 0.3 | 3 | 2 | scored | .7015<br>±.0280 | .5746<br>±.0303 | .5821<br>±.0302 | .5560<br>±.0304 | 6.22 |
| qwen-3b | qwen | 0.01 | 1.0 | 10 | 2 | scored | .6716<br>±.0287 | .5746<br>±.0303 | .5336<br>±.0305 | .5261<br>±.0306 | 6.21 |

> **Analysis.** No data yet. Single global anchor at `w_eff=10`:
> .6903 pass@gb, .5784 naive, .5597 wei, .5373 maj, 6.20
> hr/trial (`tbl-e58353`, `center=none` row).
> **Limitations / follow-up:** the three local sweeps
> (`tbl-375fa0`, `tbl-898c25`, this one) are one experiment, not
> three — the readable quantity is where each model's curve
> peaks, so a partially-filled set invites reading a
> policy-strength effect that is really a
> which-cells-finished effect. Queue them as a block or not at
> all. Full cost across all three: ~18 cells, ~5–6 hr/trial ×
> 2 trials ≈ 190 GPU-hours; the `{0.3, 1, 3}` subset on all
> three models is ~95 and still brackets the predicted optimum
> on every model.

#### lam / ds_alpha joint sweep (llama-1b, embeds_ref=relative)
<!-- table-id: tbl-ba6b11 -->
> **Compares:** the `absolute` llama-1b sweep's grid
> (`tbl-375fa0`) with `embeds_ref="relative"` substituted,
> **extended one point to `w_eff=100`**. This is the only model
> whose `absolute` local sweep is **complete (6/6)**, so every
> row here except `w_eff=100` has a measured twin and the paired
> comparison is fully determined once these cells land.
>
> **Why this model matters most.** Its `embeds_ref` comparison
> (`tbl-ea8196`) is the one that **crossed sign**: `relative`
> leads by +.0373 at `w_eff=1` (.3657 vs .3284) and trails by
> −.0560 at `w_eff=10` (.3134 vs .3694). Every other model in
> the section has `relative` ahead at both points. Two points
> can tell you a crossing happened; they cannot tell you where,
> and the whole reading — "parent-relative helps at low `w_eff`
> and hurts at high" — rests on locating it.
>
> **The two arms may peak at opposite ends.** `absolute` is
> maximal at `w_eff=10` (.3694) and minimal at 0 (.2724);
> `relative` is *worst* of its two measured points at `w_eff=10`
> (.3134) and better at 1 (.3657). If `relative` peaks below 1,
> the two arms want opposite operating points on the same model,
> which no single `w_eff` for the section can serve.
>
> ⚠️ **Named for its global counterpart, but `lam` is held at
> 0.01** — same convention as the `absolute` tables; see the
> section preamble.
>
> ⚠️ **The `w_eff=0` row is `embeds_ref`-independent by
> construction.** With `ds_alpha=0` the diversity bonus is
> multiplied by zero, so this cell must reproduce `tbl-375fa0`'s
> `w_eff=0` value (.2724) exactly. Listed for grid symmetry;
> **do not queue it** except as a plumbing check.
>
> **Fixed:** method=`mcts_sem_v02`, **`cov_scope=local`**,
> **`embeds_ref=relative`**, prm=qwen, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, cov_dtype=fp64, ds_beta=1.0,
> prm_batch_size=1, llm=llama-1b, **lam=0.01**, data.level=5,
> run.num_trials=2.
>
> ⚠️ Two cells are **already scored** and shared with the
> `embeds_ref` comparison below (`tbl-ea8196`): `w_eff=1`
> `478004b3`, `w_eff=10` `19215fb5` — one ledger entry, two
> `feeds`, not a re-run. Net new: `w_eff=0` `f19d94cf`
> (optional, see above), `0.1` `bf5f9bbc`, `0.3` `ea0bfae3`,
> `3` `a6125069`, `100` `96f7b9df`. Net cost 4 cells if the
> `w_eff=0` anchor is skipped, 5 with it. At ~4.9 hr/trial ×
> 2 trials that is ~39 GPU-hours for the four.
>
> **W&B:** mrcj4roh (`w_eff=1`), 56py1gze (`w_eff=10`),
> do0kywnf (`w_eff=0.1`), 6hd81evk (`w_eff=0.3`),
> lfq6vnc6 (`w_eff=3`), z548ni00 (`w_eff=100`).

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-1b | qwen | 0.01 | 0 | 0 | — | planned | — | — | — | — | — |
| llama-1b | qwen | 0.01 | 0.01 | 0.1 | 2 | scored | .3358<br>±.0289 | .2687<br>±.0271 | .2388<br>±.0261 | .2276<br>±.0257 | 4.69 |
| llama-1b | qwen | 0.01 | 0.03 | 0.3 | 2 | scored | .2948<br>±.0279 | .2276<br>±.0257 | .2052<br>±.0247 | .1940<br>±.0242 | 4.80 |
| llama-1b | qwen | 0.01 | 0.1 | 1 | 2 | scored | .3657<br>±.0295 | .2463<br>±.0264 | .2537<br>±.0266 | .2090<br>±.0249 | 4.98 |
| llama-1b | qwen | 0.01 | 0.3 | 3 | 2 | scored | .3731<br>±.0296 | .2873<br>±.0277 | .2537<br>±.0266 | .1903<br>±.0240 | 4.88 |
| llama-1b | qwen | 0.01 | 1.0 | 10 | 2 | scored | .3134<br>±.0284 | .2351<br>±.0260 | .2239<br>±.0255 | .1828<br>±.0237 | 4.88 |
| llama-1b | qwen | 0.01 | 10 | 100 | 2 | scored | .3246<br>±.0287 | .2164<br>±.0252 | .1978<br>±.0244 | .1679<br>±.0229 | 5.13 |

> **Analysis.** Two of seven cells measured, both inherited from
> the `embeds_ref` comparison, and they **disagree in sign**
> against `absolute` — the only model in the section where that
> happens. `relative` falls from .3657 at `w_eff=1` to .3134 at
> 10 while `absolute` climbs from .3284 to .3694 over the same
> span, so the arms are moving in opposite directions, not
> merely offset. The five unmeasured cells decide whether that
> is a real crossing or two ~1 SE wobbles.
> **Limitations / follow-up:** the low end (`w_eff` 0.1, 0.3) is
> where a `relative` peak would have to sit if the crossing is
> real, and those two are the cheapest cells in the table —
> queue them first. `w_eff=100` is the least informative here:
> `relative` is already declining by 10, and on qwen-7b and
> qwen-3b the 100 endpoint only confirmed monotone decay.
> Feeds key: `tbl-ba6b11`.

#### lam / ds_alpha joint sweep (llama-3b, embeds_ref=relative)
<!-- table-id: tbl-cf849a -->
> **Compares:** nothing yet, and that is the point to flag
> first. **There is no `absolute` local sweep for llama-3b** —
> the only `absolute` local points for this model are the two in
> its `embeds_ref` comparison (`tbl-7ee727`, `w_eff` 1 and 10),
> and both are still running. Until those land, this table is
> **unpaired** and can only be read against global scope.
>
> **Why llama-3b anyway.** llama-1b is where `embeds_ref`
> crossed sign, and the obvious explanation is that
> parent-relative displacement helps a weak policy explore and
> then starves it once `w_eff` is large. llama-3b is the
> within-family size step that tests exactly that: same
> architecture, same tokenizer, ~3× the parameters. If the
> crossing is a weak-model artifact it should weaken or vanish
> here; if it survives, it is a property of parent-relative
> geometry rather than of model strength.
>
> **Global anchors** (`tbl-591232`, `lam=0.01`, same nominal
> `w_eff`): .5261 (`w_eff=1`), .5261 (`3`), .5784 (`10`), .5485
> (`100`), .5896 (`1000`). Global peaks late on this model,
> which is the opposite of what local scope did on llama-1b.
>
> ⚠️ **Named for its global counterpart, but `lam` is held at
> 0.01** — same convention as the `absolute` tables.
>
> ⚠️ **The `w_eff=0` row is `embeds_ref`-independent by
> construction** and would reproduce the `absolute` local
> `w_eff=0` value — which for this model **has never been
> measured**. Unlike the other tables in this family, queuing it
> here would produce a genuinely new number rather than a
> plumbing check.
>
> **Fixed:** method=`mcts_sem_v02`, **`cov_scope=local`**,
> **`embeds_ref=relative`**, prm=qwen, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, cov_dtype=fp64, ds_beta=1.0,
> prm_batch_size=1, llm=llama-3b, **lam=0.01**, data.level=5,
> run.num_trials=2.
>
> ⚠️ Two cells are **already running** and shared with the
> `embeds_ref` comparison below (`tbl-7ee727`): `w_eff=1`
> `a59384e1`, `w_eff=10` `01304a84` — one ledger entry, two
> `feeds`, not a re-run. Net new: `w_eff=0` `02b5a038`,
> `0.1` `5f1b182f`, `0.3` `00defb82`, `3` `a1600163`,
> `100` `d2269ea6`. Net cost 5 cells, or 4 without the
> `w_eff=0` anchor. At ~7.5 hr/trial × 2 trials (15 h per cell)
> this is the **most expensive table in the family** — ~75
> GPU-hours for all five, ~60 without the anchor.
>
> **W&B:** 78xtrykd (`w_eff=1`), 89h5elal (`w_eff=10`),
> 67f2dbqa (`w_eff=0`), vhs7vds0 (`w_eff=0.1`),
> vdavflbd (`w_eff=0.3`), hu3gedj8 (`w_eff=3`),
> 3glufzax (`w_eff=100`).

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-3b | qwen | 0.01 | 0 | 0 | 2 | scored | .4440<br>±.0304 | .3881<br>±.0298 | .3731<br>±.0296 | .3134<br>±.0284 | 5.02 |
| llama-3b | qwen | 0.01 | 0.01 | 0.1 | 2 | scored | .5224<br>±.0306 | .4030<br>±.0300 | .3918<br>±.0299 | .3731<br>±.0296 | 6.46 |
| llama-3b | qwen | 0.01 | 0.03 | 0.3 | 2 | scored | .5709<br>±.0303 | .4627<br>±.0305 | .4254<br>±.0303 | .4067<br>±.0301 | 6.74 |
| llama-3b | qwen | 0.01 | 0.1 | 1 | 2 | scored | .5784<br>±.0302 | .4216<br>±.0302 | .4067<br>±.0301 | .3731<br>±.0296 | 6.74 |
| llama-3b | qwen | 0.01 | 0.3 | 3 | 2 | scored | .5821<br>±.0302 | .4291<br>±.0303 | .3918<br>±.0299 | .3731<br>±.0296 | 6.85 |
| llama-3b | qwen | 0.01 | 1.0 | 10 | 2 | scored | .5485<br>±.0305 | .3993<br>±.0300 | .3619<br>±.0294 | .3470<br>±.0291 | 7.08 |
| llama-3b | qwen | 0.01 | 10 | 100 | 2 | scored | .5522<br>±.0304 | .4254<br>±.0303 | .3993<br>±.0300 | .3694<br>±.0295 | 7.05 |

> **Analysis.** Complete (7/7, closed 2026-08-03). The curve
> rises steadily off the no-diversity baseline and then turns
> over: .4440 (`w_eff=0`), .5224 (`0.1`), .5709 (`0.3`), .5784
> (`1`), **.5821 (`3`, the maximum)**, .5485 (`10`), .5522
> (`100`). That is a +.138 span from 0 to the peak — the
> clearest interior optimum in the section, and ~4.5 SE, well
> clear of the noise the other tables sit inside.
> **The crossing is located.** `tbl-7ee727` had shown only that
> `relative` beats `absolute` at `w_eff=1` and loses by 10; the
> peak at 3 with a .034 drop by 10 says the turnover happens
> **between `w_eff` 3 and 10**, not below 1. So the earlier
> reading — "parent-relative helps at low `w_eff` and hurts at
> high" — survives, but the boundary is an order of magnitude
> higher than the two-point view implied, and the whole
> `w_eff` ∈ [0.3, 3] plateau (.5709–.5821, all within 0.4 SE of
> each other) is usable operating range rather than a single
> point.
> **Limitations / follow-up:** the top four cells are separated
> by ≤.011 — the peak's *location* inside [0.3, 3] is not
> resolved at n≈267 pooled over 2 trials, only its existence and
> the fall past 10. The `absolute` twin sweep still does not
> exist for this model, so the paired scope comparison remains
> one-armed except at `w_eff` 1 and 10. The tail is also flat,
> not monotone (.5485 then .5522), which is consistent with
> both settling onto the same over-diversified plateau. Feeds
> key: `tbl-cf849a`.

#### lam / ds_alpha joint sweep (qwen-3b, embeds_ref=relative)
<!-- table-id: tbl-b1cb82 -->
> **Compares:** the `absolute` qwen-3b sweep's grid
> (`tbl-fa65d4`) with `embeds_ref="relative"` substituted and
> **extended one point to `w_eff=100`** — the qwen-3b
> counterpart of the qwen-7b relative sweep above. The
> `w_eff=100` row has no `absolute` twin yet (`2efb1af5`,
> unqueued).
>
> **Why qwen-3b too.** The qwen-7b relative sweep alone cannot
> separate "relative helps" from "relative helps the strongest
> policy." qwen-3b is the mid-tier model with the most complete
> `absolute` local sweep, so every row here has a measured twin
> and the paired comparison is fully determined once these
> cells land. Its two measured `relative` points already lead:
> **+.0112** at `w_eff=1` (.7164 vs .7052) and **+.0374** at
> `w_eff=10` (.7090 vs .6716) — same sign as qwen-7b but a
> different profile, growing with `w_eff` rather than shrinking.
>
> **What the pair of sweeps buys.** If both models' `relative`
> curves peak at the same `w_eff`, the section gets one
> operating point and the model-family table below (`tbl-bf15ee`,
> currently pinned to a provisional `w_eff=1`) can be re-pointed
> once. If they peak at different `w_eff`, that table needs a
> per-model point — the same risk the `absolute` sweeps were
> queued to resolve, now inherited by the `relative` arm.
>
> ⚠️ **Named for its global counterpart, but `lam` is held at
> 0.01** — same convention as the `absolute` tables.
>
> ⚠️ **The `w_eff=0` row is `embeds_ref`-independent by
> construction** and must reproduce `tbl-fa65d4`'s `w_eff=0`
> value (.5970) exactly. Listed for grid symmetry; **do not
> queue it** except as a plumbing check.
>
> **Fixed:** method=`mcts_sem_v02`, **`cov_scope=local`**,
> **`embeds_ref=relative`**, prm=qwen, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, cov_dtype=fp64, ds_beta=1.0,
> prm_batch_size=1, llm=qwen-3b, **lam=0.01**, data.level=5,
> run.num_trials=2.
>
> ⚠️ Two cells are **already scored** and shared with the
> `embeds_ref` comparison below (`tbl-6ac460`): `w_eff=1`
> `1ef89fdc`, `w_eff=10` `df8182ca` — one ledger entry, two
> `feeds`, not a re-run. Net new: `w_eff=0` `d8b67e8b`
> (optional, see above), `0.1` `507d12ce`, `0.3` `cb6356ce`,
> `3` `65a044ea`, `100` `0a3fc03a`. Net cost 4 cells if the
> `w_eff=0` anchor is skipped, 5 with it.
>
> **W&B:** decwj7la (`w_eff=10`), x52dugmp (`w_eff=100`),
> u2i8huih (`w_eff=0.1`), spakquo1 (`w_eff=0.3`),
> ntjhh9o9 (`w_eff=3`).

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | qwen | 0.01 | 0 | 0 | — | planned | — | — | — | — | — |
| qwen-3b | qwen | 0.01 | 0.01 | 0.1 | 2 | scored | .6530<br>±.0291 | .5485<br>±.0305 | .5336<br>±.0305 | .5075<br>±.0306 | 5.77 |
| qwen-3b | qwen | 0.01 | 0.03 | 0.3 | 2 | scored | .6940<br>±.0282 | .5634<br>±.0304 | .5522<br>±.0304 | .5448<br>±.0305 | 6.05 |
| qwen-3b | qwen | 0.01 | 0.1 | 1 | 2 | scored | .7164<br>±.0276 | .5821<br>±.0302 | .5634<br>±.0304 | .5410<br>±.0305 | 6.21 |
| qwen-3b | qwen | 0.01 | 0.3 | 3 | 2 | scored | .7127<br>±.0277 | .6157<br>±.0298 | .5709<br>±.0303 | .5336<br>±.0305 | 6.35 |
| qwen-3b | qwen | 0.01 | 1.0 | 10 | 2 | scored | .7090<br>±.0278 | .5970<br>±.0300 | .5522<br>±.0304 | .5149<br>±.0306 | 6.24 |
| qwen-3b | qwen | 0.01 | 10 | 100 | 2 | scored | .6940<br>±.0282 | .5560<br>±.0304 | .5410<br>±.0305 | .5075<br>±.0306 | 6.11 |

> **Analysis.** Three of seven cells measured — `w_eff` 1 and 10
> inherited from the `embeds_ref` comparison, plus the
> `w_eff=100` endpoint (.6940, scored 2026-08-02). Like qwen-7b,
> the `relative` arm decays monotonically in `w_eff` (.7164 →
> .7090 → .6940), so neither model's optimum sits above 1.
> `relative` leads at both shared points,
> and unlike qwen-7b the gap *widens* with `w_eff` (+.0112 at 1,
> +.0374 at 10). The `absolute` arm for this model peaks at
> `w_eff=1` (.7052) and falls to .6716 by `w_eff=10`; the
> `relative` arm is nearly flat across the same span (.7164 →
> .7090). Read together, that says parent-relative mostly buys
> **robustness to over-weighting diversity**, not a higher peak
> — which is a claim the three unmeasured cells can confirm or
> kill.
> **Limitations / follow-up:** neither model's `relative` curve
> has a measured low end, so "the optimum moved left" is
> currently an inference from two points on each. `w_eff` 0.1
> and 0.3 are the cells that decide it, and they are the two
> cheapest in both tables. If budget is tight, queue `{0.1,
> 0.3}` on both models (4 cells, ~47 GPU-hours) and leave
> `w_eff=3` for later. Feeds key: `tbl-b1cb82`.

#### lam / ds_alpha joint sweep (qwen-7b gptq-int4, embeds_ref=relative)
<!-- table-id: tbl-5d64b1 -->
> **Compares:** the `absolute` qwen-7b sweep's grid
> (`tbl-898c25`) with `embeds_ref="relative"` substituted,
> **extended one point to `w_eff=100`**. Rows through `w_eff=10`
> pair with an `absolute` row at the same nominal `w_eff`, so
> the two tables read as one paired experiment; the `w_eff=100`
> row has no `absolute` twin yet (`ddf897d9`, unqueued).
>
> **Why a full sweep rather than the 2-point comparison.** The
> `embeds_ref` comparison tables below sample `w_eff` ∈ {1, 10}
> only, and for this model both points already favour
> `relative`: **+.0411** at `w_eff=1` (.7836 vs .7425, ~1.5 SE)
> and **+.0224** at `w_eff=10` (.7761 vs .7537). That is a
> consistent sign but a 2-point read cannot tell whether
> `relative` shifts the optimum or just lifts the curve. The
> `absolute` arm peaks flat across `w_eff` 3–10 (.7537 twice);
> if `relative` peaks earlier, the operating point for the whole
> section moves, which is a different conclusion from "relative
> is a bit better everywhere."
>
> **Why this model first.** It is the strongest policy in the
> grid and the one where `absolute` local scope still trails its
> own global result (.7537 vs .7687). `relative` at `w_eff=1`
> already clears that global number (.7836), so this sweep is
> the one that can settle whether local scope beats global for
> qwen-7b at all.
>
> ⚠️ **Named for its global counterpart, but `lam` is held at
> 0.01** — same convention as the `absolute` tables; see the
> section preamble.
>
> ⚠️ **The `w_eff=0` row is `embeds_ref`-independent by
> construction.** With `ds_alpha=0` the diversity bonus is
> multiplied by zero, so what `V` accumulates cannot affect
> selection and this cell must reproduce `tbl-898c25`'s
> `w_eff=0` value (.6306) exactly. It is listed for grid
> symmetry and as a plumbing check only — **do not queue it**
> unless you want that assertion tested end to end; the
> `absolute` number is the answer.
>
> **Fixed:** method=`mcts_sem_v02`, **`cov_scope=local`**,
> **`embeds_ref=relative`**, prm=qwen, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, cov_dtype=fp64, ds_beta=1.0,
> prm_batch_size=1, llm=qwen-7b gptq-int4, **lam=0.01**,
> data.level=5, run.num_trials=2.
>
> ⚠️ Two cells are **already scored** and shared with the
> `embeds_ref` comparison below (`tbl-5cf136`): `w_eff=1`
> `11ad13c7`, `w_eff=10` `b53d44dd` — one ledger entry, two
> `feeds`, not a re-run. Net new: `w_eff=0` `504ff1a8`
> (optional, see above), `0.1` `f5251f86`, `0.3` `45e2d3ac`,
> `3` `3e3fe251`, `100` `f0fcf038`. Net cost 4 cells if the
> `w_eff=0` anchor is skipped, 5 with it.
>
> **W&B:** bk2rou47 (`w_eff=1`), n2iiuppj (`w_eff=10`),
> m1g6t5dk (`w_eff=100`), 1nv4x6v7 (`w_eff=0.1`),
> pv3ievaa (`w_eff=0.3`), 1wbdyaki (`w_eff=3`).

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-7b gptq-int4 | qwen | 0.01 | 0 | 0 | — | planned | — | — | — | — | — |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.01 | 0.1 | 2 | scored | .7164<br>±.0276 | .6007<br>±.0300 | .5970<br>±.0300 | .5896<br>±.0301 | 4.70 |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.03 | 0.3 | 2 | scored | .7575<br>±.0262 | .6194<br>±.0297 | .5970<br>±.0300 | .5858<br>±.0301 | 5.15 |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.1 | 1 | 2 | scored | .7836<br>±.0252 | .6045<br>±.0299 | .5821<br>±.0302 | .5821<br>±.0302 | 5.54 |
| qwen-7b gptq-int4 | qwen | 0.01 | 0.3 | 3 | 2 | scored | .7761<br>±.0255 | .6157<br>±.0298 | .5672<br>±.0303 | .5634<br>±.0304 | 5.58 |
| qwen-7b gptq-int4 | qwen | 0.01 | 1.0 | 10 | 2 | scored | .7761<br>±.0255 | .5634<br>±.0304 | .5672<br>±.0303 | .5597<br>±.0304 | 5.59 |
| qwen-7b gptq-int4 | qwen | 0.01 | 10 | 100 | 2 | scored | .7537<br>±.0264 | .5896<br>±.0301 | .5784<br>±.0302 | .5597<br>±.0304 | 5.61 |

> **Analysis.** Three of seven cells measured — `w_eff` 1 and
> 10 from the `embeds_ref` comparison, plus the `w_eff=100`
> endpoint (.7537, scored 2026-08-02), which decays monotonically
> from `w_eff=1` and lands exactly on the `absolute` plateau.
> `relative` leads `absolute` at both shared
> points (.7836 vs .7425, .7761 vs .7537) and the `w_eff=1`
> value is the highest pass@gb recorded for this model under
> either scope, above the global `w_eff=10` anchor of .7687.
> Note the curve shape already differs: `absolute` rises to a
> flat .7537 plateau at `w_eff` 3–10, while `relative` is
> *higher at 1 than at 10*, hinting the optimum moved left.
> The three unmeasured low/mid cells are what test that.
> **Limitations / follow-up:** maj@gb tells a different story
> from pass@gb — `relative` w_eff=1 reaches .5821 maj, equal to
> `absolute` w_eff=1, so the pass@gb gain is not yet showing up
> in the aggregation that matters for a single answer. If the
> unmeasured cells keep that pattern, the honest claim is
> "relative widens the candidate set" rather than "relative is
> more accurate." Feeds key: `tbl-5d64b1`.

#### lam / ds_alpha joint sweep (qwen-math-1.5b, embeds_ref=relative)
<!-- table-id: tbl-3a76ce -->
> **Compares:** as with llama-3b, **there is no `absolute` local
> sweep for qwen-math-1.5b** — only the two `absolute` points in
> its `embeds_ref` comparison (`tbl-78da65`, `w_eff` 1 and 10),
> both still running. This table is **unpaired** until those
> land.
>
> **Why qwen-math.** It is the math-specialized model and the
> outlier of the family: strongest naive/wei/maj numbers in the
> b=320 family table (`tbl-6a015e`: .6455/.6381/.6269) despite
> being the smallest model in the grid at 1.5B. If
> parent-relative geometry interacts with *what the embeddings
> represent* rather than with model capacity, this is the model
> where it should look different — its hidden states are shaped
> by math pretraining, not general text.
>
> **Global anchors** (`tbl-a12d4f`, `lam=0.01`): .7500
> (`w_eff=10`), .7164 (`100`), .7201 (`1000`); `w_eff` 1 and 3
> are unmeasured there too. Global peaks at `w_eff=10` and
> *declines* past it — the earliest peak of any model in the
> section, which is the specific reason a low-`w_eff` local
> sweep on this model is interesting.
>
> ⚠️ **Named for its global counterpart, but `lam` is held at
> 0.01** — same convention as the `absolute` tables.
>
> ⚠️ **The `w_eff=0` row is `embeds_ref`-independent by
> construction** and, as with llama-3b, has **no measured
> `absolute` local counterpart** — queuing it here yields a new
> number, not a plumbing check.
>
> **Fixed:** method=`mcts_sem_v02`, **`cov_scope=local`**,
> **`embeds_ref=relative`**, prm=qwen, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, cov_dtype=fp64, ds_beta=1.0,
> prm_batch_size=1, llm=qwen-math-1.5b, **lam=0.01**,
> data.level=5, run.num_trials=2.
>
> ⚠️ Two cells are **already running** and shared with the
> `embeds_ref` comparison below (`tbl-78da65`): `w_eff=1`
> `43bd8117`, `w_eff=10` `70b52bb3` — one ledger entry, two
> `feeds`, not a re-run. Net new: `w_eff=0` `8eaa1d3b`,
> `0.1` `4fd0df23`, `0.3` `f9f7ff31`, `3` `e6bee4fd`,
> `100` `33b07fae`. Net cost 5 cells, or 4 without the
> `w_eff=0` anchor. At ~5.5 hr/trial × 2 trials that is ~44-55
> GPU-hours — the **cheapest** of the three new tables.
>
> **W&B:** 6cf71i7r (`w_eff=1`), 9m9rooh6 (`w_eff=10`),
> 23wzlc65 (`w_eff=0`), tkdotrkv (`w_eff=0.1`),
> jlkqzn16 (`w_eff=0.3`), u2z0lokg (`w_eff=3`),
> hv0bwrkw (`w_eff=100`).

| llm | prm | lam | ds_alpha | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen-math-1.5b | qwen | 0.01 | 0 | 0 | 2 | scored | .6791<br>±.0286 | .6082<br>±.0299 | .5970<br>±.0300 | .5634<br>±.0304 | 3.70 |
| qwen-math-1.5b | qwen | 0.01 | 0.01 | 0.1 | 2 | scored | .7090<br>±.0278 | .6231<br>±.0297 | .6269<br>±.0296 | .5933<br>±.0301 | 4.75 |
| qwen-math-1.5b | qwen | 0.01 | 0.03 | 0.3 | 2 | scored | .7537<br>±.0264 | .6269<br>±.0296 | .6343<br>±.0295 | .6045<br>±.0299 | 4.84 |
| qwen-math-1.5b | qwen | 0.01 | 0.1 | 1 | 2 | scored | .7425<br>±.0268 | .6343<br>±.0295 | .6343<br>±.0295 | .6157<br>±.0298 | 4.84 |
| qwen-math-1.5b | qwen | 0.01 | 0.3 | 3 | 2 | scored | .7239<br>±.0274 | .6082<br>±.0299 | .5933<br>±.0301 | .5970<br>±.0300 | 4.93 |
| qwen-math-1.5b | qwen | 0.01 | 1.0 | 10 | 2 | scored | .7612<br>±.0261 | .6269<br>±.0296 | .6194<br>±.0297 | .5821<br>±.0302 | 4.82 |
| qwen-math-1.5b | qwen | 0.01 | 10 | 100 | 2 | scored | .7425<br>±.0268 | .6306<br>±.0295 | .6007<br>±.0300 | .5784<br>±.0302 | 4.95 |

> **Analysis.** Complete (7/7, closed 2026-08-03): .6791
> (`w_eff=0`), .7090 (`0.1`), .7537 (`0.3`), .7425 (`1`), .7239
> (`3`), **.7612 (`10`, the maximum)**, .7425 (`100`). Only one
> comparison here is real: **turning diversity on is worth
> +.05–.08** (`w_eff=0` sits 1.5–2 SE below every other cell).
> Above that, the six non-zero cells span .7090–.7612 with SEs
> of ~.027 — a range of ~1 SE end to end, and the ordering is
> non-monotone (a dip at 3 between two higher neighbours). Read
> it as **flat from 0.3 up**, not as a peak at 10.
> **This model does not reproduce the llama pattern**, which is
> what the table was built to test: no interior optimum, no
> turnover by `w_eff=100`, where llama-3b (`tbl-cf849a`) peaks
> at 3 and drops .034 by 10. Whatever drives the llama crossing
> — family, tokenizer, or the geometry of a general-text
> embedding space — qwen-math's math-pretrained hidden states
> do not have it, so the crossing is not a universal property of
> parent-relative displacement.
> **Limitations / follow-up:** "flat" is a statement about n≈267
> pooled over 2 trials; a 4-trial rerun of `{0.3, 3, 10}` would
> settle whether the dip at 3 is noise (most likely) or a real
> notch. The `absolute` local sweep for this model still does
> not exist, so `w_eff=0` is the only scope-independent anchor
> and the `relative`-vs-`absolute` question remains open here.
> Feeds key: `tbl-3a76ce`.

#### embeds_ref comparison (llama-1b, cov_scope=local)
<!-- table-id: tbl-ea8196 -->
> **Compares:** `embeds_ref="absolute"` (the child's own pooled
> embedding) against `embeds_ref="relative"` (the displacement
> `x_child - x_parent`), under local scope.
>
> **Why this pairing.** A child's embedding is pooled over its
> whole text prefix, so siblings share a long common prefix and
> their absolute embeddings cluster tightly around the parent's
> direction — measured at 0.98 mean |cos| on synthetic clustered
> siblings, dropping to 0.07 after subtracting the parent
> (`unittests/check_cov_scope_embeds_ref.py`). That shared
> component dominates `sqrt(x^T V^-1 x)` and leaves the sibling
> *differences* — the only thing selection can act on — as a
> small perturbation. Parent-relative removes it, so the bonus
> scores step **directions**.
>
> Local scope is where this is most coherent: the parent is a
> fixed reference for the node's entire lifetime, so every
> vector folded into `V_n` shares one origin. (The existing
> `embeds_center_mode=local` knob removes a similar offset but
> recomputes the sibling-group mean at every expansion, so under
> `revisit_policy=regenerate` one node's `V` would mix origins.
> The two are mutually exclusive — `MCTS.__init__` rejects the
> combination as double-centering.)
>
> **Fixed:** as the sweep above (`cov_scope=local`, lam=0.01,
> llama-1b, b=80, level 5, 2 trials); only `embeds_ref` and
> `ds_alpha` vary.
>
> **`relative` applies at every depth, including the root.**
> The root is embedded explicitly (the question pooled with an
> empty answer, through the same pipeline as every candidate),
> so the two arms differ uniformly rather than differing at
> depths 1+ and agreeing at depth 0. Without that the
> comparison would be partly confounded — a null result could
> just mean the root, where branching is widest, was identical
> in both arms.
>
> ⚠️ The two `absolute` rows are the **same two cells** as the
> sweep table above (`3dc685e4`, `a941cc35`) — one ledger entry
> feeding two tables, not a re-run. Only the `relative` rows are
> new: `w_eff=1` `478004b3`, `w_eff=10` `19215fb5`. Net cost of
> this table is 2 cells.
>
> **W&B:** axf4tdpb (`absolute`, `w_eff=10`), mrcj4roh
> (`relative`, `w_eff=1`), 56py1gze (`relative`, `w_eff=10`).

| llm | prm | embeds_ref | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|
| llama-1b | qwen | absolute | 1 | 2 | scored | .3284<br>±.0287 | .2575<br>±.0268 | .2276<br>±.0257 | .2015<br>±.0245 | 4.73 |
| llama-1b | qwen | relative | 1 | 2 | scored | .3657<br>±.0295 | .2463<br>±.0264 | .2537<br>±.0266 | .2090<br>±.0249 | 4.98 |
| llama-1b | qwen | absolute | 10 | 2 | scored | .3694<br>±.0295 | .2799<br>±.0275 | .2127<br>±.0250 | .1754<br>±.0233 | 4.90 |
| llama-1b | qwen | relative | 10 | 2 | scored | .3134<br>±.0284 | .2351<br>±.0260 | .2239<br>±.0255 | .1828<br>±.0237 | 4.88 |

> **Analysis.** Complete (4/4, closed 2026-08-02). **The two
> points disagree in sign on pass@gb** — `relative` leads by
> +.0373 at `w_eff=1` (.3657 vs .3284) and trails by −.0560 at
> `w_eff=10` (.3134 vs .3694). This is the contingency the
> follow-up note below was written for, and it fires: on this
> model `embeds_ref` interacts with `w_eff` rather than shifting
> the curve uniformly. The crossing also runs *against* the
> other models — qwen-7b (`tbl-5cf136`) and qwen-3b
> (`tbl-6ac460`) both have `relative` ahead at both points — so
> "parent-relative helps" does not hold family-wide at the weak
> end. Note maj@gb does not cross: `relative` leads there at
> both points (.2090 vs .2015, .1828 vs .1754), so the pass@gb
> reversal at `w_eff=10` is a candidate-set effect, not a
> collapse in answer quality.
> **Limitations / follow-up:** each individual gap is ~1 SE
> (+.0373 ≈ 0.9 SE, −.0560 ≈ 1.4 SE, unpaired), so the crossing
> rests on two weak differences pointing opposite ways rather
> than on either one being solid. Confirming it needs the low
> end of a `relative` sweep on llama-1b, not more trials at
> these two points. Two `w_eff` points rather than
> one, because parent-relative changes the geometry of what `V`
> accumulates — normalized displacements fill `V` far more
> isotropically than clustered absolutes — so its own optimum
> may not sit where `absolute`'s does. Now that the two points
> disagree in sign, `embeds_ref` needs its own `w_eff` sweep
> rather than a 2-point comparison.

#### embeds_ref comparison (llama-3b, cov_scope=local)
<!-- table-id: tbl-7ee727 -->
> **Compares:** `embeds_ref="absolute"` against
> `embeds_ref="relative"` under local scope — the llama-3b
> counterpart of the llama-1b table above. Same two `w_eff`
> points, same rationale (see that table for why parent-relative
> displacement is the interesting arm: siblings' absolute
> embeddings sit at 0.98 mean |cos|, dropping to 0.07 after
> subtracting the parent).
>
> **Why llama-3b too.** The llama-1b table alone cannot separate
> "parent-relative helps" from "parent-relative helps a weak
> model." llama-3b is the within-family size step that tests
> exactly that: same tokenizer, same custom template, same
> prompt formatting, one size up. Pairing it with the qwen-3b
> table below also separates family from size — llama-3b and
> qwen-3b are the same scale under different pretraining.
>
> **Fixed:** as the llama-1b table above (`cov_scope=local`,
> lam=0.01, llama-3b, b=80, level 5, 2 trials); only
> `embeds_ref` and `ds_alpha` vary.
>
> ⚠️ The `absolute` w_eff=1 row is the **same cell** as
> llama-3b's row in the model family comparison below
> (`08e67e7a`) — one config feeding two tables, not a re-run.
> Net new here: `absolute` w_eff=10 `f45cd8a1`, `relative`
> w_eff=1 `a59384e1`, `relative` w_eff=10 `01304a84`. Net cost
> of this table is 3 cells.
>
> **W&B:** nnf53blu (`absolute`, `w_eff=1`), 78xtrykd
> (`relative`, `w_eff=1`), gku3q8ph (`absolute`, `w_eff=10`),
> 89h5elal (`relative`, `w_eff=10`).

| llm | prm | embeds_ref | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|
| llama-3b | qwen | absolute | 1 | 2 | scored | .5448<br>±.0305 | .4366<br>±.0304 | .4328<br>±.0303 | .4104<br>±.0301 | 6.67 |
| llama-3b | qwen | relative | 1 | 2 | scored | .5784<br>±.0302 | .4216<br>±.0302 | .4067<br>±.0301 | .3731<br>±.0296 | 6.74 |
| llama-3b | qwen | absolute | 10 | 2 | scored | .5672<br>±.0303 | .4030<br>±.0300 | .3955<br>±.0299 | .3731<br>±.0296 | 7.02 |
| llama-3b | qwen | relative | 10 | 2 | scored | .5485<br>±.0305 | .3993<br>±.0300 | .3619<br>±.0294 | .3470<br>±.0291 | 7.08 |

> **Analysis.** Complete (4/4, closed 2026-08-02). **The
> llama-1b sign crossing reproduces here.** `relative` leads by
> +.0336 at `w_eff=1` (.5784 vs .5448) and trails by −.0187 at
> `w_eff=10` (.5485 vs .5672) — the same direction of flip as
> llama-1b (+.0373, then −.0560). The size step from 1B to 3B
> did not remove it, so **the crossing is not a weak-model
> artifact**, which is what this table was queued to test.
> **The split is by model family, not by capacity.** Both llama
> models cross; both qwen models do not — qwen-3b (`tbl-6ac460`)
> and qwen-7b (`tbl-5cf136`) have `relative` ahead at *both*
> `w_eff` points. Two families, four models, a clean partition.
> **maj@gb is worse here than on llama-1b.** On llama-1b
> `relative` led on maj at both points, which supported reading
> the pass@gb reversal as a candidate-set effect. On llama-3b
> `relative` *loses* maj at both — .3731 vs .4104 at `w_eff=1`
> and .3470 vs .3731 at 10 — including at the point where it
> wins pass@gb. That is a sharper negative: parent-relative
> widens the candidate set without the aggregation converting
> it, and the effect grows with model size.
> **Limitations / follow-up:** every gap in this table is
> **under 1 SE** (+.0336 ≈ 0.78 SE, −.0187 ≈ 0.43 SE, unpaired),
> so llama-3b on its own confirms nothing. The value is that two
> independent models show the same sign pattern in the same
> places — an agreement argument, not a significance one.
> Also note local scope does not dominate global on this model:
> `absolute` beats its `tbl-591232` global twin at `w_eff=1`
> (.5448 vs .5261) but loses at `w_eff=10` (.5672 vs .5784).
> The queued llama-3b `relative` sweep (`tbl-cf849a`) is what
> locates the crossing rather than just detecting it.
> Feeds key: `tbl-7ee727`.

#### embeds_ref comparison (qwen-3b, cov_scope=local)
<!-- table-id: tbl-6ac460 -->
> **Compares:** `embeds_ref="absolute"` against
> `embeds_ref="relative"` under local scope — the qwen-3b
> counterpart of the llama-1b table above. Same two `w_eff`
> points, same rationale (see that table for why parent-relative
> displacement is the interesting arm: siblings' absolute
> embeddings sit at 0.98 mean |cos|, dropping to 0.07 after
> subtracting the parent).
>
> **Why qwen-3b too.** The llama-1b table alone cannot separate
> "parent-relative helps" from "parent-relative helps a weak
> model." qwen-3b is the mid-tier model with the most complete
> local sweep, so its `absolute` arm is already partly measured
> and the comparison starts half-paid.
>
> **Fixed:** as the qwen-3b sweep above (`cov_scope=local`,
> lam=0.01, qwen-3b, b=80, level 5, 2 trials); only
> `embeds_ref` and `ds_alpha` vary.
>
> ⚠️ The two `absolute` rows are the **same two cells** as the
> qwen-3b sweep table above (`77b736ec`, `83febb1b`) — one
> ledger entry feeding two tables, not a re-run. `77b736ec` is
> already scored; `83febb1b` is `inqueue` (requeued 2026-07-30
> after its allocation timed out with 1/2 trials). Only the
> `relative` rows are new: `w_eff=1` `1ef89fdc`, `w_eff=10`
> `df8182ca`. Net cost of this table is 2 cells (~12 h each).
>
> **W&B:** fgy1e9sq (`absolute`, w_eff=1), decwj7la (`relative`,
> w_eff=10).

| llm | prm | embeds_ref | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen-3b | qwen | absolute | 1 | 2 | scored | .7052<br>±.0279 | .5672<br>±.0303 | .5672<br>±.0303 | .5373<br>±.0305 | 6.12 |
| qwen-3b | qwen | relative | 1 | 2 | scored | .7164<br>±.0276 | .5821<br>±.0302 | .5634<br>±.0304 | .5410<br>±.0305 | 6.21 |
| qwen-3b | qwen | absolute | 10 | 2 | scored | .6716<br>±.0287 | .5746<br>±.0303 | .5336<br>±.0305 | .5261<br>±.0306 | 6.21 |
| qwen-3b | qwen | relative | 10 | 2 | scored | .7090<br>±.0278 | .5970<br>±.0300 | .5522<br>±.0304 | .5149<br>±.0306 | 6.24 |

> **Analysis.** One of four cells measured. The `absolute`
> w_eff=1 baseline is .7052 pass@gb — the best qwen-3b local
> number so far, and .0149 above the global w_eff=10 twin
> (.6903), which is inside one standard error. `relative` has
> to beat .7052 by more than ~.03 to be a real effect.
> **Limitations / follow-up:** the `absolute` w_eff=10 row is
> gated on the requeued sweep cell, so the two-point shape of
> the `absolute` arm is not yet known for this model. Feeds key
> is the table-id below; the two `relative` cells have no ledger
> entry yet — queue them with hashes `1ef89fdc` and `df8182ca`.

#### embeds_ref comparison (qwen-7b gptq-int4, cov_scope=local)
<!-- table-id: tbl-5cf136 -->
> **Compares:** `embeds_ref="absolute"` against
> `embeds_ref="relative"` under local scope — the qwen-7b
> counterpart of the two tables above. Same two `w_eff` points,
> same rationale.
>
> **Why qwen-7b too.** It is the strongest model in the family
> and the one whose local sweep shows the largest measured
> effect anywhere in this section (`w_eff` 0 → 3 moves pass@gb
> .6306 → .7537, ~4 SE). If parent-relative matters, the model
> with the most headroom to lose is where a regression would
> show first; if it does nothing here, the knob is unlikely to
> repay a full sweep.
>
> **Fixed:** as the qwen-7b sweep above (`cov_scope=local`,
> lam=0.01, qwen-7b gptq-int4, b=80, level 5, 2 trials); only
> `embeds_ref` and `ds_alpha` vary.
>
> ⚠️ The two `absolute` rows are the **same two cells** as the
> qwen-7b sweep table above (`d5a1327c`, `94840f6b`) — one
> ledger entry feeding two tables, not a re-run. `94840f6b` is
> already scored; `d5a1327c` is `inqueue` (requeued 2026-07-30,
> resuming from trial 1). Only the `relative` rows are new:
> `w_eff=1` `11ad13c7`, `w_eff=10` `b53d44dd`. Net cost of this
> table is 2 cells (~11 h each).
>
> **W&B:** ip6rfqxy (`absolute`, w_eff=10), bk2rou47 (`relative`,
> w_eff=1), n2iiuppj (`relative`, w_eff=10).

| llm | prm | embeds_ref | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen-7b gptq-int4 | qwen | absolute | 1 | 2 | scored | .7425<br>±.0268 | .6194<br>±.0297 | .5970<br>±.0300 | .5821<br>±.0302 | 4.97 |
| qwen-7b gptq-int4 | qwen | relative | 1 | 2 | scored | .7836<br>±.0252 | .6045<br>±.0299 | .5821<br>±.0302 | .5821<br>±.0302 | 5.54 |
| qwen-7b gptq-int4 | qwen | absolute | 10 | 2 | scored | .7537<br>±.0264 | .5709<br>±.0303 | .5597<br>±.0304 | .5448<br>±.0305 | 5.61 |
| qwen-7b gptq-int4 | qwen | relative | 10 | 2 | scored | .7761<br>±.0255 | .5634<br>±.0304 | .5672<br>±.0303 | .5597<br>±.0304 | 5.59 |

> **Analysis.** One of four cells measured. The `absolute`
> w_eff=10 baseline is .7537 pass@gb, still .0150 below this
> model's global w_eff=10 result (.7687) — so local scope has
> not yet beaten global for qwen-7b, and `relative` is one of
> the remaining candidates for closing that gap.
> **Limitations / follow-up:** the `absolute` w_eff=1 row is
> gated on the requeued sweep cell. Note this table's two
> measured points come from opposite corners of the grid
> relative to the qwen-3b table (w_eff=10 here, w_eff=1 there),
> so the three `embeds_ref` tables cannot yet be read as a
> family. The two `relative` cells have no ledger entry yet —
> queue them with hashes `11ad13c7` and `b53d44dd`.

#### embeds_ref comparison (qwen-math-1.5b, cov_scope=local)
<!-- table-id: tbl-78da65 -->
> **Compares:** `embeds_ref="absolute"` against
> `embeds_ref="relative"` under local scope — the qwen-math
> counterpart of the four tables above, which completes the
> family. Same two `w_eff` points, same rationale.
>
> **Why qwen-math too.** It is the only math-specialized model
> in the grid and the one whose generations are most
> stylistically uniform: a model fine-tuned on a narrow domain
> emits siblings that read alike, which is precisely the regime
> where absolute embeddings should cluster hardest and
> parent-relative should buy the most. It is the strongest test
> of the mechanism the other four tables assume — if subtracting
> the parent does nothing here, the knob is unlikely to matter
> anywhere.
>
> **Fixed:** as the llama-1b table above (`cov_scope=local`,
> lam=0.01, qwen-math-1.5b, b=80, level 5, 2 trials); only
> `embeds_ref` and `ds_alpha` vary. No `llm.max_model_len`
> override — that is a b=320 concern for this model (see the
> b=320 tables, where it needs 4096), not a b=80 one.
>
> ⚠️ The `absolute` w_eff=1 row is the **same cell** as
> qwen-math-1.5b's row in the model family comparison below
> (`74a4b258`) — one config feeding two tables, not a re-run.
> Net new here: `absolute` w_eff=10 `a4a229ee`, `relative`
> w_eff=1 `43bd8117`, `relative` w_eff=10 `70b52bb3`. Net cost
> of this table is 3 cells.
>
> **W&B:** t6wgq6yh (`absolute` w_eff=1), 6cf71i7r
> (`relative` w_eff=1), k34dky0k (`absolute` w_eff=10),
> 9m9rooh6 (`relative` w_eff=10).

| llm | prm | embeds_ref | w_eff | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen-math-1.5b | qwen | absolute | 1 | 2 | scored | .7351<br>±.0270 | .6493<br>±.0292 | .6306<br>±.0295 | .6119<br>±.0298 | 4.87 |
| qwen-math-1.5b | qwen | relative | 1 | 2 | scored | .7425<br>±.0268 | .6343<br>±.0295 | .6343<br>±.0295 | .6157<br>±.0298 | 4.84 |
| qwen-math-1.5b | qwen | absolute | 10 | 2 | scored | .7313<br>±.0271 | .5970<br>±.0300 | .6082<br>±.0299 | .5933<br>±.0301 | 4.82 |
| qwen-math-1.5b | qwen | relative | 10 | 2 | scored | .7612<br>±.0261 | .6269<br>±.0296 | .6194<br>±.0297 | .5821<br>±.0302 | 4.82 |

> **Analysis.** No data yet.
> **Limitations / follow-up:** this model is the one most likely
> to break the b=80 assumptions — its b=320 runs needed
> `max_model_len=4096` after the 6000 variants failed, so if
> these cells die on context length the fix is the same override
> and the comparison stays valid at a shorter budget. None of
> the four cells has a ledger entry; queue with hashes
> `74a4b258`, `43bd8117`, `a4a229ee`, `70b52bb3`. Feeds key:
> `tbl-78da65`.

#### model family comparison (QwenPRM, cov_scope=local)
<!-- table-id: tbl-bf15ee -->
> **Compares:** the standard 5-model family/size/quantization
> grid under per-node covariance — the local counterpart of
> `tbl-73533c` (global, `lam=0.01/ds_alpha=1`, `w_eff=10`)
> above, and the table that answers whether local scope is worth
> adopting.
>
> ⚠️ **Provisional operating point.** Authored at
> `lam=0.01, ds_alpha=0.1` (`w_eff=1`), the *predicted* local
> optimum, not a measured one. The joint sweep above is the
> gate: if it lands elsewhere, these cells must be re-derived at
> that point and the hashes below are void. Do not queue the
> remaining cells before that sweep reports.
>
> **Two rows are already `inqueue`** — qwen-3b and qwen-7b
> gptq-int4, queued 2026-07-28 at priority 1 *as sweep cells*
> (`tbl-898c25` / `tbl-fa65d4` each own a `w_eff=1` point, and
> one config is one ledger entry). They are not an early
> commitment to this operating point: if the sweeps peak
> elsewhere, these two rows stay valid at `w_eff=1` and the
> table's operating point moves out from under them. The other
> three — llama-1b, llama-3b, qwen-math-1.5b — remain gated.
>
> **The comparison this enables.** Local-vs-global must be read
> with **each scope at its own optimum** — global at `w_eff=10`
> (`tbl-73533c`), local at whatever the sweep says. Comparing
> local-at-global's-optimum against global-at-global's-optimum
> would understate local by construction, which is the whole
> reason this section exists rather than a single extra column
> in the section above.
>
> **Fixed:** method=`mcts_sem_v02`, **`cov_scope=local`**,
> `embeds_ref=absolute`, prm=qwen, bs-4, d-20, b=80,
> proj=sparse512, cov_update=sm, ds_beta=1.0, prm_batch_size=1,
> tmpl=model-family default, lam=0.01, ds_alpha=0.1
> (`w_eff=1`), data.level=5, run.num_trials=2.
>
> ⚠️ Hashes at the provisional point: llama-1b `3dc685e4`
> (**shared with both tables above**), llama-3b `08e67e7a`,
> qwen-3b `77b736ec`, qwen-7b gptq-int4 `d5a1327c`,
> qwen-math-1.5b `74a4b258`. Net new: 4 cells.
>
> **W&B:** 2xfjs5bq (llama-1b), nnf53blu (llama-3b),
> t6wgq6yh (qwen-math-1.5b).

| llm | prm | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | 2 | scored | .3284<br>±.0287 | .2575<br>±.0268 | .2276<br>±.0257 | .2015<br>±.0245 | 4.73 |
| llama-3b fp16 | qwen | 2 | scored | .5448<br>±.0305 | .4366<br>±.0304 | .4328<br>±.0303 | .4104<br>±.0301 | 6.67 |
| qwen-3b fp16 | qwen | 2 | scored | .7052<br>±.0279 | .5672<br>±.0303 | .5672<br>±.0303 | .5373<br>±.0305 | 6.12 |
| qwen-7b gptq-int4 | qwen | 2 | scored | .7425<br>±.0268 | .6194<br>±.0297 | .5970<br>±.0300 | .5821<br>±.0302 | 4.97 |
| qwen-math-1.5b fp16 | qwen | 2 | scored | .7351<br>±.0270 | .6493<br>±.0292 | .6306<br>±.0295 | .6119<br>±.0298 | 4.87 |

> **Analysis.** One of two queued rows landed. qwen-3b at
> `w_eff=1` local reaches **.7052** pass@gb against the global
> `w_eff=10` baseline of **.6903** — a +.0149 edge, which is
> inside one standard error (±.028) and so is not yet evidence
> that local scope helps. The qwen-7b row `failed` (allocation
> ended after trial 1 of 2), so the one model with the largest
> global baseline is still unmeasured under local scope.
> Global baselines at `w_eff=10`
> (`tbl-73533c`): llama-1b .3209, llama-3b .5784, qwen-3b .6903,
> qwen-7b gptq-int4 .7687, qwen-math-1.5b .7500 pass@gb.
> **Limitations / follow-up:** gated on the `w_eff` sweep, as
> flagged above. At ~4.8–7.0 hr/trial × 2 trials these five
> cells are ~55 GPU-hours. Expected shape if local works: the
> weaker policies (llama-1b, llama-3b) should benefit most,
> since a per-node covariance stops one subtree's folds from
> flattening another's bonus, and weak policies produce the most
> redundant branches.

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
| llama-3b | 0.0 | 0.1 | 2 | scored | .5224<br>±.0306 | .3731<br>±.0296 | .3545<br>±.0293 | .3396<br>±.0290 | 5.16 |

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
| qwen-3b | 1.0 | 0.1 | 2 | scored | .6045<br>±.0299 | .5224<br>±.0306 | .4888<br>±.0306 | .4701<br>±.0305 | 3.72 |
| qwen-3b | 0.8 | 2.0 | 2 | scored | .6157<br>±.0298 | .5261<br>±.0306 | .4925<br>±.0306 | .4701<br>±.0305 | 3.87 |
| qwen-3b | 0.8 | 0.5 | 2 | scored | .6269<br>±.0296 | .5261<br>±.0306 | .4813<br>±.0306 | .4813<br>±.0306 | — |
| qwen-3b | 0.8 | 0.1 | 2 | scored | .6418<br>±.0293 | .5224<br>±.0306 | .5037<br>±.0306 | .4851<br>±.0306 | 3.78 |
| qwen-3b | 0.5 | 2.0 | 2 | scored | .6269<br>±.0296 | .5112<br>±.0306 | .5000<br>±.0306 | .4851<br>±.0306 | 4.14 |
| qwen-3b | 0.5 | 0.5 | 2 | scored | .6231<br>±.0297 | .5187<br>±.0306 | .4925<br>±.0306 | .4888<br>±.0306 | 3.97 |
| qwen-3b | 0.5 | 0.1 | 2 | scored | .6269<br>±.0296 | .5485<br>±.0305 | .5187<br>±.0306 | .5149<br>±.0306 | 3.87 |
| qwen-3b | 0.0 | 2.0 | 2 | scored | .6231<br>±.0297 | .5410<br>±.0305 | .5037<br>±.0306 | .4925<br>±.0306 | 4.04 |
| qwen-3b | 0.0 | 0.5 | 2 | scored | .5896<br>±.0301 | .5373<br>±.0305 | .5075<br>±.0306 | .5112<br>±.0306 | 4.00 |
| qwen-3b | 0.0 | 0.1 | 2 | scored | .6343<br>±.0295 | .5672<br>±.0303 | .5448<br>±.0305 | .5485<br>±.0305 | 3.94 |

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
| qwen-7b gptq-int4 | 2 | scored | .7649<br>±.0260 | .5746<br>±.0303 | .4813<br>±.0306 | .4552<br>±.0305 | 5.75 |
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
| qwen-math-1.5b fp16 | qwen | 2 | scored | .8470<br>±.0220 | .6940<br>±.0282 | .6567<br>±.0291 | .6269<br>±.0296 | 13.48 |

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
| llama-3b fp16 | qwen | 2 | scored | .6754<br>±.0287 | .4291<br>±.0303 | .4627<br>±.0305 | .4440<br>±.0304 | 24.50 |
| qwen-3b fp16 | qwen | 2 | scored | .8097<br>±.0240 | .6157<br>±.0298 | .5858<br>±.0301 | .5784<br>±.0302 | 23.4 |
| qwen-7b gptq-int4 | qwen | 2 | scored | .7985<br>±.0245 | .6418<br>±.0293 | .6269<br>±.0296 | .6157<br>±.0298 | 14.6 |
| qwen-math-1.5b fp16 | qwen | 2 | scored (mml4096) | .8396<br>±.0225 | .6828<br>±.0285 | .6642<br>±.0289 | .6493<br>±.0292 | 18.93 |

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
| llama-3b fp16 | qwen | 2 | scored | .7164<br>±.0276 | .4254<br>±.0303 | .3955<br>±.0299 | .3433<br>±.0291 | 29.08 |
| qwen-3b fp16 | qwen | 2 | scored | .8396<br>±.0225 | .5896<br>±.0301 | .5560<br>±.0304 | .5299<br>±.0305 | 23.8 |
| qwen-7b gptq-int4 | qwen | 2 | scored | .8582<br>±.0213 | .6007<br>±.0300 | .6007<br>±.0300 | .5821<br>±.0302 | 19.1 |
| qwen-math-1.5b fp16 | qwen | 2 | scored (mml4096) | .8321<br>±.0229 | .6343<br>±.0295 | .6194<br>±.0297 | .6082<br>±.0299 | 18.50 |

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

#### model family comparison (b=320, QwenPRM, lam=0.01/ds_alpha=1, embeds_center_mode=local)
<!-- table-id: tbl-6a015e -->
> **Compares:** the `ds_alpha=1` b=320 table above (`tbl-900e87`,
> no centering) against the same five cells run with
> `embeds_center=true, embeds_center_mode=local` — sibling-group
> centering, where each expansion group's embeddings have that
> group's own mean subtracted before the diversity bonus is
> formed. Centering is the only axis that moves, so this is the
> b=320 counterpart of the b=80 `embeds_center_mode comparison
> (lam=0.01/ds_alpha=1)` table (`tbl-e58353`) above, which found
> no consistent effect at b=80. The open question this table
> answers: **does centering start to matter once the budget is
> large enough for the tree to spread?** At b=80 the covariance
> sees ~80 folded embeddings; at b=320 it sees 4×, so an
> uncentered common direction has far more opportunity to
> dominate `V` and flatten the bonus.
>
> **Fixed:** identical to `tbl-900e87` (method=`mcts_sem_v02`,
> prm=qwen, bs-4, d-20, b=320, prm_batch_size=1,
> `ds_alpha_schedule=global`, `cov_update=sm`, `cov_dtype=fp64`,
> `embeds_dim=512`/`embeds_proj=sparse`, ds_beta=1.0,
> tmpl=model-family default, **lam=0.01, ds_alpha=1.0**
> (`w_eff=10`), data.level=5, run.num_trials=2) except
> **`embeds_center=true` + `embeds_center_mode=local`**.
>
> **`max_model_len`:** 6000 for the four models that accept it,
> for the reason given in `tbl-900e87` — b=320 prompts overflow
> the 5000 default at level 5. **qwen-math-1.5b is at 4096**, not
> 6000: its `max_position_embeddings=4096` makes 6000 impossible
> at engine construction, so 6000 is not an authorable cell for
> that model. 4096 is the value its `tbl-900e87` counterpart is
> actually running at (`cfg-06533f44`), so the two qwen-math rows
> still pair — but note the pair is a 4096↔4096 comparison while
> the other four are 6000↔6000.
>
> ⚠️ All five cells are `planned` — authored 2026-07-28, **not
> queued**, so no ledger entries exist yet. Config hashes are
> resolved and recorded below so they can be queued without
> re-deriving: llama-1b `54eb87b0`, llama-3b `ba3267b4`, qwen-3b
> `991731e8`, qwen-7b gptq-int4 `2953ac47`, qwen-math-1.5b
> (mml4096) `db906620`.
>
> ⚠️ **Cost before value:** at ~15–29 hr/trial × 2 trials × 5
> cells this table is ~200–290 GPU-hours, and its b=80 twin found
> every centering gap inside ~1 SEM. Worth queueing a subset
> first — see the follow-up note below.
>
> **W&B:** `jsbk9eds` (llama-1b), `etotz0u8` (llama-3b),
> `d84v1o1a` (qwen-3b), `t3qf0n2i` (qwen-7b gptq-int4),
> `57bxhktv` (qwen-math-1.5b).

| llm | prm | center | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | local | 2 | scored | .5261<br>±.0306 | .2687<br>±.0271 | .2388<br>±.0261 | .1940<br>±.0242 | 19.96 |
| llama-3b fp16 | qwen | local | 2 | scored | .7090<br>±.0278 | .4478<br>±.0304 | .4216<br>±.0302 | .4067<br>±.0301 | 27.79 |
| qwen-3b fp16 | qwen | local | 2 | scored | .8284<br>±.0231 | .5858<br>±.0301 | .5448<br>±.0305 | .5373<br>±.0305 | 24.84 |
| qwen-7b gptq-int4 | qwen | local | 2 | scored | .8582<br>±.0213 | .5896<br>±.0301 | .5634<br>±.0304 | .5597<br>±.0304 | 18.27 |
| qwen-math-1.5b fp16 | qwen | local | 2 | scored | .8507<br>±.0218 | .6455<br>±.0293 | .6381<br>±.0294 | .6269<br>±.0296 | 19.49 |

> **Analysis.** Complete (5/5, closed 2026-08-02). Read each row
> against its `tbl-900e87` twin (same model, same everything but
> centering): llama-1b .5187, qwen-3b .8097, qwen-7b gptq-int4
> .7985 pass@gb are the uncentered baselines; llama-3b and
> qwen-math are still running there, so those two pairs cannot
> be read until `tbl-900e87` completes.
> On the three readable pairs the centering gain **grows with
> model strength**: llama-1b +.0074 (.5187 → .5261), qwen-3b
> +.0187 (.8097 → .8284), qwen-7b gptq-int4 **+.0597** (.7985 →
> .8582, ~1.8 SE). Only the qwen-7b gap reaches the ~.06
> resolution floor this table was warned about, so it is the one
> cell that is even arguably non-null.
> **This inverts the hypothesis the table was queued to test.**
> llama-1b was picked because it showed the largest b=80
> centering gain (.3209 → .3806) and was expected to show the
> effect most clearly at b=320; it instead shows the smallest
> gain of the three, essentially zero. The b=80 result did not
> reproduce at b=320 for this model, and whatever centering buys
> at large budget, it is not a weak-model effect.
> **Limitations / follow-up:** with two of five pairs still
> waiting on `tbl-900e87`, the "grows with model strength"
> reading rests on three points and one ordering; llama-3b and
> qwen-math are exactly the rows that would break or confirm it,
> and qwen-math (.8507, the second-best pass@gb here) is the
> cheapest of the two to finish. The b=80 twin (`tbl-e58353`)
> found every `local`-vs-`none` gap within ~1 SEM at 2 trials,
> so a null result here is the likely outcome and 2 trials/cell
> will not resolve anything smaller than ~.06. If the point is
> to test the budget-dependence hypothesis rather than to fill a
> grid, **queue qwen-3b and llama-1b first** — qwen-3b is the
> strongest uncentered b=320 cell and llama-1b showed the
> largest (if insignificant) b=80 centering gain (.3209→.3806),
> so they bracket the effect; the other three only become worth
> their ~150 GPU-hours if that pair moves. The qwen-math row
> additionally depends on the open 4096-window question in
> `tbl-900e87`; if 4096 turns out to be insufficient at b=320,
> that cell dies with its twin.

#### model family comparison (b=320, QwenPRM, lam=0.01/ds_alpha=10, embeds_center_mode=local)
<!-- table-id: tbl-560ce2 -->
> **Compares:** same as the `ds_alpha=1, embeds_center_mode=local`
> table above (`tbl-6a015e`), at the next `w_eff` checkpoint —
> **ds_alpha=10**, `w_eff = ds_alpha/sqrt(lam) = 100`. It is the
> centered counterpart of `tbl-01c466` (b=320, ds_alpha=10, no
> centering) and completes a 2×2: {ds_alpha 1, 10} ×
> {center none, local} at b=320. With all four tables filled,
> the centering effect can be read *at each diversity weight
> separately*, which matters because the two knobs plausibly
> interact: `ds_alpha=10` already pushes selection toward
> spread-out branches, and centering removes the common
> direction that would otherwise inflate every embedding's
> apparent novelty — so if centering ever helps, the
> high-diversity regime is where it should show up least (the
> bonus is already dominating) or most (the bonus is already
> mis-calibrated), and those two predictions are
> distinguishable.
>
> **Fixed:** identical to `tbl-6a015e` (method=`mcts_sem_v02`,
> prm=qwen, bs-4, d-20, b=320, prm_batch_size=1,
> `ds_alpha_schedule=global`, `cov_update=sm`, `cov_dtype=fp64`,
> `embeds_dim=512`/`embeds_proj=sparse`, ds_beta=1.0,
> tmpl=model-family default, **`embeds_center=true` +
> `embeds_center_mode=local`**, lam=0.01, data.level=5,
> run.num_trials=2) except **ds_alpha=10** (`w_eff=100`).
>
> **`max_model_len`:** 6000 for four models; **qwen-math-1.5b at
> 4096** for the same architectural reason as `tbl-6a015e`
> (`max_position_embeddings=4096` makes 6000 unstartable). Its
> `tbl-01c466` twin is likewise running at 4096
> (`cfg-9748b857`), so the pair is 4096↔4096 while the other
> four are 6000↔6000.
>
> Authored 2026-07-28 with all five cells `planned`; four are
> now `scored` and llama-3b is still running (2026-08-03).
> Resolved config hashes: llama-1b `d14628de`, llama-3b
> `9d60bc89`, qwen-3b `965e3b55`, qwen-7b gptq-int4 `038366a7`,
> qwen-math-1.5b (mml4096) `7e4e73e8`.
>
> ⚠️ **Cost:** ~19–29 hr/trial × 2 trials × 5 cells ≈ 200–290
> GPU-hours, on top of `tbl-6a015e`'s comparable bill. The two
> tables together are ~400–580 GPU-hours; queue selectively.
>
> ⚠️ **Baseline caveat inherited from the b=80 twin:** the b=80
> `ds_alpha=10` center-mode table (`tbl-2e75f2`) has three of
> its five `none` baselines flagged as unverified (recompute
> disagreed with the doc beyond rounding, and the doc was not
> overwritten). That flag is about b=80 bookkeeping and does not
> touch this table's own cells — but it does mean the b=80
> `ds_alpha=10` centering result is a weaker prior than the
> `ds_alpha=1` one.
>
> **W&B:** `6bf6dagw` (llama-1b), `ovav837s` (llama-3b),
> `g8gac5e5` (qwen-3b), `qpcd4vnh` (qwen-7b gptq-int4),
> `tttarieg` (qwen-math-1.5b, mml4096).

| llm | prm | center | trials | status | pass@gb | naive@gb | wei@gb | maj@gb | hr/trial |
|---|---|---|---|---|---|---|---|---|---|
| llama-1b fp16 | qwen | local | 2 | scored | .5037<br>±.0306 | .2985<br>±.0280 | .2575<br>±.0268 | .2090<br>±.0249 | 19.66 |
| llama-3b fp16 | qwen | local | 2 | scored | .7201<br>±.0275 | .4590<br>±.0305 | .3955<br>±.0299 | .3545<br>±.0293 | 27.67 |
| qwen-3b fp16 | qwen | local | 2 | scored | .8134<br>±.0238 | .6082<br>±.0299 | .5634<br>±.0304 | .5336<br>±.0305 | 25.07 |
| qwen-7b gptq-int4 | qwen | local | 2 | scored | .8694<br>±.0206 | .6119<br>±.0298 | .5672<br>±.0303 | .5522<br>±.0304 | 19.00 |
| qwen-math-1.5b fp16 | qwen | local | 2 | scored | .8321<br>±.0229 | .6381<br>±.0294 | .6343<br>±.0295 | .6082<br>±.0299 | 19.46 |

> **Analysis.** Complete (5/5, closed 2026-08-03). Read against
> the uncentered `tbl-01c466` twins at the same `w_eff=100`:
> llama-1b .5037 vs .5373, llama-3b .7201 vs .7164, qwen-3b
> .8134 vs .8396, qwen-7b gptq-int4 .8694 vs .8582. **Centering
> does not help at b=320.** Three of four measured pairs move
> against `local` (−.034, −.026, and llama-1b's −.034) and the
> one gain is +.011 on qwen-7b — every gap is inside ~1.2 SE, so
> the honest read is *no effect*, with the point estimates
> leaning slightly negative.
> **That kills the b=80 hint.** `tbl-2e75f2`'s two clean pairs
> both trended *up* under `local` (llama-3b .5485→.5896, qwen-3b
> .6642→.6903); at 4× the budget the same two models go the
> other way (llama-3b +.004, qwen-3b −.026). A hint that
> reverses sign when the budget grows is noise, not a
> budget-dependent effect. Model ranking is unchanged from every
> other b=320 table: qwen-7b > qwen-math > qwen-3b > llama-3b >
> llama-1b on pass@gb.
> **Limitations / follow-up:** n≈267 pooled over 2 trials, and
> the `tbl-01c466` qwen-math twin is still unscored, so one of
> the five pairs cannot be formed. The pass@gb-vs-maj@gb split
> is worth noting independently of centering — llama-3b reaches
> .7201 pass but only .3545 maj, the widest gap in the table,
> which says the b=320 budget is finding correct branches this
> model's aggregation cannot pick out. Nothing further is needed
> for the centering question: at two budgets and five models it
> has not produced an effect worth chasing.

---
